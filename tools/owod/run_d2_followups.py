#!/usr/bin/env python
"""Matched D2 replay controls and an independent 30-epoch GNN run."""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import math
import os
from pathlib import Path
import random
import shlex
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import stage_files
from tools.owod.run_stage1_diagnostics import file_hash, metric_rows, split_command, training_environment
from tools.owod.run_paper_baselines import (busy_gpus, payload_hash, queue_lock,
    read_json, run_child, validate_fingerprints, write_json)

PILOT = ROOT / 'exps/owod/m-owodb/order0/pilot_unverified'
ARMS = {'random': 'd2_random_k', 'long': 'd2_gnn_30ep', 'cosine': 'd2_cosine_k'}


def argument_schema():
    """Read argparse declarations without importing the CUDA training entry point."""
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == 'get_args_parser')
    result = {}
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != 'add_argument':
            continue
        flags = [arg.value for arg in node.args if isinstance(arg, ast.Constant)
                 and isinstance(arg.value, str) and arg.value.startswith('--')]
        if not flags:
            continue
        keywords = {kw.arg: kw.value for kw in node.keywords}
        dest = ast.literal_eval(keywords['dest']) if 'dest' in keywords else flags[0][2:].replace('-', '_')
        action = ast.literal_eval(keywords['action']) if 'action' in keywords else 'store'
        result[dest] = (flags[0], action)
    return result


def matched_command(recorded, output, annotation, epochs, port, gpus):
    # Pin recorded defaults as explicit arguments, including backbone LR, teacher
    # settings, augmentation and optimizer multipliers. Do not inherit new defaults.
    ignored = {'resume', 'output_dir', 'train_ann', 'epochs', 'start_epoch', 'eval',
               'log_file', 'append_log', 'no_file_log', 'world_size', 'dist_url',
               'local_rank', 'rank', 'gpu', 'distributed'}
    command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
               f'--nproc_per_node={len(gpus.split(","))}', '--master_addr=127.0.0.1',
               f'--master_port={port}', str(ROOT / 'main.py')]
    for dest, (flag, action) in argument_schema().items():
        if dest in ignored or dest not in recorded or recorded[dest] is None:
            continue
        value = recorded[dest]
        if action in ('store_true', 'store_false'):
            if bool(value) == (action == 'store_true'):
                command.append(flag)
        elif isinstance(value, (list, tuple)):
            if value:
                command.extend([flag, *map(str, value)])
        else:
            command.extend([flag, str(value)])
    command.extend(['--output_dir', str(output), '--train-ann', str(annotation),
                    '--epochs', str(epochs), '--start_epoch', '0', '--no-file-log'])
    return command


def select_images(old, old_ids, selected_classes, base, extra, seed, budget=None, current_ids=()):
    """Original independent quotas, followed by an explicit unique-budget repair.

    Base exemplars are protected when trimming; refill visits risk classes first.
    The repair is identity for the reconstructed historical GNN arm.
    """
    by_class = defaultdict(set)
    for ann in old['annotations']:
        by_class[ann['category_id']].add(ann['image_id'])
    rng = random.Random(seed)
    selected, protected, pools = set(), set(), {}
    for category in sorted(old_ids):
        pool = sorted(by_class[category])
        rng.shuffle(pool)
        pools[category] = pool
        quota = base + (extra if category in selected_classes else 0)
        selected.update(pool[:quota])
        protected.update(pool[:base])
    current_ids = set(current_ids)
    initial = len(selected - current_ids)
    if budget is not None:
        if len(protected - current_ids) > budget:
            raise ValueError('Replay budget cannot preserve all base exemplars')
        removable = sorted(selected - protected - current_ids)
        random.Random(seed + 100003).shuffle(removable)
        selected.difference_update(removable[:max(0, initial - budget)])
        priority = sorted(selected_classes) + sorted(set(old_ids) - set(selected_classes))
        cursors = {category: 0 for category in priority}
        while len(selected - current_ids) < budget:
            before = len(selected)
            for category in priority:
                pool = pools[category]
                index = cursors[category]
                while index < len(pool) and (pool[index] in selected or pool[index] in current_ids):
                    index += 1
                if index < len(pool):
                    selected.add(pool[index])
                    index += 1
                cursors[category] = index
                if len(selected - current_ids) == budget:
                    break
            if len(selected) == before:
                raise ValueError('Insufficient old images to match the D2 replay budget')
    return selected, {'before_repair': initial, 'after_repair': len(selected - current_ids),
                      'overlap_current_images': len(selected & current_ids),
                      'added': max(0, (budget if budget is not None else initial) - initial),
                      'removed': max(0, initial - (budget if budget is not None else initial))}


def merge_annotation(current, old, selected):
    """Same COCO image/annotation ordering and overlap handling as historical D2."""
    images = [{**image, 'owod_replay': False} for image in current['images']]
    names = {image['id']: image['file_name'] for image in images}
    for image in old['images']:
        if image['id'] not in selected:
            continue
        if image['id'] in names:
            if names[image['id']] != image['file_name']:
                raise ValueError('COCO image ID collision across different filenames')
        else:
            images.append({**image, 'owod_replay': True})
            names[image['id']] = image['file_name']
    annotations, used = [], set()
    for ann in current['annotations'] + [a for a in old['annotations'] if a['image_id'] in selected]:
        key = (ann['image_id'], ann['id'])
        if key not in used:
            annotations.append({**ann, 'id': len(annotations) + 1})
            used.add(key)
    categories = {cat['id']: cat for cat in old['categories'] + current['categories']}
    return {**current, 'images': images, 'annotations': annotations,
            'categories': [categories[key] for key in sorted(categories)]}


def cosine_neighbors(checkpoint_path, old_ids, new_ids, k, aggregation, top_n):
    import torch
    from torch.nn import functional as F
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint['model']
    keys = [key for key in state if key.startswith('class_embed.') and key.endswith('.weight')]
    if not keys:
        raise ValueError('Stage 0 checkpoint has no classifier rows for cosine control')
    key = sorted(keys, key=lambda value: int(value.split('.')[1]) if value.split('.')[1].isdigit() else -1)[-1]
    rows = state[key].float()
    if rows.ndim != 2 or max(old_ids + new_ids) >= len(rows) or not torch.isfinite(rows).all():
        raise ValueError('Invalid classifier rows in Stage 0 checkpoint')
    scores = F.normalize(rows[new_ids], dim=1) @ F.normalize(rows[old_ids], dim=1).T
    risk = scores.max(0).values if aggregation == 'max' else scores.topk(min(top_n, len(new_ids)), dim=0).values.mean(0)
    values = {category: float(risk[index]) for index, category in enumerate(old_ids)}
    selected = sorted(old_ids, key=lambda category: (-values[category], category))[:k]
    return selected, {'classifier_key': key, 'risk': {str(c): v for c, v in values.items()},
                      'aggregation': aggregation, 'top_n': top_n}


def complete(directory, epochs):
    marker = directory / 'training_complete.json'
    if not marker.is_file():
        return False
    row = metric_rows(directory / 'metrics.jsonl').get(epochs - 1, {})
    return (read_json(marker).get('last_epoch') == epochs - 1
            and (directory / 'checkpoint.pth').is_file()
            and all(isinstance(row.get('test_owod_' + key), (int, float))
                    and math.isfinite(row['test_owod_' + key])
                    for key in ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'h_score')))


def create_plans(args):
    d2 = args.d2_dir
    recorded = read_json(d2 / 'run_config.json')
    source_plan = read_json(d2.parent / 'diagnostic_plan.json')
    source = Path(source_plan['source_run'])
    metadata = read_json(source / 'run_config.json')
    if source_plan.get('experiment') != 'd2' or not complete(d2, 20):
        raise ValueError('Require completed original D2 with final metrics, checkpoint and diagnostic plan')
    for field, value in {'owod_stage': 1, 'epochs': 20, 'lr_drop': 15, 'lr': 1e-4,
                         'batch_size': 2, 'seed': 42, 'teacher_completion': True}.items():
        if recorded.get(field) != value:
            raise ValueError(f'Unexpected D2 configuration: {field}={recorded.get(field)}')
    if recorded.get('lr_backbone', 0) <= 0 or not 0 < recorded.get('replay_sampling_fraction', 0) < 1:
        raise ValueError('D2 must update backbone and use balanced replay')
    for field in ('neighbor_scoped_lora', 'with_tree', 'old_class_distillation', 'paper_baseline',
                  'local_margin_coef', 'off_projection_coef', 'skip_eval', 'frozen_weights', 'trainable_class_ids'):
        if recorded.get(field):
            raise ValueError(f'Unexpected active D2 setting: {field}')
    _, original_options = split_command(source_plan['command'])
    for field, (flag, action) in argument_schema().items():
        if flag not in original_options or field in ('output_dir', 'resume'):
            continue
        values = original_options[flag]
        actual = recorded.get(field)
        if action in ('store_true', 'store_false'):
            equal = actual == (action == 'store_true')
        else:
            expected = actual if isinstance(actual, list) else [actual]
            equal = len(values) == len(expected) and all(str(a) == str(b) or
                (isinstance(b, (int, float)) and float(a) == b) for a, b in zip(values, expected))
        if not equal:
            raise ValueError(f'D2 plan/config mismatch: {flag}')
    fingerprints = {}
    for item in source_plan['inputs'].values():
        path = Path(item['path'])
        if file_hash(path) != item['sha256']:
            raise ValueError(f'Original D2 input changed: {path}')
        fingerprints[str(path)] = item['sha256']
    manifest_path = Path(recorded['owod_manifest'])
    manifest, files = stage_files(manifest_path, 1, allow_unverified=True)
    _, prior = stage_files(manifest_path, 0, allow_unverified=True)
    current, old = read_json(files['increment_train']), read_json(prior['train'])
    d2_data = read_json(recorded['train_ann'])
    old_ids, new_ids = recorded['owod_previous_class_ids'], recorded['owod_current_class_ids']
    if (set(manifest['stages'][1]['classes']) != set(new_ids)
            or set(old_ids) & set(new_ids)
            or set(recorded['owod_known_class_ids']) != set(old_ids + new_ids)
            or not {ann['category_id'] for ann in current['annotations']} <= set(new_ids)
            or not {ann['category_id'] for ann in old['annotations']} <= set(old_ids)):
        raise ValueError('Manifest or annotation class sets differ from D2')
    selected_gnn = metadata['selected_replay_classes']
    if selected_gnn != source_plan['selected_replay_classes']:
        raise ValueError('GNN selection differs from the D2 plan')
    base, extra = metadata['base_exemplars_per_class'], metadata['risk_extra_exemplars_per_class']
    if (base <= 0 or extra <= 0 or not 0 < len(selected_gnn) <= len(old_ids)
            or len(set(selected_gnn)) != len(selected_gnn) or not set(selected_gnn) <= set(old_ids)):
        raise ValueError('Require the original base-plus-risk replay policy')
    current_ids = {image['id'] for image in current['images']}
    budget = sum(bool(image.get('owod_replay')) for image in d2_data['images'])
    original_ids, _ = select_images(old, old_ids, selected_gnn, base, extra, recorded['seed'])
    rebuilt = merge_annotation(current, old, original_ids)
    if any(rebuilt[field] != d2_data[field] for field in ('images', 'annotations', 'categories')):
        raise ValueError('Cannot reconstruct D2 annotation exactly; inspect source data before running controls')
    aggregation, top_n = metadata['graph_aggregation'], metadata['graph_aggregation_top_n']
    if aggregation not in ('max', 'top_mean') or top_n < 1:
        raise ValueError('Unsupported GNN aggregation')
    # Keep histories and resumed runs reproducible even when module defaults change.
    paths = [d2 / 'run_config.json', d2.parent / 'diagnostic_plan.json', source / 'run_config.json',
             files['increment_train'], prior['train'], Path(__file__), ROOT / 'main.py', ROOT / 'engine.py',
             ROOT / 'tools/owod/protocol.py', ROOT / 'tools/owod/run_stage1_diagnostics.py',
             ROOT / 'tools/owod/run_paper_baselines.py']
    for folder in ('models', 'datasets', 'util'):
        paths.extend((ROOT / folder).rglob('*.py'))
    fingerprints.update({str(path): file_hash(path) for path in paths})
    results = []
    for arm in args.experiments:
        arm_dir = args.output_dir / ARMS[arm]
        epochs = 30 if arm == 'long' else 20
        selection_info = {}
        if arm == 'long':
            selected, annotation, payload = selected_gnn, Path(recorded['train_ann']), None
            replay_info = {'before_repair': budget, 'after_repair': budget, 'added': 0, 'removed': 0}
        else:
            if arm == 'random':
                selected = random.Random(recorded['seed']).sample(sorted(old_ids), len(selected_gnn))
            else:
                selected, selection_info = cosine_neighbors(Path(recorded['pretrained']), old_ids, new_ids,
                                                            len(selected_gnn), aggregation, top_n)
            image_ids, replay_info = select_images(old, old_ids, selected, base, extra,
                                                   recorded['seed'], budget, current_ids)
            payload = merge_annotation(current, old, image_ids)
            if len(payload['images']) != len(d2_data['images']):
                raise ValueError('Control dataset length differs from D2; epoch exposure would change')
            annotation = arm_dir / 'train.json'
            replay_info['selected_old_image_ids'] = sorted(image_ids)
            replay_info['annotations_by_class'] = {str(c): sum(a['category_id'] == c for a in payload['annotations'])
                                                   for c in old_ids + new_ids}
        command = matched_command(recorded, arm_dir / 'graph', annotation, epochs, args.master_port, args.gpus)
        count_per_rank = math.ceil(len(d2_data['images']) / len(args.gpus.split(',')))
        plan = {'schema': 1, 'arm': arm, 'epochs': epochs, 'command': command, 'gpus': args.gpus,
                'd2_source': str(d2), 'paper_comparable': False, 'fingerprints': fingerprints,
                'selection': selected, 'selection_info': selection_info, 'replay': replay_info,
                'base_quota': base, 'risk_extra_quota': extra,
                'dataset_images': len(d2_data['images']), 'replay_images': budget,
                'samples_per_rank': count_per_rank,
                'replay_draws_per_rank_before_drop_last': max(1, round(count_per_rank * recorded['replay_sampling_fraction'])),
                'optimizer_steps_per_epoch': count_per_rank // recorded['batch_size'],
                'annotation_sha256': payload_hash(payload) if payload is not None else file_hash(annotation),
                'long_schedule_note': 'Independent Stage 0 initialization; StepLR period 15; extra training budget'}
        results.append((arm_dir, plan, payload))
    # Existence checks are CPU-only, and include images used by every control.
    check_payloads = [d2_data, read_json(recorded['val_ann'])] + [p for _, _, p in results if p is not None]
    missing = []
    for index, data in enumerate(check_payloads):
        folder = 'val2017' if index == 1 else 'train2017'
        for image in data['images']:
            path = Path(recorded['coco_path']) / folder / image['file_name']
            if not path.is_file():
                missing.append(str(path))
                if len(missing) == 3:
                    raise ValueError(f'Missing COCO images: {missing}')
    if missing:
        raise ValueError(f'Missing COCO images: {missing}')
    return results


def check_existing(directory, plan, resume):
    saved = directory / 'plan.json'
    if saved.is_file():
        if read_json(saved) != plan:
            raise ValueError(f'Plan/code/input changed: {directory}; choose a new output directory')
        annotation = directory / 'train.json'
        if plan['arm'] != 'long' and (not annotation.is_file() or file_hash(annotation) != plan['annotation_sha256']):
            raise ValueError(f'Generated annotation changed: {annotation}')
        graph = directory / 'graph'
        if complete(graph, plan['epochs']):
            return True
        if (graph / 'training_complete.json').exists():
            raise ValueError(f'Inconsistent completion artifacts: {graph}')
        if not resume or not (graph / 'checkpoint.pth').is_file():
            raise ValueError(f'Incomplete run: {directory}; --resume requires its own checkpoint')
    elif directory.exists() and any(directory.iterdir()):
        raise ValueError(f'Nonempty output without plan: {directory}')
    return False


def require_idle(gpus):
    busy = busy_gpus(gpus)
    if busy:
        raise RuntimeError(f'GPUs {gpus} occupied: {busy}; no new training started. Finish the existing queue first.')


def summarize(d2, output):
    reference = metric_rows(d2 / 'metrics.jsonl')
    print('AP50 / U-Recall / H in percent; delta against D2 at SAME epoch; long >20 has no matched reference.')
    print('run epoch previous current known U-Recall H delta_previous delta_current status')
    for label, directory, epochs in [('d2', d2, 20)] + [(a, output / n / 'graph', 30 if a == 'long' else 20) for a, n in ARMS.items()]:
        for epoch, row in sorted(metric_rows(directory / 'metrics.jsonl').items()):
            values = [f"{100 * row['test_owod_' + key]:.3f}" if 'test_owod_' + key in row else 'NA'
                      for key in ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'h_score')]
            ref = reference.get(epoch, {})
            delta = [f'{100 * (row[key] - ref[key]):+.3f}' if key in row and key in ref else 'NA'
                     for key in ('test_owod_previous_ap50', 'test_owod_current_ap50')]
            print(label, epoch + 1, *values, *delta, 'complete' if complete(directory, epochs) else 'incomplete')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d2-dir', type=Path, default=PILOT / 'stage1_diagnostics_v2/d2_full_finetune_no_extra_losses/graph')
    parser.add_argument('--output-dir', type=Path, default=PILOT / 'd2_followups_v1')
    parser.add_argument('--experiments', nargs='+', choices=tuple(ARMS), default=['random', 'long'],
                        help='Default: D2 Random-K control then D2-long; cosine is opt-in only')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--master-port', type=int, default=29583)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(argv)
    args.d2_dir, args.output_dir = args.d2_dir.resolve(), args.output_dir.resolve()
    if args.summarize:
        summarize(args.d2_dir, args.output_dir)
        return 0
    devices = args.gpus.split(',')
    if len(devices) != 2 or len(set(devices)) != 2 or not all(d.isdigit() for d in devices):
        raise ValueError('D2 controls require two distinct numeric GPU indices')
    if len(set(args.experiments)) != len(args.experiments):
        raise ValueError('Duplicate experiment names')
    plans = create_plans(args)
    for directory, plan, _ in plans:
        check_existing(directory, plan, args.resume)
        print(f"{plan['arm']}: selected={plan['selection']}, replay={plan['replay_images']}, "
              f"budget repair +{plan['replay']['added']}/-{plan['replay']['removed']}, epochs={plan['epochs']}")
        print(shlex.join(plan['command']))
    if args.dry_run:
        print('Dry run passed. No training/output files created. Original D2 annotation reconstructed exactly.')
        return 0
    environment = training_environment(args.gpus)
    with queue_lock(args.output_dir):
        for directory, plan, payload in plans:
            if check_existing(directory, plan, args.resume):
                print(f'Skipping completed {directory}')
                continue
            require_idle(args.gpus)
            validate_fingerprints(plan)
            graph = directory / 'graph'
            graph.mkdir(parents=True, exist_ok=True)
            write_json(directory / 'plan.json', plan)
            if payload is not None:
                write_json(directory / 'train.json', payload)
            command = list(plan['command'])
            if args.resume and (graph / 'checkpoint.pth').is_file():
                command.extend(['--resume', str(graph / 'checkpoint.pth')])
            write_json(args.output_dir / 'queue_status.json', {'status': 'running', 'arm': plan['arm'], 'pid': os.getpid()})
            try:
                run_child(command, graph / 'train.log', environment)
                if not complete(graph, plan['epochs']):
                    raise RuntimeError(f'Training exited without valid final artifacts: {graph}')
            except BaseException as error:
                write_json(args.output_dir / 'queue_status.json', {'status': 'failed', 'arm': plan['arm'], 'error': str(error)})
                raise
            summarize(args.d2_dir, args.output_dir)
        write_json(args.output_dir / 'queue_status.json', {'status': 'complete', 'experiments': args.experiments})
    return 0


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        raise SystemExit(str(error))
