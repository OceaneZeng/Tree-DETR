#!/usr/bin/env python
"""Paired, task-local GLCA / uniform replay / uniform + teacher experiments."""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import math
import os
from pathlib import Path
import random
import shlex
import signal
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import stage_files
from tools.owod.run_d2_followups import matched_command, require_idle
from tools.owod.run_paper_baselines import (
    payload_hash, queue_lock, read_json, run_child, validate_fingerprints, write_json,
)
from tools.owod.run_stage1_diagnostics import file_hash, metric_rows, training_environment

METHODS = ('glca', 'uniform', 'uniform_teacher')
LABELS = {'glca': 'GLCA-DETR', 'uniform': 'Uniform Replay',
          'uniform_teacher': 'Uniform Replay + teacher supervision'}
DEFAULT_SOURCE = ROOT / 'exps/owod/m-owodb/order0/pilot_unverified/stage1_diagnostics_v2/d2_full_finetune_no_extra_losses/graph'
DEFAULT_OUTPUT = ROOT / 'exps/owod/m-owodb/order0/pilot_unverified/replay_comparison_v1'
RUNTIME_FIELDS = {'output_dir', 'train_ann', 'resume', 'start_epoch', 'eval', 'log_file',
                  'append_log', 'no_file_log', 'world_size', 'dist_url', 'local_rank',
                  'rank', 'gpu', 'distributed'}


def detector_defaults():
    """Use the real argument parser without importing torch/CUDA extensions."""
    tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'get_args_parser')
    namespace = {'argparse': argparse, 'np': SimpleNamespace(pi=math.pi)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(ROOT / 'main.py'), 'exec'), namespace)
    return vars(namespace['get_args_parser']().parse_args([]))


def subset(data, ids):
    return {**data, 'images': [dict(i) for i in data['images'] if i['id'] in ids],
            'annotations': [dict(a) for a in data['annotations'] if a['image_id'] in ids]}


def uniform_annotation(reference, current, old, old_classes, seed):
    """Uniform without replacement over unique old images outside the increment.

    Keep all non-replay supervision byte-for-byte equivalent (apart from annotation
    IDs). Historical protocols can have overlapping images and extra old GT there;
    these labels are fixed across all arms, not silently added/removed by sampling.
    """
    current_ids = {i['id'] for i in current['images']}
    fixed_ids = {i['id'] for i in reference['images'] if not i.get('owod_replay', False)}
    if fixed_ids != current_ids:
        raise ValueError('Reference current images differ from the manifest increment')
    fixed = subset(reference, fixed_ids)
    names = {i['id']: i['file_name'] for i in reference['images']}
    if any(names[i['id']] != i['file_name'] for i in current['images']):
        raise ValueError('Current image ID/filename mismatch')
    old_names = {i['id']: i['file_name'] for i in old['images']}
    if any(names[i] != old_names[i] for i in fixed_ids & old_names.keys()):
        raise ValueError('Old/current image ID collision')
    old_classes = set(old_classes)
    if not {a['category_id'] for a in old['annotations']} <= old_classes:
        raise ValueError('Replay pool contains future/current class supervision')
    eligible = {a['image_id'] for a in old['annotations'] if a['category_id'] in old_classes}
    if not eligible <= old_names.keys():
        raise ValueError('Replay annotation references a missing image')
    candidates = sorted(eligible - current_ids)
    replay_ids = {i['id'] for i in reference['images'] if i.get('owod_replay', False)}
    if not replay_ids or not replay_ids <= set(candidates):
        raise ValueError('Reference replay images must belong to the same old image pool')
    if any(names[i] != old_names[i] for i in replay_ids):
        raise ValueError('Reference replay image ID/filename mismatch')
    chosen = set(random.Random(seed).sample(candidates, len(replay_ids)))
    replay = subset(old, chosen)
    # Stable image index order is important for identical sampler exposure.
    payload = {**reference,
               'images': [{**i, 'owod_replay': False} for i in fixed['images']]
                         + [{**i, 'owod_replay': True} for i in sorted(replay['images'], key=lambda i: i['id'])],
               'annotations': [{**a, 'id': j + 1} for j, a in enumerate(
                   fixed['annotations'] + replay['annotations'])]}
    # Reject missing/altered current labels, ignoring regenerated COCO annotation IDs.
    def boxes(annotations):
        return Counter(payload_hash({k: v for k, v in a.items() if k != 'id'}) for a in annotations)
    fixed_new = [a for a in fixed['annotations'] if a['category_id'] not in old_classes]
    if boxes(fixed_new) != boxes(current['annotations']):
        raise ValueError('Reference current supervision differs from the manifest increment')
    # GLCA replay must also use the same complete old labels as the uniform arms.
    if boxes(subset(reference, replay_ids)['annotations']) != boxes(subset(old, replay_ids)['annotations']):
        raise ValueError('Reference replay labels differ from the old pool')
    info = {'policy': 'uniform_unique_old_images_without_replacement',
            'candidate_images': len(candidates), 'selected_image_ids': sorted(chosen),
            'excluded_current_overlap_images': len(eligible & current_ids),
            'fixed_old_annotations_on_current_images': sum(a['category_id'] in old_classes for a in fixed['annotations']),
            'replay_images': len(chosen),
            'replay_annotations_by_class': dict(sorted(Counter(str(a['category_id']) for a in replay['annotations']).items()))}
    return payload, info


def graph_provenance(source, config):
    diagnostic = source.parent / 'diagnostic_plan.json'
    graph_source = source.parent
    paths = []
    if diagnostic.is_file():
        plan = read_json(diagnostic)
        if plan.get('experiment') != 'd2':
            raise ValueError('Diagnostic source must be the D2 GLCA recipe')
        graph_source = Path(plan['source_run'])
        paths.append(diagnostic)
        for item in plan.get('inputs', {}).values():
            if file_hash(item['path']) != item['sha256']:
                raise ValueError(f"Historical GLCA input changed: {item['path']}")
            paths.append(Path(item['path']))
    metadata_path = graph_source / 'run_config.json'
    metadata = read_json(metadata_path)
    if metadata.get('graph_estimator') != 'gnn':
        raise ValueError('Source must have GNN replay provenance (graph_estimator=gnn)')
    if metadata.get('checkpoint') and Path(metadata['checkpoint']).resolve() != Path(config['pretrained']).resolve():
        raise ValueError('GNN selection and detector initialization use different checkpoints')
    annotation = graph_source / 'annotations/train_graph.json'
    if annotation.is_file() and file_hash(annotation) != file_hash(config['train_ann']):
        raise ValueError('Reference annotation differs from recorded GNN replay')
    paths.append(metadata_path)
    graph = graph_source / 'graph.json'
    if graph.is_file():
        paths.append(graph)
    return paths


def create_plans(args):
    defaults = detector_defaults()
    devices = args.gpus.split(',')
    if not devices or len(set(devices)) != len(devices) or not all(d.isdigit() for d in devices):
        raise ValueError('Supply distinct numeric GPU indices')
    if args.seeds is not None and (not args.seeds or len(set(args.seeds)) != len(args.seeds)):
        raise ValueError('Supply distinct seeds')
    plans, seen, hashes = [], set(), {}
    manifest_digest = None
    code = [ROOT / 'main.py', ROOT / 'engine.py']
    for folder in ('models', 'datasets', 'util', 'tools/owod'):
        code.extend(p for p in (ROOT / folder).rglob('*.py') if 'tests' not in p.parts)
    for source in args.source_dirs:
        source = source.resolve()
        recorded = read_json(source / 'run_config.json')
        config = {**defaults, **recorded}
        stage = config['owod_stage']
        if stage not in (1, 2, 3) or stage in seen:
            raise ValueError('Supply one GLCA source per incremental stage (1, 2, 3)')
        seen.add(stage)
        if (not config['teacher_completion'] or config['lr_backbone'] <= 0
                or any(config.get(k) for k in ('neighbor_scoped_lora', 'with_tree', 'old_class_distillation',
                        'paper_baseline', 'local_margin_coef', 'off_projection_coef', 'frozen_weights',
                        'trainable_class_ids', 'skip_eval', 'eval', 'reset_classifier'))):
            raise ValueError('Require D2 GLCA: ordinary fine-tuning, teacher completion, no extra losses')
        if len(devices) != recorded.get('world_size', args.reference_world_size):
            raise ValueError('GPU count must match the source world_size to preserve the effective batch/steps')
        if not 0 < config['replay_sampling_fraction'] < 1 or config['epochs'] <= 0:
            raise ValueError('Require positive epochs and a balanced replay fraction')
        manifest_path = Path(config['owod_manifest'])
        digest = file_hash(manifest_path)
        if manifest_digest is not None and digest != manifest_digest:
            raise ValueError('All source tasks must use the same split manifest')
        manifest_digest = digest
        manifest, files = stage_files(manifest_path, stage, allow_unverified=True)
        _, prior = stage_files(manifest_path, stage - 1, allow_unverified=True)
        old_ids = config['owod_previous_class_ids'] or []
        new_ids = config['owod_current_class_ids'] or []
        previous = {c for s in manifest['stages'][:stage] for c in s['classes']}
        if (set(old_ids) != previous or set(new_ids) != set(manifest['stages'][stage]['classes'])
                or set(old_ids) & set(new_ids) or set(config['owod_known_class_ids'] or []) != set(old_ids + new_ids)
                or set(config['teacher_old_class_ids'] or []) != set(old_ids)):
            raise ValueError('Manifest, known/current/previous and teacher class sets must agree')
        if file_hash(config['val_ann']) != file_hash(files['full_val']):
            raise ValueError('Use the same full-label validation annotation as the manifest')
        reference, current, old = (read_json(p) for p in (config['train_ann'], files['increment_train'], prior['train']))
        for data in (reference, current, old):
            image_ids = {i['id'] for i in data['images']}
            if len(image_ids) != len(data['images']):
                raise ValueError('Duplicate annotation image IDs')
            if not {a['image_id'] for a in data['annotations']} <= image_ids:
                raise ValueError('Annotation references a missing image')
        if (not {a['category_id'] for a in current['annotations']} <= set(new_ids)
                or not {a['category_id'] for a in reference['annotations']} <= set(old_ids + new_ids)):
            raise ValueError('Training annotations contain future or out-of-scope supervision')
        validation = read_json(config['val_ann'])
        unknown_gt = sum(a['category_id'] not in set(old_ids + new_ids) for a in validation['annotations'])
        paths = [source / 'run_config.json', Path(config['pretrained']), Path(config['train_ann']),
                 Path(config['val_ann']), manifest_path, files['increment_train'], prior['train']]
        paths += graph_provenance(source, config) + code
        for path in paths:
            if str(path) not in hashes:
                hashes[str(path)] = file_hash(path)
        fingerprints = {str(p): hashes[str(p)] for p in paths}
        for seed in args.seeds if args.seeds is not None else [config['seed']]:
            uniform, selection = uniform_annotation(reference, current, old, old_ids, seed)
            count = math.ceil(len(reference['images']) / len(devices))
            exposure = {'dataset_images': len(reference['images']), 'replay_images': selection['replay_images'],
                        'samples_per_rank': count, 'optimizer_steps_per_epoch': count // config['batch_size'],
                        'replay_draws_per_rank_before_drop_last': max(1, round(count * config['replay_sampling_fraction']))}
            if exposure['optimizer_steps_per_epoch'] < 1:
                raise ValueError('Dataset too small for a training batch')
            for method in METHODS:
                directory = args.output_dir / f'stage_{stage}' / f'seed_{seed}' / method
                payload = reference if method == 'glca' else uniform
                replay_ids = {image['id'] for image in payload['images'] if image.get('owod_replay', False)}
                replay_counts = Counter(str(a['category_id']) for a in payload['annotations'] if a['image_id'] in replay_ids)
                effective = {**config, 'seed': seed, 'teacher_completion': method != 'uniform'}
                command = matched_command(effective, directory / 'graph', directory / 'train.json',
                                          config['epochs'], args.master_port, args.gpus)
                plan = {'schema': 1, 'method': method, 'label': LABELS[method], 'stage': stage, 'task': stage + 1,
                        'seed': seed, 'epochs': config['epochs'], 'source': str(source), 'directory': str(directory),
                        'comparison_mode': 'paired_task_local_shared_initialization', 'paper_comparable': False,
                        'unknown_gt': unknown_gt, 'gpus': args.gpus, 'command': command, 'fingerprints': fingerprints,
                        'settings': {k: effective[k] for k in defaults if k not in RUNTIME_FIELDS},
                        'defaults_missing_from_source': sorted(set(defaults) - set(recorded)),
                        'exposure': exposure, 'uniform_selection': selection,
                        'actual_replay': {'policy': 'recorded_gnn' if method == 'glca' else selection['policy'],
                                          'selected_image_ids': sorted(replay_ids),
                                          'annotations_by_class': dict(sorted(replay_counts.items()))},
                        'annotation_sha256': payload_hash(payload)}
                plans.append((directory, plan, payload))
            for data, folder in ((reference, 'train2017'), (uniform, 'train2017'), (validation, 'val2017')):
                for image in data['images']:
                    path = Path(config['coco_path']) / folder / image['file_name']
                    if not path.is_file():
                        raise ValueError(f'Missing COCO image: {path}')
    return plans


def final_metrics_valid(row, plan):
    keys = ['previous_ap50', 'current_ap50', 'known_ap50']
    if plan['unknown_gt'] > 0:
        keys += ['u_recall', 'h_score']
    return all(isinstance(row.get('test_owod_' + k), (int, float))
               and math.isfinite(row['test_owod_' + k]) and 0 <= row['test_owod_' + k] <= 1 for k in keys)


def completed(directory, plan):
    graph = directory / 'graph'
    marker = graph / 'training_complete.json'
    return (marker.is_file() and read_json(marker).get('last_epoch') == plan['epochs'] - 1
            and (graph / 'checkpoint.pth').is_file()
            and final_metrics_valid(metric_rows(graph / 'metrics.jsonl').get(plan['epochs'] - 1, {}), plan))


def validate_run_settings(directory, plan):
    path = directory / 'graph/run_config.json'
    if not path.is_file():
        raise ValueError(f'Missing training run_config: {path}')
    actual = read_json(path)
    for key, expected in plan['settings'].items():
        if actual.get(key) != expected:
            raise ValueError(f'Training setting differs from plan: {path}: {key}')
    # Use recorded absolute paths so copied logs can be reported on another OS.
    expected_annotation = plan['directory'].replace('\\', '/').rstrip('/') + '/train.json'
    if (actual['train_ann'].replace('\\', '/') != expected_annotation
            or actual.get('world_size', 1) != len(plan['gpus'].split(','))):
        raise ValueError(f'Training annotation/world_size differs from plan: {path}')


def check_existing(directory, plan, resume):
    saved = directory / 'plan.json'
    if not saved.is_file():
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f'Nonempty output without plan: {directory}')
        return False
    if read_json(saved) != plan or file_hash(directory / 'train.json') != plan['annotation_sha256']:
        raise ValueError(f'Plan/code/annotation changed: {directory}; use a new output directory')
    graph = directory / 'graph'
    if completed(directory, plan):
        validate_run_settings(directory, plan)
        return True
    if (graph / 'training_complete.json').exists():
        raise ValueError(f'Invalid completion artifacts: {graph}')
    if graph.exists() and any(graph.iterdir()):
        if not resume or not (graph / 'checkpoint.pth').is_file():
            raise ValueError(f'Incomplete run: {graph}; --resume requires its checkpoint')
        validate_run_settings(directory, plan)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dirs', type=Path, nargs='+', default=[DEFAULT_SOURCE])
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', help='Default: recorded source seed; GLCA replay selection stays fixed')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--reference-world-size', type=int, default=2, help='Fallback only for old configs without world_size')
    parser.add_argument('--master-port', type=int, default=29587)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true', help='Write tables and plots; no training/data access')
    args = parser.parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    if args.summarize:
        from tools.owod.plot_replay_comparison import build_report
        build_report(args.output_dir)
        return 0
    plans = create_plans(args)
    index = {'schema': 1, 'mode': 'paired_task_local_shared_initialization',
             'runs': [(directory / 'plan.json').relative_to(args.output_dir).as_posix()
                      for directory, _, _ in plans]}
    index_path = args.output_dir / 'comparison.json'
    if index_path.is_file() and read_json(index_path) != index:
        raise ValueError('Comparison tasks/seeds changed; choose a new output directory')
    for directory, plan, _ in plans:
        check_existing(directory, plan, args.resume)
        print(f"Task {plan['task']} seed={plan['seed']} {plan['label']}: {plan['exposure']}")
        print(shlex.join(plan['command']), flush=True)
    if args.dry_run:
        print('Preflight passed; no training or output files created. All three arms will be rerun.')
        return 0
    environment = training_environment(args.gpus)
    with queue_lock(args.output_dir):
        # Recheck under the lock before modifying any files.
        if index_path.is_file() and read_json(index_path) != index:
            raise ValueError('Comparison index changed while waiting for the queue lock')
        for directory, plan, _ in plans:
            check_existing(directory, plan, args.resume)
        write_json(index_path, index)
        for directory, plan, payload in plans:
            write_json(directory / 'plan.json', plan)
            write_json(directory / 'train.json', payload)
        for directory, plan, _ in plans:
            if check_existing(directory, plan, args.resume):
                continue
            try:
                require_idle(args.gpus)
                validate_fingerprints(plan)
                graph = directory / 'graph'
                graph.mkdir(parents=True, exist_ok=True)
                command = list(plan['command'])
                if args.resume and (graph / 'checkpoint.pth').is_file():
                    command += ['--resume', str(graph / 'checkpoint.pth')]
                write_json(args.output_dir / 'queue_status.json', {'status': 'running', 'directory': str(directory), 'pid': os.getpid()})
                run_child(command, graph / 'train.log', environment)
                if not completed(directory, plan):
                    raise RuntimeError(f'Training exited without complete final metrics/checkpoint: {graph}')
                validate_run_settings(directory, plan)
            except BaseException as error:
                write_json(args.output_dir / 'queue_status.json', {'status': 'failed', 'directory': str(directory), 'error': str(error)})
                raise
        write_json(args.output_dir / 'queue_status.json', {'status': 'complete'})
    from tools.owod.plot_replay_comparison import build_report
    build_report(args.output_dir)
    return 0


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        raise SystemExit(f'Replay comparison failed: {error}')
