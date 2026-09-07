#!/usr/bin/env python
"""Queue controlled OW-DETR/EW-DETR four-stage runs after successful D4 completion."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import stage_files, file_sha256
from tools.owod.run_stage1_diagnostics import training_environment, metric_rows

PILOT = ROOT / 'exps/owod/m-owodb/order0/pilot_unverified'
D4 = PILOT / 'stage1_diagnostics_v2/d4_frozen_backbone_finetune/graph'
METHODS = ('ow-detr', 'ew-detr')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)


def payload_hash(payload):
    return hashlib.sha256((json.dumps(payload, indent=2, sort_keys=True) + '\n').encode('utf-8')).hexdigest()


def validate_fingerprints(plan):
    for name, expected in plan['fingerprints'].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if file_sha256(path) != expected:
            raise ValueError(f'Input/code changed after planning: {path}')


def annotation_subset(payload, image_ids, classes):
    image_ids, classes = set(image_ids), set(classes)
    return {**{key: value for key, value in payload.items() if key not in ('images', 'annotations')},
            'images': [dict(image) for image in payload['images'] if image['id'] in image_ids],
            'annotations': [dict(ann) for ann in payload['annotations']
                            if ann['image_id'] in image_ids and ann['category_id'] in classes]}


def combine_annotations(current, memory):
    images = {image['id']: {**image, 'owod_replay': True} for image in memory['images']}
    images.update({image['id']: {**image, 'owod_replay': False} for image in current['images']})
    annotations = {}
    for ann in [*memory['annotations'], *current['annotations']]:
        key = (ann['image_id'], ann['category_id'], tuple(ann['bbox']), ann.get('iscrowd', 0))
        annotations[key] = dict(ann)
    return {**current, 'images': sorted(images.values(), key=lambda image: image['id']),
            'annotations': [{**ann, 'id': index + 1} for index, ann in enumerate(annotations.values())]}


def select_memory(payload, classes, budget, seed):
    """Class-balanced random exemplars, from current data plus bounded prior memory."""
    rng = random.Random(seed)
    pools = {category: set() for category in classes}
    for ann in payload['annotations']:
        if ann['category_id'] in pools and not ann.get('iscrowd', 0):
            pools[ann['category_id']].add(ann['image_id'])
    pools = {category: sorted(images) for category, images in pools.items()}
    for images in pools.values():
        rng.shuffle(images)
    selected = set()
    while len(selected) < budget:
        old_length = len(selected)
        for category in classes:
            pool = pools[category]
            while pool and pool[-1] in selected:
                pool.pop()
            if pool and len(selected) < budget:
                selected.add(pool.pop())
        if len(selected) == old_length:
            break
    return annotation_subset(payload, selected, classes)


def baseline_complete(directory, epochs, method):
    try:
        marker = read_json(directory / 'training_complete.json')
        if marker.get('last_epoch') != epochs - 1 or marker.get('epochs') != epochs:
            return False
        rows = metric_rows(directory / 'metrics.jsonl')
        row = rows.get(epochs - 1, {})
        if not all(isinstance(row.get('test_owod_' + key), (int, float))
                   and math.isfinite(row['test_owod_' + key])
                   for key in ('current_ap50', 'known_ap50')):
            return False
        if not (directory / 'checkpoint.pth').is_file():
            return False
        if method == 'ew-detr':
            merged = read_json(directory / 'consolidated_metrics.json')
            return (merged.get('epoch') == epochs - 1
                    and 'test_owod_current_ap50' in merged
                    and (directory / 'checkpoint_consolidated.pth').is_file())
        return True
    except (OSError, ValueError, KeyError):
        return False


def active_output_processes(directory):
    """Match an actual output argument, not a monitor's reference to that path."""
    if not Path('/proc').is_dir():
        return []
    matches = []
    expected = str(directory.resolve())
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
            for index, token in enumerate(argv[:-1]):
                if token == '--output_dir' and str(Path(argv[index + 1]).resolve()) == expected:
                    matches.append(int(proc.name))
        except (OSError, ValueError):
            continue
    return matches


def verify_checkpoint(directory, epochs, method, stage, environment):
    command = [sys.executable, str(ROOT / 'tools/owod/verify_paper_checkpoint.py'),
               '--directory', str(directory), '--epochs', str(epochs),
               '--method', method, '--stage', str(stage)]
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def busy_gpus(gpus):
    def query(kind, fields):
        return subprocess.check_output(['nvidia-smi', f'--query-{kind}={fields}',
                                        '--format=csv,noheader,nounits'], text=True).splitlines()
    devices = {index.strip(): uuid.strip() for index, uuid in
               (row.split(',') for row in query('gpu', 'index,uuid'))}
    if not set(gpus.split(',')).issubset(devices):
        raise ValueError(f'Unknown GPU indices: {gpus}; available: {list(devices)}')
    wanted = {devices[index] for index in gpus.split(',')}
    return [row for row in query('compute-apps', 'gpu_uuid,pid') if row.split(',')[0].strip() in wanted]


def wait_for_d4(args):
    deadline = time.monotonic() + args.wait_hours * 3600
    previous = None
    while True:
        complete = baseline_complete(args.d4_dir, 20, 'd4')
        processes = active_output_processes(args.d4_dir)
        busy = busy_gpus(args.gpus)
        state = (complete, tuple(processes), tuple(busy))
        if complete and not processes and not busy:
            print('D4 complete; selected GPUs are idle. Starting baseline queue.', flush=True)
            return
        if state != previous:
            print(f'Waiting for D4: complete={complete}, D4 processes={processes}, GPU jobs={busy}', flush=True)
            previous = state
        if time.monotonic() >= deadline:
            raise TimeoutError('D4/GPU wait expired; inspect D4 logs before restarting this queue')
        time.sleep(args.poll_seconds)


@contextlib.contextmanager
def queue_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'queue.lock').open('a+') as handle:
        if os.name == 'posix':
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            handle.write('0')
            handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        yield


def run_child(command, log_path, environment):
    with log_path.open('a', encoding='utf-8') as log:
        log.write('\nCOMMAND ' + shlex.join(command) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=os.name == 'posix')
        try:
            result = process.wait()
        except BaseException:
            if process.poll() is None:
                if os.name == 'posix':
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name == 'posix':
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                    process.wait()
            raise
    if result:
        raise RuntimeError(f'Training exited {result}; queue stopped. Log: {log_path}')


def create_plan(args):
    source = read_json(args.d4_dir / 'run_config.json')
    expected = {'owod_stage': 1, 'lr_backbone': 0, 'epochs': 20,
                'backbone': 'resnet50', 'enc_layers': 6, 'dec_layers': 6,
                'hidden_dim': 256, 'num_queries': 300, 'batch_size': 2, 'seed': 42}
    for key, value in expected.items():
        if source.get(key) != value:
            raise ValueError(f'D4 setting mismatch: {key}={source.get(key)}, expected {value}')
    if source.get('neighbor_scoped_lora') or source.get('with_tree'):
        raise ValueError('D4 source is not the conventional frozen-backbone fine-tuning arm')
    for field, expected_value in (('aux_loss', True), ('dilation', False), ('two_stage', False),
                                  ('with_box_refine', False), ('masks', False), ('no_random_crop', False),
                                  ('lightweight', False), ('no_augmentation', False), ('cache_mode', False)):
        if source.get(field, expected_value) != expected_value:
            raise ValueError(f'Unsupported D4 architecture/augmentation: {field}')
    manifest_path = Path(source['owod_manifest']).resolve()
    coco_path = Path(source['coco_path']).resolve()
    source_train = Path(source['train_ann']).resolve()
    source_annotation = read_json(source_train)
    source_memory_count = sum(bool(image.get('owod_replay')) for image in source_annotation['images'])
    if source_memory_count < 1:
        raise ValueError('D4 annotation has no tagged replay images')
    memory_budget = source_memory_count if args.memory_images is None else args.memory_images
    fingerprints = {str(manifest_path): file_sha256(manifest_path), str(source_train): file_sha256(source_train)}
    methods = {}
    memory = {'images': [], 'annotations': []}
    previous_classes = []
    previous_samples = 0
    all_images = set()
    stage_records = []
    for stage in range(4):
        manifest, files = stage_files(manifest_path, stage, allow_unverified=True)
        if len(manifest['stages']) != 4:
            raise ValueError('Expected exactly four stages')
        record = manifest['stages'][stage]
        current_classes = record['classes']
        if (len(current_classes) != 20 or len(set(current_classes)) != 20
                or any(category < 1 or category > 90 for category in current_classes)
                or set(current_classes) & set(previous_classes)):
            raise ValueError('Expected disjoint 20-class increments')
        known = previous_classes + current_classes
        increment = read_json(files['increment_train'])
        classes_in_file = {ann['category_id'] for ann in increment['annotations']}
        if not classes_in_file.issubset(set(current_classes)):
            raise ValueError(f'Stage {stage} increment contains old/future supervision')
        if stage == 1:
            d4_current_images = {image['id'] for image in source_annotation['images'] if not image.get('owod_replay')}
            if d4_current_images != {image['id'] for image in increment['images']}:
                raise ValueError('Manifest Stage 1 current images differ from D4')
            if (set(source['owod_known_class_ids']) != set(known)
                    or set(source['owod_current_class_ids']) != set(current_classes)):
                raise ValueError('D4 class lists do not match the four-stage manifest')
        for key in ('increment_train', 'full_val'):
            fingerprints[str(files[key])] = file_sha256(files[key])
        all_images.update(coco_path / 'train2017' / image['file_name'] for image in increment['images'])
        validation = read_json(files['full_val'])
        all_images.update(coco_path / 'val2017' / image['file_name'] for image in validation['images'])
        combined = combine_annotations(increment, memory)
        replay_count = sum(image['owod_replay'] for image in combined['images'])
        if stage and replay_count == 0:
            raise ValueError('No replay images remain after merging current/memory image overlap')
        stage_records.append({'stage': stage, 'current_classes': current_classes,
                              'known_classes': known, 'increment_images': len(increment['images']),
                              'ow_replay_images': replay_count, 'ew_replay_images': 0})
        for method in METHODS:
            directory = args.output_dir / method / f'stage_{stage}'
            annotation = directory / 'train.json'
            payload = combined if method == 'ow-detr' else annotation_subset(
                increment, [image['id'] for image in increment['images']], current_classes)
            command = [sys.executable, '-m', 'torch.distributed.run', '--nnodes=1',
                       '--nproc_per_node=2', '--master_addr=127.0.0.1', f'--master_port={args.master_port}',
                       str(ROOT / 'main.py'), '--paper-baseline', method,
                       '--coco_path', str(coco_path), '--train-ann', str(annotation),
                       '--val-ann', str(files['full_val']), '--output_dir', str(directory),
                       '--owod-manifest', str(manifest_path), '--owod-stage', str(stage),
                       '--num_classes', '92', '--lr_backbone', '0', '--no-file-log',
                       '--epochs', '50' if stage == 0 else '20',
                       '--lr_drop', '40' if stage == 0 else str(source['lr_drop']),
                       '--owod-known-class-ids', *map(str, known),
                       '--owod-current-class-ids', *map(str, current_classes)]
            for field in ('backbone', 'enc_layers', 'dec_layers', 'hidden_dim', 'dim_feedforward',
                          'num_queries', 'num_feature_levels', 'nheads', 'enc_n_points', 'dec_n_points',
                          'dropout', 'batch_size', 'seed', 'num_workers', 'lr', 'weight_decay',
                          'position_embedding', 'position_embedding_scale',
                          'clip_max_norm', 'class_embed_lr_mult', 'lr_linear_proj_mult',
                          'set_cost_class', 'set_cost_bbox', 'set_cost_giou',
                          'cls_loss_coef', 'bbox_loss_coef', 'giou_loss_coef', 'focal_alpha', 'eval_interval'):
                if field not in source:
                    raise ValueError(f'D4 config is missing {field}')
                command.extend(['--' + field, str(source[field])])
            command.extend(['--unknown-threshold', str(source['unknown_threshold']),
                            '--print-freq', '100', '--eval-print-freq', '100'])
            if stage:
                prior_checkpoint = ('checkpoint_consolidated.pth' if method == 'ew-detr' else 'checkpoint.pth')
                command.extend(['--pretrained', str(directory.parent / f'stage_{stage - 1}' / prior_checkpoint),
                                '--owod-previous-class-ids', *map(str, previous_classes)])
            if method == 'ow-detr' and stage:
                command.extend(['--replay-sampling-fraction', str(source['replay_sampling_fraction'])])
            if method == 'ew-detr':
                command.extend(['--ew-current-samples', str(len(increment['images'])),
                                '--ew-previous-samples', str(previous_samples)])
            methods[f'{method}/stage_{stage}'] = {'command': command, 'annotation': payload,
                                                  'annotation_sha256': payload_hash(payload),
                                                  'epochs': 50 if stage == 0 else 20}
        memory = select_memory(combined, known, memory_budget, source['seed'] + stage)
        previous_classes = known
        previous_samples += len(increment['images'])
    missing_images = [str(path) for path in all_images if not path.is_file()]
    if missing_images:
        raise ValueError(f'{len(missing_images)} images missing; first: {missing_images[:3]}')
    if len(args.gpus.split(',')) != 2 or len(set(args.gpus.split(','))) != 2:
        raise ValueError('Matched setting requires two distinct GPU indices')
    # Pin implementation and source annotations; resume never silently changes either.
    code_paths = [ROOT / 'main.py', ROOT / 'engine.py', Path(__file__),
                  ROOT / 'tools/owod/protocol.py', ROOT / 'tools/owod/verify_paper_checkpoint.py',
                  ROOT / 'tools/owod/smoke_paper_baselines.py']
    for folder in ('models', 'datasets', 'util'):
        code_paths.extend((ROOT / folder).rglob('*.py'))
    for path in code_paths:
        fingerprints[str(path.relative_to(ROOT))] = file_sha256(path)
    plan = {'schema': 1, 'paper_comparable': False, 'protocol': 'D4_matched_internal_COCO_20x4',
            'd4_source': str(args.d4_dir), 'initialization': 'torchvision_ImageNet1K_V1_ResNet50; random_detector',
            'ow_source': 'akshitac8/OW-DETR@3515ff8c36687a4f582ef6a2c83866966b56ce23',
            'ew_source': 'CVPR2026 paper + supplement; from-paper adaptation, unspecified choices documented',
            'memory_images': memory_budget, 'd4_memory_images': source_memory_count,
            'gpus': args.gpus, 'stage_records': stage_records,
            'fingerprints': fingerprints, 'runs': {key: {k: v for k, v in value.items() if k != 'annotation'}
                                                   for key, value in methods.items()}}
    return plan, methods


def summarize(output_dir):
    print('AP50 / H in percent. EW pre-merge and consolidated results are separate.')
    print('method stage epoch weights previous current known U-Recall H status')
    for method in METHODS:
        for stage in range(4):
            directory = output_dir / method / f'stage_{stage}'
            status = 'complete' if baseline_complete(directory, 50 if stage == 0 else 20, method) else 'incomplete'
            rows = [(row, 'task') for _, row in sorted(metric_rows(directory / 'metrics.jsonl').items())]
            merged = directory / 'consolidated_metrics.json'
            if method == 'ew-detr' and merged.is_file():
                rows.append((read_json(merged), 'consolidated'))
            for row, weights in rows:
                values = [f"{100 * row['test_owod_' + key]:.3f}" if 'test_owod_' + key in row else 'NA'
                          for key in ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'h_score')]
                print(method, stage, row['epoch'] + 1, weights, *values, status)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--d4-dir', type=Path, default=D4)
    parser.add_argument('--output-dir', type=Path, default=PILOT / 'paper_baselines_v1')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--master-port', type=int, default=29579)
    parser.add_argument('--memory-images', type=int, default=None,
                        help='Default: exact number of tagged replay images in D4; optional override')
    parser.add_argument('--wait-hours', type=float, default=168)
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(argv)
    args.d4_dir, args.output_dir = args.d4_dir.resolve(), args.output_dir.resolve()
    if args.summarize:
        summarize(args.output_dir)
        return 0
    if (args.memory_images is not None and args.memory_images < 80) or args.poll_seconds <= 0 or args.wait_hours <= 0:
        raise ValueError('Require overridden memory >=80 images and positive waiting intervals')
    plan, runs = create_plan(args)
    print(json.dumps({key: plan[key] for key in ('protocol', 'initialization', 'stage_records')}, indent=2))
    for key, run in runs.items():
        print(key, shlex.join(run['command']))
    plan_path = args.output_dir / 'plan.json'
    if not plan_path.exists() and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError('Output directory has files but no plan; choose an empty baseline output directory')
    if plan_path.is_file() and read_json(plan_path) != plan:
        raise ValueError('Existing plan differs; use a new output directory for changed code/configuration')
    for key, run in runs.items():
        directory = args.output_dir / key
        if (directory / 'training_complete.json').exists() and not baseline_complete(directory, run['epochs'], key.split('/')[0]):
            raise ValueError(f'Inconsistent completion artifacts: {directory}')
        if (directory / 'run_config.json').exists() and not baseline_complete(directory, run['epochs'], key.split('/')[0]):
            if not args.resume or not (directory / 'checkpoint.pth').is_file():
                raise ValueError(f'Incomplete run: {directory}; --resume requires checkpoint.pth')
    if args.dry_run:
        print('Dry run passed. No training/output files created. Source and image paths checked.')
        return 0
    environment = training_environment(args.gpus)
    with queue_lock(args.output_dir):
        write_json(plan_path, plan)
        for key, run in runs.items():
            annotation_path = args.output_dir / key / 'train.json'
            if annotation_path.is_file() and read_json(annotation_path) != run['annotation']:
                raise ValueError(f'Annotation was modified: {annotation_path}')
            write_json(annotation_path, run['annotation'])
            del run['annotation']
        write_json(args.output_dir / 'queue_status.json', {'status': 'waiting_for_d4', 'pid': os.getpid()})
        try:
            wait_for_d4(args)
            validate_fingerprints(plan)
            verify_checkpoint(args.d4_dir, 20, 'd4', 1, environment)
            run_child([sys.executable, str(ROOT / 'tools/owod/smoke_paper_baselines.py')],
                      args.output_dir / 'preflight.log', environment)
            for method in METHODS:
                for stage in range(4):
                    key = f'{method}/stage_{stage}'
                    run = runs[key]
                    directory = args.output_dir / key
                    if baseline_complete(directory, run['epochs'], method):
                        verify_checkpoint(directory, run['epochs'], method, stage, environment)
                        print(f'Skipping completed {key}', flush=True)
                        continue
                    command = list(run['command'])
                    if file_sha256(directory / 'train.json') != run['annotation_sha256']:
                        raise ValueError(f'Training annotation changed: {directory}')
                    validate_fingerprints(plan)
                    if args.resume and (directory / 'checkpoint.pth').is_file():
                        command.extend(['--resume', str(directory / 'checkpoint.pth')])
                    write_json(args.output_dir / 'queue_status.json', {'status': 'running', 'run': key, 'pid': os.getpid()})
                    print(f'Starting {key}; log: {directory / "console.log"}', flush=True)
                    run_child(command, directory / 'console.log', environment)
                    if not baseline_complete(directory, run['epochs'], method):
                        raise RuntimeError(f'{key} exited without valid completion artifacts')
                    verify_checkpoint(directory, run['epochs'], method, stage, environment)
                    write_json(directory / 'verified_completion.json', {
                        'checkpoint_sha256': file_sha256(directory / 'checkpoint.pth'),
                        'last_epoch': run['epochs'] - 1})
        except BaseException as error:
            write_json(args.output_dir / 'queue_status.json', {'status': 'failed', 'error': str(error)})
            raise
        write_json(args.output_dir / 'queue_status.json', {'status': 'complete'})
        summarize(args.output_dir)
    return 0


if __name__ == '__main__':
    def stop(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(str(error))
