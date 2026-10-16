#!/usr/bin/env python
"""Screen paired discovery variants on training-only data, then train the winner fresh."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_server_experiment import VARIANTS
from tools.owod.screening_data import write_json


def benchmark_command(args, variant, screening=True):
    routing, localization = VARIANTS[variant]
    command = [sys.executable, str(ROOT / 'tools/owod/run_benchmark.py'),
               '--manifest', str(args.manifest), '--backbone-weights', str(args.weights),
               '--output-dir', str(args.output / ('screen' if screening else 'full') / variant),
               '--recipe', 'ow-detr-paper', '--method', 'discovery',
               '--graph-routing', routing, '--graph-localization', localization,
               '--discovery-scoring', 'hierarchical', '--gpus', args.gpus,
               '--nproc-per-node', str(len(args.gpus.split(','))),
               '--batch-size', str(args.batch_size), '--num-workers', str(args.workers),
               '--seed', str(args.seed), '--activation-checkpointing']
    if screening:
        command += ['--owod-screening', str(getattr(args, 'screening_path', None) or args.output / 'data/screening.json'),
                    '--screen-increment-epochs', str(args.increment_epochs),
                    '--screen-finetune-epochs', str(args.finetune_epochs),
                    '--screen-eval-interval', str(args.eval_interval)]
        if getattr(args, 'screen_shared_task1', None):
            command += ['--screen-shared-task1', str(args.screen_shared_task1),
                        '--screen-shared-gnn', str(args.screen_shared_gnn)]
    if args.resume:
        command += ['--resume']
    return command


def select_winner(root, protocol, seed, epochs, screen_hash, increment_epochs=8, shared_identity=None):
    """Use only completed final Task 2 finetuning, never best test epoch."""
    rows = []
    for variant in VARIANTS:
        base = root / 'screen' / variant / protocol / f'seed_{seed}'
        seconds = 0
        phases = [(1, 'increment', increment_epochs), (1, 'finetune', epochs)]
        if not shared_identity:
            phases.insert(0, (0, 'increment', increment_epochs))
        for stage, phase, expected_epochs in phases:
            directory = base / f'stage_{stage}' / phase
            complete = read_json(directory / 'training_complete.json')
            config = read_json(directory / 'run_config.json')
            if (complete['last_epoch'] != expected_epochs - 1 or not (directory / 'checkpoint.pth').is_file()
                    or config.get('owod_screening_sha256') != screen_hash
                    or config.get('seed') != seed or config.get('paper_comparable_protocol') is not False
                    or (config.get('graph_routing'), config.get('graph_localization')) != VARIANTS[variant]):
                raise ValueError(f'{variant}: incomplete or mismatched screening result')
            if shared_identity and (
                    config.get('screen_branch_checkpoint_sha256') != shared_identity['checkpoint_sha256']
                    or config.get('screen_branch_gnn_sha256') != shared_identity['gnn_sha256']):
                raise ValueError(f'{variant}: common parent/GNN mismatch')
            seconds += complete['training_time_seconds']
        metrics = [json.loads(line) for line in (directory / 'metrics.jsonl').read_text().splitlines() if line.strip()]
        final = [record for record in metrics if record['epoch'] == epochs - 1]
        if not final:
            raise ValueError(f'{variant}: final epoch validation missing')
        values = {group: float(final[-1][f'test_owod_{group}_ap50'])
                  for group in ('previous', 'current', 'known')}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError(f'{variant}: invalid validation metric')
        rows.append({'variant': variant, **values,
                     'training_and_evaluation_seconds': seconds})
    # Stable tie break favors the simpler variant; noise is exposed in the margin.
    ranked = sorted(rows, key=lambda row: -row['known'])
    if ranked[0]['known'] <= 0:
        raise ValueError('All methods have zero Known AP50; no useful winner, full run not launched')
    comparisons = []
    for before, after in zip(rows, rows[1:]):
        comparisons.append({'from': before['variant'], 'to': after['variant'],
                            **{f'delta_{group}_ap50': after[group] - before[group]
                               for group in ('previous', 'current', 'known')}})
    return {'winner': ranked[0]['variant'], 'rows': rows, 'comparisons': comparisons,
            'selection_metric': 'final Task 2 train-heldout Known AP50 (author AP convention)',
            'margin_ap50': ranked[0]['known'] - ranked[1]['known'],
            'paper_comparable': False, 'unknown_metrics': 'not measured (known-only validation)',
            'screening_sha256': screen_hash,
            'scope': 'single-seed two-task diagnostic winner; full-scale ranking not established'}


def run_bounded(command, env, deadline, log_path):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('Screening budget exhausted; no partial comparison or full launch')
    print(shlex.join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            result = process.wait(timeout=remaining)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            # Terminate only this launcher's process group, including torchrun workers.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError(f'Screen stopped; checkpoints retained; inspect {log_path}')
    if result:
        raise RuntimeError(f'Command failed ({result}); inspect {log_path}; full training not launched')


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/screening-s-seed42')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-images', type=int, default=2000)
    parser.add_argument('--val-images', type=int, default=400)
    parser.add_argument('--increment-epochs', type=int, default=8)
    parser.add_argument('--finetune-epochs', type=int, default=4)
    parser.add_argument('--eval-interval', type=int, default=4)
    parser.add_argument('--budget-hours', type=float, default=12)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--screen-only', action='store_true', help='Do not launch full training after selection')
    parser.add_argument('--execute', action='store_true', help='Default only prints the workflow')
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    devices = args.gpus.split(',')
    if (len(devices) != 2 or not all(device.isdigit() for device in devices) or len(set(devices)) != len(devices)
            or min(args.batch_size, args.train_images, args.val_images, args.finetune_epochs, args.eval_interval) < 1
            or args.increment_epochs < 2 or args.workers < 0
            or not math.isfinite(args.budget_hours) or args.budget_hours <= 0):
        raise ValueError('Invalid devices or screening budget')
    args.manifest, args.weights, args.output = (path.resolve() for path in (args.manifest, args.weights, args.output))
    prepare = [sys.executable, str(ROOT / 'tools/owod/manage_data.py'), 'prepare-screen',
               '--manifest', str(args.manifest), '--screen-output', str(args.output / 'data'),
               '--train-images', str(args.train_images), '--val-images', str(args.val_images), '--seed', str(args.seed)]
    if not args.execute:
        print(shlex.join(prepare))
        for variant in VARIANTS:
            print(shlex.join(benchmark_command(args, variant)))
        print('Compare final held-out Task 2 Known AP50; then launch the winner with a NEW full-data checkpoint chain.'
              if not args.screen_only else 'Compare final held-out Task 2 Known AP50; stop after screening.')
        return 0
    if os.name != 'posix':
        raise ValueError('Execute on the Linux training server; Windows supports planning/tests only')
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    identity = {'manifest_sha256': file_sha256(args.manifest), 'weights_sha256': file_sha256(args.weights),
                **{key: getattr(args, key) for key in ('seed', 'train_images', 'val_images', 'increment_epochs',
                                                     'finetune_epochs', 'eval_interval', 'gpus', 'batch_size', 'workers')}}
    plan_path = args.output / 'screen_plan.json'
    if plan_path.exists():
        if not args.resume or read_json(plan_path) != identity:
            raise ValueError('Existing screen requires --resume and identical inputs/settings')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a clean screening output directory')
    write_json(plan_path, identity)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus)
    env.setdefault('OMP_NUM_THREADS', '2')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'models/ops'), str(ROOT), env.get('PYTHONPATH', '')])
    deadline = time.monotonic() + args.budget_hours * 3600
    run_bounded(prepare, env, deadline, args.output / 'prepare.log')
    preflight = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                 '--nproc_per_node', str(len(devices)), str(ROOT / 'tools/owod/server_cuda_check.py')]
    run_bounded(preflight, env, deadline, args.output / 'preflight.log')
    for variant in VARIANTS:
        run_bounded(benchmark_command(args, variant), env, deadline, args.output / f'{variant}.log')
    protocol = read_json(args.manifest)['protocol']
    report = select_winner(args.output, protocol, args.seed, args.finetune_epochs,
                           file_sha256(args.output / 'data/screening.json'), args.increment_epochs)
    full_command = benchmark_command(args, report['winner'], screening=False)
    report['full_command'] = full_command
    write_json(args.output / 'selection.json', report)
    print(json.dumps(report, indent=2), flush=True)
    if not args.screen_only:
        # Full execution has no diagnostic wall-clock cutoff. No screen pretrained/resume path is passed.
        print('Starting fresh full-data run:', shlex.join(full_command), flush=True)
        return subprocess.run(full_command, cwd=ROOT, env=env).returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
