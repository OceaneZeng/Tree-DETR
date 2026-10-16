#!/usr/bin/env python
"""Repeat only Task 2 from one immutable diagnostic Task 1 parent and GNN."""
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import read_json
from tools.owod.run_screening import benchmark_command, run_bounded, select_winner
from tools.owod.run_server_experiment import VARIANTS
from tools.owod.screening_data import write_json
from tools.owod.shared_task1 import validate_inputs


def configure(args):
    args.source_root = args.source_root.resolve()
    args.output = args.output.resolve()
    args.manifest, args.weights = args.manifest.resolve(), args.weights.resolve()
    if args.output == args.source_root or args.source_root in args.output.parents:
        raise ValueError('Use a separate output directory outside the original screen')
    source_plan = read_json(args.source_root / 'screen_plan.json')
    # Inherit the previous paired budget instead of silently falling back to 2k/8/4.
    for key in ('seed', 'train_images', 'val_images', 'increment_epochs', 'finetune_epochs',
                'eval_interval', 'batch_size', 'workers', 'gpus'):
        setattr(args, key, source_plan[key])
    args.screening_path = args.source_root / 'data/screening.json'
    protocol = read_json(args.manifest)['protocol']
    anchor = args.source_root / 'screen' / args.anchor_variant / protocol / f'seed_{args.seed}'
    args.screen_shared_task1 = anchor / 'stage_0/increment/checkpoint.pth'
    args.screen_shared_gnn = anchor / 'gnn/stage_0/gnn_stage0.pt'
    identity = validate_inputs(args.manifest, args.screening_path, args.screen_shared_task1,
                               args.screen_shared_gnn, args.weights, args.seed)
    if (identity['manifest_sha256'] != source_plan['manifest_sha256']
            or identity['weights_sha256'] != source_plan['weights_sha256']):
        raise ValueError('Source plan does not match manifest/pretraining inputs')
    config = read_json(anchor / 'stage_0/increment/run_config.json')
    marker = read_json(anchor / 'stage_0/increment/training_complete.json')
    if (marker['last_epoch'] != source_plan['increment_epochs'] - 1
            or config.get('owod_screening_sha256') != identity['screening_sha256']):
        raise ValueError('Common parent completion/config does not match the original screen')
    return protocol, {**identity, 'source_plan': source_plan, 'anchor_variant': args.anchor_variant,
                      'checkpoint': str(args.screen_shared_task1), 'gnn': str(args.screen_shared_gnn),
                      'screening': str(args.screening_path), 'kind': 'shared_task1_task2_comparison'}


def main(argv=None):
    import argparse
    import math

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'output/screening-s-8k')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/task2-shared-s-8k')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--anchor-variant', choices=tuple(VARIANTS), default='hard')
    parser.add_argument('--budget-hours', type=float, default=12)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.budget_hours) or args.budget_hours <= 0:
        parser.error('budget-hours must be positive and finite')
    protocol, identity = configure(args)
    commands = [benchmark_command(args, variant) for variant in VARIANTS]
    print('Common parent/GNN verified:', json.dumps(identity, indent=2), flush=True)
    if not args.execute:
        for command in commands:
            print(shlex.join(command))
        print('Only Task 2 increment + finetune; no recalibration, no Task 1 or full-data launch.')
        return 0
    if os.name != 'posix':
        raise ValueError('Execute on the Linux training server; local Windows supports validation/tests')
    plan_path = args.output / 'comparison_plan.json'
    if plan_path.exists():
        if not args.resume or read_json(plan_path) != identity:
            raise ValueError('Resume requires unchanged common parent, GNN, data and settings')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a clean Task 2 comparison output directory')
    write_json(plan_path, identity)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus, PYTHONHASHSEED=str(args.seed))
    env.setdefault('OMP_NUM_THREADS', '2')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'models/ops'), str(ROOT), env.get('PYTHONPATH', '')])
    deadline = time.monotonic() + args.budget_hours * 3600
    preflight = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', '2',
                 str(ROOT / 'tools/owod/server_cuda_check.py')]
    run_bounded(preflight, env, deadline, args.output / 'preflight.log')
    for variant, command in zip(VARIANTS, commands):
        run_bounded(command, env, deadline, args.output / f'{variant}.log')
    report = select_winner(args.output, protocol, args.seed, args.finetune_epochs,
                           identity['screening_sha256'], args.increment_epochs, shared_identity=identity)
    for row in report['rows']:
        base = args.output / 'screen' / row['variant'] / protocol / f'seed_{args.seed}' / 'stage_1'
        row['curves'] = {}
        for phase in ('increment', 'finetune'):
            records = [json.loads(line) for line in (base / phase / 'metrics.jsonl').read_text().splitlines()
                       if line.strip()]
            row['curves'][phase] = [{key: record[key] for key in
                                    ('epoch', 'test_owod_previous_ap50', 'test_owod_current_ap50', 'test_owod_known_ap50')}
                                   for record in records if 'test_owod_known_ap50' in record]
    report.update(shared_task1=identity, full_training_started=False,
                  scope='Shared-parent single-seed Task 2 diagnostic comparison; not a full-data ranking')
    write_json(args.output / 'selection.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
