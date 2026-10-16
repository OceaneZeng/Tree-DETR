#!/usr/bin/env python
"""One hard/no-output-distillation diagnostic, followed by matched score probes."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.owod.protocol import file_sha256, read_json
from tools.owod.run_task2_comparison import configure
from tools.owod.run_screening import benchmark_command, run_bounded
from tools.owod.screening_data import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'output/screening-s-8k')
    parser.add_argument('--baseline-root', type=Path, default=ROOT / 'output/task2-shared-s-8k')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/task2-final-check')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--budget-hours', type=float, default=12)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.budget_hours) or args.budget_hours <= 0:
        parser.error('budget-hours must be positive and finite')
    args.anchor_variant = 'hard'
    protocol, identity = configure(args)
    args.baseline_root = args.baseline_root.resolve()
    if (args.output == args.baseline_root or args.baseline_root in args.output.parents
            or args.output in args.baseline_root.parents or args.output in args.source_root.parents):
        raise ValueError('Use a separate output directory outside existing runs')
    baseline = read_json(args.baseline_root / 'comparison_plan.json')
    for key in ('manifest_sha256', 'screening_sha256', 'checkpoint_sha256', 'gnn_sha256', 'source_plan'):
        if baseline[key] != identity[key]:
            raise ValueError(f'Baseline differs from the paired source: {key}')
    # Verify baseline phases before spending time on training.
    from tools.owod.diagnose_learning import checked_checkpoint
    for phase in ('increment', 'finetune'):
        checkpoint = args.baseline_root / 'screen/hard' / protocol / f'seed_{args.seed}/stage_1/{phase}/checkpoint.pth'
        saved = checked_checkpoint(checkpoint, identity['manifest_sha256'], identity['screening_sha256'], 1, phase)
        if (not getattr(saved, 'old_class_distillation', False)
                or getattr(saved, 'screen_no_distill', False)
                or getattr(saved, 'screen_branch_checkpoint_sha256', None) != identity['checkpoint_sha256']
                or getattr(saved, 'screen_branch_gnn_sha256', None) != identity['gnn_sha256']):
            raise ValueError('Expected completed shared-parent hard/distill-on baseline')
    identity.update(kind='task2_final_check', screen_no_distill=True,
                    baseline_root=str(args.baseline_root),
                    baseline_checkpoints={phase: file_sha256(args.baseline_root / 'screen/hard' / protocol /
                        f'seed_{args.seed}/stage_1/{phase}/checkpoint.pth')
                        for phase in ('increment', 'finetune')})
    command = benchmark_command(args, 'hard') + ['--screen-no-distill']
    probes = []
    for treatment, root in (('baseline', args.baseline_root), ('no_distill', args.output)):
        for phase in ('increment', 'finetune'):
            directory = args.output / 'evaluation' / treatment / phase
            probe = [sys.executable, str(ROOT / 'tools/owod/diagnose_learning.py'), 'scores-task2',
                     '--source-root', str(args.source_root), '--comparison-root', str(root),
                     '--manifest', str(args.manifest), '--score-phase', phase,
                     '--gpu', args.gpus.split(',')[0], '--seed', str(args.seed),
                     '--output', str(directory)]
            probes.append((treatment, phase, directory, probe, root))
    if not args.execute:
        print(shlex.join(command))
        for _, _, _, probe, _ in probes:
            print(shlex.join(probe))
        return 0
    if os.name != 'posix':
        raise ValueError('Execute on the Linux training server')
    plan = args.output / 'comparison_plan.json'
    if plan.exists():
        if not args.resume or read_json(plan) != identity:
            raise ValueError('Resume requires unchanged inputs and settings')
    elif args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use an empty output directory')
    write_json(plan, identity)
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpus, PYTHONHASHSEED=str(args.seed))
    env.setdefault('OMP_NUM_THREADS', '2')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'models/ops'), str(ROOT), env.get('PYTHONPATH', '')])
    deadline = time.monotonic() + args.budget_hours * 3600
    preflight = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                 '--nproc_per_node', str(len(args.gpus.split(','))), str(ROOT / 'tools/owod/server_cuda_check.py')]
    run_bounded(preflight, env, deadline, args.output / 'preflight.log')
    run_bounded(command, env, deadline, args.output / 'hard-no-distill.log')
    report = {'paper_comparable': False, 'full_training_started': False, 'results': {}}
    for treatment, phase, directory, probe, root in probes:
        expected = {key: identity[key] for key in ('manifest_sha256', 'screening_sha256')}
        expected.update(parent_sha256=identity['checkpoint_sha256'], mode='scores-task2', score_phase=phase,
                        checkpoint_sha256=file_sha256(root / 'screen/hard' / protocol /
                            f'seed_{args.seed}/stage_1/{phase}/checkpoint.pth'))
        marker = directory / 'complete.json'
        if args.resume and marker.exists():
            completed = read_json(marker)
            if any(completed.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Score provenance changed: {directory}')
        else:
            # Preserve interrupted evaluation output and retry in a fresh sibling.
            if directory.exists() and any(directory.iterdir()):
                retry = 1
                while directory.with_name(f'{phase}-retry{retry}').exists():
                    retry += 1
                directory = directory.with_name(f'{phase}-retry{retry}')
                probe[probe.index('--output') + 1] = str(directory)
            run_bounded(probe, env, deadline, args.output / f'eval-{treatment}-{phase}.log')
        report['results'][f'{treatment}_{phase}'] = read_json(directory / 'summary.json')
    write_json(args.output / 'final_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
