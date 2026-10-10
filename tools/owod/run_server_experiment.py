"""Plan or execute sequential paired four-task studies on two RTX 3090 GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VARIANTS = {'hard': ('hard', 'frozen'), 'soft': ('soft', 'frozen'),
            'soft-localization': ('soft', 'bbox-last')}


def commands(args):
    selected = VARIANTS if args.variant == 'suite' else [args.variant]
    result = []
    for name in selected:
        routing, localization = VARIANTS[name]
        command = [sys.executable, str(ROOT / 'tools/owod/run_benchmark.py'),
            '--manifest', str(args.manifest.resolve()), '--backbone-weights', str(args.weights.resolve()),
            '--output-dir', str((args.output / name).resolve()), '--recipe', 'ow-detr-paper',
            '--method', 'discovery', '--graph-routing', routing, '--graph-localization', localization,
            '--discovery-scoring', 'hierarchical', '--gpus', args.gpus, '--nproc-per-node', '2',
            '--batch-size', str(args.batch_size), '--num-workers', str(args.workers),
            '--seed', str(args.seed), '--activation-checkpointing']
        if args.resume:
            command.append('--resume')
        if not args.execute:
            command.append('--dry-run')
        result.append((name, command))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/server-3090')
    parser.add_argument('--variant', choices=['suite', *VARIANTS], default='suite')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--batch-size', type=int, default=2, help='Per GPU; default global batch is 4')
    parser.add_argument('--workers', type=int, default=4, help='Per GPU')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--execute', action='store_true', help='Otherwise validate and print plans only')
    args = parser.parse_args(argv)
    devices = args.gpus.split(',')
    if (len(devices) != 2 or len(set(devices)) != 2 or not all(d.isdigit() for d in devices)
            or args.batch_size < 1 or args.workers < 0):
        parser.error('Require two distinct GPU indices, positive batch, nonnegative workers')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = args.gpus
    env.setdefault('OMP_NUM_THREADS', '2')
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'models/ops'), str(ROOT), env.get('PYTHONPATH', '')])
    from tools.owod.manage_data import verify_dataset
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    verification = verify_dataset(args.manifest.parent, manifest['protocol'])
    if verification.get('ready') is False:
        raise ValueError('Data verification failed')
    if args.execute:
        subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                        '--nproc_per_node', '2', str(ROOT / 'tools/owod/server_cuda_check.py')],
                       cwd=ROOT, env=env, check=True)
    for name, command in commands(args):
        print(f'Variant: {name}; per-GPU batch={args.batch_size}; global batch={args.batch_size * 2}', flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.execute and args.variant == 'suite':
        from tools.owod.validate_discovery import compare_runs
        suffix = Path(manifest['protocol']) / f'seed_{args.seed}'
        for baseline, treatment, label in [('hard', 'soft', 'routing'),
                                            ('soft', 'soft-localization', 'localization')]:
            report = compare_runs(args.output / baseline / suffix, args.output / treatment / suffix)
            (args.output / (label + '_comparison.json')).write_text(json.dumps(report, indent=2) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
