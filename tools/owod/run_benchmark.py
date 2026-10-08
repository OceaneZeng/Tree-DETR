#!/usr/bin/env python
"""Run official four-task increments followed by released balanced finetuning."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.owod.protocol import file_sha256, read_json, stage_files


def build_plan(args):
    manifest, _ = stage_files(args.manifest, 0)
    if args.recipe == 'ore-paper':
        if manifest['protocol'] != 'm-owodb':
            raise ValueError('ORE 2021 does not specify S-OWODB')
        if args.finetune_epochs is None:
            raise ValueError('ORE specifies Nex=50 but not a standalone finetuning epoch count. '
                             'Supply --finetune-epochs from your documented author training recipe; '
                             'this launcher will not invent that value.')
        epochs, ft_epochs, lr, ft_lr = 8, args.finetune_epochs, .01, .01
    else:
        epochs, ft_epochs, lr, ft_lr = 50, 20, 2e-4, 2e-5
        if args.finetune_epochs is not None and args.finetune_epochs != 20:
            raise ValueError('OW-DETR Appendix D requires 20 finetuning epochs')
        if not args.backbone_weights:
            raise ValueError('Original OW-DETR recipe requires DINO --backbone-weights')
    if manifest['protocol'] == 's-owodb' and not args.backbone_weights:
        raise ValueError('S-OWODB original recipe requires DINO backbone pretraining')
    if args.backbone_weights and not args.backbone_weights.is_file():
        raise FileNotFoundError(args.backbone_weights)
    image_root = Path(manifest['image_root'])
    runs, previous = [], None
    for stage in range(4):
        _, files = stage_files(args.manifest, stage)
        for key in ('increment_train', 'train', 'full_val'):
            payload = read_json(files[key])
            missing = [image['file_name'] for image in payload['images']
                       if not (image_root / image['file_name']).is_file()]
            if missing:
                raise FileNotFoundError(f'{key}: {len(missing)} images missing; examples={missing[:5]}')
        for phase in (['increment'] if stage == 0 else ['increment', 'finetune']):
            directory = args.output_dir / manifest['protocol'] / f'seed_{args.seed}' / f'stage_{stage}' / phase
            n_epochs = epochs if phase == 'increment' else ft_epochs
            n_lr = lr if phase == 'increment' else ft_lr
            command = [sys.executable]
            if args.nproc_per_node > 1:
                command += ['-m', 'torch.distributed.run', '--standalone',
                            '--nproc_per_node', str(args.nproc_per_node)]
            command += [str(ROOT / 'main.py'), '--coco_path', str(image_root),
                        '--owod-manifest', str(args.manifest), '--owod-stage', str(stage),
                        '--owod-phase', phase, '--output_dir', str(directory),
                        '--owod-recipe', args.recipe,
                        '--num_classes', '91', '--backbone', 'resnet50',
                        '--num_queries', '100', '--dim_feedforward', '1024',
                        '--epochs', str(n_epochs), '--lr', str(n_lr),
                        '--batch_size', str(args.batch_size), '--num_workers', str(args.num_workers),
                        '--seed', str(args.seed), '--lr_drop', str(max(1, n_epochs)),
                        '--eval_interval', '1']
            if args.recipe == 'ore-paper':
                if args.batch_size * args.nproc_per_node != 8:
                    raise ValueError('ORE Appendix F requires effective batch size 8')
                command += ['--sgd', '--lr_backbone', str(n_lr)]
            else:
                command += ['--weight_decay', '0.0001', '--lr_backbone', str(n_lr / 10)]
            if args.backbone_weights:
                command += ['--backbone-weights', str(args.backbone_weights)]
            if previous:
                command += ['--pretrained', str(previous)]
            runs.append({'stage': stage, 'phase': phase, 'command': command,
                         'output_dir': str(directory), 'epochs': n_epochs})
            previous = directory / 'checkpoint.pth'
    return {'protocol': manifest['protocol'], 'manifest_sha256': file_sha256(args.manifest),
            'backbone_weights_sha256': file_sha256(args.backbone_weights) if args.backbone_weights else None,
            'recipe': args.recipe, 'recipe_source': ('ORE Appendix F; separate finetuning schedule supplied explicitly'
                                                    if args.recipe == 'ore-paper' else 'OW-DETR Appendix D'),
            'model': 'Local Deformable DETR control (not an ORE/OW-DETR method reproduction)',
            'runs': runs}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--recipe', required=True, choices=('ore-paper', 'ow-detr-paper'))
    parser.add_argument('--finetune-epochs', type=int)
    parser.add_argument('--backbone-weights', type=Path)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--nproc-per-node', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpus', default='')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.nproc_per_node < 1 or args.batch_size < 1 or (args.finetune_epochs is not None and args.finetune_epochs < 1):
        raise ValueError('Process count, batch size and finetuning epochs must be positive')
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.backbone_weights:
        args.backbone_weights = args.backbone_weights.resolve()
    plan = build_plan(args)
    if args.dry_run:
        for run in plan['runs']:
            print(shlex.join(run['command']))
        return 0
    root = args.output_dir / plan['protocol'] / f'seed_{args.seed}'
    plan_path = root / 'benchmark_plan.json'
    if plan_path.is_file():
        if not args.resume or read_json(plan_path) != plan:
            raise ValueError('Output already has a plan; resume only with unchanged input and settings')
    elif root.exists() and any(root.iterdir()):
        raise ValueError('Use a clean output directory')
    root.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'models/ops') + os.pathsep + str(ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    if args.gpus:
        env['CUDA_VISIBLE_DEVICES'] = args.gpus
    for run in plan['runs']:
        directory = Path(run['output_dir'])
        marker = directory / 'training_complete.json'
        command = list(run['command'])
        if args.resume and marker.is_file():
            completed = read_json(marker)
            if completed.get('last_epoch') == run['epochs'] - 1 and (directory / 'checkpoint.pth').is_file():
                continue
            raise ValueError(f'Invalid completion marker: {marker}')
        if args.resume and (directory / 'checkpoint.pth').is_file():
            command += ['--resume', str(directory / 'checkpoint.pth')]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
