#!/usr/bin/env python
"""Continue the completed D2 Stage 1 with the existing GNN replay recipe."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.owod.protocol import stage_files
from tools.owod.run_stage1_diagnostics import file_hash, metric_rows, training_environment

PILOT = ROOT / 'exps/owod/m-owodb/order0/pilot_unverified'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def completed(path, epochs):
    marker = path / 'training_complete.json'
    return (marker.is_file() and read(marker).get('last_epoch') == epochs - 1
            and read(marker).get('epochs') == epochs
            and (path / 'checkpoint.pth').is_file()
            and epochs - 1 in metric_rows(path / 'metrics.jsonl'))


def build_command(args, d2, source, graph, previous):
    if (d2['owod_stage'] != 1 or d2['neighbor_scoped_lora'] or d2['lr_backbone'] <= 0
            or not d2['teacher_completion'] or d2['local_margin_coef']
            or d2['off_projection_coef'] or d2['old_class_distillation']):
        raise ValueError('Source must be D2: ordinary fine-tuning, positive backbone LR, teacher, no extra losses')
    expected = dict(backbone='resnet50', enc_layers=6, dec_layers=6, num_queries=300,
                    hidden_dim=256, dim_feedforward=1024, num_feature_levels=4,
                    with_tree=False, two_stage=False, with_box_refine=False,
                    lightweight=False, no_random_crop=False, no_augmentation=False,
                    aux_loss=True, class_embed_lr_mult=1.0, epochs=20,
                    weight_decay=1e-4, lr_linear_proj_mult=0.1, clip_max_norm=0.1,
                    cls_loss_coef=2, bbox_loss_coef=5, giou_loss_coef=2,
                    set_cost_class=2, set_cost_bbox=5, set_cost_giou=2, focal_alpha=0.25)
    for field, value in expected.items():
        if d2.get(field, value) != value:
            raise ValueError(f'Unsupported D2 setting: {field}={d2.get(field)}')
    command = [sys.executable, str(ROOT / 'tools/owod/run_graph_local_increment.py'),
               '--stage', str(args.stage), '--checkpoint', str(previous / 'checkpoint.pth'),
               '--output-dir', str(args.output_dir / f'stage_{args.stage}'),
               '--coco-path', d2['coco_path'], '--manifest', d2['owod_manifest'],
               '--gnn-checkpoint', graph['graph']['checkpoint'], '--control', 'graph',
               '--allow-unverified-protocol', '--no-neighbor-scoped-lora', '--teacher-completion',
               '--local-margin-coef', '0', '--off-projection-coef', '0',
               '--gpus', args.gpus, '--nproc-per-node', '2', '--master-port', str(args.master_port)]
    for field, option in [('num_classes', 'num-classes'), ('epochs', 'epochs'),
                          ('lr', 'lr'), ('lr_backbone', 'lr-backbone'), ('lr_drop', 'lr-drop'),
                          ('batch_size', 'batch-size'), ('num_workers', 'num-workers'), ('seed', 'seed'),
                          ('eval_interval', 'eval-interval'), ('unknown_threshold', 'unknown-threshold'),
                          ('replay_sampling_fraction', 'replay-sampling-fraction'),
                          ('teacher_score_threshold', 'teacher-score-threshold'),
                          ('teacher_duplicate_iou', 'teacher-duplicate-iou'),
                          ('teacher_ground_truth_iou', 'teacher-ground-truth-iou'),
                          ('teacher_max_per_image', 'teacher-max-per-image')]:
        command += ['--' + option, str(d2[field])]
    command += ['--graph-k', str(graph['graph']['requested_k']),
                '--gnn-min-score', str(graph['graph']['min_score']),
                '--graph-aggregation', source['graph_aggregation'],
                '--graph-aggregation-top-n', str(source['graph_aggregation_top_n']),
                '--sketch-max-images', str(graph['max_images_per_class']),
                '--last-decoder-layers', str(graph['last_decoder_layers'])]
    if source.get('exemplars_per_class') is not None:
        command += ['--exemplars-per-class', str(source['exemplars_per_class'])]
    else:
        command += ['--base-exemplars-per-class', str(source['base_exemplars_per_class']),
                    '--risk-extra-exemplars-per-class', str(source['risk_extra_exemplars_per_class'])]
    if args.resume:
        command += ['--resume', str(args.output_dir / f'stage_{args.stage}/graph/checkpoint.pth')]
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=int, choices=(2, 3), default=2)
    parser.add_argument('--d2-dir', type=Path, default=PILOT / 'stage1_diagnostics_v2/d2_full_finetune_no_extra_losses/graph')
    parser.add_argument('--source-run', type=Path, default=PILOT / 'full_three_module_stage1_v1')
    parser.add_argument('--output-dir', type=Path, default=PILOT / 'd2_continual_v1')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--master-port', type=int, default=29583)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args(argv)
    for field in ('d2_dir', 'source_run', 'output_dir'):
        setattr(args, field, getattr(args, field).resolve())
    if args.summarize:
        print('D2 continuation: AP50 / H in percent; Stage 3 has no unknown classes.')
        print('stage epoch previous current known H status')
        for stage, directory in [(1, args.d2_dir), *[(s, args.output_dir / f'stage_{s}/graph') for s in (2, 3)]]:
            status = 'complete' if completed(directory, 20) else 'incomplete'
            for epoch, row in sorted(metric_rows(directory / 'metrics.jsonl').items()):
                values = [f"{100 * row['test_owod_' + key]:.3f}" if 'test_owod_' + key in row else 'NA'
                          for key in ('previous_ap50', 'current_ap50', 'known_ap50', 'h_score')]
                print(stage, epoch + 1, *values, status)
        return 0
    if len(set(args.gpus.split(','))) != 2 or len(args.gpus.split(',')) != 2:
        raise ValueError('Supply exactly two distinct GPUs')
    d2 = read(args.d2_dir / 'run_config.json')
    source = read(args.source_run / 'run_config.json')
    graph = read(args.source_run / 'graph.json')
    if file_hash(Path(d2['train_ann'])) != file_hash(args.source_run / 'annotations/train_graph.json'):
        raise ValueError('D2 does not use the specified source GNN annotation')
    if not completed(args.d2_dir, d2['epochs']):
        raise ValueError('D2 has not completed its schedule/evaluation')
    previous = args.d2_dir if args.stage == 2 else args.output_dir / 'stage_2/graph'
    if not completed(previous, d2['epochs']):
        raise ValueError(f'Previous stage has not completed: {previous}')
    if args.stage == 3:
        prior = read(previous / 'run_config.json')
        if prior['owod_stage'] != 2 or prior['pretrained'] != str(args.d2_dir / 'checkpoint.pth'):
            raise ValueError('Stage 2 does not descend from the requested D2 checkpoint')
    _, current_files = stage_files(Path(d2['owod_manifest']), args.stage, allow_unverified=True)
    _, previous_files = stage_files(Path(d2['owod_manifest']), args.stage - 1, allow_unverified=True)
    command = build_command(args, d2, source, graph, previous)
    inputs = [args.d2_dir / 'run_config.json', args.source_run / 'run_config.json',
              args.source_run / 'graph.json', previous / 'checkpoint.pth',
              Path(graph['graph']['checkpoint']), Path(d2['owod_manifest']),
              current_files['increment_train'], current_files['full_val'], previous_files['train']]
    fingerprints = {str(path): file_hash(path) for path in inputs}
    for path in [ROOT / 'main.py', Path(__file__), ROOT / 'tools/owod/run_graph_local_increment.py']:
        fingerprints[str(path)] = file_hash(path)
    destination = args.output_dir / f'stage_{args.stage}'
    plan_path = destination / 'continuation_plan.json'
    canonical_command = command[:-2] if args.resume else command
    plan = {'command': canonical_command, 'input_sha256': fingerprints,
            'paper_comparable': False, 'update_policy': 'D2_partial_backbone_finetuning',
            'replay_pool_policy': 'existing_manifest_previous_train',
            'note': 'Same per-class quotas, not a fixed total memory budget'}
    if completed(destination / 'graph', d2['epochs']):
        print(f'Already complete: {destination}. Use --summarize.')
        return 0
    if args.resume:
        if not plan_path.is_file() or read(plan_path) != plan or not (destination / 'graph/checkpoint.pth').is_file():
            raise ValueError('Resume requires the unchanged original plan and current-stage checkpoint')
    elif destination.exists() and any(destination.iterdir()):
        raise ValueError('Output is not empty; use --resume with a checkpoint or a new output directory')
    print(shlex.join(command), flush=True)
    print('Replay quotas apply to all old classes; memory grows with the stage. Internal pilot only.', flush=True)
    if args.dry_run:
        print('Preflight passed; no graph extraction, training, or files created.')
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    environment = training_environment(args.gpus)
    result = subprocess.run(command, cwd=ROOT, env=environment)
    if result.returncode:
        return result.returncode
    if not completed(destination / 'graph', d2['epochs']):
        raise ValueError('Training exited without complete final evaluation artifacts')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f'D2 continuation preflight failed: {error}')
