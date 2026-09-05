# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import copy
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from models.graph_local import (freeze_for_class_ids, inject_decoder_lora,
                                merge_decoder_lora)
from util.checkpoint import matching_state_dict
from util.experiment_log import (archive_log_file, experiment_log_path,
                                 prepare_log_file, start_file_logging,
                                 stop_file_logging)
from models.owod_detector import detector_profile_dict


def load_local_checkpoint(path):
    """Load a trusted local checkpoint across PyTorch 2.4+ releases."""
    safe_globals = getattr(torch.serialization, 'safe_globals', None)
    if safe_globals is None:
        # PyTorch 2.4 lacks safe_globals. These are local experiment or
        # official model-zoo checkpoints, whose provenance is controlled here.
        return torch.load(path, map_location='cpu', weights_only=False)
    with safe_globals([argparse.Namespace]):
        return torch.load(path, map_location='cpu', weights_only=True)


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument('--with_tree', action='store_true',
                        help='Enable the experimental Tree-DETR EE-0 flat-tree losses and adapters')

    parser.add_argument('--owod-manifest', default='',
                        help='S-OWODB/M-OWODB split manifest used for this run')
    parser.add_argument('--owod-stage', default=None, type=int,
                        help='Stage index in --owod-manifest, recorded in run metadata')
    parser.add_argument('--unknown-threshold', default=0.5, type=float,
                        help='Threshold used to mark post-processed predictions as unknown')
    parser.add_argument('--owod-known-class-ids', type=int, nargs='+', default=None,
                        help='Known category IDs for U-Recall/A-OSE/WI evaluation on full labels')
    parser.add_argument('--owod-previous-class-ids', type=int, nargs='+', default=None,
                        help='Previously learned category IDs for OWOD Previous AP50')
    parser.add_argument('--owod-current-class-ids', type=int, nargs='+', default=None,
                        help='Current-stage category IDs for OWOD Current AP50')
    parser.add_argument('--replay-sampling-fraction', default=0.0, type=float,
                        help='fixed fraction of each training epoch drawn from tagged replay images')
    parser.add_argument('--old-class-distillation', action='store_true',
                        help='preserve frozen previous-stage outputs on old classes')
    parser.add_argument('--distill-old-class-ids', type=int, nargs='+', default=None)
    parser.add_argument('--distill-class-coef', default=2.0, type=float)
    parser.add_argument('--distill-bbox-coef', default=5.0, type=float)
    parser.add_argument('--distill-score-threshold', default=0.3, type=float)
    parser.add_argument('--distill-max-queries', default=20, type=int)
    parser.add_argument('--neighbor-scoped-lora', action='store_true',
                        help='train a temporary stage-level LoRA on the final decoder FFNs')
    parser.add_argument('--lora-rank', default=8, type=int)
    parser.add_argument('--lora-last-decoder-layers', default=2, type=int)
    parser.add_argument('--trainable-class-ids', type=int, nargs='+', default=None,
                        help='classifier rows trained together with the temporary LoRA')
    parser.add_argument('--graph-local-class-ids', type=int, nargs='+', default=None,
                        help='current and GNN-neighbor classes used by the local margin')
    parser.add_argument('--local-margin-coef', default=0.0, type=float)
    parser.add_argument('--local-margin', default=1.0, type=float)
    parser.add_argument('--off-neighborhood-basis', default='', type=str)
    parser.add_argument('--off-projection-coef', default=0.0, type=float)
    parser.add_argument('--teacher-completion', action='store_true',
                        help='complete missing old foreground with frozen-teacher pseudo labels')
    parser.add_argument('--teacher-old-class-ids', type=int, nargs='+', default=None)
    parser.add_argument('--teacher-score-threshold', default=0.5, type=float)
    parser.add_argument('--teacher-duplicate-iou', default=0.7, type=float)
    parser.add_argument('--teacher-ground-truth-iou', default=0.5, type=float)
    parser.add_argument('--teacher-max-per-image', default=20, type=int)

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--num_classes', default=None, type=int,
                        help='Override the class count for a remapped lightweight dataset')
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--class_embed_lr_mult', default=1.0, type=float,
                        help='learning-rate multiplier for a newly initialized classifier')
    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='./data/coco', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrained', default='',
                        help='load model weights only and start a fresh optimizer/schedule')
    parser.add_argument('--reset-classifier', action='store_true',
                        help='when loading --pretrained, discard all classifier rows to avoid future-class leakage')
    parser.add_argument('--train-ann', default='',
                        help='override the COCO training annotation JSON (supports incremental splits)')
    parser.add_argument('--val-ann', default='',
                        help='override the COCO validation annotation JSON (supports incremental splits)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--skip-eval', action='store_true',
                        help='skip validation during training; useful when DDP COCO gather is unstable')
    parser.add_argument('--eval_interval', default=1, type=int,
                        help='evaluate every N epochs; the final epoch is always evaluated')
    parser.add_argument('--print-freq', default=100, type=int,
                        help='training progress lines are printed every N iterations')
    parser.add_argument('--eval-print-freq', default=100, type=int,
                        help='evaluation progress lines are printed every N iterations')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--lightweight', action='store_true',
                        help='Use 256-384 px image scales for feasibility experiments')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Use the validation resize for training; only for overfit diagnostics')
    parser.add_argument('--no-random-crop', action='store_true',
                        help='Keep flips and multi-scale resize but disable the COCO random-crop branch')

    parser.add_argument('--log-file', default='', type=str,
                        help='Human-readable log; relative paths are placed under output_dir')
    parser.add_argument('--no-file-log', action='store_true',
                        help='Disable the rank-0 human-readable experiment log')

    return parser


def main(args):
    utils.init_distributed_mode(args)
    file_log_state = start_file_logging(args, utils.is_main_process())
    if utils.is_main_process() and args.output_dir:
        run_config = vars(args).copy()
        run_config.update({'owod_detector_profile': detector_profile_dict()})
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        history_path = Path(args.output_dir) / 'run_history' / f'{run_id}.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        for config_path in (Path(args.output_dir) / 'run_config.json', history_path):
            with config_path.open('w', encoding='utf-8') as handle:
                json.dump(run_config, handle, indent=2, sort_keys=True, default=str)
        print('OWOD detector profile:', json.dumps(detector_profile_dict(), sort_keys=True))
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Keep the original detector path as the default.  The opt-in tree path is
    # currently the EE-0 flat-tree control, not an automatically induced tree.
    epoch_state = {'value': args.start_epoch}
    tree_head = None
    if args.with_tree:
        from models.tree import build_tree
        model, criterion, postprocessors, tree_head = build_tree(
            args, epoch_getter=lambda: epoch_state['value'])
        print('Tree-DETR enabled: EE-0 flat two-level tree with reserved-gap losses.')
    else:
        model, criterion, postprocessors = build_model(args)
    model.to(device)
    criterion.to(device)

    model_without_ddp = model
    teacher_model = None
    classifier_hook_handles = []
    off_neighborhood_basis = None
    output_dir = Path(args.output_dir)

    if args.neighbor_scoped_lora and args.with_tree:
        raise ValueError('--neighbor-scoped-lora currently requires the Deformable DETR path')
    if args.neighbor_scoped_lora and not args.trainable_class_ids:
        raise ValueError('--neighbor-scoped-lora requires --trainable-class-ids')
    if args.local_margin_coef > 0 and not args.graph_local_class_ids:
        raise ValueError('--local-margin-coef requires --graph-local-class-ids')
    if args.off_projection_coef > 0 and not args.neighbor_scoped_lora:
        raise ValueError('--off-projection-coef requires --neighbor-scoped-lora')
    if args.off_projection_coef > 0 and not args.off_neighborhood_basis:
        raise ValueError('--off-projection-coef requires --off-neighborhood-basis')
    if args.teacher_completion and not args.teacher_old_class_ids:
        raise ValueError('--teacher-completion requires --teacher-old-class-ids')

    if args.frozen_weights is not None:
        checkpoint = load_local_checkpoint(args.frozen_weights)
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    # Load the previous-stage detector while it still has the plain Deformable
    # DETR module structure. The frozen teacher must not contain current-stage
    # LoRA factors or weights restored from an interrupted current-stage run.
    if args.pretrained:
        checkpoint = load_local_checkpoint(args.pretrained)
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        compatible, skipped = matching_state_dict(model_without_ddp, state_dict)
        if getattr(args, 'reset_classifier', False):
            classifier_keys = tuple(key for key in compatible
                                    if key.startswith('class_embed.') or
                                    key.startswith('transformer.decoder.class_embed.'))
            for key in classifier_keys:
                del compatible[key]
            print(f'Discarded {len(classifier_keys)} pretrained classifier tensors.')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(compatible, strict=False)
        print(f'Loaded {len(compatible)} shape-compatible pretrained tensors only; skipped={skipped}.')
        if missing_keys:
            print('Missing Keys: {}'.format(missing_keys))
        if unexpected_keys:
            print('Unexpected Keys: {}'.format(unexpected_keys))

    teacher_required = args.teacher_completion or args.old_class_distillation
    if teacher_required:
        if not args.pretrained:
            raise ValueError('teacher completion/distillation requires --pretrained')
        if args.old_class_distillation and not args.distill_old_class_ids:
            raise ValueError('--old-class-distillation requires --distill-old-class-ids')
        teacher_model = copy.deepcopy(model_without_ddp).to(device)
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad_(False)
        print('Enabled frozen previous-stage teacher:', json.dumps({
            'completion_classes': args.teacher_old_class_ids or [],
            'distillation_classes': args.distill_old_class_ids or [],
        }, sort_keys=True))

    if args.neighbor_scoped_lora:
        wrappers = inject_decoder_lora(
            model_without_ddp, rank=args.lora_rank,
            last_n=args.lora_last_decoder_layers)
        classifier_hook_handles, _trainable = freeze_for_class_ids(
            model_without_ddp, args.trainable_class_ids)
        print('Enabled neighbor-scoped stage LoRA:', json.dumps({
            'rank': args.lora_rank,
            'decoder_layers': args.lora_last_decoder_layers,
            'wrapped_linears': len(wrappers),
            'trainable_class_ids': sorted(set(args.trainable_class_ids)),
            'graph_local_class_ids': sorted(set(args.graph_local_class_ids or [])),
        }, sort_keys=True))

    if args.off_neighborhood_basis:
        off_neighborhood_basis = load_local_checkpoint(args.off_neighborhood_basis)
        if isinstance(off_neighborhood_basis, dict):
            off_neighborhood_basis = off_neighborhood_basis['basis']
        if not torch.is_tensor(off_neighborhood_basis):
            raise ValueError('--off-neighborhood-basis must contain a tensor or a basis entry')
        off_neighborhood_basis = off_neighborhood_basis.to(device)
        print('Loaded off-neighborhood basis:', tuple(off_neighborhood_basis.shape))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params:', n_parameters)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    replay_indices = [
        index for index, image_id in enumerate(getattr(dataset_train, 'ids', []))
        if bool(dataset_train.coco.imgs[int(image_id)].get('owod_replay', False))
    ]
    if args.replay_sampling_fraction > 0:
        sampler_train = samplers.ReplayBalancedSampler(
            dataset_train, replay_indices,
            replay_fraction=args.replay_sampling_fraction,
            num_replicas=utils.get_world_size(), rank=utils.get_rank(), seed=args.seed)
        print('Balanced replay sampler:', json.dumps({
            'tagged_replay_images': len(replay_indices),
            'current_images': len(dataset_train) - len(replay_indices),
            'requested_fraction': args.replay_sampling_fraction,
            'samples_per_rank': len(sampler_train),
            'replay_samples_per_rank': sampler_train.replay_per_rank,
        }, sort_keys=True))
        if args.distributed:
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    elif args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    class_head_names = ['class_embed']
    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names)
                 and not match_name_keywords(n, args.lr_linear_proj_names)
                 and not match_name_keywords(n, class_head_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, class_head_names) and p.requires_grad],
            "lr": args.lr * args.class_embed_lr_mult,
            # Row masks freeze previous-class gradients. Disabling decay keeps
            # those rows bitwise fixed even though the tensor is optimized.
            "weight_decay": 0.0 if args.neighbor_scoped_lora else args.weight_decay,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = load_local_checkpoint(args.resume)
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint['optimizer'])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg['lr'] = pg_old['lr']
                pg['initial_lr'] = pg_old['initial_lr']
            optimizer_summary = [
                {
                    'group': index,
                    'lr': group.get('lr'),
                    'initial_lr': group.get('initial_lr'),
                    'weight_decay': group.get('weight_decay'),
                    'parameter_tensors': len(group.get('params', [])),
                    'parameters': sum(parameter.numel() for parameter in group.get('params', [])),
                }
                for index, group in enumerate(optimizer.param_groups)
            ]
            print('Restored optimizer groups:', optimizer_summary)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            # todo: this is a hack for doing experiment that resume from checkpoint and also modify lr scheduler (e.g., decrease lr in advance).
            args.override_resumed_lr_drop = True
            if args.override_resumed_lr_drop:
                print('Warning: (hack) args.override_resumed_lr_drop is set to True, so args.lr_drop would override lr_drop in resumed lr_scheduler.')
                lr_scheduler.step_size = args.lr_drop
                lr_scheduler.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint['epoch'] + 1
        # check the resumed model
        if not args.eval and not getattr(args, 'skip_eval', False):
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                owod_known_class_ids=args.owod_known_class_ids,
                owod_unknown_threshold=args.unknown_threshold,
                owod_previous_class_ids=args.owod_previous_class_ids,
                owod_current_class_ids=args.owod_current_class_ids,
                print_freq=args.eval_print_freq,
            )
    
    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir,
                                              owod_known_class_ids=args.owod_known_class_ids,
                                              owod_unknown_threshold=args.unknown_threshold,
                                              owod_previous_class_ids=args.owod_previous_class_ids,
                                              owod_current_class_ids=args.owod_current_class_ids,
                                              print_freq=args.eval_print_freq)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        stop_file_logging(file_log_state)
        return

    print("Start training")
    structured_log = None
    if args.output_dir and utils.is_main_process():
        structured_log = experiment_log_path(output_dir, "metrics.jsonl")
        prepare_log_file(structured_log, append=bool(args.resume))
        completion_marker = output_dir / 'training_complete.json'
        if completion_marker.is_file():
            archive_log_file(completion_marker)
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        epoch_state['value'] = epoch
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm,
            print_freq=args.print_freq, teacher=teacher_model,
            distill_old_class_ids=args.distill_old_class_ids,
            distill_class_coef=args.distill_class_coef,
            distill_bbox_coef=args.distill_bbox_coef,
            distill_score_threshold=args.distill_score_threshold,
            distill_max_queries=args.distill_max_queries,
            teacher_completion=args.teacher_completion,
            teacher_old_class_ids=args.teacher_old_class_ids,
            teacher_score_threshold=args.teacher_score_threshold,
            teacher_duplicate_iou=args.teacher_duplicate_iou,
            teacher_ground_truth_iou=args.teacher_ground_truth_iou,
            teacher_max_per_image=args.teacher_max_per_image,
            graph_local_class_ids=args.graph_local_class_ids,
            local_margin_coef=args.local_margin_coef,
            local_margin=args.local_margin,
            off_neighborhood_basis=off_neighborhood_basis,
            off_projection_coef=args.off_projection_coef)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        should_evaluate = (not getattr(args, 'skip_eval', False) and
                            ((epoch + 1) % max(1, args.eval_interval) == 0
                             or epoch + 1 == args.epochs))
        if should_evaluate:
            test_stats, coco_evaluator = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                owod_known_class_ids=args.owod_known_class_ids,
                owod_unknown_threshold=args.unknown_threshold,
                owod_previous_class_ids=args.owod_previous_class_ids,
                owod_current_class_ids=args.owod_current_class_ids,
                print_freq=args.eval_print_freq,
            )
        else:
            test_stats, coco_evaluator = {}, None

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with structured_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if args.output_dir and utils.is_main_process():
        if args.neighbor_scoped_lora:
            merged_count = merge_decoder_lora(
                model_without_ddp, last_n=args.lora_last_decoder_layers)
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'epoch': args.epochs - 1,
                'args': args,
                'merged_lora_linears': merged_count,
                'source_checkpoint': str(output_dir / 'checkpoint.pth'),
            }, output_dir / 'checkpoint_merged.pth')
            print('Merged {} LoRA linears into checkpoint_merged.pth'.format(merged_count))
        with (output_dir / 'training_complete.json').open('w') as f:
            json.dump({'epochs': args.epochs, 'last_epoch': args.epochs - 1,
                       'training_time_seconds': int(total_time),
                       'merged_checkpoint': ('checkpoint_merged.pth'
                                             if args.neighbor_scoped_lora else None)}, f)
    for handle in classifier_hook_handles:
        handle.remove()
    stop_file_logging(file_log_state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
