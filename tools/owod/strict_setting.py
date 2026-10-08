"""Shared preflight: official task phases, labels, data hashes, and checkpoint lineage."""
from pathlib import Path

from tools.owod.protocol import file_sha256, stage_files, RETIRED_MESSAGE


def configure_run(args):
    requested = bool(args.owod_manifest or args.owod_known_class_ids is not None or args.paper_baseline)
    if not requested:
        return
    if not args.owod_manifest or args.owod_stage is None:
        raise ValueError('OWOD experiments require an author-release manifest and stage')
    manifest, files = stage_files(Path(args.owod_manifest), args.owod_stage)
    recipe = getattr(args, 'owod_recipe', None)
    if recipe not in ('ore-paper', 'ow-detr-paper'):
        raise ValueError('OWOD runs must explicitly select --owod-recipe ore-paper or ow-detr-paper')
    if recipe == 'ore-paper' and manifest['protocol'] != 'm-owodb':
        raise ValueError('ORE 2021 does not define an S-OWODB recipe')
    phase = args.owod_phase
    if phase == 'finetune' and args.owod_stage == 0:
        raise ValueError('Task 1 has no incremental balanced-finetuning phase')
    expected_train = files['increment_train' if phase == 'increment' else 'train']
    for field, expected in [('train_ann', expected_train), ('val_ann', files['full_val'])]:
        supplied = getattr(args, field, '')
        if supplied and file_sha256(Path(supplied)) != file_sha256(expected):
            raise ValueError(f'{field} does not match the official {phase} phase; {RETIRED_MESSAGE}')
        setattr(args, field, str(expected))
    record = manifest['stages'][args.owod_stage]
    previous = [c for c in manifest['category_order_source_ids'] if
                c in set(record['active_classes']) - set(record['classes'])]
    for field, values in [('owod_known_class_ids', record['active_classes']),
                          ('owod_current_class_ids', record['classes']),
                          ('owod_previous_class_ids', previous)]:
        supplied = getattr(args, field)
        if supplied is not None and set(supplied) != set(values):
            raise ValueError(f'{field} differs from the author-defined stage')
        setattr(args, field, values)
    image_root = manifest.get('image_root')
    if not image_root:
        raise ValueError('Author manifest must record image_root; re-register with --image-root')
    args.owod_image_root = str(Path(image_root).resolve())
    if not Path(args.owod_image_root).is_dir():
        raise FileNotFoundError(f'Benchmark images missing: {args.owod_image_root}')
    if args.lightweight or args.no_augmentation or args.replay_sampling_fraction:
        raise ValueError('Old lightweight/overfit/custom replay-exposure settings are retired for OWOD')
    if args.owod_stage > 0 and not (args.pretrained or args.resume):
        raise ValueError('Incremental tasks must continue the preceding task checkpoint')
    if args.owod_stage == 0 and args.pretrained:
        raise ValueError('Task 1 cannot load a COCO detector checkpoint; use backbone-only pretraining')
    if manifest['protocol'] == 's-owodb' and not args.backbone_weights:
        raise ValueError('S-OWODB original OW-DETR setting requires --backbone-weights with DINO ResNet-50 weights')
    args.owod_manifest_sha256 = file_sha256(Path(args.owod_manifest))
    args.backbone_weights_sha256 = file_sha256(Path(args.backbone_weights)) if args.backbone_weights else None
    args.owod_evaluation = manifest['evaluation']
    args.owod_protocol = manifest['protocol']
    args.owod_max_detections = 50 if (manifest['protocol'] == 's-owodb'
                                   or getattr(args, 'owod_recipe', None) == 'ow-detr-paper') else 100
    args.paper_comparable_protocol = True


def validate_checkpoint(args, checkpoint, resume=False):
    if not getattr(args, 'owod_manifest', ''):
        return
    saved = checkpoint.get('args')
    if (saved is None or getattr(saved, 'owod_manifest_sha256', None) != args.owod_manifest_sha256):
        raise ValueError('Checkpoint belongs to an old/custom/different benchmark manifest')
    if getattr(saved, 'owod_recipe', None) != args.owod_recipe:
        raise ValueError('Training recipe changed within the checkpoint chain')
    stage = args.owod_stage if resume or args.owod_phase == 'finetune' else args.owod_stage - 1
    phase = args.owod_phase if resume else ('increment' if args.owod_phase == 'finetune'
                                          or stage == 0 else 'finetune')
    if getattr(saved, 'owod_stage', None) != stage or getattr(saved, 'owod_phase', None) != phase:
        raise ValueError(f'Expected checkpoint from stage_{stage}/{phase}')
    if not resume and checkpoint.get('epoch') != getattr(saved, 'epochs', 0) - 1:
        raise ValueError('Stage transition requires the completed preceding phase checkpoint')
    if getattr(saved, 'backbone_weights_sha256', None) != getattr(args, 'backbone_weights_sha256', None):
        raise ValueError('Backbone pretraining changed within the checkpoint chain')
    if resume:
        for field in ('epochs', 'lr', 'lr_backbone', 'lr_drop', 'batch_size', 'weight_decay',
                      'seed', 'sgd', 'backbone', 'num_classes', 'num_queries', 'world_size'):
            if getattr(saved, field, None) != getattr(args, field, None):
                raise ValueError(f'Strict benchmark resume cannot change {field}')
