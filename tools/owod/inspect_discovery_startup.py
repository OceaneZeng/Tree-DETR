"""Inspect discovery gates on legal train images, without tuning on test labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/ops'))
sys.path.insert(0, str(ROOT))


def inspect(directory, images=6):
    import torch
    from main import get_args_parser
    from models.graph_local.discovery import build
    from models.graph_local.discovery_candidates import (flip_samples, mine_candidates,
                                                         saliency_in_boxes, view_matches)
    from tools.owod.gnn_calibration import build_calibration_dataset
    from tools.owod.strict_setting import configure_run
    from util.misc import collate_fn
    from util.box_ops import box_cxcywh_to_xyxy, box_iou
    status = json.loads((directory / 'status.json').read_text(encoding='utf-8'))
    paired = (directory / 'task1_args.json').is_file()
    config = (json.loads((directory / 'task1_args.json').read_text(encoding='utf-8'))
              if paired else status['configuration'])
    args = get_args_parser().parse_args([
        '--owod-manifest', config['owod_manifest'], '--owod-stage', '0',
        '--owod-recipe', config['owod_recipe'], '--backbone-weights', config['backbone_weights'],
        '--owod-discovery', '--activation-checkpointing', '--num_classes', '91',
        '--num_queries', '100', '--dim_feedforward', '1024'])
    configure_run(args)
    checkpoint = torch.load(directory / ('graph_increment_diagnostic.pt' if paired else 'discovery_diagnostic.pt'),
                            map_location='cpu', weights_only=True)
    if (checkpoint.get('diagnostic_only') is not True
            or checkpoint['manifest_sha256'] != args.owod_manifest_sha256
            or status['backbone_sha256'] != args.backbone_weights_sha256):
        raise ValueError('Require this diagnostic and matching verified data/weights')
    torch.set_num_threads(2)
    model, criterion, _ = build(args)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.cuda().eval()
    criterion.cuda().eval()
    dataset = build_calibration_dataset(Path(args.owod_image_root), Path(args.train_ann))
    if any(a['category_id'] not in set(args.owod_known_class_ids) for a in dataset.coco.anns.values()):
        raise ValueError('Future train labels found')
    stream_file = 'task1_stream.json' if paired else 'training_stream.json'
    stream_key = 'image_ids' if paired else 'consumed_image_ids'
    selected = json.loads((directory / stream_file).read_text())[stream_key][:images]
    positions = {i: index for index, i in enumerate(dataset.ids)}
    rows = []
    with torch.no_grad():
        for image_id in selected:
            samples, targets = collate_fn([dataset[positions[image_id]]])
            samples = samples.to('cuda')
            targets = [{k: v.cuda() if torch.is_tensor(v) else v for k, v in target.items()}
                       for target in targets]
            output = model(samples)
            # Fresh trained self-teacher isolates view/gate geometry from EMA lag.
            # This is not the actual teacher used in the training trajectory.
            flipped = model(flip_samples(samples))
            indices = criterion.matcher(output, targets)
            mined = mine_candidates(output, targets, indices, args.owod_known_class_ids,
                                    flipped, args.discovery_top_k, args.discovery_saliency_threshold,
                                    args.discovery_consistency_threshold)
            saliency = saliency_in_boxes(output['attention_feature'], output['pred_boxes'],
                                        output['image_sizes'], output['padded_size'])[0]
            mapping, consistency, _ = view_matches(output, flipped)
            teacher_saliency = saliency_in_boxes(flipped['attention_feature'], flipped['pred_boxes'],
                                                flipped['image_sizes'], flipped['padded_size'])[0]
            boxes = output['pred_boxes'][0]
            valid = torch.isfinite(boxes).all(-1) & (boxes[:, 2:] > .005).all(-1)
            valid &= boxes[:, 2:].prod(-1) < .85
            excluded = torch.zeros_like(valid)
            excluded[indices[0][0]] = True
            if targets[0]['boxes'].numel():
                excluded |= box_iou(box_cxcywh_to_xyxy(boxes),
                                    box_cxcywh_to_xyxy(targets[0]['boxes']))[0].amax(-1) > .3
            eligible = valid & ~excluded
            known = output['pred_logits'][0, :, args.owod_known_class_ids].sigmoid().amax(-1)
            teacher_known = flipped['pred_logits'][0, :, args.owod_known_class_ids].sigmoid().amax(-1)
            semantic = eligible & (known < .3) & (teacher_known[mapping[0].clamp_min(0)] < .3)
            salient = eligible & (saliency >= args.discovery_saliency_threshold)
            both_salient = salient & (teacher_saliency[mapping[0].clamp_min(0)] >= args.discovery_saliency_threshold)
            stable = both_salient & (consistency[0] >= args.discovery_consistency_threshold)
            rows.append({'train_image_id': image_id, 'valid': int(valid.sum()),
                         'outside_known_GT': int(eligible.sum()), 'legacy_low_known_score': int(semantic.sum()),
                         'salient': int(salient.sum()), 'both_views_salient': int(both_salient.sum()),
                         'view_consistent': int(stable.sum()), 'selected_pseudo': int(mined['pseudo'].sum()),
                         'max_eligible_saliency': float(saliency[eligible].max()) if eligible.any() else None,
                         'max_eligible_consistency': float(consistency[0][eligible].max()) if eligible.any() else None})
    counts = model.memory.counts[model.memory.known_slots:].cpu().tolist()
    return {'scope': 'Training-only startup diagnostic; no future GT or test tuning.',
            'candidate_rule_version': args.discovery_version,
            'teacher': 'trained self-teacher on flipped inputs, not training EMA',
            'transforms': 'deterministic official val resize on legal training images',
            'anonymous_support': counts, 'anonymous_slots_with_support': sum(c > 0 for c in counts),
            'rows': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--images', default=6, type=int)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.images < 1:
        parser.error('Images must be positive')
    report = inspect(options.directory, options.images)
    output = options.output or options.directory / 'startup_probe.json'
    output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
