"""Read-only training-image probe; no optimizer, split changes or AP selection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'models/ops')]


def main():
    import torch
    from tools.owod.protocol import file_sha256, stage_files
    from tools.owod.gnn_calibration import load_detector, build_calibration_dataset, move_targets
    from tools.owod.supervision_probe import probe_batch, summarize
    from models.graph_local.discovery_candidates import mine_candidates
    from util.box_ops import box_iou, box_cxcywh_to_xyxy
    from util.misc import collate_fn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--images', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    options = parser.parse_args()
    if options.images < 1:
        parser.error('--images must be positive')
    torch.set_num_threads(2)
    identity = file_sha256(options.checkpoint)
    payload = torch.load(options.checkpoint, map_location='cpu', weights_only=False)
    saved = payload['args']
    if (not getattr(saved, 'owod_discovery', False) or saved.owod_stage != 0
            or saved.owod_phase != 'increment' or getattr(saved, 'owod_screening', None)):
        raise ValueError('Expected a full-data discovery Task 1 checkpoint')
    if file_sha256(options.manifest) != saved.owod_manifest_sha256:
        raise ValueError('Checkpoint manifest mismatch')
    manifest, files = stage_files(options.manifest, 0)
    known = manifest['stages'][0]['active_classes']
    if list(saved.owod_known_class_ids) != known:
        raise ValueError('Checkpoint class order mismatch')
    device = torch.device(options.device)
    model, criterion = load_detector(saved, options.checkpoint, device)
    model.eval()
    dataset = build_calibration_dataset(Path(manifest['image_root']), files['increment_train'])
    rows, image_ids = [], []
    counts = dict(queries=0, background_without_teacher=0, unmatched_known_overlap=0)
    with torch.no_grad():
        for index in range(min(options.images, len(dataset))):
            samples, targets = collate_fn([dataset[index]])
            targets = move_targets(targets, device)
            outputs = model(samples.to(device))
            rows.extend(probe_batch(outputs, targets, criterion.matcher, known))
            image_ids.append(int(targets[0]['image_id'].item()))
            indices = criterion.matcher(outputs, targets)
            candidates = mine_candidates(outputs, targets, indices, known)
            overlaps = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0]),
                               box_cxcywh_to_xyxy(targets[0]['boxes']))[0]
            duplicate = (overlaps.amax(-1) > .3 if overlaps.shape[1] else
                         torch.zeros(overlaps.shape[0], device=device, dtype=torch.bool))
            duplicate[indices[0][0]] = False
            counts['queries'] += outputs['pred_boxes'].shape[1]
            counts['background_without_teacher'] += int(candidates['background'].sum())
            counts['unmatched_known_overlap'] += int(duplicate.sum())
    if identity != file_sha256(options.checkpoint):
        raise ValueError('Checkpoint changed during probe; use a numbered epoch checkpoint')
    print(json.dumps(dict(checkpoint_sha256=identity, epoch=payload['epoch'],
                          discovery_version=getattr(saved, 'discovery_version', None),
                          image_ids=image_ids, summary=summarize(rows), supervision=counts,
                          note='Training-image diagnostic only, not AP or U-Recall. '
                               'Deterministic dataset prefix; not representative accuracy. '
                               'No EMA used: background count is an upper bound on two-view trust. '
                               'Coverage includes any correctly labelled IoU>0.5 candidate, '
                               'not one-to-one AP matching; score variants are diagnostic only.'),
                     indent=2))


if __name__ == '__main__':
    main()
