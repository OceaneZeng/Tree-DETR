#!/usr/bin/env python
"""Re-evaluate existing full-data control weights without optimizer changes."""
import argparse
import io
import json
import hashlib
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models/ops'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'output/task1-control-full/s-owodb/seed_42/stage_0/increment/checkpoint.pth')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a new empty evaluation directory')
    os.environ.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=args.gpu)
    import torch
    from tools.owod.protocol import stage_files, file_sha256
    from tools.owod.diagnose_learning import make_loader
    from tools.owod.screening_data import write_json
    from models.deformable_detr import build, PostProcess
    from models.owod_benchmark_eval import evaluate_author_benchmark
    # Read once: a running trainer may replace checkpoint.pth between epochs.
    content = args.checkpoint.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    payload = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    del content
    saved = payload['args']
    if (getattr(saved, 'owod_screening_sha256', None) or saved.owod_stage != 0
            or saved.owod_phase != 'increment' or saved.graph_routed_adapter or saved.paper_baseline
            or saved.owod_discovery):
        raise ValueError('Only full-data Task 1 supervised control checkpoints are supported')
    manifest_path = Path(saved.owod_manifest)
    manifest, files = stage_files(manifest_path, 0)
    if file_sha256(manifest_path) != saved.owod_manifest_sha256:
        raise ValueError('Checkpoint manifest changed')
    if file_sha256(Path(saved.backbone_weights)) != saved.backbone_weights_sha256:
        raise ValueError('Backbone input changed')
    if list(saved.owod_known_class_ids) != manifest['stages'][0]['active_classes']:
        raise ValueError('Checkpoint class order changed')
    saved.device = 'cuda'
    model, _, _ = build(saved)
    model.load_state_dict(payload['model'], strict=True)
    model.cuda().eval()
    dataset, loader = make_loader(manifest['image_root'], files['full_val'], 2, args.workers)
    known = saved.owod_known_class_ids
    processors = {name: PostProcess(known_ids=known, max_detections=saved.owod_max_detections,
                                   include_unknown=enabled)
                  for name, enabled in [('legacy_complement', True), ('known_sigmoid_v2', False)]}
    predictions = {name: {} for name in processors}
    with torch.no_grad():
        for index, (samples, targets) in enumerate(loader, 1):
            outputs = model(samples.to('cuda'))
            sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
            for name, processor in processors.items():
                for target, result in zip(targets, processor(outputs, sizes)):
                    predictions[name][int(target['image_id'])] = {k: v.cpu() for k, v in result.items()}
            if index % 100 == 0:
                print(f'Evaluation {index}/{len(loader)}', flush=True)
    report = {'checkpoint_sha256': digest, 'epoch': payload['epoch'],
              'manifest_sha256': saved.owod_manifest_sha256, 'postprocess_version': 'known-sigmoid-topk-v2',
              'note': 'Same raw outputs and Top-K; corrected control has no unknown detector. No training performed.',
              'metrics': {name: evaluate_author_benchmark(pred, dataset.coco, known, [], known, manifest['evaluation'])
                          for name, pred in predictions.items()},
              'unknown_predictions': {name: sum(int((r['labels'] == 91).sum()) for r in pred.values())
                                      for name, pred in predictions.items()}}
    write_json(args.output / 'reevaluation.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
