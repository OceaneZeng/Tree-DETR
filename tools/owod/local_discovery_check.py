"""Budgeted real-image diagnostic; never produces official training checkpoints.

Keeps the verified author splits, transforms and full test evaluation. Only a
prefix of a seeded train permutation is consumed. This is interrupted training,
not a new benchmark split or an author schedule reproduction.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/ops'))
sys.path.insert(0, str(ROOT))


def save(path, payload):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def metric_delta(control, treatment):
    keys = ('previous_ap50', 'known_ap50', 'current_ap50', 'u_recall', 'h_score', 'wi', 'a_ose', 'unknown_precision')
    return {k: {'control': control[k], 'discovery': treatment[k],
                'delta': treatment[k] - control[k]} for k in keys if k in control and k in treatment}


def run(options):
    import gc
    import torch
    from torch.utils.data import DataLoader
    from datasets.coco import CocoDetection, make_coco_transforms
    from engine import train_one_epoch
    from main import get_args_parser
    from models.deformable_detr import build as build_base, PostProcess
    from models.graph_local.discovery import DiscoveryDetector, DiscoveryCriterion, DiscoveryPostProcess, make_ema
    from models.owod_benchmark_eval import evaluate_author_benchmark
    from tools.owod.gnn_calibration import set_seed
    from tools.owod.manage_data import verify_dataset
    from tools.owod.strict_setting import configure_run
    from util.misc import collate_fn

    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise ValueError('Use a fresh diagnostic output directory; no implicit resume')
    status = {'status': 'running', 'scope': 'Task 1 short-training diagnostic',
              'official_training_result': False, 'ap_measured': False,
              'steps_per_method': options.steps, 'batch_size': 2, 'seed': options.seed,
              'task2_promotion_tested': False, 'methods': {}}
    def update(**kwargs):
        status.update(kwargs)
        status['updated_unix'] = time.time()
        save(out / 'status.json', status)
    update(phase='verify_data')
    try:
        manifest = json.loads(options.manifest.read_text(encoding='utf-8'))
        status['data_verification'] = verify_dataset(options.manifest.parent, manifest['protocol'])
        args = get_args_parser().parse_args([
            '--owod-manifest', str(options.manifest.resolve()), '--owod-stage', '0',
            '--owod-recipe', 'ow-detr-paper', '--backbone-weights', str(options.weights.resolve()),
            '--activation-checkpointing', '--num_classes', '91', '--num_queries', '100',
            '--dim_feedforward', '1024', '--seed', str(options.seed)])
        configure_run(args)
        # Identical official datasets for both methods, with train future labels absent.
        train = CocoDetection(args.owod_image_root, args.train_ann,
                              make_coco_transforms('train'), return_masks=False)
        val = CocoDetection(args.owod_image_root, args.val_ann,
                            make_coco_transforms('val'), return_masks=False)
        if options.steps * 2 > len(train):
            raise ValueError('Update budget exceeds a single training stream; use a smaller diagnostic budget')
        released = set(args.owod_known_class_ids)
        if any(a['category_id'] not in released for a in train.coco.anns.values()):
            raise ValueError('Future-class supervision found in train annotations')
        generator = torch.Generator().manual_seed(options.seed)
        order = torch.randperm(len(train), generator=generator).tolist()
        consumed = order[:options.steps * 2]
        ids = [train.ids[i] for i in consumed]
        save(out / 'training_stream.json', {'official_train_annotation': args.train_ann,
                                          'consumed_image_ids': ids})
        status.update(manifest_sha256=args.owod_manifest_sha256,
                      backbone_sha256=args.backbone_weights_sha256,
                      train_stream_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                      full_test_images=len(val), full_train_images=len(train),
                      configuration=vars(args),
                      limitations=['Single seed and only 300 updates by default; no convergence evidence.',
                                   'Task 1 only: cannot substantiate unknown-to-known transfer or retention.',
                                   'Control and discovery differ in added capacity and prediction scoring.',
                                   'No test-based epoch, threshold or hyperparameter selection.'])
        torch.set_num_threads(2)
        set_seed(options.seed)
        base, base_criterion, _ = build_base(args)
        initial = copy.deepcopy(base.state_dict())
        del base
        for method in ('control', 'discovery'):
            update(phase='train', method=method)
            set_seed(options.seed)
            base, criterion, _ = build_base(args)
            base.load_state_dict(initial, strict=True)
            if method == 'discovery':
                model = DiscoveryDetector(base, args)
                criterion = DiscoveryCriterion(criterion, model, args)
                post = DiscoveryPostProcess(known_ids=args.owod_known_class_ids, max_detections=50)
            else:
                model = base
                post = PostProcess(known_ids=args.owod_known_class_ids, max_detections=50)
            model.cuda()
            criterion.cuda()
            teacher = make_ema(model) if method == 'discovery' else None
            groups = {}
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    lr = args.lr_backbone if 'backbone.0' in name else (
                        args.lr * args.lr_linear_proj_mult if any(k in name for k in args.lr_linear_proj_names)
                        else args.lr * args.class_embed_lr_mult if 'class_embed' in name else args.lr)
                    groups.setdefault(lr, []).append(parameter)
            optimizer = torch.optim.AdamW([{'params': p, 'lr': lr} for lr, p in groups.items()],
                                          lr=args.lr, weight_decay=args.weight_decay)
            training = []
            # Chunks are diagnostic warmup windows, NOT completed official epochs.
            chunk_size = options.steps // 3
            for chunk in range(3):
                set_seed(options.seed + chunk)
                batches = [consumed[i:i + 2] for i in
                           range(chunk * chunk_size * 2, (chunk + 1) * chunk_size * 2, 2)]
                loader = DataLoader(train, batch_sampler=batches, num_workers=options.workers,
                                    collate_fn=collate_fn, pin_memory=True)
                if method == 'discovery':
                    criterion.epoch = chunk
                result = train_one_epoch(model, criterion, loader, optimizer, torch.device('cuda'),
                                         chunk, max_norm=args.clip_max_norm, print_freq=25,
                                         discovery_teacher=teacher)
                training.append(result)
                status['methods'][method] = {'completed_steps': (chunk + 1) * chunk_size,
                                            'train': training}
                update()
                del loader
            if method == 'discovery':
                counts = model.memory.counts[model.memory.known_slots:].detach().cpu()
                status['methods'][method]['anonymous_support'] = counts.tolist()
                status['methods'][method]['populated_anonymous_slots'] = int((counts > 0).sum())
            del optimizer, teacher
            gc.collect()
            torch.cuda.empty_cache()
            # Diagnostic state cannot be mistaken for main.py benchmark checkpoint.
            torch.save({'diagnostic_only': True, 'manifest_sha256': args.owod_manifest_sha256,
                        'method': method, 'updates': options.steps, 'model': model.state_dict()},
                       out / (method + '_diagnostic.pt'))
            update(phase='full_test', method=method)
            model.eval()
            predictions = {}
            loader = DataLoader(val, batch_size=2, shuffle=False, num_workers=options.workers,
                                collate_fn=collate_fn, pin_memory=True)
            with torch.no_grad():
                for index, (samples, targets) in enumerate(loader):
                    outputs = model(samples.to('cuda'))
                    sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
                    results = post(outputs, sizes)
                    for target, prediction in zip(targets, results):
                        predictions[int(target['image_id'])] = {
                            k: v.cpu() for k, v in prediction.items() if k in ('scores', 'labels', 'boxes')}
                    if index % 100 == 0:
                        print(f'{method}: evaluated {len(predictions)}/{len(val)}', flush=True)
                        update(evaluated_images=len(predictions))
                    del outputs, results
            metrics = evaluate_author_benchmark(predictions, val.coco, args.owod_known_class_ids,
                                                [], args.owod_current_class_ids, args.owod_evaluation)
            torch.save(predictions, out / (method + '_predictions.pt'))
            status['methods'][method].update(metrics=metrics)
            save(out / (method + '_metrics.json'), metrics)
            print(method, json.dumps(metrics), flush=True)
            update()
            del model, base, criterion, post, predictions, loader
            gc.collect()
            torch.cuda.empty_cache()
        differences = metric_delta(status['methods']['control']['metrics'],
                                   status['methods']['discovery']['metrics'])
        result = {'status': 'measured_short_training', 'official_training_result': False,
                  'protocol': manifest['protocol'], 'task': 1, 'steps': options.steps,
                  'full_test_images': len(val), 'metrics': differences,
                  'scale': 'AP50, recall, H-Score are fractions; multiply by 100 for points.',
                  'ap_and_hscore_improved': differences['known_ap50']['delta'] > 0
                                          and differences['h_score']['delta'] > 0,
                  'idea_effectiveness': 'not_established_by_short_training_alone',
                  'diagnosis': {
                      'control_all_50_predictions_per_image_unknown':
                          status['methods']['control']['metrics']['unknown_predictions'] == 50 * len(val),
                      'discovery_unknown_predictions': status['methods']['discovery']['metrics']['unknown_predictions'],
                      'training_pseudo_candidates': sum(
                          chunk['discovery_pseudo_count'] * chunk_size
                          for chunk in status['methods']['discovery']['train']),
                      'populated_anonymous_slots': status['methods']['discovery']['populated_anonymous_slots'],
                      'task2_transfer_tested': False},
                  'limitations': status['limitations']}
        save(out / 'comparison.json', result)
        update(status='completed', phase='finished', ap_measured=True)
        print(json.dumps(result, indent=2), flush=True)
        return result
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/discovery/local-check')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    options = parser.parse_args()
    if options.steps < 3 or options.steps % 3 or options.workers < 0:
        parser.error('Steps must be a positive multiple of 3; workers must be nonnegative')
    run(options)


if __name__ == '__main__':
    main()
