"""Isolated, budgeted S/M diagnostics: graph messages and anonymous transfer.

Uses verified author annotations, ordinary train augmentation and full test.
Never creates formal checkpoints or permits them to resume these short runs.
"""
import argparse
import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'models/ops'))
sys.path.insert(0, str(ROOT))
from tools.owod.local_discovery_check import save, metric_delta


def state_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def require_parent(checkpoint, manifest_hash, version):
    if (checkpoint.get('diagnostic_only') is not True
            or checkpoint.get('manifest_sha256') != manifest_hash
            or checkpoint.get('discovery_version') != version
            or checkpoint.get('task') != 1 or checkpoint.get('phase') != 'increment'
            or checkpoint.get('method') != 'graph'):
        raise ValueError('Diagnostic transition requires its matching Task 1 graph parent')


def audit_transition(directory, methods=('transfer', 'scratch')):
    """Read completed diagnostic states; distinguish inheritance from learning."""
    import torch
    parent = torch.load(directory / 'graph_increment_diagnostic.pt', map_location='cpu', weights_only=True)
    config = json.loads((directory / 'task2_args.json').read_text(encoding='utf-8'))
    old, new = config['owod_previous_class_ids'], config['owod_current_class_ids']
    report = {'scope': 'Parameter audit, not AP evidence', 'methods': {}}
    for method in methods:
        payload = torch.load(directory / (method + '_increment_diagnostic.pt'), map_location='cpu', weights_only=True)
        if (not payload.get('diagnostic_only') or payload.get('task') != 2
                or payload.get('phase') != 'increment' or payload.get('method') != method
                or payload.get('manifest_sha256') != parent['manifest_sha256']
                or payload.get('discovery_version') != parent['discovery_version']):
            raise ValueError('Cannot audit mismatched diagnostic states')
        state = payload['model']
        promotion = payload['spec']['promotion']
        promoted = {p['class_id']: p for p in promotion}
        old_changes, new_changes, box_changes, adapters = [], [], [], []
        for name, tensor in state.items():
            if 'class_embed' in name and name in parent['model']:
                difference = tensor - parent['model'][name]
                old_changes.append(float(difference[old].abs().max()))
                new_changes.append(float(difference[new].norm()))
            if 'bbox_embed' in name and name in parent['model']:
                box_changes.append(float((tensor - parent['model'][name]).abs().max()))
            if '.graph_adapter.stages.0.adapters.' in name and name.endswith('.up.weight'):
                index = int(name.split('.adapters.')[1].split('.')[0])
                effective = tensor @ state[name.replace('.up.weight', '.down.weight')]
                initial = torch.zeros_like(effective)
                record = promoted[new[index]]
                if record['inherited']:
                    prefix = name.split('.graph_adapter.')[0] + '.discovery_adapter.'
                    for slot, weight in zip(record['anonymous_slots'], record['weights']):
                        initial += weight * (parent['model'][prefix + 'snapshot_up'][slot]
                                             @ parent['model'][prefix + 'snapshot_down'][slot])
                    u, singular, vh = torch.linalg.svd(initial, full_matrices=False)
                    rank = payload['spec']['rank']
                    initial = (u[:, :rank] * singular[:rank]) @ vh[:rank]
                adapters.append({'parameter': name, 'class_id': new[index], 'delta_norm': float(effective.norm()),
                                 'change_from_initial_norm': float((effective - initial).norm())})
        report['methods'][method] = dict(old_classifier_max_change=max(old_changes),
            new_classifier_change_norm=max(new_changes), bbox_head_max_change=max(box_changes),
            nonzero_adapter_classes=sorted({a['class_id'] for a in adapters if a['delta_norm'] > 1e-10}),
            updated_adapter_classes=sorted({a['class_id'] for a in adapters if a['change_from_initial_norm'] > 1e-6}),
            adapter_deltas=adapters, inherited_classes=sum(p['inherited'] for p in promotion),
            inherited_slot_sets=[p['anonymous_slots'] for p in promotion if p['inherited']],
            effective_risk_beta=payload['spec']['risk_beta'])
    save(directory / 'transition_audit.json', report)
    return report


def run(options):
    import torch
    from torch.utils.data import DataLoader
    from datasets.coco import CocoDetection, make_coco_transforms
    from engine import train_one_epoch
    from main import get_args_parser
    from models.graph_local.discovery import build, make_ema, DISCOVERY_VERSION
    from models.graph_local.routed_adapter import append_stage, configure_trainable, RiskRoutedStage
    from models.owod_benchmark_eval import evaluate_author_benchmark
    from tools.owod.gnn_calibration import set_seed
    from tools.owod.graph_adaptation import collect_promotion_features
    from tools.owod.manage_data import verify_dataset
    from tools.owod.protocol import stage_files, file_sha256
    from tools.owod.strict_setting import configure_run
    from util.misc import collate_fn

    out = options.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'status.json').exists():
        raise ValueError('Use a fresh output directory; implicit diagnostic resume is forbidden')
    status = dict(status='running', official_training_result=False, discovery_version=DISCOVERY_VERSION,
                  seed=options.seed, study=options.study, methods={}, limitations=[
                      'One seed; interrupted training, not author training schedule or convergence.',
                      'Three short Task 1 warmup windows are not full official epochs.',
                      'Task 2 risk suppression disabled identically in both arms to isolate transfer.',
                      'No test-based model, threshold or hyperparameter selection.'])
    optimization = options.study == 'optimization'
    status['budgets'] = dict(task1=options.steps, task2_increment=options.task2_steps,
                             task2_finetune=options.finetune_steps)
    if optimization:
        status['limitations'][2] = 'Task 2 risk suppression disabled identically in both arms to isolate routing.'
        status['prespecified_comparisons'] = ['same_weights_focal_vs_hierarchical',
                                              'same_parent_soft_vs_hard_no_transfer']
        status['limitations'].append('Known-only scoring is a diagnostic, not an OWOD result.')
    def update(**values):
        status.update(values)
        status['updated_unix'] = time.time()
        save(out / 'status.json', status)
    def clean():
        gc.collect()
        torch.cuda.empty_cache()
    def dataset(args, key, mode):
        return CocoDetection(args.owod_image_root, key, make_coco_transforms(mode), False)
    def stream(ds, steps, seed, name):
        if steps * 2 > len(ds):
            raise ValueError('Budget exceeds one permutation of the author split')
        indices = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed)).tolist()[:steps * 2]
        save(out / (name + '_stream.json'), {'image_ids': [ds.ids[i] for i in indices], 'indices': indices})
        return indices
    def optimize(model, args, increment=False):
        groups = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            lr = args.lr_backbone if 'backbone.0' in name else (
                args.lr * args.lr_linear_proj_mult if any(k in name for k in args.lr_linear_proj_names)
                else args.lr * args.class_embed_lr_mult if 'class_embed' in name else args.lr)
            decay = 0. if increment and 'class_embed' in name else args.weight_decay
            groups.setdefault((lr, decay), []).append(parameter)
        return torch.optim.AdamW([dict(params=p, lr=lr, weight_decay=d) for (lr, d), p in groups.items()])
    def train(model, criterion, args, ds, indices, name, windows, teacher=None, finetune=False):
        update(phase='train', method=name)
        optimizer = optimize(model, args, args.owod_stage > 0)
        ema = make_ema(model)
        records = []
        size = len(indices) // windows
        for window in range(windows):
            set_seed(options.seed + args.owod_stage * 100 + window + (1000 if finetune else 0))
            selected = indices[window * size:(window + 1) * size]
            loader = DataLoader(ds, batch_sampler=[selected[i:i + 2] for i in range(0, len(selected), 2)],
                                num_workers=options.workers, collate_fn=collate_fn, pin_memory=True)
            criterion.epoch = window if args.owod_stage == 0 else args.discovery_warmup
            records.append(train_one_epoch(
                model, criterion, loader, optimizer, torch.device('cuda'), window,
                max_norm=args.clip_max_norm, print_freq=50, discovery_teacher=ema,
                teacher=teacher, distill_old_class_ids=args.owod_previous_class_ids,
                distill_class_coef=args.distill_class_coef, distill_bbox_coef=args.distill_bbox_coef,
                distill_score_threshold=args.distill_score_threshold, distill_max_queries=args.distill_max_queries))
            status['methods'].setdefault(name, {})['train'] = records
            update()
        counts = model.memory.counts[model.memory.known_slots:].detach().cpu()
        status['methods'][name].update(anonymous_support=counts.tolist(),
                                     populated_anonymous_slots=int((counts > 0).sum()),
                                     pseudo_candidates=sum(r['discovery_pseudo_count'] * (size // 2) for r in records))
        del ema, optimizer, loader
        clean()
    def checkpoint(model, args, method, phase, spec=None):
        path = out / (method + '_' + phase + '_diagnostic.pt')
        torch.save(dict(diagnostic_only=True, manifest_sha256=args.owod_manifest_sha256,
                        discovery_version=DISCOVERY_VERSION, task=args.owod_stage + 1,
                        phase=phase, method=method, spec=spec, model=model.state_dict()), path)
        return path
    def evaluate(model, args, ds, name):
        update(phase='full_test', method=name, evaluated_images=0)
        model.eval()
        from models.graph_local.discovery import DiscoveryPostProcess
        modes = ('hierarchical', 'focal', 'known_only') if optimization else ('hierarchical',)
        posts = {mode: DiscoveryPostProcess(known_ids=args.owod_known_class_ids,
                 max_detections=args.owod_max_detections,
                 scoring='hierarchical' if mode == 'hierarchical' else 'focal') for mode in modes}
        # Same images/transforms; deterministic aspect sorting reduces padded pixels.
        order = sorted(range(len(ds)), key=lambda i: ds.coco.imgs[ds.ids[i]]['width'] / ds.coco.imgs[ds.ids[i]]['height'])
        loader = DataLoader(ds, batch_sampler=[order[i:i + 2] for i in range(0, len(order), 2)],
                            num_workers=options.workers, collate_fn=collate_fn, pin_memory=True)
        predictions = {mode: {} for mode in modes}
        coverage = dict(known_gt=0, localized_known_gt=0, localized_correct_class_gt=0)
        with torch.no_grad():
            for index, (samples, targets) in enumerate(loader):
                outputs = model(samples.to('cuda'))
                sizes = torch.stack([t['orig_size'] for t in targets]).cuda()
                for mode, post in posts.items():
                    scored = outputs if mode != 'known_only' else dict(outputs,
                              pred_novelty=torch.full_like(outputs['pred_novelty'], -100.))
                    results = post(scored, sizes)
                    for target, prediction in zip(targets, results):
                        predictions[mode][int(target['image_id'])] = {k: v.cpu() for k, v in prediction.items()
                                                                    if k in ('scores', 'labels', 'boxes')}
                if optimization:
                    from util.box_ops import box_cxcywh_to_xyxy, box_iou
                    known_ids = torch.tensor(args.owod_known_class_ids, device='cuda')
                    for batch, target in enumerate(targets):
                        labels = target['labels'].cuda()
                        keep = torch.isin(labels, known_ids)
                        labels = labels[keep]
                        truth = target['boxes'].cuda()[keep]
                        coverage['known_gt'] += len(truth)
                        if len(truth):
                            iou = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][batch]),
                                          box_cxcywh_to_xyxy(truth))[0]
                            predicted = known_ids[outputs['pred_logits'][batch, :, known_ids].argmax(-1)]
                            coverage['localized_known_gt'] += int((iou.amax(0) >= .5).sum())
                            coverage['localized_correct_class_gt'] += int(((iou >= .5) &
                                                                          (predicted[:, None] == labels)).any(0).sum())
                if index % 200 == 0:
                    count = len(predictions[modes[0]])
                    print(f'{name}: evaluated {count}/{len(ds)}', flush=True)
                    update(evaluated_images=count)
        all_metrics = {}
        for mode in modes:
            metrics = evaluate_author_benchmark(predictions[mode], ds.coco, args.owod_known_class_ids,
                                               args.owod_previous_class_ids, args.owod_current_class_ids,
                                               args.owod_evaluation)
            label = name + '_' + mode if optimization else name
            torch.save(predictions[mode], out / (label + '_predictions.pt'))
            save(out / (label + '_metrics.json'), metrics)
            all_metrics[mode] = metrics
        status['methods'].setdefault(name, {})['metrics'] = all_metrics['focal' if optimization else 'hierarchical']
        if optimization:
            status['methods'][name]['scoring_metrics'] = all_metrics
            save(out / (name + '_localization_audit.json'), dict(**coverage,
                 scope='All-query known-GT diagnostic, normalized IoU; not author AP evaluator'))
        update(evaluated_images=len(ds))
        print(name, json.dumps(all_metrics), flush=True)
        return all_metrics

    try:
        update(phase='verify_data')
        manifest = json.loads(options.manifest.read_text(encoding='utf-8'))
        status['verification'] = verify_dataset(options.manifest.parent, manifest['protocol'])
        args = get_args_parser().parse_args([
            '--owod-manifest', str(options.manifest.resolve()), '--owod-stage', '0', '--owod-discovery',
            '--owod-recipe', 'ow-detr-paper', '--backbone-weights', str(options.weights.resolve()),
            '--activation-checkpointing', '--num_classes', '91', '--num_queries', '100',
            '--dim_feedforward', '1024', '--seed', str(options.seed)])
        configure_run(args)
        status.update(manifest_sha256=args.owod_manifest_sha256, backbone_sha256=args.backbone_weights_sha256)
        save(out / 'task1_args.json', vars(args))
        train1 = dataset(args, args.train_ann, 'train')
        val = dataset(args, args.val_ann, 'val')
        indices1 = stream(train1, options.steps, options.seed, 'task1')
        torch.set_num_threads(2)
        set_seed(options.seed)
        template, criterion, _ = build(args)
        initial = copy.deepcopy(template.state_dict())
        status['initial_state_sha256'] = state_digest(initial)
        status['parameter_count'] = sum(p.numel() for p in template.parameters())
        del template, criterion
        for method in (('graph',) if optimization else ('graph', 'self_loop')):
            set_seed(options.seed)
            args.discovery_no_message = method == 'self_loop'
            model, criterion, _ = build(args)
            model.load_state_dict(initial, strict=True)
            assert state_digest(model.state_dict()) == status['initial_state_sha256']
            model.cuda(); criterion.cuda()
            train(model, criterion, args, train1, indices1, method, 3)
            checkpoint(model, args, method, 'increment')
            evaluate(model, args, val, method)
            del model, criterion
            clean()
        del train1, initial
        clean()
        update(phase='prepare_task2')
        manifest2, files = stage_files(options.manifest, 1)
        args2 = copy.deepcopy(args)
        args2.discovery_no_message = False
        args2.owod_stage = 1
        args2.owod_known_class_ids = manifest2['stages'][1]['active_classes']
        args2.owod_current_class_ids = manifest2['stages'][1]['classes']
        args2.owod_previous_class_ids = manifest2['stages'][0]['active_classes']
        args2.train_ann = str(files['increment_train'])
        args2.graph_sketch_max_images = options.promotion_images
        args2.graph_risk_beta = 0.
        args2.num_workers = options.workers
        save(out / 'task2_args.json', vars(args2))
        train2 = dataset(args2, files['increment_train'], 'train')
        ft = dataset(args2, files['train'], 'train')
        indices2 = stream(train2, options.task2_steps, options.seed + 1, 'task2_increment')
        indices_ft = stream(ft, options.finetune_steps, options.seed + 2, 'task2_finetune')
        parent_path = out / 'graph_increment_diagnostic.pt'
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)
        require_parent(parent, args.owod_manifest_sha256, DISCOVERY_VERSION)
        args.discovery_no_message = False
        teacher, _, _ = build(args)
        teacher.load_state_dict(parent['model'], strict=True)
        teacher.cuda().eval().requires_grad_(False)
        spec = dict(stage=1, class_ids=args2.owod_current_class_ids, old_ids=args2.owod_previous_class_ids,
                    rank=args2.graph_adapter_rank, last_n=args2.graph_adapter_last_n, risk_beta=0.,
                    risk=[[0.] * len(args2.owod_previous_class_ids) for _ in args2.owod_current_class_ids],
                    score_semantics='disabled_for_paired_transfer_diagnostic',
                    source_checkpoint_sha256=file_sha256(parent_path))
        means = route_means = None
        task2_methods = ('hard', 'soft') if optimization else ('transfer', 'scratch')
        for method in task2_methods:
            set_seed(options.seed + 1)
            args2.discovery_no_transfer = optimization or method == 'scratch'
            args2.graph_routing = method if optimization else 'hard'
            model, criterion, _ = build(args2)
            model.load_state_dict(parent['model'], strict=True)
            model.cuda(); criterion.cuda()
            local_spec = copy.deepcopy(spec)
            local_spec['routing'] = args2.graph_routing
            append_stage(model, local_spec)
            if means is None:
                update(phase='collect_current_train_features', method=method)
                means, route_means = collect_promotion_features(model, criterion, args2)
            promotion = model.promote(means, local_spec, route_means)
            deltas = [float((a.up.weight @ a.down.weight).norm())
                      for m in model.modules() if isinstance(m, RiskRoutedStage) for a in m.adapters]
            status['methods'][method] = dict(promotion=promotion, initial_adapter_delta_norms=deltas,
                                            promotion_class_coverage=len(means),
                                            inherited_classes=sum(r['inherited'] for r in promotion))
            save(out / (method + '_promotion.json'), status['methods'][method])
            save(out / (method + '_args.json'), vars(args2))
            hooks = configure_trainable(model, args2.owod_current_class_ids, args2.owod_known_class_ids)
            criterion.weight_dict['loss_adapter_route'] = args2.graph_route_loss_coef
            model.train_discovery()
            train(model, criterion, args2, train2, indices2, method, 1, teacher)
            checkpoint(model, args2, method, 'increment', local_spec)
            for handle in hooks:
                handle.remove()
            # Separate phase, separate official balanced-memory loader and optimizer.
            hooks = configure_trainable(model, args2.owod_current_class_ids, args2.owod_known_class_ids, True)
            model.train_discovery()
            train(model, criterion, args2, ft, indices_ft, method + '_finetune', 1, teacher, True)
            checkpoint(model, args2, method, 'finetune', local_spec)
            evaluate(model, args2, val, method)
            for handle in hooks:
                handle.remove()
            del model, criterion
            clean()
        if optimization:
            comparisons = {'task1_focal_vs_hierarchical': metric_delta(
                status['methods']['graph']['scoring_metrics']['hierarchical'],
                status['methods']['graph']['scoring_metrics']['focal']),
                'task2_soft_vs_hard': metric_delta(status['methods']['hard']['metrics'],
                                                  status['methods']['soft']['metrics']),
                'task2_soft_vs_hard_hierarchical': metric_delta(
                    status['methods']['hard']['scoring_metrics']['hierarchical'],
                    status['methods']['soft']['scoring_metrics']['hierarchical']),
                'task2_combined_vs_reference': metric_delta(
                    status['methods']['hard']['scoring_metrics']['hierarchical'],
                    status['methods']['soft']['scoring_metrics']['focal'])}
        else:
            comparisons = {name: metric_delta(status['methods'][control]['metrics'], status['methods'][treatment]['metrics'])
                           for name, control, treatment in [('graph_vs_self_loop', 'self_loop', 'graph'),
                                                           ('transfer_vs_scratch', 'scratch', 'transfer')]}
        save(out / 'comparison.json', dict(official_training_result=False, full_test_images=len(val),
             comparisons=comparisons, scale='AP/recall/H-Score fractions; WI ratio; A-OSE count',
             limitations=status['limitations']))
        audit_transition(out, task2_methods)
        update(status='completed', phase='finished', ap_measured=True)
    except BaseException as error:
        update(status='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=('transition', 'optimization'), default='transition')
    parser.add_argument('--manifest', type=Path, default=ROOT / 'data/owod/s-owodb/split_manifest.json')
    parser.add_argument('--weights', type=Path, default=ROOT / 'pretrained/dino_resnet50.pth')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/discovery/local-transition-v3')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--task2-steps', type=int, default=150)
    parser.add_argument('--finetune-steps', type=int, default=60)
    parser.add_argument('--promotion-images', type=int, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    options = parser.parse_args()
    if (options.steps < 3 or options.steps % 3 or options.task2_steps < 1 or options.finetune_steps < 1
            or options.promotion_images < 1 or options.workers < 0):
        parser.error('Require positive budgets, Task 1 steps divisible by 3, nonnegative workers')
    run(options)
