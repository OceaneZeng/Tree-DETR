"""Disposable parameter-capacity checks; never alter production training policy."""
import torch


def open_detection_parameters(model):
    """Expose base decoder and complete bbox MLPs; keep existing row/bank masks."""
    detector = model.detr
    for name, parameter in detector.transformer.decoder.named_parameters():
        if not any(part in name for part in ('graph_adapter', 'discovery_adapter', 'class_embed')):
            parameter.requires_grad_(True)
    detector.bbox_embed.requires_grad_(True)


def parameter_group(name):
    if 'class_embed' in name:
        return 'classifier'
    if 'bbox_embed' in name:
        return 'bbox_heads'
    if 'graph_adapter' in name:
        return 'class_adapters'
    if 'discovery_adapter' in name or 'affinity_graph' in name:
        return 'discovery'
    if 'transformer.decoder' in name:
        return 'base_decoder'
    return 'other'


def gradient_paths(model, losses, weights):
    """Measure task-specific gradients before clipping, without optimizer updates."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    result = {}
    for group, prefixes in (('classification', ('loss_ce',)), ('localization', ('loss_bbox', 'loss_giou'))):
        terms = [value * weights[key] for key, value in losses.items()
                 if key in weights and key.startswith(prefixes)]
        total = sum(terms)
        norms = {}
        if terms and total.requires_grad:
            gradients = torch.autograd.grad(total, [p for _, p in named], retain_graph=True, allow_unused=True)
            for (name, _), gradient in zip(named, gradients):
                key = parameter_group(name)
                norms[key] = norms.get(key, 0.) + (float(gradient.detach().square().sum()) if gradient is not None else 0.)
        result[group] = norms
    return result


@torch.no_grad()
def probe_training(model, criterion, loader, known, device):
    from tools.owod.supervision_probe import probe_batch, summarize
    from tools.owod.gnn_calibration import move_targets
    model.eval()
    rows = []
    for samples, targets in loader:
        rows.extend(probe_batch(model(samples.to(device)), move_targets(targets, torch.device(device)),
                                criterion.matcher, known))
    return summarize(rows)
