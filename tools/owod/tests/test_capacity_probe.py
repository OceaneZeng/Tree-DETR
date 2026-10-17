import torch
from torch import nn

from tools.owod.capacity_probe import open_detection_parameters, gradient_paths


def test_open_scope_preserves_classifier_masks_and_adapter_policy():
    model = nn.Module()
    model.detr = nn.Module()
    model.detr.transformer = nn.Module()
    decoder = nn.Module()
    model.detr.transformer.decoder = decoder
    decoder.base = nn.Linear(2, 2)
    decoder.graph_adapter = nn.Linear(2, 2)
    decoder.discovery_adapter = nn.Linear(2, 2)
    model.detr.class_embed = nn.Linear(2, 3)
    model.detr.bbox_embed = nn.Linear(2, 4)
    model.detr.backbone = nn.Linear(2, 2)
    model.requires_grad_(False)
    model.detr.class_embed.requires_grad_(True)
    mask = torch.tensor([[0.], [1.], [0.]])
    hook = model.detr.class_embed.weight.register_hook(lambda g: g * mask)
    open_detection_parameters(model)
    assert all(p.requires_grad for p in decoder.base.parameters())
    assert all(p.requires_grad for p in model.detr.bbox_embed.parameters())
    assert not any(p.requires_grad for p in decoder.graph_adapter.parameters())
    assert not any(p.requires_grad for p in decoder.discovery_adapter.parameters())
    assert not any(p.requires_grad for p in model.detr.backbone.parameters())
    x = decoder.base(torch.ones(1, 2))
    losses = {'loss_ce': model.detr.class_embed(x).square().sum(),
              'loss_bbox': model.detr.bbox_embed(x).square().sum()}
    paths = gradient_paths(model, losses, dict.fromkeys(losses, 1.))
    assert paths['localization']['bbox_heads'] > 0
    assert paths['localization']['base_decoder'] > 0
    assert paths['localization']['classifier'] == 0
    assert all(p.grad is None for p in model.parameters())
    sum(losses.values()).backward()
    assert model.detr.class_embed.weight.grad[0].abs().sum() == 0
    assert model.detr.class_embed.weight.grad[1].abs().sum() > 0
    hook.remove()
