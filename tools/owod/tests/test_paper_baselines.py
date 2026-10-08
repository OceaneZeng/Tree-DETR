import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from torch import nn

from main import get_args_parser
from models.deformable_detr import SetCriterion
from models.matcher import HungarianMatcher
from models.paper_baselines.detector import PaperCriterion, PaperPostProcess, attention_pseudo_queries
from models.paper_baselines.ew_modules import DualLoRA, QueryNormEUMix, inject_dual_lora, merge_beta


def criterion(method='ow-detr'):
    base = SetCriterion(92, HungarianMatcher(), {'loss_ce': 2, 'loss_bbox': 5, 'loss_giou': 2},
                        ['labels', 'boxes', 'cardinality'])
    return PaperCriterion(base, SimpleNamespace(paper_baseline=method, owod_known_class_ids=[2, 9],
                                               ow_pseudo_warmup=9, ow_top_unknown=2, dec_layers=3))


def test_attention_excludes_matched_and_invalid_and_ranks_box_means():
    feature = torch.tensor([[[[1., 1., 10., 10.], [1., 1., 10., 10.]]]])
    boxes = torch.tensor([[[.25, .5, .5, 1], [.75, .5, .5, 1], [.75, .5, .5, 1], [0., 0., 0., 0.]]])
    matched = [(torch.tensor([1]), torch.tensor([0]))]
    result = attention_pseudo_queries(feature, boxes, matched, (2, 4), 5)
    assert result[0].tolist() == [2, 0]


def test_ow_aux_losses_and_no_pseudo_box_regression_or_future_leak():
    torch.manual_seed(1)
    loss = criterion()
    boxes = torch.tensor([[[.5, .5, .2, .2], [.2, .2, .1, .1], [.8, .8, .1, .1]]], requires_grad=True)
    layer = {'pred_logits': torch.randn(1, 3, 92, requires_grad=True), 'pred_boxes': boxes,
             'pred_objectness': torch.zeros(1, 3, 1, requires_grad=True)}
    output = {**layer, 'attention_feature': torch.rand(1, 2, 8, 8), 'padded_size': (32, 32),
              'aux_outputs': [dict(layer), dict(layer)]}
    target = {'labels': torch.tensor([2, 77]), 'boxes': torch.tensor([[.5, .5, .2, .2], [.8, .8, .1, .1]])}
    original = copy.deepcopy(target)
    loss.epoch = 0
    before = loss(output, [target])
    loss.epoch = 9
    after = loss(output, [target])
    assert {'loss_NC', 'loss_NC_0', 'loss_NC_1', 'loss_ce_0', 'loss_ce_1'} <= after.keys()
    torch.testing.assert_close(before['loss_bbox'], after['loss_bbox'])
    torch.testing.assert_close(before['loss_giou'], after['loss_giou'])
    only_known = loss(output, [{'labels': target['labels'][:1], 'boxes': target['boxes'][:1]}])
    for key in after:
        torch.testing.assert_close(after[key], only_known[key])
    torch.testing.assert_close(target['labels'], original['labels'])
    torch.testing.assert_close(target['boxes'], original['boxes'])
    sum(value for key, value in after.items() if key.startswith('loss')).backward()
    assert layer['pred_objectness'].grad is not None


def test_ew_handles_empty_known_targets():
    output = {'pred_logits': torch.randn(1, 3, 92, requires_grad=True),
              'pred_boxes': torch.rand(1, 3, 4, requires_grad=True)}
    target = {'labels': torch.tensor([77]), 'boxes': torch.rand(1, 4)}
    losses = criterion('ew-detr')(output, [target])
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert losses['loss_bbox'] == 0


def test_dual_lora_svd_merge_and_reset_preserve_rank_and_restart_gradients():
    torch.manual_seed(3)
    adapter = DualLoRA(torch.zeros(4, 5), rank=4)
    with torch.no_grad():
        adapter.aggregate_a.normal_()
        adapter.aggregate_b.normal_()
        adapter.task_b.normal_()
    expected = .7 * adapter.aggregate_b @ adapter.aggregate_a + .3 * adapter.task_b @ adapter.task_a
    adapter.consolidate(.3)
    torch.testing.assert_close(adapter(torch.zeros(4, 5)), expected)
    assert adapter.task_b.count_nonzero() == 0
    adapter(torch.zeros(4, 5)).sum().backward()
    assert adapter.task_b.grad.abs().sum() > 0
    assert not adapter.aggregate_a.requires_grad
    assert merge_beta(10, 0) == 1
    assert merge_beta(400, 100) == .2
    assert merge_beta(50, 100) == pytest.approx(.5)


class AttentionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(8, 8)
        self.transformer = nn.Module()
        self.transformer.encoder = nn.Sequential(nn.Linear(8, 8))
        self.transformer.decoder = nn.MultiheadAttention(8, 2, batch_first=True)
        self.class_embed = nn.Linear(8, 3)
        self.bbox_embed = nn.Linear(8, 4)


def test_functional_attention_uses_lora_for_packed_qkv_and_output():
    torch.manual_seed(4)
    model = AttentionDetector()
    names = inject_dual_lora(model, 2)
    assert any('in_proj_weight' in name for name in names)
    assert any('out_proj.weight' in name for name in names)
    attention = model.transformer.decoder
    x = torch.randn(2, 4, 8)
    original = attention(x, x, x)[0]
    original.square().sum().backward()
    for adapter in (attention.parametrizations.in_proj_weight[0], attention.out_proj.parametrizations.weight[0]):
        assert adapter.task_b.grad.abs().sum() > 0
        with torch.no_grad():
            adapter.task_b.add_(.1)
    assert not torch.allclose(attention(x, x, x)[0], original)
    assert not model.backbone.weight.requires_grad
    assert not attention.parametrizations.in_proj_weight.original.requires_grad
    assert model.class_embed.weight.requires_grad
    restored = AttentionDetector()
    inject_dual_lora(restored, 2)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.transformer.decoder(x, x, x)[0], attention(x, x, x)[0])


def test_eumix_calibration_gradients_and_explicit_unknown_score():
    torch.manual_seed(8)
    module = QueryNormEUMix(8, [2, 9])
    classifier = nn.Linear(8, 92)
    features = torch.randn(2, 5, 8)
    logits = module(features, classifier)
    assert (logits[..., 77] == -1e8).all()
    logits[..., [2, 9, 91]].square().mean().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    output = {'pred_logits': logits, 'pred_boxes': torch.rand(2, 5, 4)}
    result = PaperPostProcess([2, 9], .5)(output, torch.tensor([[32, 32], [32, 32]]))
    assert set(result[0]['labels'].tolist()) == {2, 9, 91}
    assert result[0]['unknown_scores'].max() <= 1


def payload(classes, offset=0):
    return {'images': [{'id': offset + i, 'file_name': f'{offset + i}.jpg'} for i in range(len(classes))],
            'annotations': [{'id': i, 'image_id': offset + i, 'category_id': c, 'bbox': [0, 0, 1, 1]}
                            for i, c in enumerate(classes)],
            'categories': [{'id': c, 'name': str(c)} for c in classes]}
