import pytest
import torch
from tools.owod.supervision_probe import probe_batch, summarize


def fixture():
    outputs = {'pred_logits': torch.tensor([0., 5., 0.]).repeat(1, 100, 1),
               'pred_boxes': torch.tensor([.5, .5, .2, .2]).repeat(1, 100, 1),
               'pred_objectness': torch.full((1, 100), 7.),
               'pred_novelty': torch.full((1, 100), 4.59512)}
    targets = [{'image_id': torch.tensor([7]), 'labels': torch.tensor([1]),
                'boxes': torch.tensor([[.5, .5, .2, .2]])}]
    matcher = lambda *_: [(torch.tensor([0]), torch.tensor([0]))]
    return outputs, targets, matcher


def test_gate_and_truncation_are_distinct_from_geometry():
    outputs, targets, matcher = fixture()
    row = probe_batch(outputs, targets, matcher, [1, 2])[0]
    assert row['any_box_iou50'] and row['correct_argmax_box_iou50']
    assert not row['hierarchical_top50']
    assert row['hierarchical_top300']
    assert row['hierarchical_without_novelty_gate_top50']
    assert row['matched_novelty'] == pytest.approx(.99)
    assert summarize([row])['percent_of_gt']['hierarchical_top50'] == 0


def test_wrong_class_and_no_box_are_reported_separately():
    outputs, targets, matcher = fixture()
    targets[0]['labels'][0] = 2
    row = probe_batch(outputs, targets, matcher, [1, 2])[0]
    assert row['any_box_iou50'] and not row['correct_argmax_box_iou50']
    outputs['pred_boxes'][..., :2] = .1
    row = probe_batch(outputs, targets, matcher, [1, 2])[0]
    assert not row['any_box_iou50'] and not row['focal_top300']


def test_future_labels_rejected_and_empty_summary_explicit():
    outputs, targets, matcher = fixture()
    targets[0]['labels'][0] = 91
    with pytest.raises(ValueError, match='outside'):
        probe_batch(outputs, targets, matcher, [1, 2])
    assert summarize([])['matched_means']['matched_novelty'] is None
