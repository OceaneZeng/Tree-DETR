"""Regression coverage for zero-union IoU in cross-view assignment."""
import pytest
import torch

from models.graph_local.discovery_candidates import mine_candidates, view_matches
from models.graph_local.tests.test_discovery import evidence
from util.box_ops import box_cxcywh_to_xyxy, box_iou


@pytest.mark.parametrize('width', [0., 1e-12])
def test_degenerate_boxes_do_not_enter_assignment(width):
    outputs, _ = evidence()
    teacher = outputs['discovery_teacher']
    with torch.no_grad():
        outputs['pred_boxes'][0, 1, 2] = width
        teacher['pred_boxes'][0, 1, 2] = width
    flipped = teacher['pred_boxes'].clone()
    flipped[..., 0] = 1 - flipped[..., 0]
    # The old implementation passed this NaN directly to SciPy.
    raw = box_iou(box_cxcywh_to_xyxy(outputs['pred_boxes'][0]),
                  box_cxcywh_to_xyxy(flipped[0]))[0]
    assert torch.isnan(raw[1, 1])
    mapping, confidence, _ = view_matches(outputs, teacher)
    assert mapping.tolist() == [[0, -1, 2]]
    assert torch.allclose(confidence, torch.tensor([[1., 0., 1.]]))


def test_regular_assignment_is_preserved():
    outputs, _ = evidence()
    mapping, confidence, _ = view_matches(outputs, outputs['discovery_teacher'])
    assert mapping.tolist() == [[0, 1, 2]]
    assert torch.allclose(confidence, torch.ones_like(confidence))


@pytest.mark.parametrize('side', ['student', 'teacher'])
def test_all_degenerate_boxes_leave_no_matches(side):
    outputs, _ = evidence()
    teacher = outputs['discovery_teacher']
    selected = outputs if side == 'student' else teacher
    with torch.no_grad():
        selected['pred_boxes'][..., 2:] = 0
    mapping, confidence, _ = view_matches(outputs, teacher)
    assert (mapping == -1).all()
    assert (confidence == 0).all()


def test_unmatched_queries_are_not_background_or_pseudo_labels():
    outputs, targets = evidence()
    teacher = outputs['discovery_teacher']
    teacher['pred_boxes'][..., 2:] = 0
    teacher['attention_feature'].zero_()
    found = mine_candidates(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))],
                            [1], teacher, consistency_threshold=0)
    assert not found['pseudo'].any()
    assert not found['background'].any()
    assert torch.isfinite(found['weights']).all()


@pytest.mark.parametrize('side', ['student', 'teacher'])
@pytest.mark.parametrize('key', ['pred_boxes', 'query_features'])
@pytest.mark.parametrize('value', [float('nan'), float('inf')])
def test_nonfinite_model_outputs_are_not_silently_hidden(side, key, value):
    outputs, _ = evidence()
    teacher = outputs['discovery_teacher']
    selected = outputs if side == 'student' else teacher
    with torch.no_grad():
        selected[key][0, 1, 0] = value
    with pytest.raises(ValueError, match=f'non-finite {side} boxes/features in batch 0'):
        view_matches(outputs, teacher)
