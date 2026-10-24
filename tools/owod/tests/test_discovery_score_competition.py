import torch

from tools.owod.diagnose_discovery_task1 import score_competition, summarize_competition


def example():
    outputs = dict(pred_logits=torch.tensor([[[0., 8., -8.], [0., -8., -8.]]]),
                   pred_boxes=torch.tensor([[[.2, .2, .2, .2], [.8, .8, .1, .1]]]),
                   pred_objectness=torch.full((1, 2), 8.),
                   pred_novelty=torch.tensor([[2., 8.]]))
    targets = [dict(labels=torch.tensor([1]), boxes=outputs['pred_boxes'][0, :1])]
    return outputs, targets


def test_same_forward_isolates_unknown_competition_without_changing_boxes():
    outputs, targets = example()
    before = {k: v.clone() for k, v in outputs.items()}
    result = score_competition(outputs, targets, [(torch.tensor([0]), torch.tensor([0]))], [1, 2], limit=1)
    summary = summarize_competition([result])
    assert summary['coverage'] == dict(raw=1, joint=0, known_hierarchical=1, known_sigmoid=1)
    assert summary['unknown'] == summary['detections'] == 1
    assert summary['groups']['matched']['count'] == 1
    assert summary['groups']['unmatched_no_known_overlap']['count'] == 1
    for key in outputs:
        assert torch.equal(outputs[key], before[key])


def test_empty_gt_retains_unmatched_scores_without_claiming_background_labels():
    outputs, _ = example()
    targets = [dict(labels=torch.empty(0, dtype=torch.long), boxes=torch.empty(0, 4))]
    empty = torch.empty(0, dtype=torch.long)
    result = score_competition(outputs, targets, [(empty, empty)], [1, 2], limit=50)
    summary = summarize_competition([result])
    assert summary['gt'] == 0
    assert summary['groups']['matched']['mean'] is None
    assert summary['groups']['unmatched_no_known_overlap']['count'] == 2
