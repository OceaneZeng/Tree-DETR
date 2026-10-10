from tools.owod.local_discovery_check import metric_delta
from tools.owod.local_transition_check import require_parent, state_digest, audit_transition
import json
import pytest
import torch


def test_short_run_comparison_preserves_negative_ap_and_hscore_changes():
    before = {'known_ap50': .25, 'current_ap50': .25, 'u_recall': .1,
              'h_score': 2 * .25 * .1 / .35, 'wi': .2, 'a_ose': 10.}
    after = {'known_ap50': .20, 'current_ap50': .20, 'u_recall': .15,
             'h_score': 2 * .20 * .15 / .35, 'wi': .25, 'a_ose': 12.}
    result = metric_delta(before, after)
    assert result['known_ap50']['delta'] < 0
    assert result['u_recall']['delta'] > 0
    assert result['h_score']['delta'] > 0
    assert result['a_ose']['delta'] == 2.
    assert result['wi']['delta'] > 0


def test_transition_rejects_wrong_manifest_version_task_and_method():
    parent = dict(diagnostic_only=True, manifest_sha256='verified', discovery_version=2,
                  task=1, phase='increment', method='graph')
    require_parent(parent, 'verified', 2)
    for key, value in [('diagnostic_only', False), ('manifest_sha256', 'other'),
                       ('discovery_version', 1), ('task', 2), ('phase', 'finetune'),
                       ('method', 'self_loop')]:
        with pytest.raises(ValueError, match='matching Task 1'):
            require_parent({**parent, key: value}, 'verified', 2)


def test_pair_initialization_digest_detects_head_difference():
    a = {'head': torch.zeros(2), 'base': torch.ones(3)}
    assert state_digest(a) == state_digest(dict(reversed(list(a.items()))))
    assert state_digest(a) != state_digest({**a, 'head': torch.ones(2)})


def test_parameter_audit_does_not_call_unchanged_inheritance_learning(tmp_path):
    prefix = 'detr.transformer.decoder.layers.0'
    base = {'detr.class_embed.0.weight': torch.zeros(4, 2),
            'detr.bbox_embed.0.weight': torch.ones(2, 2),
            prefix + '.discovery_adapter.snapshot_up': torch.eye(2)[None],
            prefix + '.discovery_adapter.snapshot_down': torch.eye(2)[None]}
    parent = dict(diagnostic_only=True, manifest_sha256='same', discovery_version=3,
                  task=1, phase='increment', method='graph', model=base)
    torch.save(parent, tmp_path / 'graph_increment_diagnostic.pt')
    (tmp_path / 'task2_args.json').write_text(json.dumps(dict(
        owod_previous_class_ids=[1], owod_current_class_ids=[2, 3])))
    for method in ('transfer', 'scratch'):
        state = {k: v.clone() for k, v in base.items()}
        state['detr.class_embed.0.weight'][2:] += .1
        for index in (0, 1):
            stem = prefix + f'.graph_adapter.stages.0.adapters.{index}'
            state[stem + '.down.weight'] = torch.eye(2)
            state[stem + '.up.weight'] = (torch.eye(2) if index or method == 'transfer' else torch.zeros(2, 2))
        spec = dict(rank=2, risk_beta=0., promotion=[
            dict(class_id=2, anonymous_slots=[0], weights=[1.], inherited=method == 'transfer'),
            dict(class_id=3, anonymous_slots=[], weights=[], inherited=False)])
        torch.save({**parent, 'task': 2, 'method': method, 'model': state, 'spec': spec},
                   tmp_path / (method + '_increment_diagnostic.pt'))
    audit = audit_transition(tmp_path)
    for result in audit['methods'].values():
        assert result['updated_adapter_classes'] == [3]
        assert result['old_classifier_max_change'] == 0
        assert result['new_classifier_change_norm'] > 0
