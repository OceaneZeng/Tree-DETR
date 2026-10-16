from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tools.owod.diagnose_learning import (checked_checkpoint, evaluate_views, parameter_snapshot,
                                         parameter_report, score_variants)
from tools.owod.tests.test_screening import source


def test_score_probes_share_one_forward_and_keep_old_gt_comparable():
    class Model(nn.Module):
        calls = 0
        def forward(self, samples):
            self.calls += 1
            return {'pred_logits': torch.zeros(1, 100, 4),
                    'pred_boxes': torch.tensor([.5, .5, .2, .2]).repeat(1, 100, 1),
                    'pred_objectness': torch.full((1, 100), 5.),
                    'pred_novelty': torch.full((1, 100), -5.)}
    model = Model()
    target = {'image_id': torch.tensor([1]), 'orig_size': torch.tensor([100, 100]),
              'labels': torch.tensor([1]), 'boxes': torch.tensor([[.5, .5, .2, .2]])}
    coco = SimpleNamespace(imgs={1: {'owod_source_id': 'one'}},
                           imgToAnns={1: [{'category_id': 1, 'voc_bbox': [41, 41, 60, 60]}]})
    spec = {'image_source_ids': ['one'], 'ap_mode': 'integral', 'iou_threshold': .5,
            'wi_recall': .8, 'known_only_screening': True}
    result = evaluate_views(model, [(torch.zeros(1), [target])], coco, score_variants([1], [1, 2]),
                            spec, [1], [2], 'cpu')
    assert model.calls == 1
    assert result['raw_old_box_recall_iou50']['recall'] == 1
    assert result['task1_hierarchical_top50']['composition']['detections'] == 50
    assert result['expanded_hierarchical_top300']['composition']['detections'] == 300
    assert result['expanded_hierarchical_top50']['composition']['mean_old_score'] < result['task1_hierarchical_top50']['composition']['mean_old_score']
    assert 'u_recall' not in result['expanded_hierarchical_top50']['metrics']


def test_gradient_monitor_distinguishes_old_and_new_rows():
    model = nn.Module()
    model.class_embed = nn.Linear(2, 4)
    initial = parameter_snapshot(model, [1], [2])
    model.class_embed(torch.ones(1, 2))[:, 2].sum().backward()
    before = parameter_report(model, [1], [2], initial)
    assert before['new_classifier']['gradient_l2_squared'] > 0
    assert before['old_classifier']['gradient_l2_squared'] == 0
    torch.optim.SGD(model.parameters(), lr=.1).step()
    after = parameter_report(model, [1], [2], initial)
    assert after['new_classifier']['delta_l2_squared'] > 0
    assert after['old_classifier']['delta_l2_squared'] == 0


def test_diagnostic_rejects_full_or_unfinished_checkpoints(tmp_path):
    path = tmp_path / 'checkpoint.pth'
    saved = SimpleNamespace(owod_manifest_sha256='manifest', owod_screening_sha256='screen',
        owod_stage=1, owod_phase='increment', owod_discovery=True, paper_comparable_protocol=False,
        epochs=12, graph_routing='hard', graph_localization='frozen')
    torch.save({'args': saved, 'epoch': 11}, path)
    assert checked_checkpoint(path, 'manifest', 'screen', 1).epochs == 12
    torch.save({'args': saved, 'epoch': 10}, path)
    with pytest.raises(ValueError, match='completed'):
        checked_checkpoint(path, 'manifest', 'screen', 1)
    saved.owod_screening_sha256 = None
    torch.save({'args': saved, 'epoch': 11}, path)
    with pytest.raises(ValueError, match='provenance'):
        checked_checkpoint(path, 'manifest', 'screen', 1)


def test_paired_fit_exports_metrics_not_weights(tmp_path, monkeypatch):
    import copy
    import json
    from tools.owod import diagnose_learning as diag
    from util.misc import nested_tensor_from_tensor_list
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.class_embed = nn.Linear(1, 4)
        def forward(self, samples):
            batch = samples.tensors.shape[0]
            return {'pred_logits': self.class_embed(torch.ones(batch, 1, 1)),
                    'pred_boxes': torch.full((batch, 1, 4), .3)}
        def train_discovery(self):
            pass
        def observe_training(self, evidence):
            pass
    class Criterion(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_dict = {'loss_ce': 1.}
            self.evidence = None
        def forward(self, outputs, targets):
            return {'loss_ce': (outputs['pred_logits'][..., 2] - 1).square().mean()}
    torch.manual_seed(42)
    initial = Model()
    monkeypatch.setattr('tools.owod.gnn_calibration.load_detector', lambda *a: (copy.deepcopy(initial), Criterion()))
    monkeypatch.setattr('models.graph_local.routed_adapter.configure_trainable', lambda *a, **k: [])
    monkeypatch.setattr('tools.owod.manage_data.prepare_fit_data', lambda *a: tmp_path / 'train.json')
    dataset = SimpleNamespace(coco=SimpleNamespace(dataset={'images': [{'owod_source_id': 'one'}]}))
    batch = (nested_tensor_from_tensor_list([torch.zeros(3, 4, 4)]), [{'labels': torch.tensor([2])}])
    monkeypatch.setattr(diag, 'make_loader', lambda *a, **k: (dataset, [batch]))
    monkeypatch.setattr(diag, 'evaluate_views', lambda model, *a: {'weight': float(model.class_embed.weight.sum())})
    screen = tmp_path / 'screen.json'
    screen.write_text(json.dumps({'heldout_source_ids': ['one']}))
    saved = SimpleNamespace(owod_previous_class_ids=[1], owod_current_class_ids=[2], owod_known_class_ids=[1, 2],
        graph_route_loss_coef=1., epochs=12, lr=.01, weight_decay=.0001, class_embed_lr_mult=1.,
        lr_backbone_names=['backbone'], lr_backbone=.001, lr_linear_proj_names=[], lr_linear_proj_mult=.1,
        distill_score_threshold=.3, distill_max_queries=20, distill_class_coef=2., distill_bbox_coef=5., clip_max_norm=.1)
    args = SimpleNamespace(mode='distill', manifest=tmp_path / 'manifest', screening=screen, output=tmp_path / 'result',
        images=1, seed=42, steps=2, batch_size=1, workers=0, device='cpu', report_interval=1)
    args.output.mkdir()
    diag.fit(args, saved, tmp_path / 'checkpoint', tmp_path / 'parent',
             {'image_root': str(tmp_path), 'evaluation': {}}, {'full_val': tmp_path / 'val'}, {})
    for treatment in ('distill_on', 'distill_off'):
        assert (args.output / treatment / 'after.json').is_file()
        records = [json.loads(line) for line in (args.output / treatment / 'updates.jsonl').read_text().splitlines()]
        assert records[-1]['parameters']['new_classifier']['delta_l2_squared'] > 0
        assert ('loss_distill_cls' in records[-1]['weighted_losses']) == (treatment == 'distill_on')
    assert not list(args.output.rglob('*.pth'))
    assert (args.output / 'distill_on/before.json').read_bytes() == (args.output / 'distill_off/before.json').read_bytes()


def test_fit_data_keeps_new_labels_and_excludes_holdout(source, tmp_path):
    from tools.owod.screening_data import prepare_screen, read_json, file_sha256
    from tools.owod.manage_data import prepare_fit_data
    parent, _ = source
    screen = prepare_screen(parent, tmp_path / 'screen', 8, 4, 42)
    path = prepare_fit_data(parent, screen, tmp_path / 'fit', 4, 42)
    payload = read_json(path)
    names = {image['owod_source_id'] for image in payload['images']}
    assert len(names) == 4
    assert not names & set(read_json(screen)['heldout_source_ids'])
    assert {ann['category_id'] for ann in payload['annotations']} == {3, 4}
    assert read_json(path.parent / 'diagnostic_data.json')['annotation_sha256'] == file_sha256(path)
    with pytest.raises(ValueError, match='seed'):
        prepare_fit_data(parent, screen, tmp_path / 'other', 4, 43)
    with pytest.raises(ValueError, match='empty'):
        prepare_fit_data(parent, screen, tmp_path / 'fit', 4, 42)
