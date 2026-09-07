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
from tools.owod.run_paper_baselines import (annotation_subset, baseline_complete, combine_annotations,
                                           create_plan, select_memory, wait_for_d4, payload_hash, write_json)


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


def test_memory_is_bounded_balanced_and_never_recovers_dropped_exemplars():
    first = payload([1, 1, 2, 2])
    memory = select_memory(first, [1, 2], 2, 42)
    assert len(memory['images']) == 2
    assert {ann['category_id'] for ann in memory['annotations']} == {1, 2}
    combined = combine_annotations(payload([3, 3], 10), memory)
    assert sum(image['owod_replay'] for image in combined['images']) == 2
    next_memory = select_memory(combined, [1, 2, 3], 3, 43)
    assert {image['id'] for image in next_memory['images']}.issubset(
        {image['id'] for image in memory['images']} | {10, 11})
    current = annotation_subset(combined, [10, 11], [3])
    assert {ann['category_id'] for ann in current['annotations']} == {3}


def test_marker_alone_cannot_start_queue_or_skip_stage(tmp_path):
    (tmp_path / 'training_complete.json').write_text(json.dumps({'epochs': 20, 'last_epoch': 19}))
    assert not baseline_complete(tmp_path, 20, 'ow-detr')
    (tmp_path / 'checkpoint.pth').touch()
    (tmp_path / 'metrics.jsonl').write_text(json.dumps({'epoch': 19, 'test_owod_current_ap50': .5,
                                                     'test_owod_known_ap50': .4}) + '\n')
    assert baseline_complete(tmp_path, 20, 'ow-detr')
    assert not baseline_complete(tmp_path, 50, 'ow-detr')
    assert not baseline_complete(tmp_path, 20, 'ew-detr')


def test_generated_annotation_hash_is_platform_independent(tmp_path):
    from tools.owod.protocol import file_sha256
    path = tmp_path / 'annotation.json'
    value = {'images': [{'id': 1}], 'annotations': []}
    write_json(path, value)
    assert file_sha256(path) == payload_hash(value)


@pytest.mark.parametrize('fail_stage', [None, 'ow-detr/stage_1'])
def test_queue_executes_all_eight_stages_in_order_and_stops_on_failure(tmp_path, fail_stage):
    from tools.owod import run_paper_baselines as runner
    output = tmp_path / 'queue'
    annotation = {'images': [], 'annotations': []}
    runs = {}
    for method in runner.METHODS:
        for stage in range(4):
            key = f'{method}/stage_{stage}'
            runs[key] = {'epochs': 50 if stage == 0 else 20, 'annotation': annotation,
                         'annotation_sha256': payload_hash(annotation),
                         'command': ['python', 'main.py', '--output_dir', str(output / key)]}
    plan = {'protocol': 'fixture', 'initialization': 'fixture', 'stage_records': [], 'fingerprints': {}}
    completed = set()
    launched = []
    def execute(command, log, environment):
        if '--output_dir' not in command:
            return
        directory = Path(command[command.index('--output_dir') + 1])
        key = directory.relative_to(output).as_posix()
        launched.append(key)
        if key == fail_stage:
            raise RuntimeError('fixture training failure')
        (directory / 'checkpoint.pth').write_bytes(b'fixture checkpoint')
        completed.add(key)
    def complete(directory, epochs, method):
        return directory.is_relative_to(output) and directory.relative_to(output).as_posix() in completed
    with mock.patch.object(runner, 'create_plan', return_value=(plan, runs)), \
         mock.patch.object(runner, 'training_environment', return_value={}), \
         mock.patch.object(runner, 'wait_for_d4') as wait, \
         mock.patch.object(runner, 'verify_checkpoint'), \
         mock.patch.object(runner, 'baseline_complete', side_effect=complete), \
         mock.patch.object(runner, 'run_child', side_effect=execute):
        if fail_stage:
            with pytest.raises(RuntimeError, match='fixture training failure'):
                runner.main(['--output-dir', str(output)])
            assert launched == ['ow-detr/stage_0', 'ow-detr/stage_1']
            assert json.loads((output / 'queue_status.json').read_text())['status'] == 'failed'
        else:
            runner.main(['--output-dir', str(output)])
            assert launched == list(runs)
            assert json.loads((output / 'queue_status.json').read_text())['status'] == 'complete'
        assert wait.call_count == 1


def test_wait_requires_d4_success_and_idle_gpus(tmp_path):
    args = SimpleNamespace(d4_dir=tmp_path, gpus='0,1', wait_hours=1, poll_seconds=1)
    with mock.patch('tools.owod.run_paper_baselines.baseline_complete', side_effect=[False, True, True]), \
         mock.patch('tools.owod.run_paper_baselines.active_output_processes', return_value=[]), \
         mock.patch('tools.owod.run_paper_baselines.busy_gpus', side_effect=[[], ['busy'], []]), \
         mock.patch('tools.owod.run_paper_baselines.time.sleep') as sleep:
        wait_for_d4(args)
        assert sleep.call_count == 2


def test_four_stage_plan_parses_and_chains_own_weights(tmp_path):
    source = vars(get_args_parser().parse_args([]))
    source.update(owod_stage=1, lr_backbone=0, epochs=20, batch_size=2, seed=42,
                  coco_path=str(tmp_path / 'coco'), owod_manifest=str(tmp_path / 'manifest.json'))
    source['train_ann'] = str(tmp_path / 'd4_train.json')
    (tmp_path / 'd4_train.json').write_text(json.dumps({'images': [{'id': 0, 'owod_replay': True}]}))
    d4 = tmp_path / 'd4'
    d4.mkdir()
    (d4 / 'run_config.json').write_text(json.dumps(source))
    records = []
    for stage in range(4):
        classes = list(range(stage * 20 + 1, stage * 20 + 21))
        increment = payload(classes, stage * 100)
        ann = tmp_path / f'inc{stage}.json'
        ann.write_text(json.dumps(increment))
        for split in ('train2017', 'val2017'):
            (tmp_path / 'coco' / split).mkdir(parents=True, exist_ok=True)
            for image in increment['images']:
                (tmp_path / 'coco' / split / image['file_name']).touch()
        records.append({'classes': classes, 'files': {'increment_train': str(ann), 'full_val': str(ann)}})
        if stage == 1:
            (tmp_path / 'd4_train.json').write_text(json.dumps({'images': [
                {'id': 0, 'owod_replay': True}, *increment['images']]}))
    source['owod_known_class_ids'] = list(range(1, 41))
    source['owod_current_class_ids'] = list(range(21, 41))
    (d4 / 'run_config.json').write_text(json.dumps(source))
    (tmp_path / 'manifest.json').write_text(json.dumps({'stages': records}))
    args = SimpleNamespace(d4_dir=d4, output_dir=tmp_path / 'output', memory_images=398,
                           gpus='0,1', master_port=29579)
    plan, runs = create_plan(args)
    assert len(runs) == 8
    assert not args.output_dir.exists()
    for key, run in runs.items():
        command = run['command']
        parsed = get_args_parser().parse_args(command[command.index(str(Path(__file__).resolve().parents[3] / 'main.py')) + 1:])
        stage = int(key[-1])
        assert len(parsed.owod_known_class_ids) == 20 * (stage + 1)
        assert parsed.epochs == (50 if stage == 0 else 20)
        assert parsed.lr_backbone == 0
        if stage == 0:
            assert not parsed.pretrained
        else:
            assert key.split('/')[0] in parsed.pretrained
            assert f'stage_{stage - 1}' in parsed.pretrained
        if key.startswith('ew'):
            assert not parsed.replay_sampling_fraction
            assert {ann['category_id'] for ann in run['annotation']['annotations']}.issubset(parsed.owod_current_class_ids)
    assert not plan['paper_comparable']
