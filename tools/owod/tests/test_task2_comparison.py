import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.owod import shared_task1, run_benchmark
from tools.owod.protocol import file_sha256
from tools.owod.run_screening import benchmark_command, get_parser, select_winner
from tools.owod.screening_data import write_json
from tools.owod.strict_setting import validate_checkpoint
from tools.owod.tests.test_screening import results


@pytest.fixture
def parent(tmp_path, monkeypatch):
    manifest, screen, weights, train = [tmp_path / name for name in ('manifest', 'screen', 'dino', 'train')]
    for path in (manifest, screen, weights, train):
        path.write_text('{}')
    record = {'classes': [3], 'active_classes': [1, 3]}
    spec = {'protocol': 's-owodb', 'stages': [{'active_classes': [1]}, record]}
    monkeypatch.setattr(shared_task1, 'screening_files', lambda *args: (spec, {'train': train}))
    monkeypatch.setattr('tools.owod.strict_setting.stage_files', lambda *args: (spec, {}))
    saved = SimpleNamespace(owod_manifest=str(manifest), owod_manifest_sha256=file_sha256(manifest),
        owod_screening_sha256=file_sha256(screen), backbone_weights_sha256=file_sha256(weights),
        owod_stage=0, owod_phase='increment', owod_recipe='ow-detr-paper', seed=42,
        owod_discovery=True, graph_routed_adapter=True, graph_routing='hard', graph_localization='frozen',
        graph_adapter_specs=[], graph_adapter_last_n=2, epochs=12, paper_comparable_protocol=False)
    checkpoint, gnn = tmp_path / 'checkpoint.pth', tmp_path / 'gnn.pt'
    payload = {'args': saved, 'epoch': 11, 'model': {'weight': torch.ones(1)}}
    torch.save(payload, checkpoint)
    metadata = {'checkpoint_sha256': file_sha256(checkpoint), 'manifest_sha256': file_sha256(manifest),
                'screening_sha256': file_sha256(screen), 'annotation_sha256': file_sha256(train),
                'stage': 0, 'production_ready': True, 'uses_validation_labels': False,
                'supervision': 'empirical_train_loss_increase', 'last_decoder_layers': 2,
                'training_classes': [1]}
    torch.save({'extra': metadata, 'config': {}, 'state_dict': {}}, gnn)
    current = copy.deepcopy(saved)
    current.owod_stage = 1
    current.graph_routing, current.graph_localization = 'soft', 'bbox-last'
    current.pretrained = str(checkpoint)
    current.screen_branch_checkpoint_sha256 = file_sha256(checkpoint)
    current.screen_branch_gnn_sha256 = file_sha256(gnn)
    return SimpleNamespace(manifest=manifest, screen=screen, weights=weights, checkpoint=checkpoint,
                           gnn=gnn, payload=payload, metadata=metadata, current=current)


def test_only_shared_task1_transition_can_change_strategy(parent):
    validate_checkpoint(parent.current, parent.payload)
    with pytest.raises(ValueError, match='chain mismatch'):
        validate_checkpoint(parent.current, parent.payload, resume=True)
    args = copy.deepcopy(parent.current)
    args.owod_phase = 'finetune'
    with pytest.raises(ValueError, match='chain mismatch'):
        validate_checkpoint(args, parent.payload)
    args = copy.deepcopy(parent.current)
    del args.screen_branch_checkpoint_sha256
    del args.screen_branch_gnn_sha256
    with pytest.raises(ValueError, match='graph_routing'):
        validate_checkpoint(args, parent.payload)


def test_no_distill_only_changes_at_shared_parent_transition(parent):
    parent.current.screen_no_distill = True
    validate_checkpoint(parent.current, parent.payload)
    payload = copy.deepcopy(parent.payload)
    payload['args'] = copy.deepcopy(parent.current)
    payload['args'].screen_no_distill = False
    with pytest.raises(ValueError, match='screen_no_distill'):
        validate_checkpoint(parent.current, payload, resume=True)
    parent.current.owod_phase = 'finetune'
    with pytest.raises(ValueError, match='screen_no_distill'):
        validate_checkpoint(parent.current, payload)


def test_no_distill_rejected_outside_diagnostic():
    from tools.owod.strict_setting import configure_run
    with pytest.raises(ValueError, match='screen-no-distill'):
        configure_run(SimpleNamespace(screen_no_distill=True, owod_screening=''))


def test_full_data_and_wrong_parent_cannot_use_branch_exception(parent, tmp_path):
    args = copy.deepcopy(parent.current)
    args.owod_screening_sha256 = None
    with pytest.raises(ValueError, match='restricted'):
        validate_checkpoint(args, parent.payload)
    other = tmp_path / 'wrong.pth'
    other.write_bytes(b'wrong')
    args = copy.deepcopy(parent.current)
    args.pretrained = str(other)
    with pytest.raises(ValueError, match='checkpoint or seed mismatch'):
        validate_checkpoint(args, parent.payload)
    payload = copy.deepcopy(parent.payload)
    payload['epoch'] = 10
    with pytest.raises(ValueError, match='completed preceding'):
        validate_checkpoint(parent.current, payload)


def test_shared_gnn_is_bound_to_exact_parent(parent):
    shared_task1.validate_inputs(parent.manifest, parent.screen, parent.checkpoint, parent.gnn, parent.weights, 42)
    parent.metadata['checkpoint_sha256'] = 'different-checkpoint'
    torch.save({'extra': parent.metadata, 'config': {}, 'state_dict': {}}, parent.gnn)
    with pytest.raises(ValueError, match='exactly the common'):
        shared_task1.validate_inputs(parent.manifest, parent.screen, parent.checkpoint, parent.gnn, parent.weights, 42)


def test_task2_plan_has_no_task1_or_calibration(parent, monkeypatch, tmp_path):
    image = tmp_path / 'one.jpg'
    image.touch()
    ann = tmp_path / 'ann.json'
    write_json(ann, {'images': [{'file_name': image.name}]})
    spec = {'protocol': 's-owodb', 'image_root': str(tmp_path), 'category_order_source_ids': [1, 3],
            'stages': [{'active_classes': [1]}, {'classes': [3], 'active_classes': [1, 3]}]}
    monkeypatch.setattr(run_benchmark, 'stage_files', lambda *a: (spec, dict.fromkeys(('increment_train','train','full_val'), ann)))
    monkeypatch.setattr('tools.owod.screening_data.screening_files', lambda *a: (spec, {}))
    args = SimpleNamespace(manifest=parent.manifest, owod_screening=parent.screen,
        screen_shared_task1=parent.checkpoint, screen_shared_gnn=parent.gnn,
        backbone_weights=parent.weights, recipe='ow-detr-paper', finetune_epochs=None,
        method='discovery', screen_increment_epochs=12, screen_finetune_epochs=6, screen_eval_interval=2,
        seed=42, output_dir=tmp_path / 'output', nproc_per_node=2, batch_size=2, num_workers=4,
        graph_adapter_rank=8, graph_adapter_last_n=2, graph_risk_beta=2., graph_route_loss_coef=1.,
        graph_sketch_max_images=12, graph_routing='soft', graph_localization='bbox-last')
    plan = run_benchmark.build_plan(args)
    assert [(r['stage'],r['phase']) for r in plan['runs']] == [(1,'increment'),(1,'finetune')]
    first, second = [r['command'] for r in plan['runs']]
    assert first[first.index('--pretrained') + 1] == str(parent.checkpoint)
    assert second[second.index('--pretrained') + 1] == str(Path(plan['runs'][0]['output_dir']) / 'checkpoint.pth')
    assert all(c[c.index('--graph-gnn-checkpoint') + 1] == str(parent.gnn) for c in (first,second))
    assert plan['shared_task1']['checkpoint_sha256'] == file_sha256(parent.checkpoint)
    args.screen_no_distill = True
    for run in run_benchmark.build_plan(args)['runs']:
        assert '--screen-no-distill' in run['command']
        assert '--old-class-distillation' not in run['command']
        assert '--discovery-no-keep' not in run['command']
    args.screen_no_distill = False
    args.owod_screening = None
    with pytest.raises(ValueError, match='require screening'):
        run_benchmark.build_plan(args)


def test_selection_checks_common_parent_for_every_phase(tmp_path):
    results(tmp_path)
    identity = {'checkpoint_sha256': 'parent', 'gnn_sha256': 'gnn'}
    with pytest.raises(ValueError, match='common parent/GNN'):
        select_winner(tmp_path, 's-owodb', 42, 4, 'screen', shared_identity=identity)
    for path in tmp_path.glob('screen/*/s-owodb/seed_42/stage_1/*/run_config.json'):
        import json
        record = json.loads(path.read_text())
        record.update(screen_branch_checkpoint_sha256='parent', screen_branch_gnn_sha256='gnn')
        write_json(path, record)
    report = select_winner(tmp_path, 's-owodb', 42, 4, 'screen', shared_identity=identity)
    assert report['winner'] == 'soft'
    assert all(row['training_and_evaluation_seconds'] == 20 for row in report['rows'])


def test_full_command_never_inherits_branch_flags(tmp_path):
    args = get_parser().parse_args([])
    args.screen_shared_task1, args.screen_shared_gnn = tmp_path / 'parent', tmp_path / 'gnn'
    args.screening_path = tmp_path / 'old-screen.json'
    command = benchmark_command(args, 'soft')
    assert command[command.index('--owod-screening')+1] == str(args.screening_path)
    full = benchmark_command(args, 'soft', screening=False)
    assert '--screen-shared-task1' not in full and '--owod-screening' not in full
