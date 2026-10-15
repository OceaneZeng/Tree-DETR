import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.owod import screening_data as data
from tools.owod.run_screening import benchmark_command, get_parser, select_winner
from tools.owod import run_screening
from tools.owod.run_server_experiment import VARIANTS
from tools.owod.strict_setting import validate_checkpoint


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Overlapping official-task identities, disjoint current class labels."""
    manifest = {'schema_version': 3, 'protocol': 's-owodb',
                'stages': [{'classes': [1, 2], 'active_classes': [1, 2]},
                           {'classes': [3, 4], 'active_classes': [1, 2, 3, 4]}]}
    path = tmp_path / 'manifest.json'
    data.write_json(path, manifest)
    def payload(classes, ids):
        return {'categories': [{'id': c} for c in range(1, 5)],
                'images': [dict(id=i, owod_source_id=str(i), file_name=f'{i}.jpg') for i in ids],
                'annotations': [dict(id=i * 10 + c, image_id=i, category_id=c,
                                     owod_source_annotation_id=i * 10 + c,
                                     bbox=[0, 0, 2, 2], voc_bbox=[1, 1, 2, 2]) for i in ids for c in classes]}
    files = []
    for stage, record in enumerate(manifest['stages']):
        train = tmp_path / f'{stage}-increment.json'
        memory = tmp_path / f'{stage}-memory.json'
        data.write_json(train, payload(record['classes'], range(1, 31)))
        data.write_json(memory, payload(record['active_classes'], range(25, 31)) if stage else
                        payload(record['classes'], range(1, 31)))
        files.append({'increment_train': train, 'train': memory})
    monkeypatch.setattr(data, 'stage_files', lambda p, stage: (manifest, files[stage]))
    return path, files


def test_screen_is_disjoint_covers_classes_and_preserves_memory(source, tmp_path):
    parent, original = source
    spec = data.prepare_screen(parent, tmp_path / 'screen', 8, 4, 42)
    heldout = set(data.read_json(spec)['heldout_source_ids'])
    for stage in range(2):
        manifest, files = data.screening_files(parent, spec, stage)
        train = data.read_json(files['increment_train'])
        assert len(train['images']) == 8
        assert not data.source_ids(train) & heldout
        assert {a['category_id'] for a in train['annotations']} == set(manifest['stages'][stage]['classes'])
        val = data.read_json(files['full_val'])
        assert data.source_ids(val) == heldout
        assert {a['category_id'] for a in val['annotations']} == set(manifest['stages'][stage]['active_classes'])
    assert data.read_json(files['train']) == data.read_json(original[1]['train'])
    assert data.prepare_screen(parent, tmp_path / 'screen', 8, 4, 42) == spec
    other = data.prepare_screen(parent, tmp_path / 'same', 8, 4, 42)
    assert data.read_json(other) == data.read_json(spec)


def test_changed_screen_inputs_are_rejected(source, tmp_path):
    parent, _ = source
    spec = data.prepare_screen(parent, tmp_path / 'screen', 8, 4, 42)
    with pytest.raises(ValueError, match='settings changed'):
        data.prepare_screen(parent, tmp_path / 'screen', 9, 4, 42)
    _, files = data.screening_files(parent, spec, 0)
    files['increment_train'].write_text('{}')
    with pytest.raises(ValueError, match='annotation changed'):
        data.screening_files(parent, spec, 0)


def test_rehashed_future_labels_still_rejected(source, tmp_path):
    parent, _ = source
    spec = data.prepare_screen(parent, tmp_path / 'screen', 8, 4, 42)
    record = data.read_json(spec)
    path = spec.parent / record['stages'][0]['files']['full_val']
    payload = data.read_json(path)
    payload['annotations'][0]['category_id'] = 4
    data.write_json(path, payload)
    record['stages'][0]['sha256']['full_val'] = data.file_sha256(path)
    data.write_json(spec, record)
    with pytest.raises(ValueError, match='validation'):
        data.screening_files(parent, spec, 0)


@pytest.mark.parametrize('saved_hash,current_hash', [('abc', None), (None, 'abc'), ('abc', 'def')])
def test_screen_checkpoint_cannot_initialize_full_or_another_screen(saved_hash, current_hash):
    args = SimpleNamespace(owod_manifest='manifest', owod_screening_sha256=current_hash)
    saved = SimpleNamespace(owod_screening_sha256=saved_hash)
    with pytest.raises(ValueError, match='Cannot mix screening'):
        validate_checkpoint(args, {'args': saved})


def test_full_command_starts_separate_chain_and_only_variants_change(tmp_path):
    args = get_parser().parse_args(['--output', str(tmp_path)])
    normalized = []
    for variant in ('hard', 'soft', 'soft-localization'):
        command = benchmark_command(args, variant)
        for flag in ('--output-dir', '--graph-routing', '--graph-localization'):
            command[command.index(flag) + 1] = '<variant>'
        normalized.append(command)
    assert normalized[0] == normalized[1] == normalized[2]
    full = benchmark_command(args, 'soft', screening=False)
    assert '--owod-screening' not in full and '--pretrained' not in full and '--resume' not in full
    assert '--screen-increment-epochs' not in full
    assert full[full.index('--output-dir') + 1] == str(tmp_path / 'full/soft')


def results(root):
    for variant, value in [('hard', .2), ('soft', .3), ('soft-localization', .25)]:
        for stage, phase, epoch in [(0, 'increment', 7), (1, 'increment', 7), (1, 'finetune', 3)]:
            directory = root / 'screen' / variant / 's-owodb/seed_42' / f'stage_{stage}' / phase
            data.write_json(directory / 'training_complete.json', {'last_epoch': epoch, 'training_time_seconds': 10})
            data.write_json(directory / 'run_config.json', {'owod_screening_sha256': 'screen', 'seed': 42,
                'paper_comparable_protocol': False, 'graph_routing': VARIANTS[variant][0],
                'graph_localization': VARIANTS[variant][1]})
            (directory / 'checkpoint.pth').touch()
        (directory / 'metrics.jsonl').write_text(json.dumps({'epoch': 3, 'test_owod_known_ap50': value,
            'test_owod_previous_ap50': value, 'test_owod_current_ap50': value}) + '\n')


def test_winner_requires_all_completed_final_results(tmp_path):
    results(tmp_path)
    report = select_winner(tmp_path, 's-owodb', 42, 4, 'screen')
    assert report['winner'] == 'soft'
    assert report['margin_ap50'] == pytest.approx(.05)
    marker = tmp_path / 'screen/hard/s-owodb/seed_42/stage_1/finetune/training_complete.json'
    data.write_json(marker, {'last_epoch': 2})
    with pytest.raises(ValueError, match='incomplete'):
        select_winner(tmp_path, 's-owodb', 42, 4, 'screen')


def test_no_winner_from_nonfinite_results(tmp_path):
    results(tmp_path)
    path = tmp_path / 'screen/soft/s-owodb/seed_42/stage_1/finetune/metrics.jsonl'
    value = json.loads(path.read_text())
    value['test_owod_known_ap50'] = float('nan')
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match='invalid validation metric'):
        select_winner(tmp_path, 's-owodb', 42, 4, 'screen')


def test_timeout_terminates_only_launched_process_group(tmp_path, monkeypatch):
    calls = []
    class Process:
        pid = 12345
        def wait(self, timeout=None):
            calls.append(('wait', timeout))
            if len([call for call in calls if call[0] == 'wait']) == 1:
                raise subprocess.TimeoutExpired('train', timeout)
            return -15
    def launch(*args, **kwargs):
        assert kwargs['start_new_session'] is True
        return Process()
    monkeypatch.setattr(run_screening.subprocess, 'Popen', launch)
    monkeypatch.setattr(run_screening.os, 'killpg', lambda pid, sig: calls.append(('kill', pid)), raising=False)
    with pytest.raises(TimeoutError, match='checkpoints retained'):
        run_screening.run_bounded(['train'], {}, run_screening.time.monotonic() + 10, tmp_path / 'run.log')
    assert ('kill', 12345) in calls


def test_expired_budget_never_starts_another_variant(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Expired budget must not start a subprocess')
    monkeypatch.setattr(run_screening.subprocess, 'Popen', unexpected)
    with pytest.raises(TimeoutError, match='budget exhausted'):
        run_screening.run_bounded(['train'], {}, 0, tmp_path / 'run.log')
