"""Data preparation is atomic, repeatable, and rejects changed source annotations."""
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import pytest

from tools.owod import manage_data
from tools.owod.benchmark_spec import categories
from tools.owod.tests.test_protocol import fixture_split


def source_tree(root):
    (root / 'Annotations').mkdir(parents=True)
    (root / 'JPEGImages').mkdir()
    objects = ''.join(
        f'<object><name>{c["name"]}</name><difficult>0</difficult>'
        f'<bndbox><xmin>{j+1}</xmin><ymin>{c["id"]+1}</ymin>'
        f'<xmax>{j+5}</xmax><ymax>{c["id"]+5}</ymax></bndbox></object>'
        for c in categories() for j in range(50))
    xml = '<annotation><size><width>128</width><height>128</height></size>' + objects + '</annotation>'
    for stem in ('100001', '100002', '900001'):
        (root / 'Annotations' / (stem + '.xml')).write_text(xml, encoding='utf-8')
        Image.new('RGB', (128, 128)).save(root / 'JPEGImages' / (stem + '.jpg'))


@pytest.fixture
def released_fixture():
    with patch('tools.owod.manage_data.official_split', fixture_split), \
            patch('tools.owod.prepare_benchmarks.official_split', fixture_split), \
            patch('tools.owod.protocol.official_split', fixture_split):
        yield


def test_publish_paths_reuse_and_changed_raw_source(tmp_path, released_fixture):
    source, output = tmp_path / 'ore', tmp_path / 'derived' / 'm-owodb'
    source_tree(source)
    report = manage_data.prepare_dataset('m-owodb', source, output, decode_images=True)
    assert report['ready'] and report['action'] == 'prepared'
    assert report['images_decoded'] and report['unique_image_files_verified'] == 3
    manifest = json.loads((output / 'split_manifest.json').read_text())
    assert not Path(manifest['stages'][0]['files']['train']).is_absolute()
    assert manifest['annotation_root'] == str(output.resolve())
    second = manage_data.prepare_dataset('m-owodb', source, output)
    assert second['action'] == 'reused'
    assert second['manifest_sha256'] == report['manifest_sha256']
    changed = source / 'Annotations' / '100001.xml'
    changed.write_text(changed.read_text() + '\n')
    with pytest.raises(ValueError, match='Raw source annotation changed'):
        manage_data.prepare_dataset('m-owodb', source, output)


def test_failed_conversion_leaves_no_published_or_staging_dataset(tmp_path, released_fixture):
    source, output = tmp_path / 'ore', tmp_path / 'derived' / 'm-owodb'
    source_tree(source)
    def fail(protocol, data_root, staging):
        (staging / 'partial.json').write_text('{}')
        raise ValueError('invalid author annotation')
    with patch.object(manage_data, 'prepare', side_effect=fail):
        with pytest.raises(ValueError, match='invalid author annotation'):
            manage_data.prepare_dataset('m-owodb', source, output)
    assert not output.exists()
    assert not list(output.parent.glob('.prepare-*'))


def test_plan_verifies_real_lists_without_sources_or_outputs(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        code = manage_data.main(['plan', '--output-root', str(tmp_path / 'unused')])
    result = json.loads(captured.getvalue())
    assert code == 0 and not (tmp_path / 'unused').exists()
    assert [len(task['new_classes']) for task in result['benchmarks'][1]['tasks']] == [19, 21, 20, 20]
    assert result['benchmarks'][0]['test_entries'] == 10246


def test_both_missing_sources_report_failure_without_creating_data(tmp_path):
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        code = manage_data.main(['check', '--ore-root', str(tmp_path / 'ore'),
                                 '--coco-root', str(tmp_path / 'coco'),
                                 '--output-root', str(tmp_path / 'derived')])
    reports = json.loads(captured.getvalue())['benchmarks']
    assert code == 1 and len(reports) == 2
    assert all(not report['ready'] and report['missing_count'] for report in reports)
    assert not (tmp_path / 'derived').exists()


def test_old_output_and_another_source_are_never_overwritten(tmp_path, released_fixture):
    source, output = tmp_path / 'ore', tmp_path / 'derived'
    source_tree(source)
    output.mkdir()
    sentinel = output / 'old-setting.json'
    sentinel.write_text('user data')
    with pytest.raises(FileExistsError, match='Refusing to replace'):
        manage_data.prepare_dataset('m-owodb', source, output)
    assert sentinel.read_text() == 'user data'
    with pytest.raises(ValueError, match='without nesting'):
        manage_data.prepare_dataset('m-owodb', source, source / 'generated')
    assert not (source / 'generated').exists()
    prepared = tmp_path / 'valid'
    manage_data.prepare_dataset('m-owodb', source, prepared)
    with pytest.raises(ValueError, match='another raw source root'):
        manage_data.prepare_dataset('m-owodb', tmp_path / 'other-ore', prepared)
