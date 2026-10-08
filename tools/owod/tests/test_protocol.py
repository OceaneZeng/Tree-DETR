import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.owod.benchmark_spec import categories, specification, official_split
from tools.owod.protocol import build_official_manifest, stage_files


def fixture_split(protocol, name):
    if name in ('all_task_test', 'test'):
        return ['900001']
    task = int(name[1])
    return ['100001'] if name.endswith('train') else ['100001', '100002']


def write_fixture(path, classes, stems, instances=1):
    images = [{'id': int(stem), 'file_name': stem + '.jpg', 'owod_source_id': stem}
              for stem in stems]
    anns = [{'id': len(classes) * j + i + 1, 'image_id': int(stems[0]),
             'category_id': c, 'bbox': [j * 2, i * 2, 1, 1], 'voc_bbox': [j * 2 + 1, i * 2 + 1, j * 2 + 1, i * 2 + 1]}
            for j in range(instances) for i, c in enumerate(classes)]
    path.write_text(json.dumps({'images': images, 'annotations': anns, 'categories': categories()}), encoding='utf-8')


def make_tree(root, protocol='m-owodb'):
    spec = specification(protocol)
    all_classes = [c for task in spec['task_class_ids'] for c in task]
    active = []
    for i, current in enumerate(spec['task_class_ids']):
        active += current
        stage = root / f'stage_{i}'
        stage.mkdir()
        write_fixture(stage / 'instances_increment_train2017.json', current, fixture_split(protocol, f't{i+1}_train'))
        write_fixture(stage / 'instances_train2017.json', active,
                      fixture_split(protocol, f't{i+1}_train' if i == 0 else f't{i+1}_ft'), instances=50)
        write_fixture(stage / 'instances_val2017.json', active, ['900001'])
        write_fixture(stage / 'instances_val2017_full.json', all_classes, ['900001'])


class OfficialProtocolTests(unittest.TestCase):
    def test_pinned_real_release_and_class_boundaries(self):
        self.assertEqual(list(map(len, specification('m-owodb')['task_class_ids'])), [20] * 4)
        self.assertEqual(list(map(len, specification('s-owodb')['task_class_ids'])), [19, 21, 20, 20])
        self.assertEqual(len(official_split('m-owodb', 'all_task_test')), 10246)
        self.assertEqual(len(set(official_split('m-owodb', 'all_task_test'))), 9119)
        self.assertEqual(len(official_split('s-owodb', 'test')), 4952)

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_imports_both_official_category_orders_and_legal_image_overlap(self):
        for protocol in ('m-owodb', 's-owodb'):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_tree(root, protocol)
                manifest = build_official_manifest(root, protocol, 'author-release-fixture')
                self.assertEqual(manifest['schema_version'], 3)
                self.assertGreater(manifest['stages'][1]['cross_task_image_overlap'], 0)
                path = root / 'split_manifest.json'
                path.write_text(json.dumps(manifest), encoding='utf-8')
                loaded, files = stage_files(path, 1)
                self.assertEqual(loaded['stages'][1]['classes'], specification(protocol)['task_class_ids'][1])
                self.assertTrue(files['train'].is_file())

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_rejects_future_class_supervision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            p = root / 'stage_0/instances_increment_train2017.json'
            payload = json.loads(p.read_text())
            payload['annotations'][0]['category_id'] = 8  # truck is future in M
            p.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'future class'):
                build_official_manifest(root, 'm-owodb', 'author-release-fixture')

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_rejects_heuristic_image_membership_and_duplicate_annotations(self):
        for mutation, expected in [('image', 'official split'), ('annotation', 'duplicate annotation')]:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_tree(root)
                p = root / 'stage_0/instances_increment_train2017.json'
                data = json.loads(p.read_text())
                if mutation == 'image':
                    data['images'][0]['owod_source_id'] = 'custom'
                else:
                    data['annotations'].append(dict(data['annotations'][0]))
                p.write_text(json.dumps(data), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, expected):
                    build_official_manifest(root, 'm-owodb', 'author-release-fixture')

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_distinct_source_objects_with_same_integer_box_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            path = root / 'stage_0/instances_increment_train2017.json'
            data = json.loads(path.read_text())
            data['annotations'][0]['owod_source_annotation_id'] = 'original-object'
            other = {**data['annotations'][0], 'id': 999,
                     'owod_source_annotation_id': 'distinct-object'}
            data['annotations'].append(other)
            path.write_text(json.dumps(data), encoding='utf-8')
            build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            other['owod_source_annotation_id'] = 'original-object'
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'duplicate annotation'):
                build_official_manifest(root, 'm-owodb', 'author-release-fixture')

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_loader_detects_post_registration_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            p = root / 'stage_0/instances_increment_train2017.json'
            p.write_text(p.read_text() + '\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed after registration'):
                stage_files(path, 0)

    def test_old_manifests_and_all_unverified_overrides_are_retired(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.json'
            path.write_text(json.dumps({'schema_version': 2, 'official_annotations': True}), encoding='utf-8')
            for bypass in (False, True):
                with self.assertRaisesRegex(ValueError, 'retired'):
                    stage_files(path, 0, allow_unverified=bypass)


if __name__ == '__main__':
    unittest.main()
