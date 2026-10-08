import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.owod.benchmark_spec import specification
from tools.owod.prepare_benchmarks import prepare
from tools.owod.tests.test_protocol import fixture_split


class PreparationTests(unittest.TestCase):
    @patch('tools.owod.prepare_benchmarks.official_split', lambda p, n: [s.zfill(12) for s in fixture_split(p, n)])
    @patch('tools.owod.protocol.official_split', lambda p, n: [s.zfill(12) for s in fixture_split(p, n)])
    def test_coco_to_s_matches_author_integer_coordinates_and_19_21_tasks(self):
        from tools.owod.benchmark_spec import categories
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'coco', root / 's-output'
            (source / 'annotations').mkdir(parents=True)
            objects = [{'id': len(categories()) * j + i + 1, 'image_id': 100001,
                        'category_id': c['id'], 'bbox': [j + .8, c['id'] + .6, 4.2, 4.2],
                        'iscrowd': 1 if j == 0 else 0}
                       for j in range(50) for i, c in enumerate(categories())]
            for subset, stems in [('train2017', ['000000100001', '000000100002']), ('val2017', ['000000900001'])]:
                (source / subset).mkdir()
                images = [{'id': int(stem), 'file_name': stem + '.jpg', 'width': 128, 'height': 128}
                          for stem in stems]
                anns = [{**ann, 'image_id': int(stems[0])} for ann in objects]
                payload = {'images': images, 'annotations': anns, 'categories': categories()}
                (source / 'annotations' / ('instances_' + subset + '.json')).write_text(json.dumps(payload))
                for stem in stems:
                    Image.new('RGB', (128, 128)).save(source / subset / (stem + '.jpg'))
            manifest = prepare('s-owodb', source, output)
            self.assertEqual(manifest['integrity']['task_class_counts'], [19, 21, 20, 20])
            full = json.loads(Path(manifest['stages'][0]['files']['full_val']).read_text())
            ann = full['annotations'][0]
            self.assertEqual(ann['voc_bbox'], [1, 2, 6, 6])
            self.assertEqual(ann['iscrowd'], 0)  # author XML conversion includes crowd as normal
            self.assertEqual(ann['difficult'], 0)

    @patch('tools.owod.prepare_benchmarks.official_split', fixture_split)
    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_voc_serialization_retains_geometry_difficult_and_label_visibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / 'source', root / 'output'
            (source / 'Annotations').mkdir(parents=True)
            (source / 'JPEGImages').mkdir()
            from tools.owod.benchmark_spec import categories
            objects = []
            for c in categories():
                for j in range(50):
                    objects.append(f"<object><name>{c['name']}</name><difficult>0</difficult>"
                                   f'<bndbox><xmin>{j+1}</xmin><ymin>{c["id"]+1}</ymin>'
                                   f'<xmax>{j+5}</xmax><ymax>{c["id"]+5}</ymax></bndbox></object>')
            xml = '<annotation><size><width>128</width><height>128</height></size>' + ''.join(objects) + '</annotation>'
            for stem in ('100001', '100002', '900001'):
                (source / 'Annotations' / (stem + '.xml')).write_text(xml, encoding='utf-8')
                Image.new('RGB', (128, 128)).save(source / 'JPEGImages' / (stem + '.jpg'))
            manifest = prepare('m-owodb', source, output)
            self.assertTrue(manifest['images_verified'])
            stage = manifest['stages'][0]
            increment = json.loads(Path(stage['files']['increment_train']).read_text())
            full = json.loads(Path(stage['files']['full_val']).read_text())
            self.assertEqual({a['category_id'] for a in increment['annotations']}, set(stage['classes']))
            self.assertEqual(len({a['category_id'] for a in full['annotations']}), 80)
            ann = increment['annotations'][0]
            self.assertEqual(ann['bbox'][0] + 1, ann['voc_bbox'][0])
            self.assertEqual(ann['bbox'][2], ann['voc_bbox'][2] - ann['voc_bbox'][0] + 1)
            with self.assertRaisesRegex(FileExistsError, 'clean output'):
                prepare('m-owodb', source, output)

    def test_missing_real_release_fails_before_outputs_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            with self.assertRaisesRegex(FileNotFoundError, 'VOC-style release is incomplete'):
                prepare('m-owodb', root, output)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
