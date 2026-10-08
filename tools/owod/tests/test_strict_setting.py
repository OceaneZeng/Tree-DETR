import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.owod.protocol import build_official_manifest, file_sha256
from tools.owod.strict_setting import configure_run, validate_checkpoint
from tools.owod.tests.test_protocol import make_tree, fixture_split


def arguments(manifest, stage=0, phase='increment'):
    return SimpleNamespace(owod_manifest=str(manifest), owod_stage=stage, owod_phase=phase,
                           owod_recipe='ore-paper',
                           owod_known_class_ids=None, owod_previous_class_ids=None,
                           owod_current_class_ids=None, paper_baseline=None,
                           train_ann='', val_ann='', lightweight=False, no_augmentation=False,
                           replay_sampling_fraction=0, pretrained='', resume='', backbone_weights='')


class StrictSettingTests(unittest.TestCase):
    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_phase_files_derive_labels_and_reject_old_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            args = arguments(path)
            configure_run(args)
            self.assertEqual(len(args.owod_known_class_ids), 20)
            self.assertTrue(args.train_ann.endswith('instances_increment_train2017.json'))
            self.assertEqual(args.owod_manifest_sha256, file_sha256(path))
            with self.assertRaisesRegex(ValueError, 'old/custom'):
                validate_checkpoint(args, {'args': SimpleNamespace(owod_stage=0)})
            args = arguments(path, stage=1, phase='finetune')
            args.pretrained = 'stage_1_increment.pth'
            configure_run(args)
            self.assertTrue(args.train_ann.endswith('instances_train2017.json'))
            source = SimpleNamespace(owod_manifest_sha256=args.owod_manifest_sha256,
                                     owod_stage=1, owod_phase='increment', epochs=50,
                                     owod_recipe='ore-paper')
            validate_checkpoint(args, {'args': source, 'epoch': 49})
            with self.assertRaisesRegex(ValueError, 'completed preceding phase'):
                validate_checkpoint(args, {'args': source, 'epoch': 48})
            source.owod_recipe = 'ow-detr-paper'
            with self.assertRaisesRegex(ValueError, 'recipe changed'):
                validate_checkpoint(args, {'args': source, 'epoch': 49})
            source.owod_recipe = 'ore-paper'
            source.owod_phase = 'finetune'
            with self.assertRaisesRegex(ValueError, 'Expected checkpoint'):
                validate_checkpoint(args, {'args': source, 'epoch': 49})

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_future_class_metadata_and_custom_replay_setting_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            args = arguments(path)
            args.owod_known_class_ids = list(range(1, 21))
            with self.assertRaisesRegex(ValueError, 'author-defined'):
                configure_run(args)
            args = arguments(path)
            args.replay_sampling_fraction = .1
            with self.assertRaisesRegex(ValueError, 'retired'):
                configure_run(args)

    @patch('tools.owod.protocol.official_split', fixture_split)
    def test_run_plan_has_separate_phases_and_previous_finetuning_chain(self):
        from tools.owod.run_benchmark import build_plan
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_tree(root)
            manifest = build_official_manifest(root, 'm-owodb', 'author-release-fixture')
            manifest['image_root'] = str(root)
            path = root / 'split_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            for stem in ('100001', '100002', '900001'):
                (root / (stem + '.jpg')).write_bytes(b'fixture path only')
            weights = root / 'dino.pth'
            weights.write_bytes(b'fixture path only')
            args = SimpleNamespace(manifest=path, recipe='ow-detr-paper', finetune_epochs=None,
                                   backbone_weights=weights, output_dir=root / 'output', seed=42,
                                   nproc_per_node=2, batch_size=2, num_workers=0)
            plan = build_plan(args)
            self.assertEqual([r['phase'] for r in plan['runs']],
                             ['increment', 'increment', 'finetune', 'increment', 'finetune', 'increment', 'finetune'])
            self.assertEqual([r['epochs'] for r in plan['runs']], [50, 50, 20, 50, 20, 50, 20])
            command = plan['runs'][3]['command']
            self.assertIn('stage_1/finetune/checkpoint.pth', command[command.index('--pretrained') + 1].replace('\\', '/'))


if __name__ == '__main__':
    unittest.main()
