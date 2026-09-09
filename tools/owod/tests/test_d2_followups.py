import argparse
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.owod import run_d2_followups as runner


def dataset(ids, category):
    return {'images': [{'id': i, 'file_name': f'{i}.jpg'} for i in ids],
            'annotations': [{'id': i, 'image_id': i, 'category_id': category,
                             'bbox': [1, 1, 4, 4], 'area': 16, 'iscrowd': 0} for i in ids],
            'categories': [{'id': category, 'name': str(category)}]}


class D2FollowupTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'source'
        source.mkdir()
        d2 = root / 'd2' / 'graph'
        d2.mkdir(parents=True)
        current = dataset(range(100, 104), 3)
        old = dataset(range(1, 11), 1)
        second = dataset(range(11, 21), 2)
        for key in ('images', 'annotations', 'categories'):
            old[key] += second[key]
        selected, _ = runner.select_images(old, [1, 2], [1], 1, 2, 42)
        combined = runner.merge_annotation(current, old, selected)
        files = {'current.json': current, 'old.json': old, 'val.json': current, 'train.json': combined}
        for name, payload in files.items():
            runner.write_json(source / name, payload)
        runner.write_json(source / 'manifest.json', {'stages': [
            {'classes': [1, 2], 'files': {'train': str(source / 'old.json')}},
            {'classes': [3], 'files': {'increment_train': str(source / 'current.json'),
                                     'full_val': str(source / 'val.json')}}]})
        (source / 'stage0.pth').write_bytes(b'fixture')
        runner.write_json(source / 'graph.json', {'selected_replay_classes': [1]})
        runner.write_json(source / 'run_config.json', {
            'selected_replay_classes': [1], 'base_exemplars_per_class': 1,
            'risk_extra_exemplars_per_class': 2, 'graph_aggregation': 'top_mean',
            'graph_aggregation_top_n': 3})
        config = {'owod_stage': 1, 'epochs': 20, 'lr_drop': 15, 'lr': 1e-4, 'lr_backbone': 2e-5,
                  'batch_size': 2, 'seed': 42, 'teacher_completion': True, 'no_file_log': True,
                  'replay_sampling_fraction': 0.1, 'owod_previous_class_ids': [1, 2],
                  'owod_current_class_ids': [3], 'owod_known_class_ids': [1, 2, 3],
                  'pretrained': str(source / 'stage0.pth'), 'train_ann': str(source / 'train.json'),
                  'val_ann': str(source / 'val.json'), 'owod_manifest': str(source / 'manifest.json'),
                  'coco_path': str(root / 'coco'), 'start_epoch': 0,
                  'teacher_score_threshold': 0.5, 'local_margin_coef': 0.0, 'off_projection_coef': 0.0,
                  'unknown_threshold': 0.5, 'aux_loss': True, 'resume': '', 'eval_interval': 5}
        runner.write_json(d2 / 'run_config.json', config)
        self.finish(d2, 20)
        command = runner.matched_command(config, d2, source / 'train.json', 20, 29567, '0,1')
        plan = {'experiment': 'd2', 'source_run': str(source), 'command': command,
                'selected_replay_classes': [1], 'inputs': {name: {'path': str(source / name),
                    'sha256': runner.file_hash(source / name)} for name in
                    ('train.json', 'val.json', 'stage0.pth', 'manifest.json', 'graph.json')}}
        runner.write_json(d2.parent / 'diagnostic_plan.json', plan)
        for folder, data in [('train2017', old), ('train2017', current), ('val2017', current)]:
            directory = root / 'coco' / folder
            directory.mkdir(parents=True, exist_ok=True)
            for image in data['images']:
                (directory / image['file_name']).write_bytes(b'image fixture')
        args = argparse.Namespace(d2_dir=d2, output_dir=root / 'output', experiments=['random', 'long'],
                                  gpus='0,1', master_port=29583, resume=False)
        return args

    def finish(self, directory, epochs):
        directory.mkdir(parents=True, exist_ok=True)
        runner.write_json(directory / 'training_complete.json', {'last_epoch': epochs - 1})
        row = {'epoch': epochs - 1, **{'test_owod_' + key: 0.5 for key in
                                     ('previous_ap50', 'current_ap50', 'known_ap50', 'u_recall', 'h_score')}}
        (directory / 'metrics.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        (directory / 'checkpoint.pth').write_bytes(b'checkpoint fixture')

    def test_budget_repair_refills_overlap_and_preserves_base(self):
        old = dataset(range(1, 11), 1)
        for ann in list(old['annotations']):
            old['annotations'].append({**ann, 'id': ann['id'] + 20, 'category_id': 2})
        original, _ = runner.select_images(old, [1, 2], [1], 2, 2, 42)
        base, _ = runner.select_images(old, [1, 2], [], 2, 0, 42)
        repaired, report = runner.select_images(old, [1, 2], [1], 2, 2, 42, 8, [1])
        self.assertEqual(len(repaired - {1}), 8)
        self.assertTrue(base <= repaired)
        self.assertGreater(report['added'], 0)
        trimmed, report = runner.select_images(old, [1, 2], [1], 2, 2, 42, len(base - {1}), [1])
        self.assertTrue(base <= trimmed)
        self.assertEqual(len(trimmed - {1}), len(base - {1}))
        self.assertEqual(repaired, runner.select_images(old, [1, 2], [1], 2, 2, 42, 8, [1])[0])
        with self.assertRaisesRegex(ValueError, 'base exemplars'):
            runner.select_images(old, [1, 2], [1], 2, 2, 42, 1)

    def test_overlap_keeps_current_tag_and_adds_old_annotation_once(self):
        current, old = dataset([1], 3), dataset([1, 2], 1)
        old['annotations'][0]['id'] = 11
        merged = runner.merge_annotation(current, old, {1, 2})
        self.assertEqual([i['owod_replay'] for i in merged['images']], [False, True])
        self.assertEqual(len(merged['annotations']), 3)
        old['images'][0]['file_name'] = 'collision.jpg'
        with self.assertRaisesRegex(ValueError, 'collision'):
            runner.merge_annotation(current, old, {1})

    def test_plans_match_budget_steps_and_initial_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            plans = runner.create_plans(args)
            random_plan, long_plan = plans[0][1], plans[1][1]
            self.assertEqual(random_plan['replay_images'], long_plan['replay_images'])
            self.assertEqual(random_plan['optimizer_steps_per_epoch'], long_plan['optimizer_steps_per_epoch'])
            _, random_options = runner.split_command(random_plan['command'])
            _, long_options = runner.split_command(long_plan['command'])
            self.assertEqual(long_options['--epochs'], ['30'])
            self.assertEqual(long_options['--lr_drop'], ['15'])
            for key in ('--pretrained', '--lr', '--lr_backbone', '--teacher-completion', '--teacher-score-threshold'):
                self.assertEqual(random_options[key], long_options[key])
            self.assertNotIn('--resume', long_options)
            self.assertEqual(long_options['--start_epoch'], ['0'])
            self.assertEqual(plans[1][2], None)
            self.assertEqual(len(plans[0][2]['images']), random_plan['dataset_images'])

    def test_dry_run_readonly_and_all_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            with mock.patch.object(runner, 'cosine_neighbors', return_value=([2], {'risk': {'2': 1.0}})), \
                    contextlib.redirect_stdout(io.StringIO()), mock.patch.object(runner, 'run_child') as train:
                code = runner.main(['--d2-dir', str(args.d2_dir), '--output-dir', str(args.output_dir),
                                    '--experiments', 'random', 'long', 'cosine', '--dry-run'])
            self.assertEqual(code, 0)
            self.assertFalse(args.output_dir.exists())
            train.assert_not_called()

    def test_changed_source_and_frozen_backbone_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            config_file = args.d2_dir / 'run_config.json'
            config = runner.read_json(config_file)
            config['lr_backbone'] = 0
            runner.write_json(config_file, config)
            with self.assertRaisesRegex(ValueError, 'update backbone'):
                runner.create_plans(args)
            config['lr_backbone'] = 2e-5
            runner.write_json(config_file, config)
            Path(config['train_ann']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'input changed'):
                runner.create_plans(args)

    def test_reconstruction_rejects_changed_prior_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            old_file = Path(tmp) / 'source' / 'old.json'
            old = runner.read_json(old_file)
            for ann in old['annotations']:
                ann['bbox'][0] += 1
            runner.write_json(old_file, old)
            with self.assertRaisesRegex(ValueError, 'reconstruct D2'):
                runner.create_plans(args)

    def test_resume_artifacts_tampering_and_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            directory, plan, payload = runner.create_plans(args)[0]
            runner.write_json(directory / 'plan.json', plan)
            runner.write_json(directory / 'train.json', payload)
            with self.assertRaisesRegex(ValueError, 'Incomplete run'):
                runner.check_existing(directory, plan, False)
            self.finish(directory / 'graph', 20)
            self.assertTrue(runner.check_existing(directory, plan, False))
            (directory / 'train.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'annotation changed'):
                runner.check_existing(directory, plan, False)

    def test_busy_gpu_and_failed_training_do_not_start_next_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            argv = ['--d2-dir', str(args.d2_dir), '--output-dir', str(args.output_dir), '--experiments', 'random', 'long']
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(runner, 'training_environment', return_value={}), \
                    mock.patch.object(runner, 'busy_gpus', return_value=['occupied']), \
                    mock.patch.object(runner, 'run_child') as train:
                with self.assertRaisesRegex(RuntimeError, 'occupied'):
                    runner.main(argv)
                train.assert_not_called()
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(runner, 'training_environment', return_value={}), \
                    mock.patch.object(runner, 'busy_gpus', return_value=[]), \
                    mock.patch.object(runner, 'run_child', side_effect=RuntimeError('training failure')) as train:
                with self.assertRaisesRegex(RuntimeError, 'training failure'):
                    runner.main(argv)
                self.assertEqual(train.call_count, 1)
                self.assertFalse((args.output_dir / runner.ARMS['long']).exists())

    def test_long_summary_does_not_compare_epoch_30_to_d2_epoch_20(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            self.finish(args.output_dir / runner.ARMS['long'] / 'graph', 30)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                runner.summarize(args.d2_dir, args.output_dir)
            row = next(line for line in out.getvalue().splitlines() if line.startswith('long 30 '))
            self.assertTrue(row.endswith('NA NA complete'))

    def test_successful_queue_runs_in_order_and_skips_completed_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = self.fixture(Path(tmp))
            order = []
            def train(command, log_path, environment):
                _, options = runner.split_command(command)
                directory = Path(options['--output_dir'][0])
                order.append(directory.parent.name)
                self.assertNotIn('--resume', options)
                self.assertTrue(options['--pretrained'][0].endswith('stage0.pth'))
                self.finish(directory, int(options['--epochs'][0]))
            argv = ['--d2-dir', str(args.d2_dir), '--output-dir', str(args.output_dir)]
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(runner, 'training_environment', return_value={}), \
                    mock.patch.object(runner, 'busy_gpus', return_value=[]), \
                    mock.patch.object(runner, 'cosine_neighbors') as cosine, \
                    mock.patch.object(runner, 'run_child', side_effect=train) as launch:
                runner.main(argv)
                self.assertEqual(order, [runner.ARMS['random'], runner.ARMS['long']])
                runner.main(argv)
                self.assertEqual(launch.call_count, 2)
                cosine.assert_not_called()
                self.assertFalse((args.output_dir / runner.ARMS['cosine']).exists())
            self.assertEqual(runner.read_json(args.output_dir / 'queue_status.json')['status'], 'complete')

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU torch required for checkpoint scoring')
    def test_cosine_reads_last_classifier_and_aggregates_new_class_rows(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / 'stage0.pth'
            rows = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 0.], [1., 0.]])
            torch.save({'model': {'class_embed.0.weight': rows.flip(0), 'class_embed.5.weight': rows},
                        'args': argparse.Namespace(owod_stage=0)}, checkpoint)
            selected, details = runner.cosine_neighbors(checkpoint, [1, 2], [3, 4], 1, 'top_mean', 3)
            self.assertEqual(selected, [1])
            self.assertEqual(details['classifier_key'], 'class_embed.5.weight')
            self.assertEqual(details, json.loads(json.dumps(details)))
            self.assertAlmostEqual(details['risk']['1'], 1.)
            self.assertAlmostEqual(details['risk']['2'], 0.)


if __name__ == '__main__':
    unittest.main()
