"""CPU-only checks for experimental fairness, recovery and scientific reporting."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import shutil
from unittest import mock

from tools.owod import run_replay_comparison as runner
from tools.owod import plot_replay_comparison as plot
from tools.owod.run_stage1_diagnostics import split_command
from tools.owod.tests import test_d2_followups as fixtures


class ReplayComparisonTests(unittest.TestCase):
    def fixture(self, root):
        original = fixtures.D2FollowupTests().fixture(root)
        path = original.d2_dir / 'run_config.json'
        config = runner.read_json(path)
        config.update(teacher_old_class_ids=[1, 2], world_size=2)
        runner.write_json(path, config)
        metadata_path = root / 'source/run_config.json'
        metadata = runner.read_json(metadata_path)
        metadata.update(graph_estimator='gnn', checkpoint=config['pretrained'])
        runner.write_json(metadata_path, metadata)
        return argparse.Namespace(source_dirs=[original.d2_dir], output_dir=root / 'comparison',
                                  seeds=None, gpus='0,1', reference_world_size=2, master_port=29587)

    def materialize(self, args, plans):
        runner.write_json(args.output_dir / 'comparison.json', {
            'schema': 1, 'mode': 'paired_task_local_shared_initialization',
            'runs': [(d / 'plan.json').relative_to(args.output_dir).as_posix() for d, _, _ in plans]})
        for directory, plan, payload in plans:
            runner.write_json(directory / 'plan.json', plan)
            runner.write_json(directory / 'train.json', payload)

    def finish(self, directory, plan, known=.6, recall=.3, finish=True, epoch=None):
        graph = directory / 'graph'
        graph.mkdir(parents=True, exist_ok=True)
        epoch = plan['epochs'] if epoch is None else epoch
        row = {'epoch': epoch - 1, 'test_owod_previous_ap50': known - .1,
               'test_owod_current_ap50': known + .1, 'test_owod_known_ap50': known,
               'test_owod_u_recall': recall, 'test_owod_unknown_gt': plan['unknown_gt']}
        if plan['unknown_gt']:
            row['test_owod_h_score'] = 2 * known * recall / (known + recall)
        (graph / 'metrics.jsonl').write_text(json.dumps(row) + '\n', encoding='utf-8')
        runner.write_json(graph / 'run_config.json', {**plan['settings'],
            'train_ann': str(directory / 'train.json'), 'world_size': 2})
        if finish:
            runner.write_json(graph / 'training_complete.json', {'last_epoch': epoch - 1})
            (graph / 'checkpoint.pth').write_bytes(b'test fixture, not a trained model')

    def test_three_arms_share_settings_steps_and_uniform_annotations(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            plans = runner.create_plans(args)
            self.assertEqual([p['method'] for _, p, _ in plans], list(runner.METHODS))
            self.assertEqual(plans[1][2], plans[2][2])
            self.assertEqual(plans[1][1]['annotation_sha256'], plans[2][1]['annotation_sha256'])
            commands = [split_command(p['command'])[1] for _, p, _ in plans]
            for command in commands:
                self.assertNotIn('--resume', command)
                self.assertEqual(command['--epochs'], ['20'])
                self.assertEqual(command['--lr_drop'], ['15'])
                self.assertEqual(command['--start_epoch'], ['0'])
            for ignored in ('--output_dir', '--train-ann', '--teacher-completion'):
                for command in commands:
                    command.pop(ignored, None)
            self.assertEqual(commands[0], commands[1])
            self.assertEqual(commands[1], commands[2])
            self.assertEqual(plans[0][1]['exposure'], plans[2][1]['exposure'])
            self.assertFalse(plans[1][1]['settings']['teacher_completion'])
            self.assertTrue(plans[2][1]['settings']['teacher_completion'])
            self.assertEqual(plans[0][2], runner.read_json(runner.read_json(args.source_dirs[0] / 'run_config.json')['train_ann']))

    def test_uniform_is_deterministic_unique_and_not_class_quota(self):
        current = fixtures.dataset([100], 3)
        old = fixtures.dataset(range(1, 11), 1)
        reference = fixtures.runner.merge_annotation(current, old, {1, 2, 3, 4})
        first, info = runner.uniform_annotation(reference, current, old, [1, 2], 42)
        second, _ = runner.uniform_annotation(reference, current, old, [1, 2], 42)
        self.assertEqual(first, second)
        self.assertEqual(len(set(info['selected_image_ids'])), 4)
        # A class with no candidate images is permitted: this is image-uniform sampling.
        self.assertEqual(info['candidate_images'], 10)
        other, _ = runner.uniform_annotation(reference, current, old, [1, 2], 7)
        self.assertNotEqual(first['images'], other['images'])

    def test_overlap_keeps_same_labels_without_extra_images(self):
        current, old = fixtures.dataset([1], 3), fixtures.dataset(range(1, 9), 1)
        current['annotations'][0]['id'] = 99
        reference = fixtures.runner.merge_annotation(current, old, {1, 2, 3})
        payload, info = runner.uniform_annotation(reference, current, old, [1], 42)
        self.assertEqual(len(payload['images']), len(reference['images']))
        self.assertEqual(info['fixed_old_annotations_on_current_images'], 1)
        self.assertNotIn(1, info['selected_image_ids'])
        self.assertEqual([a['category_id'] for a in payload['annotations'] if a['image_id'] == 1], [3, 1])

    def test_dry_run_readonly_and_world_size_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            argv = ['--source-dirs', str(args.source_dirs[0]), '--output-dir', str(args.output_dir), '--dry-run']
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(runner, 'run_child') as child:
                self.assertEqual(runner.main(argv), 0)
            child.assert_not_called()
            self.assertFalse(args.output_dir.exists())
            args.gpus = '0'
            with self.assertRaisesRegex(ValueError, 'GPU count'):
                runner.create_plans(args)

    def test_resume_keeps_original_teacher_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            plans = runner.create_plans(args)
            self.materialize(args, plans)
            directory, plan, payload = plans[0]
            self.assertFalse(runner.check_existing(directory, plan, False))  # queued but never started
            self.finish(directory, plan)
            self.assertTrue(runner.check_existing(directory, plan, True))
            actual_path = directory / 'graph/run_config.json'
            actual = runner.read_json(actual_path)
            actual['pretrained'] = str(directory / 'graph/checkpoint.pth')
            runner.write_json(actual_path, actual)
            with self.assertRaisesRegex(ValueError, 'pretrained'):
                runner.check_existing(directory, plan, True)
            runner.write_json(directory / 'train.json', {})
            with self.assertRaisesRegex(ValueError, 'changed'):
                runner.check_existing(directory, plan, True)

    def test_failed_training_stops_queue(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            argv = ['--source-dirs', str(args.source_dirs[0]), '--output-dir', str(args.output_dir)]
            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(runner, 'training_environment', return_value={}), \
                    mock.patch.object(runner, 'require_idle'), \
                    mock.patch.object(runner, 'run_child', side_effect=RuntimeError('training failed')) as child:
                with self.assertRaisesRegex(RuntimeError, 'training failed'):
                    runner.main(argv)
            self.assertEqual(child.call_count, 1)
            self.assertEqual(runner.read_json(args.output_dir / 'queue_status.json')['status'], 'failed')
            self.assertFalse((args.output_dir / 'stage_1/seed_42/uniform/graph').exists())

    def test_invalid_metrics_and_no_unknown_task(self):
        row = {'test_owod_known_ap50': .6, 'test_owod_u_recall': .3, 'test_owod_h_score': .4}
        self.assertAlmostEqual(plot.normalized_metrics(row, {'unknown_gt': 10})['h_score'], 40)
        self.assertIsNone(plot.normalized_metrics(row, {'unknown_gt': 0})['h_score'])
        row['test_owod_h_score'] = .7
        with self.assertRaisesRegex(ValueError, 'H-score'):
            plot.normalized_metrics(row, {'unknown_gt': 10})
        row['test_owod_u_recall'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            plot.normalized_metrics(row, {'unknown_gt': 10})

    def test_successful_queue_resume_and_portable_log_reporting(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            argv = ['--source-dirs', str(args.source_dirs[0]), '--output-dir', str(args.output_dir)]
            calls = []

            def child(command, log_path, environment):
                directory = log_path.parent.parent
                plan = runner.read_json(directory / 'plan.json')
                calls.append(command)
                self.finish(directory, plan)

            with contextlib.redirect_stdout(io.StringIO()), \
                    mock.patch.object(runner, 'training_environment', return_value={}), \
                    mock.patch.object(runner, 'require_idle'), \
                    mock.patch.object(runner, 'run_child', side_effect=child), \
                    mock.patch.object(plot, 'build_report'):
                runner.main(argv)
                self.assertEqual(len(calls), 3)
                runner.main(argv)
                self.assertEqual(len(calls), 3)  # completed arms are not rerun
                first = args.output_dir / 'stage_1/seed_42/glca'
                plan = runner.read_json(first / 'plan.json')
                (first / 'graph/training_complete.json').unlink()
                self.finish(first, plan, finish=False, epoch=5)
                runner.main(argv + ['--resume'])
            self.assertEqual(len(calls), 4)
            self.assertEqual(calls[-1][-2:], ['--resume', str(first / 'graph/checkpoint.pth')])
            resumed = split_command(calls[-1])[1]
            self.assertEqual(resumed['--pretrained'], [plan['settings']['pretrained']])
            # Plotting can use copied logs without checkpoint weights or source data.
            copied = Path(temp) / 'copied'
            shutil.copytree(args.output_dir, copied, ignore=shutil.ignore_patterns('*.pth'))
            groups = plot.collect(copied)
            self.assertTrue(all(r['complete'] for r in groups[(1, 42)].values()))

    def test_report_rejects_teacher_setting_mismatch_and_changed_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            plans = runner.create_plans(args)
            self.materialize(args, plans)
            for directory, plan, _ in plans:
                self.finish(directory, plan)
            directory = plans[1][0]
            path = directory / 'graph/run_config.json'
            actual = runner.read_json(path)
            actual['teacher_completion'] = True
            runner.write_json(path, actual)
            with self.assertRaisesRegex(ValueError, 'teacher_completion'):
                plot.collect(args.output_dir)
            config_path = args.source_dirs[0] / 'run_config.json'
            config = runner.read_json(config_path)
            validation = Path(temp) / 'different_val.json'
            runner.write_json(validation, {})
            config['val_ann'] = str(validation)
            runner.write_json(config_path, config)
            with self.assertRaisesRegex(ValueError, 'full-label validation'):
                runner.create_plans(args)

    def test_partial_and_unpaired_seeds_do_not_enter_final_bars(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            args.seeds = [42, 43]
            plans = runner.create_plans(args)
            self.materialize(args, plans)
            for directory, plan, _ in plans:
                incomplete = plan['seed'] == 43 and plan['method'] == 'uniform'
                self.finish(directory, plan, known=.6 if plan['seed'] == 42 else .9,
                            finish=not incomplete, epoch=5 if incomplete else 20)
            with mock.patch.object(plot, 'draw_figures', return_value=[]), contextlib.redirect_stdout(io.StringIO()):
                result = plot.build_report(args.output_dir)
            rows = [r for r in result['finals'] if r['metric'] == 'known_ap50']
            self.assertTrue(all(r['n_seeds'] == 1 and r['mean_percent'] == 60 for r in rows))
            self.assertTrue(all(r['std_percent'] is None for r in rows))

    def test_report_renders_all_requested_metrics_and_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            plans = runner.create_plans(args)
            # Synthetic metrics only, generated inside a temporary test directory.
            plans2 = []
            for directory, plan, payload in plans:
                plan.update(unknown_gt=10)
                other = {**plan, 'stage': 3, 'task': 4, 'unknown_gt': 0}
                other_dir = args.output_dir / 'stage_3/seed_42' / plan['method']
                other['directory'] = str(other_dir)
                plans2.append((other_dir, other, payload))
            plans += plans2
            self.materialize(args, plans)
            for directory, plan, _ in plans:
                self.finish(directory, plan)
            with contextlib.redirect_stdout(io.StringIO()):
                result = plot.build_report(args.output_dir)
            self.assertEqual(len(result['figures']), 8)
            for name in result['figures']:
                self.assertGreater((result['report_dir'] / name).stat().st_size, 1000)
            last_unknown = [r for r in result['finals'] if r['task'] == 4 and r['metric'] == 'u_recall']
            self.assertTrue(all(r['mean_percent'] is None for r in last_unknown))


if __name__ == '__main__':
    unittest.main()
