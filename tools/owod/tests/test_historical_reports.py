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

from tools.analysis import replay_report as runner
from tools.analysis import plot_replay_comparison as plot


def historical_plans(args):
    """Synthetic archived log records, independent of any training launcher."""
    from tools.owod.protocol import file_sha256
    plans = []
    for seed in args.seeds or [42]:
        for method in runner.METHODS:
            directory = args.output_dir / f'stage_1/seed_{seed}' / method
            payload = {'images': [], 'annotations': [], 'categories': []}
            temporary = args.output_dir.parent / 'annotation-fixture.json'
            runner.write_json(temporary, payload)
            plan = {'method': method, 'stage': 1, 'task': 2, 'seed': seed,
                    'epochs': 20, 'unknown_gt': 10, 'gpus': '0,1',
                    'source': 'synthetic historical log fixture',
                    'comparison_mode': 'historical', 'exposure': {}, 'fingerprints': {},
                    'annotation_sha256': file_sha256(temporary), 'directory': str(directory),
                    'settings': {'seed': seed, 'teacher_completion': method != 'uniform'}}
            plans.append((directory, plan, payload))
    return plans


class ReplayComparisonTests(unittest.TestCase):
    def fixture(self, root):
        return argparse.Namespace(output_dir=root / 'comparison', seeds=None)

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


    def test_report_rejects_teacher_setting_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            plans = historical_plans(args)
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
    def test_partial_and_unpaired_seeds_do_not_enter_final_bars(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.fixture(Path(temp))
            args.seeds = [42, 43]
            plans = historical_plans(args)
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
            plans = historical_plans(args)
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
            self.assertEqual(len(result['figures']), 28)
            for name in result['figures']:
                self.assertGreater((result['report_dir'] / name).stat().st_size, 1000)
            last_unknown = [r for r in result['finals'] if r['task'] == 4 and r['metric'] == 'u_recall']
            self.assertTrue(all(r['mean_percent'] is None for r in last_unknown))


if __name__ == '__main__':
    unittest.main()
