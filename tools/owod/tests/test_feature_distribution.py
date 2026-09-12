"""Sampling and paired-instance checks without loading detector checkpoints."""

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.owod import plot_feature_distribution as plotting


def record(class_id, gt_index, prediction=None, confidence=0.9, query_id=0):
    return {'class_id': class_id, 'gt_index': gt_index, 'image_id': 10,
            'query_id': query_id, 'confidence': confidence,
            'predicted_class_id': class_id if prediction is None else prediction}


class FeatureDistributionTests(unittest.TestCase):
    def test_rare_class_does_not_limit_other_classes(self):
        records = [record(1, i) for i in range(250)] + [record(2, 0)]
        selected, audit = plotting.select_class_records(records, {1, 2, 3}, 0.3, 200, 42)
        self.assertEqual(len(selected), 201)
        self.assertEqual(audit['1']['plotted'], 200)
        self.assertEqual(audit['2']['plotted'], 1)
        self.assertEqual(audit['3']['eligible'], 0)
        self.assertEqual(len({(r['class_id'], r['gt_index']) for r in selected}), 201)

    def test_correct_filter_is_optional(self):
        records = [record(1, 0, prediction=2), record(1, 1, confidence=0.1), record(1, 2)]
        all_records, _ = plotting.select_class_records(records, {1}, 0.3, 200, 42)
        correct, _ = plotting.select_class_records(records, {1}, 0.3, 200, 42, 'correct')
        self.assertEqual(len(all_records), 3)
        self.assertEqual([r['gt_index'] for r in correct], [2])

    def test_pair_by_gt_not_query_or_prediction(self):
        before = [record(1, 0, prediction=9, query_id=2), record(1, 1), record(2, 0)]
        after = [record(1, 0, query_id=40), record(1, 2), record(2, 0)]
        left, right, audit = plotting.paired_class_records(before, after, {1}, 200, 42)
        self.assertEqual(len(left), 1)
        self.assertEqual(left[0]['gt_index'], right[0]['gt_index'])
        self.assertEqual((left[0]['query_id'], right[0]['query_id']), (2, 40))
        self.assertEqual(audit['1']['paired'], 1)

    def test_known_merge_preserves_original_labels(self):
        groups = np.array(['Previous'] * 5 + ['Current'] * 5 + ['Unknown'] * 6 + ['Background'] * 6)
        selected = plotting.group_balanced_indices(groups, 6, 42)
        merged = plotting.merge_known_groups(groups)
        self.assertEqual({g: int(sum(merged[selected] == g)) for g in plotting.GROUP_ORDER},
                         {'Known': 6, 'Unknown': 6, 'Background': 6})
        self.assertEqual(groups[0], 'Previous')

    def test_task_sampling_keeps_dense_known_and_caps_other_groups(self):
        records = [{**record(1, i), 'group': 'Previous'} for i in range(250)]
        records += [{**record(2, 0), 'group': 'Current'}]
        records += [{**record(3, i), 'group': 'Unknown'} for i in range(12)]
        records += [{**record(-1, i), 'group': 'Background'} for i in range(2)]
        args = SimpleNamespace(class_confidence=0.3, max_per_class=200, seed=42,
                               class_filter='matched', max_per_group=10)
        selected, audit, counts = plotting.select_task_records(records, {1, 2}, args)
        self.assertEqual(counts, {'Known': 201, 'Unknown': 10, 'Background': 2})
        self.assertEqual(audit['1']['plotted'], 200)
        self.assertEqual(len(selected), 213)
        self.assertEqual(records[0]['group'], 'Previous')
        again, _, _ = plotting.select_task_records(records, {1, 2}, args)
        self.assertEqual(selected, again)
        _, _, final_counts = plotting.select_task_records(
            [r for r in records if r['group'] != 'Unknown'], {1, 2}, args)
        self.assertEqual(final_counts['Unknown'], 0)

    def test_four_task_manifest_and_cli(self):
        manifest = {'stages': [{'classes': [str(i)],
                                'active_classes': [str(c) for c in range(i + 1)]}
                               for i in range(4)]}
        self.assertEqual(plotting.four_task_class_sets(manifest),
                         [[0], [0, 1], [0, 1, 2], [0, 1, 2, 3]])
        manifest['stages'][3]['classes'] = [2]
        with self.assertRaises(ValueError):
            plotting.four_task_class_sets(manifest)
        args = plotting.parse_args(['--run-dir', 'run', '--output-dir', 'out',
                                    '--task-checkpoints', 'a', 'b', 'c', 'd'])
        self.assertEqual(len(args.task_checkpoints), 4)

    def test_four_task_layout_has_eight_panels_one_legend_and_paired_coordinates(self):
        from matplotlib.figure import Figure
        task_classes = [list(range(20 * (i + 1))) for i in range(4)]
        names = {i: f'category {i}' for i in range(80)}
        names[0], names[1] = 'traffic light', 'refrigerator'
        rng = np.random.default_rng(42)
        points, tasks, groups, ids = [], [], [], []
        for i, known in enumerate(task_classes):
            local_ids = list(np.repeat(known, 3)) + [-1] * 4
            local_groups = ['Known'] * (len(known) * 3) + ['Background'] * 4
            if i < 3:
                local_ids += [79] * 4
                local_groups += ['Unknown'] * 4
            points.extend(rng.normal(size=(len(local_ids), 2)) * (i + 1) + i * 10)
            tasks.extend([f'Task {i + 1}'] * len(local_ids))
            groups.extend(local_groups)
            ids.extend(local_ids)
        embedding, tasks, groups, ids = map(np.asarray, (points, tasks, groups, ids))
        original = embedding.copy()
        checks = []

        def inspect_figure(figure, *args, **kwargs):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            self.assertEqual(len(figure.axes), 8)
            self.assertEqual(len(figure.legends), 1)
            legend = figure.legends[0].get_window_extent(renderer)
            self.assertGreaterEqual(legend.x0, 0)
            self.assertGreaterEqual(legend.y0, 0)
            self.assertLessEqual(legend.x1, figure.bbox.x1)
            self.assertEqual(len(figure.legends[0].texts), 83)
            for index in range(4):
                top, bottom = figure.axes[2 * index:2 * index + 2]
                self.assertIsNone(top.get_legend())
                self.assertIsNone(bottom.get_legend())
                self.assertEqual(top.get_xlim(), bottom.get_xlim())
                self.assertEqual(top.get_ylim(), bottom.get_ylim())
                self.assertAlmostEqual(top.get_position().x0, bottom.get_position().x0)
                self.assertGreater(top.get_position().y0, bottom.get_position().y1)
                self.assertGreater(bottom.get_tightbbox(renderer).y0, legend.y1)
                np.testing.assert_array_equal(
                    np.concatenate([item.get_offsets() for item in top.collections]),
                    bottom.collections[-1].get_offsets())
                for collection in top.collections:
                    self.assertLessEqual(len(collection.get_offsets()), 3)
                self.assertEqual(len(bottom.collections), 2 if index == 3 else 3)
                np.testing.assert_array_equal(top.collections[0].get_facecolors(),
                                              figure.axes[0].collections[0].get_facecolors())
                for axis in (top, bottom):
                    bounds = axis.get_tightbbox(renderer)
                    self.assertGreaterEqual(bounds.x0, 0)
                    self.assertLessEqual(bounds.x1, figure.bbox.x1)
                    self.assertLessEqual(bounds.y1, figure.bbox.y1)
            checks.append(True)

        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, 'savefig', inspect_figure):
            plotting.save_four_task_plot(embedding, tasks, groups, ids, names,
                                         task_classes, Path(directory))
        self.assertEqual(len(checks), 2)
        np.testing.assert_array_equal(embedding, original)

    def test_comparison_layout_and_coordinates(self):
        from matplotlib.figure import Figure
        rng = np.random.default_rng(42)
        embedding = rng.normal(size=(400, 2))
        original = embedding.copy()
        groups = np.array(['Before'] * 200 + ['After'] * 200)
        ids = np.tile(np.repeat(np.arange(20), 10), 2)
        names = {i: f'category {i}' for i in range(20)}
        names[0] = 'traffic light'
        names[1] = 'refrigerator'
        checks = []

        def inspect_figure(figure, *args, **kwargs):
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            axes = figure.axes
            for plot_axis, legend_axis in ((axes[0], axes[2]), (axes[1], axes[3])):
                legend = legend_axis.get_legend().get_window_extent(renderer)
                plot = plot_axis.get_window_extent(renderer)
                self.assertGreaterEqual(legend.x0, plot.x1)
                self.assertGreaterEqual(legend.y0, 0)
                self.assertLessEqual(legend.y1, figure.bbox.y1)
                self.assertLessEqual(legend.x1, figure.bbox.x1)
            for a, b in zip(axes[0].collections, axes[1].collections):
                np.testing.assert_equal(a.get_facecolors(), b.get_facecolors())
            checks.append(True)

        with tempfile.TemporaryDirectory() as directory, patch.object(Figure, 'savefig', inspect_figure):
            plotting.save_class_plot(embedding, groups, ids, names, Path(directory),
                                     panel_groups=('Before', 'After'),
                                     titles=('Before training', 'After training'), shared_colors=True)
        self.assertEqual(len(checks), 2)
        np.testing.assert_equal(embedding, original)


if __name__ == '__main__':
    unittest.main()
