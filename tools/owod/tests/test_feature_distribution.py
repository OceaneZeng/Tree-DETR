"""Sampling and paired-instance checks without loading detector checkpoints."""

from pathlib import Path
import tempfile
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
