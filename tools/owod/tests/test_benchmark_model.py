import unittest

import torch

from models.deformable_detr import PostProcess, SetCriterion


class BenchmarkModelTests(unittest.TestCase):
    def test_control_background_cannot_displace_known_detections(self):
        logits = torch.full((1, 100, 91), -10.)
        logits[0, 0, 1] = 2.
        outputs = {'pred_logits': logits, 'pred_boxes': torch.full((1, 100, 4), .3)}
        sizes = torch.tensor([[100, 100]])
        legacy = PostProcess(known_ids=[1], max_detections=50)(outputs, sizes)[0]
        self.assertTrue((legacy['labels'] == 91).all())
        fixed = PostProcess(known_ids=[1], max_detections=50, include_unknown=False)(outputs, sizes)[0]
        self.assertTrue((fixed['labels'] == 1).all())
        self.assertAlmostEqual(float(fixed['scores'][0]), float(torch.tensor(2.).sigmoid()))
        self.assertFalse(fixed['unknown_mask'].any())

    def test_unintroduced_logits_cannot_change_known_or_unknown_outputs(self):
        logits = torch.full((1, 2, 91), -20.)
        logits[..., 1] = torch.tensor([4., -4.])
        logits[..., 8] = 100.  # unknown truck must never be predicted by name
        outputs = {'pred_logits': logits, 'pred_boxes': torch.full((1, 2, 4), .5)}
        result = PostProcess(known_ids=[1], max_detections=50)(outputs, torch.tensor([[100, 100]]))[0]
        self.assertEqual(set(result['labels'].tolist()), {1, 91})
        self.assertLess(float(result['unknown_scores'][result['labels'] == 1].min()), .1)
        self.assertGreater(float(result['scores'][result['labels'] == 91].max()), .9)

    def test_unintroduced_classifier_rows_receive_no_supervised_gradient(self):
        criterion = SetCriterion(91, None, {}, ['labels'])
        criterion.active_class_ids = [1, 5]
        logits = torch.zeros((1, 2, 91), requires_grad=True)
        targets = [{'labels': torch.tensor([1])}]
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        loss = criterion.loss_labels({'pred_logits': logits}, targets, indices, 1, log=False)['loss_ce']
        loss.backward()
        self.assertGreater(float(logits.grad[..., 1].abs().sum()), 0.)
        self.assertEqual(float(logits.grad[..., 8].abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
