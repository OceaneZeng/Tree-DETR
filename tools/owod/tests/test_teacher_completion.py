import unittest

import torch

from models.graph_local.pseudo_labels import select_teacher_pseudo_labels


class TeacherCompletionTests(unittest.TestCase):
    def test_non_contiguous_old_class_ids_are_preserved(self):
        outputs = {
            "pred_logits": torch.tensor([[[0.1, 0.2, 4.0, 0.3, 3.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        }
        targets = [{
            "boxes": torch.zeros((0, 4)),
            "labels": torch.zeros((0,), dtype=torch.long),
            "image_id": torch.tensor([1]),
        }]
        completed, counts = select_teacher_pseudo_labels(
            outputs, targets, old_class_ids=[2, 4], score_threshold=0.9)

        self.assertEqual(counts, [1])
        self.assertEqual(completed[0]["labels"].tolist(), [2])


if __name__ == "__main__":
    unittest.main()
