import json
import tempfile
import unittest
from pathlib import Path

import torch

from tools.iod.analyze_risk_drop import analyze
from tools.iod.build_replay_annotation import allocate_quotas, build_annotation, select_images
from tools.iod.estimate_conflict_risk import target_for_class


class ReplayAllocationTest(unittest.TestCase):
    def test_uniform_quotas_keep_exact_budget(self):
        quotas = allocate_quotas([1, 2, 3], 10, None, 1e-3)
        self.assertEqual(sum(quotas.values()), 10)
        self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)

    def test_risk_quotas_prioritize_high_risk_class(self):
        quotas = allocate_quotas([1, 2, 3], 12, {1: 0.0, 2: 1.0, 3: 3.0}, 1e-3)
        self.assertEqual(sum(quotas.values()), 12)
        self.assertGreater(quotas[3], quotas[2])
        self.assertGreater(quotas[2], quotas[1])

    def test_selection_uses_unique_images(self):
        coco = {
            "annotations": [
                {"image_id": 10, "category_id": 1},
                {"image_id": 10, "category_id": 2},
                {"image_id": 11, "category_id": 1},
                {"image_id": 12, "category_id": 2},
            ]
        }
        selected, _ = select_images(coco, [1, 2], 3, 42, None, 1e-3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len(set(selected)), 3)

    def test_combined_annotation_keeps_old_and_new_categories(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base_path = root / "base.json"
            new_path = root / "new.json"
            output_path = root / "combined.json"
            base_path.write_text(json.dumps({
                "images": [{"id": 10, "file_name": "old.jpg"}],
                "annotations": [{"id": 1, "image_id": 10, "category_id": 1}],
                "categories": [{"id": 1, "name": "old"}],
            }), encoding="utf-8")
            new_path.write_text(json.dumps({
                "images": [{"id": 20, "file_name": "new.jpg"}],
                "annotations": [{"id": 2, "image_id": 20, "category_id": 2}],
                "categories": [{"id": 2, "name": "new"}],
            }), encoding="utf-8")
            build_annotation(new_path, base_path, output_path, [10], [1], 42,
                             "uniform", {1: 1})
            combined = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual([category["id"] for category in combined["categories"]], [1, 2])
            self.assertEqual(len(combined["images"]), 2)
            self.assertEqual(len(combined["annotations"]), 2)


class RiskAnalysisTest(unittest.TestCase):
    def test_target_for_class_preserves_image_metadata(self):
        target = {
            "labels": torch.tensor([1, 2]),
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.1, 0.1, 0.9, 0.9]]),
            "area": torch.tensor([1.0, 2.0]),
            "iscrowd": torch.tensor([0, 0]),
            "image_id": torch.tensor([17]),
            "orig_size": torch.tensor([480, 640]),
            "size": torch.tensor([480, 640]),
        }
        filtered = target_for_class(target, 2)
        self.assertEqual(filtered["labels"].tolist(), [2])
        self.assertEqual(tuple(filtered["boxes"].shape), (1, 4))
        self.assertEqual(filtered["image_id"].tolist(), [17])
        self.assertEqual(filtered["orig_size"].tolist(), [480, 640])
        self.assertEqual(filtered["size"].tolist(), [480, 640])

    def test_marks_floor_saturated_forgetting(self):
        base = {index: 0.1 + index / 100.0 for index in range(1, 6)}
        after = {index: 0.0 for index in range(1, 6)}
        risk = {"old_classes": list(range(1, 6)), "risk": [1, 2, 3, 4, 5]}
        result = analyze(base, after, risk, top_k=2)
        self.assertEqual(result["diagnostic_status"], "saturated_forgetting")
        self.assertEqual(result["zero_after_count"], 5)
        self.assertEqual(result["zero_after_fraction"], 1.0)
        self.assertAlmostEqual(result["mean_after_ap50"], 0.0)

    def test_keeps_informative_nonzero_distribution(self):
        base = {index: 0.5 for index in range(1, 6)}
        after = {index: index / 20.0 for index in range(1, 6)}
        risk = {"old_classes": list(range(1, 6)), "risk": [5, 4, 3, 2, 1]}
        result = analyze(base, after, risk, top_k=2)
        self.assertEqual(result["diagnostic_status"], "informative")
        self.assertEqual(result["zero_after_count"], 0)
        self.assertAlmostEqual(result["top_k_random_expectation"], 0.4)


if __name__ == "__main__":
    unittest.main()
