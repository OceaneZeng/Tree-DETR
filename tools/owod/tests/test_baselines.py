import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.owod_baselines import baseline_config, normalize_baseline, unknown_score
from models.owod_metrics import compute_owod_metrics
from tools.owod.run_baseline import build_command, write_run_metadata


class OWODBaselineTests(unittest.TestCase):
    def test_names_and_aliases(self):
        self.assertEqual(normalize_baseline("ore"), "ore_star")
        self.assertTrue(baseline_config("prob").has_objectness_head)
        self.assertTrue(baseline_config("oracle").oracle)

    def test_unknown_score_is_bounded(self):
        outputs = {
            "pred_logits": torch.tensor([[[0.0, 2.0], [-2.0, 0.0]]]),
            "pred_objectness": torch.tensor([[[2.0], [-2.0]]]),
        }
        score = unknown_score(outputs, "prob")
        self.assertEqual(tuple(score.shape), (1, 2))
        self.assertTrue(bool(((score >= 0) & (score <= 1)).all()))

    def test_run_metadata_and_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = SimpleNamespace(coco_path=Path(tmp) / "coco", train_ann=Path(tmp) / "train.json",
                                   val_ann=Path(tmp) / "val.json", output_dir=out,
                                   pretrained=None, num_classes=91, epochs=1, batch_size=1,
                                   num_workers=0, seed=42, device="cpu", method="prob",
                                   extra_main_arg=[])
            command = build_command(args)
            write_run_metadata(args, command, {"protocol": "S-OWODB"})
            payload = json.loads((out / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["method"], "prob")
            self.assertTrue((out / "command.txt").exists())
            self.assertIn("--owod-baseline prob", (out / "command.txt").read_text(encoding="utf-8"))
            self.assertIn("--no-file-log", command)

    def test_unknown_metrics_match_unknown_prediction(self):
        prediction = {"boxes": torch.tensor([[45.0, 45.0, 55.0, 55.0]]),
                      "labels": torch.tensor([2]), "scores": torch.tensor([0.9]),
                      "unknown_scores": torch.tensor([0.9])}
        target = {"boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
                  "labels": torch.tensor([5]), "orig_size": torch.tensor([100, 100])}
        metrics = compute_owod_metrics([prediction], [target], known_class_ids=[2], threshold=0.5)
        self.assertEqual(metrics["unknown_gt"], 1.0)
        self.assertEqual(metrics["U-Recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
