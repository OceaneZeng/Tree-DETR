import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from models.owod_detector import detector_profile_dict, unknown_score
from models.owod_metrics import table1_summary
from tools.owod.run_detector_control import build_command
from tools.owod.table1_reference import load_reference, validate_reference


class Table1PipelineTests(unittest.TestCase):
    def test_reference_table_is_complete_and_consistent(self):
        payload = load_reference()
        validate_reference(payload)
        for protocol in ("m-owodb", "s-owodb"):
            self.assertEqual(len(payload["protocols"][protocol]["methods"]), 8)
        deus = payload["protocols"]["m-owodb"]["methods"]["DEUS"]
        self.assertEqual(deus["tasks"][2]["u_rec"], 69.0)
        self.assertEqual(deus["tasks"][3]["known_map"], 46.0)

    def test_local_detector_is_not_labeled_as_a_paper_baseline(self):
        profile = detector_profile_dict()
        self.assertEqual(profile["name"], "deformable_detr_control")
        self.assertFalse(profile["paper_baseline"])
        scores = unknown_score({"pred_logits": torch.tensor([[[4.0, -4.0]]])})
        self.assertTrue(torch.all((0.0 <= scores) & (scores <= 1.0)))

    def test_table1_summary_uses_percent_and_harmonic_score(self):
        summary = table1_summary(
            {"Previous": 0.6, "Current": 0.4, "Known": 0.5},
            {"U-Recall": 0.25, "unknown_gt": 10.0},
        )
        self.assertEqual(summary["Known mAP"], 50.0)
        self.assertAlmostEqual(summary["H-Score"], 100.0 / 3.0)

    def test_control_command_uses_manifest_metadata_without_method_alias(self):
        args = SimpleNamespace(
            coco_path=Path("coco"), manifest=Path("manifest.json"), stage=1,
            output_dir=Path("out"), num_classes=91, epochs=20, batch_size=2,
            num_workers=4, seed=42, unknown_threshold=0.5, print_freq=100,
            eval_print_freq=100, lr_drop=15, eval_interval=5,
            resume=None, pretrained=Path("base.pth"),
            nproc_per_node=1, master_port=29521,
        )
        record = {"classes": [3, 4], "active_classes": [1, 2, 3, 4]}
        files = {"train": Path("train.json"), "full_val": Path("full.json")}
        joined = " ".join(build_command(args, record, files))
        self.assertIn("--owod-manifest manifest.json", joined)
        self.assertIn("--lr_drop 15", joined)
        self.assertIn("--eval_interval 5", joined)
        self.assertNotIn("--owod-baseline", joined)


if __name__ == "__main__":
    unittest.main()
