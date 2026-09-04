import tempfile
import unittest
from pathlib import Path

from tools.owod.audit_server_workspace import apply_safe, audit


class WorkspaceAuditTests(unittest.TestCase):
    def test_apply_removes_only_caches_and_exact_log_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").touch()
            (root / "tools/owod").mkdir(parents=True)
            cache = root / "util/__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.pyc").write_bytes(b"cache")
            experiment = root / "exps/run"
            experiment.mkdir(parents=True)
            (experiment / "train.log").write_text("canonical\n", encoding="utf-8")
            (experiment / "console.log").write_text("canonical\n", encoding="utf-8")
            (experiment / "launcher.log").write_text("extra context\n", encoding="utf-8")
            checkpoint = experiment / "checkpoint0004.pth"
            checkpoint.write_bytes(b"weights")

            report = audit(root, large_log_mib=1)
            duplicate_paths = {
                item["path"] for item in report["safe_delete"]["exact_duplicate_logs"]
            }
            self.assertEqual(duplicate_paths, {"exps/run/console.log"})
            self.assertIn("exps/run/launcher.log", {
                item["path"] for item in report["review_only"]
            })
            self.assertIn("exps/run/checkpoint0004.pth", {
                item["path"] for item in report["review_only"]
            })

            apply_safe(report)
            self.assertFalse(cache.exists())
            self.assertFalse((experiment / "console.log").exists())
            self.assertTrue((experiment / "train.log").exists())
            self.assertTrue((experiment / "launcher.log").exists())
            self.assertTrue(checkpoint.exists())

    def test_legacy_central_log_requires_an_exact_canonical_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").touch()
            (root / "tools/owod").mkdir(parents=True)
            canonical = root / "exps/run/train.log"
            legacy = root / "log/exps/run/train.log"
            canonical.parent.mkdir(parents=True)
            legacy.parent.mkdir(parents=True)
            canonical.write_text("same\n", encoding="utf-8")
            legacy.write_text("same\n", encoding="utf-8")

            report = audit(root)
            self.assertEqual(
                [item["path"] for item in report["safe_delete"]["exact_duplicate_logs"]],
                ["log/exps/run/train.log"])


if __name__ == "__main__":
    unittest.main()
