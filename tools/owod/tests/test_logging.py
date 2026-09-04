import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from util.experiment_log import experiment_log_path, start_file_logging, stop_file_logging


class FileLoggingTests(unittest.TestCase):
    def test_fresh_run_archives_old_log_and_writes_inside_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log = output / "train.log"
            log.write_text("old run\n", encoding="utf-8")
            state = start_file_logging(SimpleNamespace(
                output_dir=output, log_file="", no_file_log=False, resume=""))
            try:
                print("new-run-marker")
            finally:
                stop_file_logging(state)

            self.assertEqual(experiment_log_path(output, "metrics.jsonl"),
                             output / "metrics.jsonl")
            self.assertIn("new-run-marker", log.read_text(encoding="utf-8"))
            archived = list((output / "log_archive").glob("*/train.log"))
            self.assertEqual(len(archived), 1)
            self.assertIn("old run", archived[0].read_text(encoding="utf-8"))

    def test_resume_appends_without_archiving(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            log = output / "train.log"
            log.write_text("first part\n", encoding="utf-8")
            state = start_file_logging(SimpleNamespace(
                output_dir=output, log_file="", no_file_log=False,
                resume="checkpoint.pth"))
            try:
                print("resumed part")
            finally:
                stop_file_logging(state)

            content = log.read_text(encoding="utf-8")
            self.assertIn("first part", content)
            self.assertIn("resumed part", content)
            self.assertFalse((output / "log_archive").exists())


if __name__ == "__main__":
    unittest.main()
