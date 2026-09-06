import json
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.owod.run_stage1_diagnostics import (ARM_NAMES, build_command, load_source,
                                               main, metric_rows, split_command)


def source_command():
    return ["python", "-m", "torch.distributed.run", "--nproc_per_node", "2",
            "--master_port", "29565", "main.py", "--train-ann", "train.json",
            "--val-ann", "val.json", "--pretrained", "stage0.pth",
            "--output_dir", "original/graph", "--epochs", "20", "--lr", "0.0001",
            "--seed", "42", "--teacher-completion", "--replay-sampling-fraction", "0.1",
            "--neighbor-scoped-lora", "--trainable-class-ids", "3", "6", "80",
            "--off-neighborhood-basis", "basis.pt", "--off-projection-coef", "0.1",
            "--local-margin-coef", "0.5"]


class DiagnosticTests(unittest.TestCase):
    def make_source(self, directory):
        root = Path(directory) / "source"
        (root / "graph").mkdir(parents=True)
        command = source_command()
        for option, name in (("--train-ann", "train.json"), ("--val-ann", "val.json"),
                             ("--pretrained", "stage0.pth")):
            path = root / name
            path.write_text("fixture")
            command[command.index(option) + 1] = str(path)
        manifest = root / "manifest.json"
        manifest.write_text("{}")
        command.extend(["--owod-manifest", str(manifest)])
        (root / "graph.json").write_text('{"selected_replay_classes": [2]}')
        (root / "run_config.json").write_text(json.dumps({
            "command": command, "selected_replay_classes": [2]}))
        recorded = dict(owod_stage=1, neighbor_scoped_lora=True, teacher_completion=True,
                        resume="", epochs=20, seed=42, lr=0.0001,
                        train_ann=str(root / "train.json"), val_ann=str(root / "val.json"),
                        pretrained=str(root / "stage0.pth"))
        (root / "graph/run_config.json").write_text(json.dumps(recorded))
        (root / "graph/training_complete.json").write_text(json.dumps({"last_epoch": 19}))
        return root

    def test_d1_preserves_data_teacher_and_new_row_mask(self):
        _, original = split_command(source_command())
        _, changed = split_command(build_command(source_command(), "d1", Path("d1/graph"), 29567))
        for key in ("--train-ann", "--val-ann", "--pretrained", "--epochs", "--lr",
                    "--seed", "--teacher-completion", "--replay-sampling-fraction",
                    "--neighbor-scoped-lora", "--trainable-class-ids"):
            self.assertEqual(original[key], changed[key])
        self.assertEqual(changed["--off-projection-coef"], ["0"])
        self.assertEqual(changed["--local-margin-coef"], ["0"])
        self.assertNotIn("--off-neighborhood-basis", changed)
        self.assertNotIn("--resume", changed)

    def test_d2_unfreezes_detector_without_replacing_teacher(self):
        _, changed = split_command(build_command(source_command(), "d2", Path("d2/graph"), 29567))
        self.assertNotIn("--neighbor-scoped-lora", changed)
        self.assertNotIn("--trainable-class-ids", changed)
        self.assertEqual(changed["--pretrained"], ["stage0.pth"])
        self.assertEqual(changed["--off-projection-coef"], ["0"])

    def test_resume_keeps_original_teacher_and_restores_only_arm_checkpoint(self):
        _, changed = split_command(build_command(
            source_command(), "d1", Path("d1/graph"), 29567, resume=True))
        self.assertEqual(changed["--resume"], [str(Path("d1/graph/checkpoint.pth"))])
        self.assertEqual(changed["--pretrained"], ["stage0.pth"])

    def test_partial_metrics_and_resumed_duplicate_epochs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            path.write_text('\n'.join([
                json.dumps({"epoch": 4, "test_owod_current_ap50": 0.1}),
                json.dumps({"epoch": 5, "train_loss": 1}),
                json.dumps({"epoch": 4, "test_owod_current_ap50": 0.2}),
                '{"epoch":']), encoding="utf-8")
            rows = metric_rows(path)
            self.assertEqual(list(rows), [4])
            self.assertEqual(rows[4]["test_owod_current_ap50"], 0.2)

    def test_mismatched_source_config_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph").mkdir()
            (root / "run_config.json").write_text(json.dumps({"command": source_command()}))
            recorded = dict(owod_stage=1, neighbor_scoped_lora=True, teacher_completion=True,
                            resume="", epochs=20, train_ann="different.json")
            (root / "graph/run_config.json").write_text(json.dumps(recorded))
            (root / "graph/training_complete.json").write_text(json.dumps({"last_epoch": 19}))
            with self.assertRaisesRegex(ValueError, "differ: --train-ann"):
                load_source(root)

    def test_both_arm_dry_run_does_not_create_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(directory)
            output = Path(directory) / "outputs"
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):
                code = main(["--source-run", str(source), "--output-dir", str(output), "--dry-run"])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())
            self.assertIn("d1: exact D0", captured.getvalue())
            self.assertIn("d2: exact D0", captured.getvalue())

    def test_nonempty_output_and_source_resume_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(directory)
            output = Path(directory) / "outputs"
            arm = output / ARM_NAMES["d1"]
            arm.mkdir(parents=True)
            (arm / "checkpoint.pth").write_text("existing result")
            with self.assertRaisesRegex(ValueError, "Nonempty output"):
                main(["--source-run", str(source), "--output-dir", str(output), "--dry-run"])
            recorded = json.loads((source / "graph/run_config.json").read_text())
            recorded["resume"] = "some_checkpoint.pth"
            (source / "graph/run_config.json").write_text(json.dumps(recorded))
            with self.assertRaisesRegex(ValueError, "fresh Stage 1"):
                load_source(source)

    def test_training_launch_isolated_and_failure_does_not_start_d2(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_source(directory)
            output = Path(directory) / "outputs"
            process = mock.MagicMock()
            process.__enter__.return_value = process
            process.stdout = iter(["Received Signals.SIGHUP\n"])
            process.wait.return_value = 1
            module = "tools.owod.run_stage1_diagnostics"
            with mock.patch(module + ".training_environment", return_value={}), \
                    mock.patch(module + ".subprocess.Popen", return_value=process) as launch, \
                    contextlib.redirect_stdout(io.StringIO()):
                code = main(["--source-run", str(source), "--output-dir", str(output)])
            self.assertEqual(code, 1)
            launch.assert_called_once()
            self.assertEqual(launch.call_args.kwargs["start_new_session"], os.name == "posix")
            self.assertFalse((output / ARM_NAMES["d2"]).exists())
            log = (output / ARM_NAMES["d1"] / "graph/train.log").read_text()
            self.assertIn("Received Signals.SIGHUP", log)
            self.assertIn("diagnostic exit 1", log)


if __name__ == "__main__":
    unittest.main()
