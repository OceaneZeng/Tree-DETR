"""Shared rank-0 console logging for reproducible experiments."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "log"


class TeeStream:
    """Mirror a console stream into a line-buffered experiment log."""

    def __init__(self, stream, log_handle):
        self.stream = stream
        self.log_handle = log_handle

    def write(self, value):
        self.stream.write(value)
        self.log_handle.write(value)
        self.flush()

    def flush(self):
        self.stream.flush()
        self.log_handle.flush()

    def isatty(self):
        return self.stream.isatty()

    def fileno(self):
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def experiment_log_path(output_dir, filename: str) -> Path:
    """Return a central log path mirroring the experiment output directory."""
    output_path = Path(output_dir).resolve()
    try:
        relative_output = output_path.relative_to(PROJECT_ROOT)
    except ValueError:
        relative_output = Path(output_path.name)
    return LOG_ROOT / relative_output / filename


def start_file_logging(args, is_main_process: bool = True):
    """Capture stdout/stderr to the root ``log`` directory and keep the console."""
    if not getattr(args, "output_dir", "") or not is_main_process:
        return None
    if getattr(args, "no_file_log", False):
        return None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_path = getattr(args, "log_file", "")
    log_path = (Path(requested_path) if Path(requested_path).is_absolute()
                else LOG_ROOT / Path(requested_path)) if requested_path else \
        experiment_log_path(output_dir, "train.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    handle.write(f"\n===== experiment start {timestamp} =====\n")
    handle.flush()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, handle)
    sys.stderr = TeeStream(original_stderr, handle)
    print(f"File log: {log_path}")
    return original_stdout, original_stderr, handle


def stop_file_logging(state):
    if state is None:
        return
    original_stdout, original_stderr, handle = state
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    handle.write("===== experiment end =====\n")
    handle.close()
