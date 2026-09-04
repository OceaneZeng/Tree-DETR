"""Compact, restart-aware experiment logging."""

from __future__ import annotations

import datetime
import sys
from pathlib import Path


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
    """Keep all logs beside the checkpoints they describe."""
    return Path(output_dir).resolve() / filename


def archive_log_file(path: str | Path) -> Path | None:
    """Move a stale log aside instead of appending an unrelated run to it."""
    source = Path(path)
    if not source.is_file():
        return None
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    archive_dir = source.parent / "log_archive" / stamp
    archive_dir.mkdir(parents=True, exist_ok=False)
    destination = archive_dir / source.name
    source.replace(destination)
    return destination


def prepare_log_file(path: str | Path, append: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not append:
        archive_log_file(destination)
    return destination


def start_file_logging(args, is_main_process: bool = True):
    """Capture one run in ``output_dir/train.log``; append only for resume."""
    if not getattr(args, "output_dir", "") or not is_main_process:
        return None
    if getattr(args, "no_file_log", False):
        return None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_path = getattr(args, "log_file", "")
    if requested_path:
        requested = Path(requested_path)
        log_path = requested if requested.is_absolute() else output_dir / requested
    else:
        log_path = experiment_log_path(output_dir, "train.log")
    append = bool(getattr(args, "resume", "") or getattr(args, "append_log", False))
    prepare_log_file(log_path, append=append)
    handle = log_path.open("a" if append else "w", encoding="utf-8", buffering=1)
    timestamp = datetime.datetime.now().isoformat(timespec="seconds")
    event = "resume" if append else "start"
    handle.write(f"\n===== experiment {event} {timestamp} =====\n")
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
