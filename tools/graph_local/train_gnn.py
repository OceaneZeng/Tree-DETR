#!/usr/bin/env python
"""Train a class-interference GNN from prior graph-local stage artifacts.

Each stage artifact is produced by ``run_increment.py`` as ``gnn_stage.pt``.
Only measured source rows are marked valid, so an unprobed class is not
silently converted into a zero-harm training label.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.graph_local.gnn import (  # noqa: E402
    ClassInterferenceGNN,
    fit_interference_gnn,
    save_gnn_checkpoint,
)
from util.experiment_log import start_file_logging, stop_file_logging


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_stage(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a stage artifact mapping")
    for key in ("features", "harm"):
        if key not in payload or not torch.is_tensor(payload[key]):
            raise ValueError(f"{path} is missing tensor field '{key}'")
    features = payload["features"].float()
    harm = payload["harm"].float()
    if features.ndim != 2 or harm.ndim != 2 or harm.shape[0] != harm.shape[1]:
        raise ValueError(f"{path} has invalid feature/harm shapes")
    if features.shape[0] != harm.shape[0]:
        raise ValueError(f"{path} class count differs between features and harm")
    valid_mask = payload.get("valid_mask")
    if valid_mask is None:
        valid_mask = torch.ones_like(harm, dtype=torch.bool)
    if not torch.is_tensor(valid_mask) or valid_mask.shape != harm.shape:
        raise ValueError(f"{path} has invalid valid_mask")
    return {"features": features, "harm": harm, "valid_mask": valid_mask.bool()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", type=Path, required=True,
                        help="one or more prior gnn_stage.pt files")
    parser.add_argument("--output", type=Path, required=True,
                        help="output ClassInterferenceGNN checkpoint")
    parser.add_argument("--log-file", type=Path, default=None,
                        help="human-readable log path; defaults beside the checkpoint")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-steps", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir = args.output.parent
    args.no_file_log = False
    args.log_file = str(args.log_file) if args.log_file is not None else ""
    log_state = start_file_logging(args, is_main_process=True)
    try:
        return _run(args, parser)
    finally:
        stop_file_logging(log_state)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    set_seed(args.seed)

    stages = [load_stage(path) for path in args.stages]
    input_dim = int(stages[0]["features"].shape[1])
    if any(int(stage["features"].shape[1]) != input_dim for stage in stages):
        parser.error("all stage artifacts must use the same feature dimension")
    model = ClassInterferenceGNN(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        message_steps=args.message_steps,
        dropout=args.dropout,
    )
    history = fit_interference_gnn(
        model, stages, epochs=args.epochs, lr=args.lr,
        weight_decay=args.weight_decay, grad_clip=args.grad_clip,
    )
    save_gnn_checkpoint(model, args.output, extra={
        "stage_files": [str(path.resolve()) for path in args.stages],
        "seed": args.seed,
        "epochs": args.epochs,
        "final_loss": history[-1],
    })
    print(json.dumps({
        "output": str(args.output.resolve()),
        "stages": len(stages),
        "input_dim": input_dim,
        "final_loss": history[-1],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
