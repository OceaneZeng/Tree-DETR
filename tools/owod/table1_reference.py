#!/usr/bin/env python
"""Inspect and validate the external baselines transcribed from DEUS Table 1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE_PATH = Path(__file__).with_name("deus_table1.json")


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def harmonic_score(known_map: float, unknown_recall: float) -> float:
    denominator = known_map + unknown_recall
    return 0.0 if denominator == 0 else 2.0 * known_map * unknown_recall / denominator


def validate_reference(payload: dict, tolerance: float = 0.15) -> None:
    if payload.get("metric_unit") != "percent":
        raise ValueError("DEUS Table 1 values must be stored as percentages")
    for protocol, block in payload["protocols"].items():
        for method, record in block["methods"].items():
            tasks = record["tasks"]
            if [task["task"] for task in tasks] != [1, 2, 3, 4]:
                raise ValueError(f"{protocol}/{method} must contain Tasks 1-4")
            for task in tasks[:3]:
                known = task.get("known_map", task["current_map"])
                expected = harmonic_score(known, task["u_rec"])
                if abs(expected - task["h_score"]) > tolerance:
                    raise ValueError(
                        f"{protocol}/{method}/Task {task['task']} has an inconsistent H-Score")


def markdown_table(payload: dict, protocol: str) -> str:
    methods = payload["protocols"][protocol]["methods"]
    lines = [
        "| Method | T1 Known | T1 U-Rec | T1 H | T2 Known | T2 U-Rec | T2 H | "
        "T3 Known | T3 U-Rec | T3 H | T4 Known |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, record in methods.items():
        tasks = record["tasks"]
        label = name + (" dagger" if record.get("dagger") else "")
        values = [label]
        for task in tasks[:3]:
            values.extend([
                f"{task.get('known_map', task['current_map']):.1f}",
                f"{task['u_rec']:.1f}", f"{task['h_score']:.1f}",
            ])
        values.append(f"{tasks[3]['known_map']:.1f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("m-owodb", "s-owodb"),
                        default="m-owodb")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    payload = load_reference()
    validate_reference(payload)
    if args.json:
        print(json.dumps(payload["protocols"][args.protocol], indent=2))
    else:
        print(markdown_table(payload, args.protocol))


if __name__ == "__main__":
    main()
