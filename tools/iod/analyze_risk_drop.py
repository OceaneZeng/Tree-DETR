#!/usr/bin/env python3
"""Compare M1 risk with observed old-class AP50 drop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def load_ap50(path: Path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    precision = np.asarray(value["precision"])
    cat_ids = [int(x) for x in value["params"].catIds]
    result = {}
    for index, category_id in enumerate(cat_ids):
        values = precision[0, :, index, 0, 2]
        values = values[values >= 0]
        result[category_id] = float(values.mean()) if len(values) else float("nan")
    return result


def safe_spearman(left, right):
    if len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return None, None
    rho, p_value = spearmanr(left, right)
    return float(rho), float(p_value)


def analyze(base, after, risk_payload, top_k=10):
    old_classes = [int(x) for x in risk_payload["old_classes"]]
    risks = np.asarray(risk_payload["risk"], dtype=float)
    if len(old_classes) != len(risks):
        raise ValueError("risk JSON old_classes and risk have different lengths")
    rows = []
    for category_id, risk in zip(old_classes, risks):
        before = base.get(category_id, float("nan"))
        current = after.get(category_id, float("nan"))
        rows.append({"category_id": category_id, "risk": float(risk),
                     "base_ap50": before, "after_ap50": current,
                     "drop_ap50": before - current})
    rows = [row for row in rows if np.isfinite(row["drop_ap50"]) and np.isfinite(row["risk"])]
    if not rows:
        raise ValueError("no finite old-class AP50/risk pairs were found")

    risk_values = np.asarray([row["risk"] for row in rows])
    drop_values = np.asarray([row["drop_ap50"] for row in rows])
    base_values = np.asarray([row["base_ap50"] for row in rows])
    after_values = np.asarray([row["after_ap50"] for row in rows])
    rho, p_value = safe_spearman(risk_values, drop_values)
    base_rho, base_p = safe_spearman(risk_values, base_values)
    top_k = min(top_k, len(rows))
    high_risk = {row["category_id"] for row in sorted(rows, key=lambda x: x["risk"], reverse=True)[:top_k]}
    high_drop = {row["category_id"] for row in sorted(rows, key=lambda x: x["drop_ap50"], reverse=True)[:top_k]}

    zero_after_count = int(np.count_nonzero(np.isclose(after_values, 0.0, atol=1e-12)))
    zero_after_fraction = zero_after_count / len(rows)
    saturated = zero_after_fraction >= 0.8 or float(np.std(after_values)) < 1e-6
    payload = {
        "schema_version": 2,
        "count": len(rows),
        "diagnostic_status": "saturated_forgetting" if saturated else "informative",
        "mean_base_ap50": float(np.mean(base_values)),
        "mean_after_ap50": float(np.mean(after_values)),
        "mean_drop_ap50": float(np.mean(drop_values)),
        "median_drop_ap50": float(np.median(drop_values)),
        "after_ap50_std": float(np.std(after_values)),
        "zero_after_count": zero_after_count,
        "zero_after_fraction": zero_after_fraction,
        "spearman_risk_drop": rho,
        "spearman_p_value": p_value,
        "spearman_risk_base_ap50": base_rho,
        "spearman_risk_base_p_value": base_p,
        "top_k": top_k,
        "top_k_random_expectation": top_k / len(rows),
        "top_k_harm_coverage": len(high_risk & high_drop) / max(1, top_k),
        "rows": sorted(rows, key=lambda x: x["risk"], reverse=True),
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval", type=Path, required=True)
    parser.add_argument("--increment-eval", type=Path, required=True)
    parser.add_argument("--risk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    base = load_ap50(args.base_eval)
    after = load_ap50(args.increment_eval)
    risk_payload = json.loads(args.risk.read_text(encoding="utf-8"))
    payload = analyze(base, after, risk_payload, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
