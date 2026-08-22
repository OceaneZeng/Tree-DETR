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
    old_classes = [int(x) for x in risk_payload["old_classes"]]
    risks = np.asarray(risk_payload["risk"], dtype=float)
    rows = []
    for category_id, risk in zip(old_classes, risks):
        before = base.get(category_id, float("nan"))
        current = after.get(category_id, float("nan"))
        rows.append({"category_id": category_id, "risk": float(risk),
                     "base_ap50": before, "after_ap50": current,
                     "drop_ap50": before - current})
    rows = [row for row in rows if np.isfinite(row["drop_ap50"]) and np.isfinite(row["risk"])]
    risk_values = np.asarray([row["risk"] for row in rows])
    drop_values = np.asarray([row["drop_ap50"] for row in rows])
    base_values = np.asarray([row["base_ap50"] for row in rows])
    rho, p_value = spearmanr(risk_values, drop_values)
    base_rho, base_p = spearmanr(risk_values, base_values)
    top_k = min(args.top_k, len(rows))
    high_risk = {row["category_id"] for row in sorted(rows, key=lambda x: x["risk"], reverse=True)[:top_k]}
    high_drop = {row["category_id"] for row in sorted(rows, key=lambda x: x["drop_ap50"], reverse=True)[:top_k]}
    payload = {"schema_version": 1, "count": len(rows), "spearman_risk_drop": float(rho),
               "spearman_p_value": float(p_value), "spearman_risk_base_ap50": float(base_rho),
               "spearman_risk_base_p_value": float(base_p), "top_k": top_k,
               "top_k_harm_coverage": len(high_risk & high_drop) / max(1, top_k),
               "rows": sorted(rows, key=lambda x: x["risk"], reverse=True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in payload if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
