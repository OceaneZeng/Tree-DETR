"""Presentation helpers for OWOD metric tables; evaluation lives in owod_benchmark_eval."""

from __future__ import annotations

from typing import Mapping




def harmonic_score(known_map: float, unknown_recall: float) -> float:
    denominator = known_map + unknown_recall
    return 0.0 if denominator == 0 else 2.0 * known_map * unknown_recall / denominator


def table1_summary(class_ap50: Mapping[str, float], owod_metrics: Mapping[str, float]
                   ) -> dict[str, float]:
    """Format the DEUS Table 1 columns in percentage points."""
    result = {
        f"{name} mAP": 100.0 * float(value)
        for name, value in class_ap50.items()
        if name in {"Previous", "Current", "Known"}
    }
    if float(owod_metrics.get("unknown_gt", 0.0)) > 0:
        unknown_recall = float(owod_metrics["U-Recall"])
        known_map = float(class_ap50.get("Known", class_ap50.get("Current", 0.0)))
        result["U-Rec"] = 100.0 * unknown_recall
        result["H-Score"] = 100.0 * harmonic_score(known_map, unknown_recall)
    return result
