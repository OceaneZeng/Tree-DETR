"""Shared configuration and scoring rules for the main OWOD baselines.

The detector remains a Deformable-DETR implementation, while this module
keeps the OWOD-specific policy explicit.  It is used by both the model and the
runner so that method names cannot silently change the evaluation protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class OWODBaselineConfig:
    name: str
    paper: str
    has_objectness_head: bool
    unknown_score: str
    oracle: bool = False


BASELINES = {
    "vanilla_d_detr": OWODBaselineConfig(
        "vanilla_d_detr", "Vanilla Deformable DETR", False, "one_minus_known"),
    "ore_star": OWODBaselineConfig(
        "ore_star", "ORE*", False, "energy"),
    "ow_detr": OWODBaselineConfig(
        "ow_detr", "OW-DETR", True, "novelty"),
    "prob": OWODBaselineConfig(
        "prob", "PROB", True, "probabilistic_objectness"),
    "oracle": OWODBaselineConfig(
        "oracle", "Oracle", False, "none", oracle=True),
}

ALIASES = {
    "vanilla": "vanilla_d_detr",
    "d-detr": "vanilla_d_detr",
    "d_detr": "vanilla_d_detr",
    "ore": "ore_star",
    "ow-detr": "ow_detr",
}


def normalize_baseline(name: str) -> str:
    key = str(name).strip().lower()
    key = ALIASES.get(key, key)
    if key not in BASELINES:
        choices = ", ".join(sorted(BASELINES))
        raise ValueError(f"unknown OWOD baseline {name!r}; choose one of {choices}")
    return key


def baseline_config(name: str) -> OWODBaselineConfig:
    return BASELINES[normalize_baseline(name)]


def baseline_config_dict(name: str) -> Mapping[str, object]:
    return asdict(baseline_config(name))


def unknown_score(outputs: Mapping[str, torch.Tensor], name: str) -> torch.Tensor:
    """Return a per-query unknown score in [0, 1].

    This is deliberately kept separate from COCO post-processing.  The score
    is exported in evaluation results and can be consumed by OWOD metrics,
    while the normal known-class predictions remain compatible with COCOeval.
    """
    config = baseline_config(name)
    logits = outputs["pred_logits"]
    known_prob = logits.sigmoid().amax(dim=-1)
    if config.unknown_score == "none":
        return torch.zeros_like(known_prob)
    if config.unknown_score == "probabilistic_objectness":
        if "pred_objectness" not in outputs:
            raise KeyError("PROB scoring requires pred_objectness from the model")
        objectness = outputs["pred_objectness"].sigmoid().squeeze(-1)
        return (objectness * (1.0 - known_prob)).clamp(0.0, 1.0)
    if config.unknown_score == "novelty":
        if "pred_objectness" in outputs:
            return (1.0 - outputs["pred_objectness"].sigmoid().squeeze(-1)).clamp(0.0, 1.0)
        return (1.0 - known_prob).clamp(0.0, 1.0)
    if config.unknown_score == "energy":
        energy = -torch.logsumexp(logits, dim=-1)
        return torch.sigmoid(energy)
    if config.unknown_score == "one_minus_known":
        return (1.0 - known_prob).clamp(0.0, 1.0)
    raise RuntimeError(f"unhandled OWOD score rule: {config.unknown_score}")
