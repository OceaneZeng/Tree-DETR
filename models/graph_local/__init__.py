"""Graph-local continual adaptation for Deformable DETR."""

from .interference import (
    build_conflict_matrix,
    build_off_neighborhood_basis,
    evaluate_neighborhood,
    select_positive_neighbors,
)
from .lora import (
    LoRALinear,
    expand_classification_head,
    freeze_for_increment,
    inject_decoder_lora,
    lora_delta_vector,
    merge_decoder_lora,
)
from .losses import local_margin_loss, projection_loss, weighted_detection_loss
from .pseudo_labels import complete_targets_with_teacher
from .replay import build_increment_annotation, select_replay_images

__all__ = [
    "LoRALinear",
    "build_conflict_matrix",
    "build_increment_annotation",
    "build_off_neighborhood_basis",
    "complete_targets_with_teacher",
    "evaluate_neighborhood",
    "expand_classification_head",
    "freeze_for_increment",
    "inject_decoder_lora",
    "local_margin_loss",
    "lora_delta_vector",
    "merge_decoder_lora",
    "projection_loss",
    "select_positive_neighbors",
    "select_replay_images",
    "weighted_detection_loss",
]
