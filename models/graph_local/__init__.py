"""Graph-local continual adaptation for Deformable DETR."""

from .interference import (
    build_conflict_matrix,
    build_off_neighborhood_basis,
    evaluate_neighborhood,
    select_positive_neighbors,
)
from .gnn import (
    ClassInterferenceGNN,
    compress_gradient_sketches,
    fit_interference_gnn,
    harm_prediction_loss,
    load_gnn_checkpoint,
    save_gnn_checkpoint,
    select_gnn_neighbors,
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
    "ClassInterferenceGNN",
    "build_conflict_matrix",
    "build_increment_annotation",
    "build_off_neighborhood_basis",
    "complete_targets_with_teacher",
    "compress_gradient_sketches",
    "evaluate_neighborhood",
    "expand_classification_head",
    "freeze_for_increment",
    "fit_interference_gnn",
    "harm_prediction_loss",
    "inject_decoder_lora",
    "local_margin_loss",
    "load_gnn_checkpoint",
    "lora_delta_vector",
    "merge_decoder_lora",
    "projection_loss",
    "select_positive_neighbors",
    "select_gnn_neighbors",
    "select_replay_images",
    "save_gnn_checkpoint",
    "weighted_detection_loss",
]
