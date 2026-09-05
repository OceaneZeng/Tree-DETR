"""Graph-local continual adaptation for Deformable DETR."""

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
    freeze_for_class_ids,
    freeze_for_increment,
    inject_decoder_lora,
    lora_delta_vector,
    merge_decoder_lora,
)
from .losses import weighted_detection_loss
from .distillation import old_class_distillation_losses
from .replay import (build_increment_annotation, select_replay_images,
                     select_replay_images_by_quota)

__all__ = [
    "LoRALinear",
    "ClassInterferenceGNN",
    "build_increment_annotation",
    "compress_gradient_sketches",
    "expand_classification_head",
    "freeze_for_class_ids",
    "freeze_for_increment",
    "fit_interference_gnn",
    "harm_prediction_loss",
    "inject_decoder_lora",
    "load_gnn_checkpoint",
    "lora_delta_vector",
    "merge_decoder_lora",
    "old_class_distillation_losses",
    "select_gnn_neighbors",
    "select_replay_images",
    "select_replay_images_by_quota",
    "save_gnn_checkpoint",
    "weighted_detection_loss",
]
