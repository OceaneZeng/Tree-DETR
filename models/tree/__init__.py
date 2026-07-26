# ------------------------------------------------------------------------
# Tree-DETR : recognition cascade with sibling-local adapters
# ------------------------------------------------------------------------
# A self-contained implementation of the method described in the note
#   recognition-cascade-with-sibling-local-adapters.md
#
# The hierarchy is used as a *parameter-allocation* structure, not an output
# vocabulary.  Modules (one file each):
#   config.py         - TreeConfig, every fixed hyperparameter
#   geometry.py       - cosine-cone primitives (angles, gaps)             [E1/E2]
#   losses.py         - reserved-gap loss (Module E)                      [E3-E8]
#   cone_head.py      - cone embedding + objectness heads (A1, B1)
#   tree_structure.py - confusability tree: induction & insertion (Module A)
#   cascade.py        - beam descent, halt depth, path score (Module B)
#   calibration.py    - scale radius, temperature, invariance (Module D)
#   adapter.py        - sibling-local parallel adapters (Module C)
#   tree_detr.py      - thin, opt-in integration with DeformableDETR
# ------------------------------------------------------------------------
from .config import TreeConfig, DEFAULT
from .topology import TreeTopology
from .geometry import (
    project_to_sphere, stable_angle, angle_arccos, cone_contains,
    angular_margin_ratio, cone_gap, theta_from_logit,
)
from .cone_head import ConeEmbedHead, ObjectnessHead
from .losses import ConeField, ReservedGapLoss, node_gaps
from .tree_structure import (
    ConfusabilityTree, confusion_to_rates, build_affinity, induce_tree,
    insert_class, offdiagonal_confusion_mass,
)
from .cascade import Cascade, HaltResult
from .calibration import (
    ScaleConditionedRadius, band, fit_node_temperature, recalibrate_tau0,
    threshold_transfer_gap, ece_per_depth, S_BAND, M_BAND, L_BAND,
)
from .adapter import (
    ParallelFFNAdapter, AdapterBank, adapter_width, SiblingRadiusReparam,
    attach_adapters, insertion_param_groups, configure_insertion,
)
from .tree_detr import (
    TreeHead, TreeCriterion, attach_tree_head, build_tree, flat_two_level_tree,
)

__all__ = [
    "TreeConfig", "DEFAULT", "TreeTopology",
    "project_to_sphere", "stable_angle", "angle_arccos", "cone_contains",
    "angular_margin_ratio", "cone_gap", "theta_from_logit",
    "ConeEmbedHead", "ObjectnessHead",
    "ConeField", "ReservedGapLoss", "node_gaps",
    "ConfusabilityTree", "confusion_to_rates", "build_affinity", "induce_tree",
    "insert_class", "offdiagonal_confusion_mass",
    "Cascade", "HaltResult",
    "ScaleConditionedRadius", "band", "fit_node_temperature", "recalibrate_tau0",
    "threshold_transfer_gap", "ece_per_depth", "S_BAND", "M_BAND", "L_BAND",
    "ParallelFFNAdapter", "AdapterBank", "adapter_width", "SiblingRadiusReparam",
    "attach_adapters", "insertion_param_groups", "configure_insertion",
    "TreeHead", "TreeCriterion", "attach_tree_head", "build_tree",
    "flat_two_level_tree",
]
