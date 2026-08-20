# ------------------------------------------------------------------------
# Tree-DETR : thin, opt-in integration with Deformable-DETR
# ------------------------------------------------------------------------
# Nothing in this file imports the compiled MSDeformAttn ops at module load, so
# the tree package stays importable/testable in a CPU/no-ops environment.  The
# base detector and its SetCriterion are pulled in lazily inside build_tree().
#
# Two integration points:
#   TreeHead      - bundles cone/objectness heads, cones, radius, cascade,
#                   adapter bank, and the confusability tree; captures the last
#                   decoder-layer query features via a forward hook.
#   TreeCriterion - wraps a base SetCriterion; adds the Module-E reserved-gap
#                   losses (on matched-query cone embeddings) and the depth-0
#                   objectness loss, keeping the base loss dict intact.
# ------------------------------------------------------------------------
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TreeConfig, DEFAULT
from .cone_head import ConeEmbedHead, ObjectnessHead
from .losses import ConeField, ReservedGapLoss
from .calibration import ScaleConditionedRadius, band as band_of, M_BAND
from .cascade import Cascade
from .adapter import AdapterBank, attach_adapters
from .tree_structure import ConfusabilityTree


class TreeHead(nn.Module):
    """All tree-side learnable state for one detector.

    Given the last-decoder-layer query feature h it produces the cone embedding
    z (A1), the objectness probability (B1), and can run the cascade (Module B).
    """

    def __init__(self, tree: ConfusabilityTree, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.tree = tree
        self.topo = tree.topology()
        self.cone_head = ConeEmbedHead(cfg)
        self.obj_head = ObjectnessHead(cfg)
        self.cones = ConeField(self.topo.num_nodes, cfg)
        self.radius = ScaleConditionedRadius(self.topo.num_nodes, cfg)
        self.cascade = Cascade(self.topo, self.cones, self.radius, cfg)
        self.gap_loss = ReservedGapLoss(cfg)
        self.adapters = AdapterBank(cfg.d_model, cfg)
        self._hs_cache: Optional[torch.Tensor] = None   # last transformer hs

    # -- feature capture ----------------------------------------------------
    def register_capture(self, transformer: nn.Module):
        """Install a forward hook that stores the decoder feature stack ``hs``
        ([n_layers, bs, nq, d]) so the criterion can read the last layer without
        changing the detector's forward signature."""
        def hook(_module, _inp, output):
            # DeformableTransformer returns (hs, init_ref, inter_ref, ...)
            self._hs_cache = output[0]
        transformer.register_forward_hook(hook)

    @property
    def last_query_features(self) -> Optional[torch.Tensor]:
        return None if self._hs_cache is None else self._hs_cache[-1]   # (bs, nq, d)

    # -- convenience --------------------------------------------------------
    def embed(self, h: torch.Tensor) -> torch.Tensor:
        return self.cone_head(h)

    @torch.no_grad()
    def infer(self, h: torch.Tensor, box_area: Optional[float] = None):
        """Full per-query inference: cascade descent + detection score (B5)."""
        z = self.cone_head(h)
        res = self.cascade.descend_one(z, box_area)
        obj_p = float(self.obj_head(h.unsqueeze(0)).item())
        # depth-0 objectness corroborates: background if the gate says "not a thing"
        if obj_p < self.cfg.t0:
            res.halt_depth = 0
            res.is_known = False
            res.leaf_class = None
        res.path_score = self.cascade.detection_score(obj_p, res.path_score)
        return res


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


class TreeCriterion(nn.Module):
    """Wrap a base SetCriterion and add the tree losses.

    forward(outputs, targets) returns the base loss dict updated with
    loss_contain/nest/gap/sib (Module E, Eqs E3-E6) and loss_obj (the depth-0
    objectness BCE).  The Module-E weights alpha/beta/eta/nu and the objectness
    weight go into the training weight_dict (see ``tree_weight_dict``).
    """

    def __init__(self, base_criterion: nn.Module, tree_head: TreeHead,
                 cfg: TreeConfig = DEFAULT, obj_loss_coef: float = 1.0,
                 epoch_getter=None):
        super().__init__()
        self.base = base_criterion
        self.head = tree_head
        self.cfg = cfg
        self.obj_loss_coef = obj_loss_coef
        self.matcher = base_criterion.matcher
        # engine.py reads criterion.weight_dict directly.  Keep the wrapper
        # on the exact same dictionary that the base criterion uses.
        self.weight_dict = base_criterion.weight_dict
        self.epoch_getter = epoch_getter        # callable -> current epoch (for warmup)

    def forward(self, outputs: Dict, targets: List[Dict]) -> Dict[str, torch.Tensor]:
        losses = self.base(outputs, targets)

        hs_last = self.head.last_query_features
        if hs_last is None:
            return losses                        # capture not wired; base only

        indices = self.matcher({k: v for k, v in outputs.items()
                                if k not in ("aux_outputs", "enc_outputs")}, targets)
        bidx, sidx = _get_src_permutation_idx(indices)
        h_matched = hs_last[bidx, sidx]                                   # (M, d)
        y = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])  # (M,)

        # ----- Module E : reserved-gap losses on matched cone embeddings -----
        z = self.head.cone_head(h_matched)
        epoch = self.epoch_getter() if self.epoch_getter is not None else None
        tree_terms = self.head.gap_loss(self.head.cones, self.head.topo, z, y, epoch)
        losses.update(tree_terms)

        # ----- depth-0 objectness BCE : matched = thing, unmatched = bg -------
        bs, nq, d = hs_last.shape
        obj_logits = self.head.obj_head.logit(hs_last.reshape(bs * nq, d))   # (bs*nq,)
        tgt = torch.zeros(bs * nq, device=obj_logits.device)
        flat_matched = (bidx * nq + sidx).to(obj_logits.device)
        tgt[flat_matched] = 1.0
        losses["loss_obj"] = F.binary_cross_entropy_with_logits(obj_logits, tgt)
        return losses

    def tree_weight_dict(self) -> Dict[str, float]:
        w = self.head.gap_loss.weight_dict()
        w["loss_obj"] = self.obj_loss_coef
        return w


def attach_tree_head(model: nn.Module, tree: ConfusabilityTree,
                     cfg: TreeConfig = DEFAULT, with_adapters: bool = True) -> TreeHead:
    """Attach a TreeHead to a live DeformableDETR instance (opt-in).

    Wires the decoder-feature capture hook and, if requested, wraps the last
    L=2 decoder FFNs with the (initially empty, hence identity) adapter bank.
    The base model's forward and outputs are unchanged.
    """
    head = TreeHead(tree, cfg)
    model.tree_head = head
    head.register_capture(model.transformer)
    if with_adapters:
        attach_adapters(model.transformer.decoder, head.adapters, cfg)
    return head


def build_tree(args, tree: Optional[ConfusabilityTree] = None,
               cfg: Optional[TreeConfig] = None, epoch_getter=None):
    """Factory: build the vanilla Deformable-DETR, then attach the tree head.

    Returns (model, tree_criterion, postprocessors, tree_head).  If ``tree`` is
    None a trivial flat 2-level tree over all classes is used (the EE-0 setting)
    - replace it with an induced ConfusabilityTree once Stage-0 has run.
    """
    from models.deformable_detr import build as build_base   # lazy: pulls in ops
    cfg = cfg or DEFAULT
    model, base_criterion, postprocessors = build_base(args)

    if tree is None:
        num_classes = model.class_embed[0].out_features if hasattr(model.class_embed, "__getitem__") \
            else model.class_embed.out_features
        tree = flat_two_level_tree(num_classes)

    head = attach_tree_head(model, tree, cfg)
    criterion = TreeCriterion(base_criterion, head, cfg, epoch_getter=epoch_getter)

    # fold tree-loss weights into the criterion's weight_dict so the training
    # loop scales them like any other DETR loss.
    if hasattr(base_criterion, "weight_dict"):
        base_criterion.weight_dict.update(criterion.tree_weight_dict())
    return model, criterion, postprocessors, head


def flat_two_level_tree(num_classes: int) -> ConfusabilityTree:
    """A root with all classes as direct children (depth-1 leaves).  The minimal
    tree on which L_gap is meaningful (arm EE-0), and a safe default before the
    confusability tree is induced."""
    children = {0: list(range(1, num_classes + 1))}
    leaf_class = {}
    for c in range(num_classes):
        node = c + 1
        children[node] = []
        leaf_class[node] = c
    return ConfusabilityTree(children, leaf_class, root=0)
