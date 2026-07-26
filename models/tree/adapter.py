# ------------------------------------------------------------------------
# Tree-DETR : Module C - sibling-local adapter insertion (the incremental mech)
# ------------------------------------------------------------------------
# When an unknown that halted at node n gets its label revealed, insert a new
# child adapter under n.  Its discriminative burden is ONLY against its
# siblings; every other branch is untouched, so forgetting protection is local
# by construction, not a global freeze.  The halt site *is* the insertion site.
#
# Equations:
#   C1  h' = h + s_a * W_up * GELU(W_down * LN(h)),  W_up zero-init, s_a = 1
#   C2  r_k = clip(round(r0 (1 + sum_{c in sib(k)} A(k,c))), 4, 64)
#   C3  trainable set at insertion: {W_down^k, W_up^k, mu_k, theta_k, theta_c(sib),
#       tau_n, det-head rows for k};  everything else frozen (incl. mu_n)
#   C4  old siblings may yield territory but never move: mu_c fixed,
#       theta_c = theta_c_old * sigmoid(b_c)  (shrink-only)
#   C5  locality prediction: |Delta mAP(n')| <= 0.5 for off-branch subtrees
#
# Deliberately NOT LoRA on Q/K/V (LEA's worst PEFT form): the ancestors'
# attention *is* the discrimination being reused, so mutating it is
# self-defeating in a cascade.  Parallel FFN bottleneck, base path preserved.
# ------------------------------------------------------------------------
from typing import Dict, List, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TreeConfig, DEFAULT
from .geometry import stable_angle


# ========================================================================
# C1 : the parallel FFN bottleneck adapter
# ========================================================================
class ParallelFFNAdapter(nn.Module):
    """h' = h + s_a * W_up * GELU(W_down * LN(h))   (Eq C1).

    ``W_up`` is zero-initialised and s_a = 1, so at insertion the adapter is
    *exactly* identity and cannot perturb any existing branch before training
    begins.  Base path (the argument h) is always preserved by the residual.
    """

    def __init__(self, dim: int, rank: int, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.rank = rank
        self.ln = nn.LayerNorm(dim)
        self.w_down = nn.Linear(dim, rank, bias=False)
        self.w_up = nn.Linear(rank, dim, bias=False)
        self.s_a = cfg.s_a
        nn.init.kaiming_uniform_(self.w_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.w_up.weight)          # identity at step 0

    def delta(self, h: torch.Tensor) -> torch.Tensor:
        """The additive correction s_a * W_up(GELU(W_down(LN(h)))) (== 0 at init)."""
        return self.s_a * self.w_up(F.gelu(self.w_down(self.ln(h))))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.delta(h)


# ========================================================================
# C2 : width from LOCAL confusability, not total class count
# ========================================================================
def adapter_width(sibling_affinities: List[float], cfg: TreeConfig = DEFAULT) -> int:
    """r_k = clip(round(r0 (1 + sum_{c in sib(k)} A(k,c))), 4, 64)   (Eq C2).

    ``sibling_affinities`` = [A(k, c) for c in sib(k)].  Total adapter params
    thus grow with *local* confusability, not with the global class count.
    """
    s = float(sum(sibling_affinities))
    r = round(cfg.r0 * (1.0 + s))
    return int(min(max(r, cfg.r_min), cfg.r_max))


def bootstrap_affinity_from_cones(mean_z_k: torch.Tensor,
                                  sibling_mu: torch.Tensor,
                                  cfg: TreeConfig = DEFAULT) -> torch.Tensor:
    """Chicken-and-egg bootstrap (Fixed design C): before class k is learned,
    estimate A(k,c) ~ exp(-angle(zbar_k, mu_c)) from the discovered unknowns'
    mean cone embedding.  Returns (num_siblings,) affinities in (0,1]."""
    ang = stable_angle(mean_z_k.unsqueeze(0), sibling_mu, cfg.eps)   # (k,)
    return torch.exp(-ang)


# ========================================================================
# C4 : shrink-only sibling radius reparameterisation
# ========================================================================
class SiblingRadiusReparam(nn.Module):
    """theta_c = theta_c_old * sigmoid(b_c), b_c learnable  (Eq C4).

    A sibling can only *give up* space (sigmoid in (0,1)), and the total it can
    give up is bounded by theta_c_old.  This makes "the new region is carved
    out of the annulus, not out of the siblings" a hard constraint: axes stay
    fixed (mu_c frozen), radii can only shrink.
    """

    def __init__(self, theta_old: torch.Tensor):
        super().__init__()
        self.register_buffer("theta_old", theta_old.detach().clone())
        # b init large-positive so sigmoid ~ 1  ->  theta starts ~ theta_old.
        self.b = nn.Parameter(torch.full_like(theta_old, 4.0))

    def forward(self) -> torch.Tensor:
        return self.theta_old * torch.sigmoid(self.b)


# ========================================================================
# The adapter bank : one adapter per inserted class, keyed by class id
# ========================================================================
class AdapterBank(nn.Module):
    """Holds the sibling-local adapters and applies them to decoder queries.

    Because every adapter is zero-initialised (identity at insertion), the
    default additive application  h + sum_k delta_k(h)  is safe: only trained
    adapters contribute.  For strict sibling-local *routing*, ``apply`` accepts
    an explicit ``active`` class list so only the adapters on the descended
    branch fire.
    """

    def __init__(self, dim: int, cfg: TreeConfig = DEFAULT):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.adapters = nn.ModuleDict()             # str(class_id) -> ParallelFFNAdapter

    def insert(self, cls: int, rank: int) -> ParallelFFNAdapter:
        ad = ParallelFFNAdapter(self.dim, rank, self.cfg)
        self.adapters[str(int(cls))] = ad
        return ad

    def get(self, cls: int) -> Optional[ParallelFFNAdapter]:
        key = str(int(cls))
        # nn.ModuleDict has no .get(); use explicit membership.
        return self.adapters[key] if key in self.adapters else None

    def apply(self, h: torch.Tensor, active: Optional[List[int]] = None) -> torch.Tensor:
        """Return h + sum of the (active) adapters' deltas.  ``active=None``
        applies all inserted adapters (safe under zero-init)."""
        keys = [str(int(c)) for c in active] if active is not None else list(self.adapters.keys())
        out = h
        for k in keys:
            if k in self.adapters:
                out = out + self.adapters[k].delta(h)
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ========================================================================
# Attaching the bank to the last L decoder layers
# ========================================================================
class AdaptedDecoderFFN(nn.Module):
    """Wrap a decoder layer's forward_ffn so the adapter bank is applied after
    the base FFN, preserving the base path.  Non-invasive: the original layer's
    forward_ffn is retained and called unchanged."""

    def __init__(self, base_forward_ffn, bank: AdapterBank):
        super().__init__()
        self.base_forward_ffn = base_forward_ffn
        self.bank = bank

    def forward(self, tgt, active: Optional[List[int]] = None):
        tgt = self.base_forward_ffn(tgt)
        return self.bank.apply(tgt, active)


def attach_adapters(decoder, bank: AdapterBank, cfg: TreeConfig = DEFAULT) -> List[int]:
    """Wrap the last L = cfg.adapter_layers decoder layers with the adapter
    bank (cheapest placement that still reaches the query features the cones
    read).  Returns the indices of the wrapped layers.

    ``decoder`` is a DeformableTransformerDecoder (has ``.layers``).  Each
    wrapped layer's ``forward_ffn`` is replaced by an AdaptedDecoderFFN bound to
    the same bank, so all adapters share one bank across the L layers.
    """
    layers = decoder.layers
    n = len(layers)
    wrapped = list(range(max(0, n - cfg.adapter_layers), n))
    for i in wrapped:
        layer = layers[i]
        if not isinstance(layer.forward_ffn, AdaptedDecoderFFN):
            layer.forward_ffn = AdaptedDecoderFFN(layer.forward_ffn, bank)
    return wrapped


# ========================================================================
# C3 : trainable / frozen partition at insertion
# ========================================================================
def insertion_param_groups(tree, node: int, cls: int) -> Dict[str, object]:
    """Return the specification of Eq C3 for inserting ``cls`` under ``node``.

    Trainable rows/objects:
      - the new class's adapter (W_down^k, W_up^k)
      - mu_k, theta_k                (the new leaf's cone; only row k of mu)
      - theta_c for c in C(node)      (siblings' radii - shrink only, via C4)
      - tau_node                      (this node's temperature)
      - det-head rows for k           (handled by the detector head, row k)
    Everything else - backbone, encoder, all other adapters, mu_node, and all
    cones outside C(node) u {node} - is frozen.
    """
    leaf_k = tree.class_leaf[cls]
    children_n = list(tree.children.get(node, []))
    return {
        "adapter_classes": [cls],
        "mu_trainable_rows": [leaf_k],                       # mu_k only (mu_n frozen)
        "theta_trainable_rows": sorted(set(children_n + [leaf_k])),  # theta_k + siblings
        "tau_trainable_rows": [node],
        "det_head_rows": [cls],
        "frozen_note": "backbone/encoder/other-adapters/mu_node frozen",
    }


def _row_mask_hook(mask: torch.Tensor):
    def hook(grad):
        return grad * mask
    return hook


def configure_insertion(spec: Dict[str, object], adapter_bank: AdapterBank,
                        cones, radius, cascade,
                        extra_frozen: Optional[List[nn.Module]] = None) -> List:
    """Apply the Eq C3 partition: freeze everything on the tree head, then
    unfreeze exactly the objects in ``spec`` and install per-row gradient masks
    so only the permitted rows of the shared cone/radius/tau parameters learn.

    Returns the list of registered hook handles (remove them before the next
    insertion).  ``mu_node`` stays frozen because its row is absent from the
    mu mask.
    """
    handles = []

    # 1) freeze all tree-head params
    for m in [adapter_bank, cones, radius, cascade] + list(extra_frozen or []):
        for p in m.parameters():
            p.requires_grad_(False)

    # 2) unfreeze the new class's adapter(s)
    for cls in spec["adapter_classes"]:
        ad = adapter_bank.get(cls)
        if ad is not None:
            for p in ad.parameters():
                p.requires_grad_(True)

    # 3) mu: only row(s) mu_k trainable  (mu_n stays frozen)
    N, m = cones.mu_raw.shape
    mu_mask = torch.zeros(N, 1, device=cones.mu_raw.device)
    for r in spec["mu_trainable_rows"]:
        mu_mask[r] = 1.0
    cones.mu_raw.requires_grad_(True)
    handles.append(cones.mu_raw.register_hook(_row_mask_hook(mu_mask)))

    # 4) theta base (cones.a): rows theta_k + siblings trainable
    a_mask = torch.zeros(N, device=cones.a.device)
    for r in spec["theta_trainable_rows"]:
        a_mask[r] = 1.0
    cones.a.requires_grad_(True)
    handles.append(cones.a.register_hook(_row_mask_hook(a_mask)))

    # 5) scale weights w: same rows as theta (radii absorb scale per node)
    w_mask = torch.zeros(N, 1, device=radius.w.device)
    for r in spec["theta_trainable_rows"]:
        w_mask[r] = 1.0
    radius.w.requires_grad_(True)
    handles.append(radius.w.register_hook(_row_mask_hook(w_mask)))

    # 6) tau_node trainable
    tau_mask = torch.zeros_like(cascade.tau)
    for r in spec["tau_trainable_rows"]:
        tau_mask[r] = 1.0
    cascade.tau.requires_grad_(True)
    handles.append(cascade.tau.register_hook(_row_mask_hook(tau_mask)))

    return handles
