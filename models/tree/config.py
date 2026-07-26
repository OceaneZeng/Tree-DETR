# ------------------------------------------------------------------------
# Tree-DETR : Recognition cascade with sibling-local adapters
# Central configuration - single source of truth for every fixed value.
# ------------------------------------------------------------------------
# Every hyperparameter below is quoted verbatim from the "Fixed design"
# blocks of the note
#   recognition-cascade-with-sibling-local-adapters.md
# The equation / table reference is given next to each field so the code
# can be checked against the spec without opening the note.
# ------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Tuple
import math


@dataclass
class TreeConfig:
    """All fixed constants for Modules A-E in one place.

    Grouped by module; the note reference is in the comment.  Nothing here is
    learnable - learnable quantities (mu_n, theta_n, tau_n, adapter weights,
    ...) live inside the nn.Modules and are *initialised* from these values.
    """

    # ----- global geometry (notation table) ------------------------------
    m: int = 64                 # cone-embedding dim, z in S^{m-1}
    d_model: int = 256          # detector hidden dim, h in R^256
    eps: float = 1e-6           # arccos clamp epsilon

    # ----- Module A : confusability tree ---------------------------------
    delta_h: float = 0.05       # (A4) binary->n-ary collapse fraction of dendrogram height
    max_children: int = 6       # (A4) repair: shrink delta_h if a node exceeds this
    d_max_20: int = 4           # (A4) max depth for 20 classes (excl. depth-0 root)
    d_max_80: int = 5           # (A4)                 for 80 classes
    repair_iters: int = 5       # (A4) at most 5 repair iterations
    affinity_min_area: float = 32.0 ** 2   # (A2) induce on medium+large boxes only
    insert_aff_margin: float = 0.10        # (A5) override n_vote by n_aff only if margin exceeds this
    dag_mass_threshold: float = 0.40       # (F1b) above this a DAG is required

    # ----- Module B : cascade --------------------------------------------
    beam: int = 2               # (B3) beam width B
    t0: float = 0.5             # (B1) objectness decision threshold on sigma(z_obj)
    gate_threshold: float = 1.0 # (B2) the ratio r_c <= 1 gate; one threshold everywhere

    # ----- Module C : sibling-local adapters -----------------------------
    adapter_layers: int = 2     # (C1) inserted at the last L=2 decoder layers
    r0: int = 8                 # (C2) base rank
    r_min: int = 4              # (C2) clip lower bound
    r_max: int = 64             # (C2) clip upper bound
    s_a: float = 1.0            # (C1) adapter scale (identity at init with zero W_up)
    warmup_epochs_adapter: int = 1  # (C2) re-size affinity after a 1-epoch warm-up

    # ----- Module D : calibration / invariance ---------------------------
    # COCO area thresholds delimiting the scale bands {S, M, L} (notation table)
    area_small: float = 32.0 ** 2      # area < 32^2  -> S
    area_large: float = 96.0 ** 2      # area >= 96^2 -> L ; between -> M
    ece_bins: int = 15                 # (D4) ECE_d uses 15 bins

    # ----- Module E : reserved-gap loss ----------------------------------
    # Fixed hyperparameter table at the end of "Fixed design E".
    alpha: float = 1.0          # L_contain weight (E7)
    beta: float = 1.0           # L_nest weight   (E7)
    eta: float = 0.5            # L_gap weight    (E7)
    nu: float = 1.0             # L_sib weight    (E7)
    lam: float = 0.10           # (E6) sibling separation margin lambda, in radians
    rho: float = 0.15           # (E8) open-space budget ratio gamma_n = rho * theta_n
    warm_epochs: int = 5        # (schedule) enable L_gap & L_sib only after E_warm epochs

    # ----- objectness gate norm scale (B1) -------------------------------
    tau0_init: float = 1.0      # (B1) initial magnitude scale; recalibrated per domain (D3)

    def d_max(self, num_classes: int) -> int:
        """(A4) max tree depth as a function of class count."""
        return self.d_max_20 if num_classes <= 20 else self.d_max_80

    @property
    def half_pi(self) -> float:
        return math.pi / 2.0


# A module-level default instance for convenience; callers that need to sweep
# hyperparameters (e.g. the rho sweep EE-4) should construct their own.
DEFAULT = TreeConfig()
