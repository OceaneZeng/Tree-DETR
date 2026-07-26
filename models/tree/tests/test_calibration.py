"""Module D — scale bands, theta_c(s) range, temperature reduces NLL, metrics."""
import math
import numpy as np
import torch
import torch.nn.functional as F

from models.tree.config import TreeConfig
from models.tree.calibration import (
    band, S_BAND, M_BAND, L_BAND, ScaleConditionedRadius,
    fit_node_temperature, recalibrate_tau0, best_f1_threshold,
    threshold_transfer_gap, ece_per_depth,
)
from models.tree.cone_head import ObjectnessHead


def test_band_thresholds_scalar():
    cfg = TreeConfig()
    assert band(10 * 10, cfg) == S_BAND        # < 32^2
    assert band(50 * 50, cfg) == M_BAND        # between
    assert band(200 * 200, cfg) == L_BAND      # >= 96^2


def test_band_thresholds_tensor():
    cfg = TreeConfig()
    areas = torch.tensor([10.0 * 10, 50.0 * 50, 200.0 * 200])
    b = band(areas, cfg)
    assert b.tolist() == [S_BAND, M_BAND, L_BAND]


def test_scale_radius_range_and_L_pinned():
    cfg = TreeConfig()
    n = 5
    rad = ScaleConditionedRadius(n, cfg)
    a_base = torch.zeros(n)
    node_ids = torch.arange(n)
    for bnd in (S_BAND, M_BAND, L_BAND):
        bands = torch.full((n,), bnd, dtype=torch.long)
        th = rad.theta(a_base, node_ids, bands)
        assert (th > 0).all() and (th < math.pi / 2).all()
    # L column pinned to 0: theta at L equals base (pi/2)*sigmoid(a_base).
    th_L = rad.theta_all(a_base, L_BAND)
    assert torch.allclose(th_L, torch.full((n,), math.pi / 4), atol=1e-6)


def test_scale_radius_reduces_to_base_when_w_zero():
    cfg = TreeConfig()
    rad = ScaleConditionedRadius(4, cfg)                 # w init zeros
    a_base = torch.randn(4)
    for bnd in (S_BAND, M_BAND, L_BAND):
        th = rad.theta_all(a_base, bnd)
        base = cfg.half_pi * torch.sigmoid(a_base)
        assert torch.allclose(th, base, atol=1e-6)


def test_fit_node_temperature_reduces_nll():
    torch.manual_seed(0)
    # margins correlate with descend labels but are over-confident -> T>?.
    margins = torch.linspace(-3, 3, 200)
    probs = torch.sigmoid(margins * 4.0)
    descends = (torch.rand(200) < probs).float()

    def nll(tau):
        logit = margins / tau
        return float(F.binary_cross_entropy_with_logits(logit, descends))

    tau = fit_node_temperature(margins, descends, iters=300, lr=0.05)
    assert tau > 0
    assert nll(tau) <= nll(1.0) + 1e-4                   # never worse than T=1


def test_recalibrate_tau0_runs_and_writes_back():
    cfg = TreeConfig()
    head = ObjectnessHead(cfg)
    norms = torch.randn(128).abs() + 0.5
    val = recalibrate_tau0(head, norms, iters=50, lr=0.1)
    assert val > 0
    assert abs(float(head.tau0) - val) < 1e-6            # written back into head


def test_best_f1_threshold_perfect_separation():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    thr, f1 = best_f1_threshold(scores, labels)
    assert abs(f1 - 1.0) < 1e-9


def test_threshold_transfer_gap_nonnegative_on_shift():
    rng = np.random.RandomState(0)
    src_scores = np.concatenate([rng.normal(0.3, 0.1, 100), rng.normal(0.7, 0.1, 100)])
    src_labels = np.array([0] * 100 + [1] * 100)
    # target shifted higher -> source threshold transfers imperfectly.
    tgt_scores = np.concatenate([rng.normal(0.5, 0.1, 100), rng.normal(0.9, 0.1, 100)])
    tgt_labels = np.array([0] * 100 + [1] * 100)
    ttg = threshold_transfer_gap(src_scores, src_labels, tgt_scores, tgt_labels)
    assert ttg >= -1e-9                                  # optimal >= transferred


def test_ece_per_depth_keys_and_range():
    conf = [0.9, 0.8, 0.4, 0.6, 0.95, 0.2]
    correct = [1, 1, 0, 1, 1, 0]
    depths = [1, 1, 1, 2, 2, 2]
    out = ece_per_depth(conf, correct, depths, TreeConfig())
    assert set(out) == {1, 2}
    for v in out.values():
        assert 0.0 <= v <= 1.0
