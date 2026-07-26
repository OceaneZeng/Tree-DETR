"""Module C — zero-init identity at step 0, width clip bounds, insertion masks."""
import torch

from models.tree.config import TreeConfig
from models.tree.topology import TreeTopology
from models.tree.losses import ConeField
from models.tree.calibration import ScaleConditionedRadius
from models.tree.cascade import Cascade
from models.tree.adapter import (
    ParallelFFNAdapter, AdapterBank, adapter_width, SiblingRadiusReparam,
    insertion_param_groups, configure_insertion,
)
from models.tree.tree_structure import ConfusabilityTree


def test_adapter_identity_at_init():
    cfg = TreeConfig()
    ad = ParallelFFNAdapter(cfg.d_model, rank=8, cfg=cfg)
    h = torch.randn(5, cfg.d_model)
    assert torch.allclose(ad.delta(h), torch.zeros_like(h), atol=1e-7)
    assert torch.allclose(ad(h), h, atol=1e-7)                 # residual identity


def test_adapter_nonzero_after_perturbing_wup():
    cfg = TreeConfig()
    ad = ParallelFFNAdapter(cfg.d_model, rank=8, cfg=cfg)
    with torch.no_grad():
        ad.w_up.weight.add_(0.1)
    h = torch.randn(5, cfg.d_model)
    assert not torch.allclose(ad.delta(h), torch.zeros_like(h))


def test_adapter_width_clip_bounds():
    cfg = TreeConfig()
    assert adapter_width([], cfg) == cfg.r0                      # r0*(1+0)=8
    assert adapter_width([0.0, 0.0], cfg) == cfg.r0
    # huge affinities clip to r_max
    assert adapter_width([10.0] * 10, cfg) == cfg.r_max
    # zero base still floored at r_min if r0 ever below (defensive)
    assert cfg.r_min <= adapter_width([0.1], cfg) <= cfg.r_max


def test_sibling_radius_reparam_shrinks_only():
    theta_old = torch.tensor([1.0, 0.8, 0.5])
    rep = SiblingRadiusReparam(theta_old)
    theta = rep()
    assert torch.all(theta <= theta_old + 1e-6)                 # never grows
    assert torch.all(theta > 0)


def test_adapter_bank_insert_and_apply():
    cfg = TreeConfig()
    bank = AdapterBank(cfg.d_model, cfg)
    bank.insert(cls=7, rank=8)
    h = torch.randn(3, cfg.d_model)
    # zero-init -> apply is identity even with an inserted adapter.
    assert torch.allclose(bank.apply(h), h, atol=1e-7)
    assert bank.get(7) is not None
    assert bank.param_count() > 0


def _small_head():
    cfg = TreeConfig()
    children = {0: [1, 2], 1: [], 2: []}
    leaf_class = {1: 0, 2: 1}
    tree = ConfusabilityTree(children, leaf_class, root=0)
    topo = tree.topology()
    cones = ConeField(topo.num_nodes, cfg)
    radius = ScaleConditionedRadius(topo.num_nodes, cfg)
    casc = Cascade(topo, cones, radius, cfg)
    bank = AdapterBank(cfg.d_model, cfg)
    return cfg, tree, cones, radius, casc, bank


def test_insertion_param_groups_spec():
    cfg, tree, *_ = _small_head()
    # insert a brand-new class 2 under node 0 (a new leaf must exist first).
    new_leaf = tree.add_leaf(parent=0, cls=2)
    spec = insertion_param_groups(tree, node=0, cls=2)
    assert spec["mu_trainable_rows"] == [new_leaf]              # only mu_k
    assert new_leaf in spec["theta_trainable_rows"]
    assert 0 not in spec["mu_trainable_rows"]                   # mu_node frozen


def test_configure_insertion_row_masking():
    cfg, tree, cones, radius, casc, bank = _small_head()
    new_leaf = tree.add_leaf(parent=0, cls=2)
    # grow cone/radius/tau tensors to cover the new node id.
    n_new = tree.num_nodes
    cones2 = ConeField(n_new, cfg)
    radius2 = ScaleConditionedRadius(n_new, cfg)
    topo2 = tree.topology()
    casc2 = Cascade(topo2, cones2, radius2, cfg)
    bank.insert(cls=2, rank=adapter_width([0.5], cfg))
    spec = insertion_param_groups(tree, node=0, cls=2)
    handles = configure_insertion(spec, bank, cones2, radius2, casc2)

    # a backward through mu_raw must leave frozen rows with zero grad.
    z = cones2.mu[0].detach()                                   # some direction
    loss = (cones2.mu.sum() + cones2.a.sum())
    loss.backward()
    mu_grad = cones2.mu_raw.grad
    # only row new_leaf should carry gradient (others masked to 0).
    nonzero_rows = (mu_grad.abs().sum(dim=1) > 0).nonzero().flatten().tolist()
    assert nonzero_rows == [new_leaf]
    for h in handles:
        h.remove()
