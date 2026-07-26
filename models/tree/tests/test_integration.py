"""Integration — TreeHead/flat tree wiring without the compiled detector ops.

We do NOT build a full DeformableDETR here (its MSDeformAttn op may be
uncompiled in the test env).  Instead we exercise the parts of tree_detr.py that
do not need the base model: the flat-tree factory, TreeHead heads, and the
inference path on random query features.
"""
import torch

from models.tree.config import TreeConfig
from models.tree.tree_detr import TreeHead, flat_two_level_tree
from models.tree.cascade import HaltResult


def test_flat_two_level_tree_shape():
    tree = flat_two_level_tree(num_classes=5)
    assert tree.root == 0
    assert len(tree.children[0]) == 5                     # all classes direct children
    assert sorted(tree.leaf_class.values()) == [0, 1, 2, 3, 4]
    tree.topology().validate()


def test_tree_head_embed_on_sphere():
    cfg = TreeConfig()
    head = TreeHead(flat_two_level_tree(4), cfg)
    h = torch.randn(7, cfg.d_model)
    z = head.embed(h)
    assert z.shape == (7, cfg.m)
    assert torch.allclose(z.norm(dim=-1), torch.ones(7), atol=1e-5)


def test_tree_head_infer_returns_haltresult():
    cfg = TreeConfig()
    head = TreeHead(flat_two_level_tree(4), cfg)
    h = torch.randn(cfg.d_model)
    res = head.infer(h, box_area=48 * 48)
    assert isinstance(res, HaltResult)
    assert 0.0 <= res.path_score <= 1.0
    assert res.halt_depth >= 0


def test_tree_weight_dict_includes_obj():
    cfg = TreeConfig()
    head = TreeHead(flat_two_level_tree(4), cfg)
    wd = head.gap_loss.weight_dict()
    wd["loss_obj"] = 1.0
    assert set(wd) >= {"loss_contain", "loss_nest", "loss_gap", "loss_sib", "loss_obj"}
