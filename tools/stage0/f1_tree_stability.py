#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / F1 : is the confusability tree STABLE across data splits?
# ------------------------------------------------------------------------
# A parameter-allocation structure is only worth building if it is reproducible:
# induce the tree on two disjoint halves of the detections and compare them.  If
# the tree reshuffles every split, the "sibling" relation is noise.
#
# Metric: stability = 1 - RF_norm, where RF_norm is the Robinson-Foulds distance
# (symmetric difference of the induced bipartition sets) normalised by its max.
# Gate (note): stability >= 0.70.
# ------------------------------------------------------------------------
import argparse
import numpy as np

import common
from models.tree import build_affinity, confusion_to_rates, induce_tree


def bipartitions(tree, num_classes):
    """Set of frozenset leaf-class clusters induced by each internal node
    (the clades), excluding the trivial full set and singletons."""
    clades = set()
    for node in tree.children:
        if tree.is_leaf(node):
            continue
        leaves = frozenset(tree.leaf_classes(node))
        if 1 < len(leaves) < num_classes:
            clades.add(leaves)
    return clades


def rf_stability(t1, t2, num_classes):
    c1, c2 = bipartitions(t1, num_classes), bipartitions(t2, num_classes)
    rf = len(c1 ^ c2)                       # symmetric difference
    denom = len(c1) + len(c2)
    return 1.0 - (rf / denom if denom else 0.0)


def split_confusion(feats, half):
    """Confusion counts from one half of the KNOWN detections (deterministic
    split by detection index parity)."""
    K = feats.num_known
    C = np.zeros((K, K))
    idx = np.where(feats.kind == common.KNOWN)[0]
    sel = idx[idx % 2 == half]
    y = feats.labels[sel].astype(int)
    pred = feats.posteriors[sel].argmax(1).astype(int)
    for yi, pj in zip(y, pred):
        if 0 <= yi < K and 0 <= pj < K:
            C[yi, pj] += 1.0
    return C


def main():
    ap = argparse.ArgumentParser(description="F1: confusability tree stability (Robinson-Foulds)")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    trees = []
    for half in (0, 1):
        C = confusion_to_rates(split_confusion(feats, half))
        _A, D = build_affinity(C)
        trees.append(induce_tree(D, num_classes=feats.num_known))

    stab = rf_stability(trees[0], trees[1], feats.num_known)
    return common.verdict("F1 tree stability (1 - RF_norm)", stab,
                          stab >= 0.70, "stability >= 0.70")


if __name__ == "__main__":
    raise SystemExit(main())
