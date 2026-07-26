#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / F3 : does HALT DEPTH carry novelty information?
# ------------------------------------------------------------------------
# The central claim of Module B is that *where* an object stops descending the
# tree is a novelty signal.  Before training anything, test it with a proxy
# cascade built from class prototypes: descend by nearest-prototype gates and
# record the halt depth; correlate that depth with a novelty label
# (0 = known, 1 = unknown-GT).
#
# Metric: Spearman rho between halt depth and novelty label.
# Gate (note): rho >= 0.30.
# ------------------------------------------------------------------------
import argparse
import numpy as np

import common
from models.tree import (build_affinity, confusion_to_rates, induce_tree)


def spearman(a, b):
    """Spearman rank correlation (no scipy)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def proxy_halt_depth(z, tree, protos, P, tau=0.5):
    """Descend the tree: at each node, a child 'claims' z if z is closer (in
    angle) to that child's mean prototype direction than a per-depth threshold.
    A node's direction = mean of its leaf-class prototypes.  Halt = no child
    claims.  Returns the halt depth (0 = claimed by nothing at the root)."""
    def node_dir(n):
        cls = tree.leaf_classes(n)
        v = protos[cls].mean(0)
        return v / (np.linalg.norm(v) + 1e-9)

    node = tree.root
    depth = 0
    while not tree.is_leaf(node):
        ch = tree.children[node]
        dirs = np.stack([node_dir(c) for c in ch])
        angs = common.angle(z[None, :], dirs)
        j = int(np.argmin(angs))
        # claim threshold grows with depth (deeper => stricter); tau in radians
        if angs[j] > tau + 0.15 * depth:
            break
        node = ch[j]
        depth += 1
    return depth


def main():
    ap = argparse.ArgumentParser(description="F3: halt-depth vs novelty (Spearman)")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    C = confusion_to_rates(common.confusion_counts(feats))
    _A, D = build_affinity(C)
    tree = induce_tree(D, num_classes=feats.num_known)
    protos, P = common.class_prototypes(feats, m=64, seed=args.seed)

    sel = np.isin(feats.kind, [common.KNOWN, common.UNKNOWN])
    z = common.project(feats.h[sel], P)
    novelty = (feats.kind[sel] == common.UNKNOWN).astype(float)
    depths = np.array([proxy_halt_depth(z[i], tree, protos, P) for i in range(z.shape[0])])
    # known objects should descend deeper (reach a leaf); unknowns halt early ->
    # novelty should correlate NEGATIVELY with depth; F3 reports |rho|.
    rho = spearman(depths, novelty)
    return common.verdict("F3 halt-depth/novelty |Spearman|", abs(rho),
                          abs(rho) >= 0.30, "|rho| >= 0.30")


if __name__ == "__main__":
    raise SystemExit(main())
