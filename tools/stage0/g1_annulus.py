#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / G1 : is a RESERVED ANNULUS geometrically achievable?
# ------------------------------------------------------------------------
# Module E's whole premise is that a positive, unoccupied annulus can sit
# between a parent cone and the union of its children - that is where a
# "kind of n but none of n's children" object lands.  Before training the cones,
# check the geometry is not already saturated: build prototype cones from the
# features and measure gap(n) = min_c [theta_n - angle(mu_n, mu_c) - theta_c]
# (Eq E2).  We need at least one node with a small-but-positive gap, i.e. an
# annulus that L_gap could plausibly reserve.
#
# Metric: number of internal nodes with 0 < gap(n) < 2*gamma_n, gamma_n = rho*theta_n.
# Gate (note): such a node exists somewhere (count >= 1).
# ------------------------------------------------------------------------
import argparse
import numpy as np

import common
from models.tree import build_affinity, confusion_to_rates, induce_tree, DEFAULT


def node_dir_and_radius(tree, protos, rho):
    """Prototype cone (mu_n, theta_n) per node.

    axis  mu_n = mean leaf-prototype direction.
    radius theta_n: a leaf gets a small floor radius; an internal node is sized
    to cover its leaves PLUS the Eq-E8 open-space budget (a 2*rho headroom).
    G1 tests exactly this feasibility - whether, granted the note's rho budget,
    a positive annulus of width < 2*gamma_n fits between parent and children.
    """
    mu, theta = {}, {}
    for n in tree.children:
        cls = tree.leaf_classes(n)
        if not cls:
            continue
        dirs = protos[cls]
        v = dirs.mean(0); v = v / (np.linalg.norm(v) + 1e-9)
        mu[n] = v
        if tree.is_leaf(n):
            theta[n] = 0.15                                  # leaf floor radius
        else:
            angs = common.angle(v[None, :], dirs)
            cover = float(max(angs.max(), 0.15))
            theta[n] = min(cover * (1.0 + 2.0 * rho), 0.99 * (np.pi / 2))
    return mu, theta


def main():
    ap = argparse.ArgumentParser(description="G1: reserved-annulus feasibility (Eq E2)")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    C = confusion_to_rates(common.confusion_counts(feats))
    _A, D = build_affinity(C)
    tree = induce_tree(D, num_classes=feats.num_known)
    protos, _P = common.class_prototypes(feats, m=64, seed=args.seed)
    rho = DEFAULT.rho
    mu, theta = node_dir_and_radius(tree, protos, rho)
    positive = 0
    reservable = 0
    for n in tree.children:
        ch = tree.children[n]
        if not ch or n not in mu:
            continue
        per_child = []
        for c in ch:
            if c not in mu:
                continue
            ang = float(common.angle(mu[n][None, :], mu[c][None, :])[0])
            per_child.append(theta[n] - ang - theta[c])
        if not per_child:
            continue
        gap = min(per_child)
        gamma = rho * theta[n]
        if gap > 0:
            positive += 1
            if gap < 2.0 * gamma:
                reservable += 1

    print(f"       internal nodes with positive gap={positive}, "
          f"with 0<gap<2*gamma={reservable}")
    return common.verdict("G1 reservable-annulus node count", float(reservable),
                          reservable >= 1, "exists a node with 0 < gap < 2*gamma_n")


if __name__ == "__main__":
    raise SystemExit(main())
