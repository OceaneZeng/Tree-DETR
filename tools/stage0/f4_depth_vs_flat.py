#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / F4 : does the HIERARCHY beat a FLAT novelty score?
# ------------------------------------------------------------------------
# Even if halt depth carries novelty info (F3), it must beat the standard flat
# baseline - free energy (-logsumexp of the class logits) - or the tree is an
# expensive way to reproduce what a single scalar already gives.
#
# Metric: AUROC(unknown vs known) of the depth-based score minus AUROC of the
# free-energy score.
# Gate (note): depth AUROC >= flat AUROC + 0.02.
# ------------------------------------------------------------------------
import argparse
import numpy as np

import common
from models.tree import build_affinity, confusion_to_rates, induce_tree
from f3_halt_depth_novelty import proxy_halt_depth


def auroc(scores, labels):
    """AUROC via the rank-sum (Mann-Whitney) identity; no sklearn."""
    scores = np.asarray(scores, float); labels = np.asarray(labels, int)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    r_pos = ranks[labels == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser(description="F4: hierarchical vs flat (free-energy) AUROC")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    C = confusion_to_rates(common.confusion_counts(feats))
    _A, D = build_affinity(C)
    tree = induce_tree(D, num_classes=feats.num_known)
    protos, P = common.class_prototypes(feats, m=64, seed=args.seed)

    sel = np.isin(feats.kind, [common.KNOWN, common.UNKNOWN])
    labels = (feats.kind[sel] == common.UNKNOWN).astype(int)   # 1 = unknown
    z = common.project(feats.h[sel], P)

    # depth score: shallower halt = more novel -> use (max_depth - depth)
    depths = np.array([proxy_halt_depth(z[i], tree, protos, P) for i in range(z.shape[0])])
    depth_score = depths.max() - depths                        # high => novel

    # flat baseline: free energy E = -logsumexp(logits); logits ~ log posteriors
    post = np.clip(feats.posteriors[sel], 1e-9, 1.0)
    logits = np.log(post)
    free_energy = -(np.log(np.exp(logits).sum(1)))             # high => novel
    # (with normalised posteriors logsumexp is ~0; use max-logit margin instead)
    flat_score = -np.sort(logits, axis=1)[:, -1]               # low top-logit => novel

    a_depth = auroc(depth_score, labels)
    a_flat = auroc(flat_score, labels)
    delta = a_depth - a_flat
    print(f"       depth AUROC={a_depth:.4f}  flat AUROC={a_flat:.4f}")
    return common.verdict("F4 AUROC(depth) - AUROC(flat)", delta,
                          delta >= 0.02, "depth AUROC >= flat AUROC + 0.02")


if __name__ == "__main__":
    raise SystemExit(main())
