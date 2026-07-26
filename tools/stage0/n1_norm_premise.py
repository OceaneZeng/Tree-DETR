#!/usr/bin/env python
# ------------------------------------------------------------------------
# Stage-0 / N1 : is the MAGNITUDE premise of the depth-0 gate true?
# ------------------------------------------------------------------------
# Module B's objectness gate (Eq B1) decides "is this a thing at all?" from the
# feature norm ||h|| alone.  That only works if real objects - including UNKNOWN
# ones - have larger norm than background.  This checks the premise directly.
#
# Metric: mode(||h|| | unknown-GT)  -  mode(||h|| | background), estimated by a
# histogram peak.  Must be positive.
# Gate (note): unknown-GT norm mode strictly above the background norm mode.
# ------------------------------------------------------------------------
import argparse
import numpy as np

import common


def hist_mode(x, bins=40):
    x = np.asarray(x, float)
    if x.size == 0:
        return 0.0
    counts, edges = np.histogram(x, bins=bins)
    k = int(np.argmax(counts))
    return float(0.5 * (edges[k] + edges[k + 1]))


def main():
    ap = argparse.ArgumentParser(description="N1: unknown-GT norm above background norm")
    common.add_common_args(ap)
    args = ap.parse_args()
    feats = common.load_or_synth(args)

    bg = feats.norms[feats.kind == common.BACKGROUND]
    unk = feats.norms[feats.kind == common.UNKNOWN]
    if unk.size == 0:
        print("[SKIP] N1: no unknown-GT detections in features")
        return 0

    mode_bg = hist_mode(bg)
    mode_unk = hist_mode(unk)
    gap = mode_unk - mode_bg
    print(f"       mode(||h|| bg)={mode_bg:.3f}  mode(||h|| unknown)={mode_unk:.3f}")
    return common.verdict("N1 unknown-minus-background norm mode", gap,
                          gap > 0.0, "unknown-GT norm mode > background norm mode")


if __name__ == "__main__":
    raise SystemExit(main())
