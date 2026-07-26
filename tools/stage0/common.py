# ------------------------------------------------------------------------
# Tree-DETR Stage-0 : shared feature extraction + fixtures
# ------------------------------------------------------------------------
# The Stage-0 suite is the note's falsification battery: it runs on an ALREADY
# TRAINED vanilla detector, *before* any tree training, and every check has a
# hard pass/fail gate.  If the cheap checks fail, the whole method is refuted
# and no training is warranted.
#
# This module provides the data those checks consume:
#   * a fixed feature schema (Stage0Features) of per-detection quantities,
#   * ``extract_features`` - pull them from a trained DeformableDETR (needs the
#     full training env: the compiled MSDeformAttn op + matching torchvision),
#   * ``make_synthetic_features`` - a self-contained fixture so every metric
#     script can be exercised end-to-end without a detector or GPU.
#
# The heavy detector imports are done lazily inside ``extract_features`` so the
# metric scripts import and run in a torch+numpy-only environment.
# ------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import importlib
import os
import sys
import types
import numpy as np


def bootstrap() -> str:
    """Put the repo root on sys.path and make ``models.tree`` importable even in
    an environment where the base ``models/__init__.py`` cannot load (e.g. no
    compiled MSDeformAttn op / mismatched torchvision).

    Returns the repo root.  Safe to call repeatedly.  If the real ``models``
    package imports cleanly it is left untouched; otherwise a bare stub with the
    correct ``__path__`` is registered so ``models.tree.*`` submodules resolve
    while the broken top-level body is skipped.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if "models" not in sys.modules:
        try:
            importlib.import_module("models")
        except Exception:
            stub = types.ModuleType("models")
            stub.__path__ = [os.path.join(repo_root, "models")]
            sys.modules["models"] = stub
    return repo_root


bootstrap()

# per-detection "kind"
BACKGROUND = 0     # unmatched query (no GT) -> background
KNOWN = 1          # matched to a GT of a *known* (training) class
UNKNOWN = 2        # matched to a GT of a held-out (unknown) class


@dataclass
class Stage0Features:
    """Per-detection quantities extracted from a trained detector.

    N detections total (matched + a sample of unmatched/background).  All arrays
    share the leading dimension N.
    """
    h: np.ndarray                       # (N, D)   matched decoder query feature
    norms: np.ndarray                   # (N,)     ||h||
    labels: np.ndarray                  # (N,)     GT class id, or -1 for background
    kind: np.ndarray                    # (N,)     BACKGROUND / KNOWN / UNKNOWN
    posteriors: np.ndarray              # (N, K)   known-class posteriors (softmax)
    areas: np.ndarray                   # (N,)     GT (or predicted) box area in px^2
    num_known: int                      # K, number of known classes
    class_super: Optional[np.ndarray] = None   # (K,) semantic supercategory id per known class
    unknown_classes: Optional[np.ndarray] = None  # ids of the held-out classes
    meta: Dict[str, object] = field(default_factory=dict)

    # -- persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        d = dict(h=self.h, norms=self.norms, labels=self.labels, kind=self.kind,
                 posteriors=self.posteriors, areas=self.areas,
                 num_known=np.int64(self.num_known))
        if self.class_super is not None:
            d["class_super"] = self.class_super
        if self.unknown_classes is not None:
            d["unknown_classes"] = self.unknown_classes
        np.savez_compressed(path, **d)

    @classmethod
    def load(cls, path: str) -> "Stage0Features":
        z = np.load(path, allow_pickle=False)
        return cls(
            h=z["h"], norms=z["norms"], labels=z["labels"], kind=z["kind"],
            posteriors=z["posteriors"], areas=z["areas"],
            num_known=int(z["num_known"]),
            class_super=z["class_super"] if "class_super" in z.files else None,
            unknown_classes=z["unknown_classes"] if "unknown_classes" in z.files else None,
        )

    # -- convenience masks --------------------------------------------------
    def mask(self, kind: int) -> np.ndarray:
        return self.kind == kind

    def known_h(self) -> np.ndarray:
        return self.h[self.kind == KNOWN]

    def known_labels(self) -> np.ndarray:
        return self.labels[self.kind == KNOWN]


# ========================================================================
# Confusion / prototype helpers used across the metric scripts
# ========================================================================
def confusion_counts(feats: Stage0Features) -> np.ndarray:
    """(K, K) confusion-count matrix from the known-class detections:
    C[i, j] = #{GT class i detections whose argmax known-posterior is j}."""
    K = feats.num_known
    C = np.zeros((K, K), dtype=np.float64)
    m = feats.kind == KNOWN
    y = feats.labels[m].astype(int)
    pred = feats.posteriors[m].argmax(axis=1).astype(int)
    for yi, pj in zip(y, pred):
        if 0 <= yi < K and 0 <= pj < K:
            C[yi, pj] += 1.0
    return C


def class_prototypes(feats: Stage0Features, m: int = 64, seed: int = 0) -> np.ndarray:
    """Unit-sphere prototype direction per known class from the mean projected
    feature.  A random-but-fixed projection R^D -> R^m stands in for the
    (untrained) cone-embedding head, so the Stage-0 proxy cascade is exemplar
    free and reproducible."""
    rng = np.random.RandomState(seed)
    D = feats.h.shape[1]
    P = rng.randn(D, m) / np.sqrt(D)
    K = feats.num_known
    protos = np.zeros((K, m), dtype=np.float64)
    for c in range(K):
        sel = (feats.kind == KNOWN) & (feats.labels == c)
        if sel.any():
            v = (feats.h[sel] @ P).mean(axis=0)
            protos[c] = v / (np.linalg.norm(v) + 1e-9)
        else:
            protos[c] = rng.randn(m)
            protos[c] /= np.linalg.norm(protos[c]) + 1e-9
    return protos, P


def project(h: np.ndarray, P: np.ndarray) -> np.ndarray:
    z = h @ P
    return z / (np.linalg.norm(z, axis=-1, keepdims=True) + 1e-9)


def angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Stable angle (matches geometry.stable_angle) for unit-row arrays."""
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    diff = np.linalg.norm(a - b, axis=-1)
    summ = np.linalg.norm(a + b, axis=-1)
    return 2.0 * np.arctan2(diff, summ)


# ========================================================================
# Synthetic fixture (lets every metric script run without a detector)
# ========================================================================
def make_synthetic_features(seed: int = 0, num_known: int = 12, num_unknown: int = 4,
                            per_class: int = 60, n_bg: int = 400, D: int = 256,
                            n_super: int = 4) -> Stage0Features:
    """Generate a plausible fixture in which the note's premises *hold*, so a
    green run confirms the plumbing (not the science).

    Construction:
      * ``num_known`` class means in R^D arranged in confusable blocks (nearby
        means share a block -> they confuse, giving a tree-shaped affinity);
      * known-class detections cluster tightly around their mean with LARGE norm;
      * unknown-GT detections sit *between* blocks with intermediate-large norm;
      * background sits near the origin with SMALL norm (validates N1);
      * a semantic supercategory assignment deliberately *cross-cuts* the
        confusability blocks (so F2 overlap is low).
    """
    rng = np.random.RandomState(seed)
    K = num_known
    block_size = max(2, K // 3)
    # block (confusability) centres
    n_blocks = int(np.ceil(K / block_size))
    block_dirs = rng.randn(n_blocks, D)
    block_dirs /= np.linalg.norm(block_dirs, axis=1, keepdims=True)
    means = np.zeros((K, D))
    for c in range(K):
        b = c // block_size
        means[c] = block_dirs[b] * 6.0 + rng.randn(D) * 0.6   # tight within block

    hs, norms, labels, kinds, areas = [], [], [], [], []

    def push(vecs, lab, kind):
        for v in vecs:
            hs.append(v); norms.append(np.linalg.norm(v))
            labels.append(lab); kinds.append(kind)
            areas.append(float(rng.uniform(20, 200) ** 2))

    # known detections: tight cluster, large magnitude
    for c in range(K):
        v = means[c][None, :] + rng.randn(per_class, D) * 0.5
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(9, 12, (per_class, 1))
        push(v, c, KNOWN)

    # unknown-GT detections: between two random blocks, intermediate-large norm
    for _ in range(num_unknown):
        b1, b2 = rng.choice(n_blocks, size=2, replace=True)
        mid = (block_dirs[b1] + block_dirs[b2]); mid /= np.linalg.norm(mid) + 1e-9
        v = mid[None, :] * 6.0 + rng.randn(per_class, D) * 1.0
        v = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(6, 9, (per_class, 1))
        push(v, -1, UNKNOWN)

    # background: near origin, small magnitude
    v = rng.randn(n_bg, D)
    v = v / np.linalg.norm(v, axis=1, keepdims=True) * rng.uniform(0.5, 3.0, (n_bg, 1))
    push(v, -1, BACKGROUND)

    h = np.asarray(hs, dtype=np.float32)
    norms = np.asarray(norms, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    kind = np.asarray(kinds, dtype=np.int64)
    areas = np.asarray(areas, dtype=np.float32)

    # known-class posteriors: softmax of negative distance to each class mean.
    logits = -((h[:, None, :] - means[None, :, :]) ** 2).sum(-1) / (2.0 * D)
    logits -= logits.max(axis=1, keepdims=True)
    post = np.exp(logits); post /= post.sum(axis=1, keepdims=True)

    # A closed-set head *confidently misclassifies* unknowns: sharpen every
    # unknown-GT posterior onto its nearest known class.  This is the realistic
    # OSR failure mode (high max-softmax on an unknown) that defeats a flat
    # free-energy detector (F4) while the geometric cascade, reading the
    # projected feature, still halts them shallow.
    unk_rows = np.where(kind == UNKNOWN)[0]
    for r in unk_rows:
        top = int(post[r].argmax())
        peaked = np.full(K, 0.02 / max(1, K - 1))
        peaked[top] = 0.98
        post[r] = peaked

    # semantic supercategories that cross-cut the confusability blocks:
    # class c -> c % n_super  (blocks are contiguous, so this scatters them).
    class_super = np.array([c % n_super for c in range(K)], dtype=np.int64)

    return Stage0Features(
        h=h, norms=norms, labels=labels, kind=kind,
        posteriors=post.astype(np.float32), areas=areas,
        num_known=K, class_super=class_super,
        unknown_classes=np.array([], dtype=np.int64),
        meta={"synthetic": True, "seed": seed},
    )


# ========================================================================
# Extraction from a trained detector (scaffold; needs the full training env)
# ========================================================================
def extract_features(args) -> Stage0Features:
    """Pull Stage0Features from a trained DeformableDETR checkpoint.

    This is the real-env path and is intentionally a thin scaffold: it wires the
    pieces (build the model, load the checkpoint, run the val loader, hook the
    last decoder layer for hs, Hungarian-match to label KNOWN/UNKNOWN, sample
    BACKGROUND from unmatched queries).  It requires the compiled MSDeformAttn op
    and a matching torchvision, so it is NOT exercised by the unit tests; the
    metric scripts default to ``make_synthetic_features`` unless given a
    ``--features`` npz produced here.

    Expected ``args`` attributes: dataset_file, coco_path, resume (checkpoint),
    device, plus the standard Deformable-DETR build args; ``unknown_class_ids``
    (a list) marks which COCO categories are treated as unknown.
    """
    import torch                                            # noqa: F401
    from models import build_model                          # lazy: pulls in ops
    from datasets import build_dataset
    from torch.utils.data import DataLoader
    import util.misc as utils

    device = torch.device(getattr(args, "device", "cpu"))
    model, _criterion, _pp = build_model(args)
    model.to(device).eval()
    ckpt = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)

    # capture last-decoder-layer hs via a forward hook on the transformer
    cache = {}

    def hook(_m, _i, o):
        cache["hs"] = o[0]                                  # [n_layers, bs, nq, d]
    model.transformer.register_forward_hook(hook)

    dataset = build_dataset(image_set="val", args=args)
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        collate_fn=utils.collate_fn, num_workers=0)

    from models.matcher import build_matcher
    matcher = build_matcher(args)
    unknown_ids = set(getattr(args, "unknown_class_ids", []) or [])

    H, NRM, LAB, KND, POST, AR = [], [], [], [], [], []
    with torch.no_grad():
        for samples, targets in loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(samples)
            hs = cache["hs"][-1]                            # (bs, nq, d)
            indices = matcher(out, targets)
            post = out["pred_logits"].softmax(-1)           # (bs, nq, K(+1))
            for b, (src, tgt) in enumerate(indices):
                matched = set(src.tolist())
                for qi in range(hs.shape[1]):
                    h = hs[b, qi].cpu().numpy()
                    p = post[b, qi].cpu().numpy()
                    if qi in matched:
                        ti = tgt[(src == qi).nonzero().item()].item()
                        cls = int(targets[b]["labels"][ti].item())
                        box = targets[b]["boxes"][ti].cpu().numpy()
                        ar = float(box[2] * box[3])          # cxcywh (normalised)
                        kind = UNKNOWN if cls in unknown_ids else KNOWN
                        H.append(h); NRM.append(float(np.linalg.norm(h)))
                        LAB.append(cls); KND.append(kind); POST.append(p); AR.append(ar)
                    elif np.random.rand() < 0.02:            # sample a few backgrounds
                        H.append(h); NRM.append(float(np.linalg.norm(h)))
                        LAB.append(-1); KND.append(BACKGROUND); POST.append(p); AR.append(0.0)

    K = int(model.num_classes) if hasattr(model, "num_classes") else POST[0].shape[0]
    return Stage0Features(
        h=np.asarray(H, np.float32), norms=np.asarray(NRM, np.float32),
        labels=np.asarray(LAB, np.int64), kind=np.asarray(KND, np.int64),
        posteriors=np.asarray(POST, np.float32)[:, :K], areas=np.asarray(AR, np.float32),
        num_known=K, unknown_classes=np.asarray(sorted(unknown_ids), np.int64),
    )


# ========================================================================
# CLI helper shared by the metric scripts
# ========================================================================
def load_or_synth(args) -> Stage0Features:
    """Resolve the feature source for a metric script: an npz via --features,
    else a synthetic fixture (default) via --synthetic/--seed."""
    if getattr(args, "features", None):
        return Stage0Features.load(args.features)
    return make_synthetic_features(seed=getattr(args, "seed", 0))


def add_common_args(parser) -> None:
    parser.add_argument("--features", type=str, default=None,
                        help="path to a Stage0Features .npz (from extract_features); "
                             "if omitted a synthetic fixture is used")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the synthetic fixture")


def verdict(name: str, value: float, gate_ok: bool, gate_desc: str) -> int:
    tag = "PASS" if gate_ok else "FAIL"
    print(f"[{tag}] {name}: {value:.4f}   gate: {gate_desc}")
    return 0 if gate_ok else 1
