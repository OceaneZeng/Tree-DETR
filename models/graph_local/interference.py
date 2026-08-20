"""Stage-local class-interference graph utilities."""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


def flatten_gradients(gradients: Iterable[torch.Tensor | None],
                      parameters: Sequence[torch.Tensor]) -> torch.Tensor:
    chunks = []
    for gradient, parameter in zip(gradients, parameters):
        if gradient is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(gradient.detach().reshape(-1))
    if not chunks:
        raise ValueError("No target gradients were provided")
    return torch.cat(chunks)


def build_conflict_matrix(sketches: Mapping[int, torch.Tensor]
                          ) -> Tuple[List[int], torch.Tensor]:
    """Return ``max(0, -cos(g_i, g_j))`` in sorted class order."""
    if len(sketches) < 2:
        raise ValueError("At least two class sketches are required")
    class_ids = sorted(int(class_id) for class_id in sketches)
    rows = torch.stack([sketches[class_id].float().cpu() for class_id in class_ids])
    rows = F.normalize(rows, dim=1, eps=1e-12)
    conflict = torch.clamp(-(rows @ rows.t()), min=0.0)
    conflict.fill_diagonal_(0.0)
    return class_ids, conflict


def select_positive_neighbors(class_ids: Sequence[int], conflict: torch.Tensor,
                              source_class: int, k: int = 5,
                              min_conflict: float = 0.0) -> List[Tuple[int, float]]:
    """Select up to ``k`` outgoing positive-conflict neighbors."""
    if k <= 0:
        return []
    try:
        source_index = list(class_ids).index(int(source_class))
    except ValueError as exc:
        raise KeyError(f"Unknown source class {source_class}") from exc
    candidates = []
    for index, class_id in enumerate(class_ids):
        if index == source_index:
            continue
        score = float(conflict[source_index, index])
        if score > min_conflict:
            candidates.append((int(class_id), score))
    candidates.sort(key=lambda item: (-item[1], item[0]))
    return candidates[:k]


def build_off_neighborhood_basis(sketches: Mapping[int, torch.Tensor],
                                 excluded_classes: Iterable[int],
                                 max_rank: int = 8,
                                 eps: float = 1e-8) -> torch.Tensor:
    """Build an orthonormal dense basis from off-neighborhood gradients.

    The eigendecomposition is performed in the small class-by-class Gram
    matrix, avoiding an SVD over the full detector-weight dimension.
    """
    if not sketches:
        raise ValueError("At least one sketch is required")
    excluded = {int(class_id) for class_id in excluded_classes}
    selected = [sketches[class_id].float().cpu() for class_id in sorted(sketches)
                if int(class_id) not in excluded]
    feature_dim = next(iter(sketches.values())).numel()
    if not selected or max_rank <= 0:
        return torch.empty(feature_dim, 0)
    rows = F.normalize(torch.stack(selected), dim=1, eps=eps)
    gram = rows @ rows.t()
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    keep = torch.nonzero(eigenvalues > eps, as_tuple=False).flatten()
    if keep.numel() == 0:
        return torch.empty(feature_dim, 0)
    keep = keep[-min(max_rank, keep.numel()):]
    values = eigenvalues[keep]
    vectors = eigenvectors[:, keep]
    basis = rows.t() @ (vectors / values.sqrt().unsqueeze(0))
    return basis.contiguous()


def evaluate_neighborhood(predicted: Sequence[int], harm: Mapping[int, float],
                          k: int | None = None, random_trials: int = 1000,
                          seed: int = 42) -> Dict[str, float | List[int]]:
    """Compare predicted neighbors with the oracle positive-harm classes."""
    class_ids = sorted(int(class_id) for class_id in harm)
    if k is None:
        k = len(predicted)
    k = min(max(int(k), 0), len(class_ids))
    ranked = sorted(class_ids, key=lambda class_id: (-max(0.0, float(harm[class_id])), class_id))
    oracle = ranked[:k]
    predicted = [int(class_id) for class_id in predicted[:k]]
    total_harm = sum(max(0.0, float(value)) for value in harm.values())

    def concentration(classes: Sequence[int]) -> float:
        if total_harm <= 0:
            return 0.0
        return sum(max(0.0, float(harm.get(class_id, 0.0)))
                   for class_id in classes) / total_harm

    intersection = len(set(predicted) & set(oracle))
    rng = random.Random(seed)
    random_concentrations = []
    for _ in range(max(1, random_trials)):
        sample = rng.sample(class_ids, k) if k else []
        random_concentrations.append(concentration(sample))
    random_mean = sum(random_concentrations) / len(random_concentrations)
    random_sorted = sorted(random_concentrations)
    q95_index = min(len(random_sorted) - 1, int(0.95 * len(random_sorted)))
    return {
        "predicted_neighbors": predicted,
        "oracle_neighbors": oracle,
        "predicted_concentration": concentration(predicted),
        "oracle_concentration": concentration(oracle),
        "random_concentration_mean": random_mean,
        "random_concentration_q95": random_sorted[q95_index],
        "precision_at_k": intersection / max(1, len(predicted)),
        "recall_at_k": intersection / max(1, len(oracle)),
        "total_positive_harm": total_harm,
    }


def mean_pairwise_jaccard(neighbor_sets: Sequence[Sequence[int]]) -> float:
    values = []
    for left_index in range(len(neighbor_sets)):
        left = set(neighbor_sets[left_index])
        for right_index in range(left_index + 1, len(neighbor_sets)):
            right = set(neighbor_sets[right_index])
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return sum(values) / len(values) if values else 1.0
