"""Trainable class-level interference prediction for graph-local adaptation.

The GNN in this module is deliberately a side-car controller.  It does not
consume image/query features during detector inference and it is not inserted
into the backbone.  A frozen detector supplies class gradient sketches; the
GNN predicts a directed harm score for each ordered class pair.  The predicted
neighborhood can then condition replay and the incremental objective.

Training data are stage artifacts from *previous* increments.  Each artifact
contains node features, an empirical one-step harm matrix, and an optional
validity mask.  The mask is important because a stage normally probes only
the newly added source class; unmeasured source rows must not be treated as
zero-harm labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


def compress_gradient_sketches(
    sketches: Mapping[int, torch.Tensor], output_dim: int = 128,
) -> Tuple[List[int], torch.Tensor]:
    """Pool dense weight-delta sketches into fixed-size GNN node features.

    The target decoder FFN gradients can contain hundreds of thousands of
    coordinates.  Adaptive pooling is deterministic, cheap, and avoids a
    million-to-hidden-dimension learned projection in the side-car.  The
    class order returned here must be used for harm-matrix rows and columns.
    """
    if not sketches:
        raise ValueError("At least one gradient sketch is required")
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    class_ids = sorted(int(class_id) for class_id in sketches)
    rows = []
    feature_dim = None
    for class_id in class_ids:
        row = sketches[class_id].detach().float().reshape(-1).cpu()
        feature_dim = row.numel() if feature_dim is None else feature_dim
        if row.numel() != feature_dim:
            raise ValueError("All gradient sketches must have the same dimension")
        rows.append(row)
    stacked = torch.stack(rows)
    pooled = F.adaptive_avg_pool1d(stacked.unsqueeze(1), output_dim).squeeze(1)
    return class_ids, F.normalize(pooled, dim=1, eps=1e-8)


class FixedNodeProjection(nn.Module):
    """Parameter-free dimensionality match for the node-encoder ablation."""

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = int(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] == self.output_dim:
            return features
        return F.adaptive_avg_pool1d(
            features.unsqueeze(1), self.output_dim).squeeze(1)


class ClassInterferenceGNN(nn.Module):
    """A small dense directed GNN for ordered class-pair harm prediction.

    ``edge_logits[i, j]`` always means "an update for source class ``i``
    harms target class ``j``".  The model uses a dense candidate graph by
    default; callers may provide a boolean mask to restrict candidate edges.
    This keeps the learned graph explicit and makes random/prototype controls
    straightforward to evaluate beside it.
    """

    format_version = 1

    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 message_steps: int = 2, dropout: float = 0.0,
                 use_node_encoder: bool = True):
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if message_steps < 0:
            raise ValueError("message_steps must be non-negative")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.message_steps = int(message_steps)
        self.dropout_probability = float(dropout)
        self.use_node_encoder = bool(use_node_encoder)
        self.node_dim = self.hidden_dim
        self.input_projection = (
            nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )
            if self.use_node_encoder else FixedNodeProjection(self.hidden_dim)
        )
        self.message_edges = nn.ModuleList([
            self._edge_mlp() for _ in range(self.message_steps)
        ])
        self.message_updates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * self.node_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Dropout(self.dropout_probability),
                nn.Linear(self.hidden_dim, self.node_dim),
            ) for _ in range(self.message_steps)
        ])
        self.final_edge = self._edge_mlp()

    def _edge_mlp(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(3 * self.node_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout_probability),
            nn.Linear(self.hidden_dim, 1),
        )

    @staticmethod
    def _pair_features(nodes: torch.Tensor) -> torch.Tensor:
        count = nodes.shape[0]
        source = nodes[:, None, :]
        target = nodes[None, :, :]
        return torch.cat((source.expand(-1, count, -1), target.expand(count, -1, -1),
                          source - target), dim=-1)

    @staticmethod
    def _candidate_mask(nodes: torch.Tensor,
                        candidate_mask: torch.Tensor | None) -> torch.Tensor:
        count = nodes.shape[0]
        if count < 2:
            raise ValueError("At least two class nodes are required")
        if candidate_mask is None:
            mask = torch.ones(count, count, dtype=torch.bool, device=nodes.device)
        else:
            mask = candidate_mask.to(device=nodes.device, dtype=torch.bool)
            if mask.shape != (count, count):
                raise ValueError("candidate_mask must have shape [num_nodes, num_nodes]")
            mask = mask.clone()
        mask.fill_diagonal_(False)
        if not mask.any():
            raise ValueError("candidate_mask must contain at least one off-diagonal edge")
        return mask

    def forward(self, node_features: torch.Tensor,
                candidate_mask: torch.Tensor | None = None
                ) -> Dict[str, torch.Tensor]:
        """Return directed edge scores and node embeddings.

        Args:
            node_features: ``[num_classes, input_dim]`` class-level features.
            candidate_mask: optional boolean ``[N, N]`` edge mask.
        """
        if node_features.ndim != 2 or node_features.shape[1] != self.input_dim:
            raise ValueError(
                f"node_features must have shape [N, {self.input_dim}], "
                f"got {tuple(node_features.shape)}"
            )
        nodes = self.input_projection(node_features)
        mask = self._candidate_mask(nodes, candidate_mask)
        for edge_mlp, update_mlp in zip(self.message_edges, self.message_updates):
            pair_logits = edge_mlp(self._pair_features(nodes)).squeeze(-1)
            masked_logits = pair_logits.masked_fill(~mask, float("-inf"))
            attention = torch.softmax(masked_logits, dim=-1)
            message = attention @ nodes
            nodes = nodes + update_mlp(torch.cat((nodes, message), dim=-1))
        edge_logits = self.final_edge(self._pair_features(nodes)).squeeze(-1)
        edge_logits = edge_logits.masked_fill(~mask, float("-inf"))
        return {
            "edge_logits": edge_logits,
            "edge_prob": torch.sigmoid(edge_logits),
            "node_embeddings": nodes,
            "candidate_mask": mask,
        }

    def config(self) -> Dict[str, int | float]:
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "message_steps": self.message_steps,
            "dropout": self.dropout_probability,
            "use_node_encoder": self.use_node_encoder,
        }


def harm_prediction_loss(edge_logits: torch.Tensor, harm: torch.Tensor,
                         valid_mask: torch.Tensor | None = None,
                         positive_threshold: float = 0.0,
                         ranking_margin: float = 0.25,
                         ranking_weight: float = 1.0
                         ) -> torch.Tensor:
    """Fit continuous harm values and their within-source ranking.

    ``harm`` is allowed to contain unmeasured entries; pass ``valid_mask`` to
    exclude them.  Per-source normalization makes stages with different loss
    scales comparable while the ranking term preserves the top-k objective.
    """
    if edge_logits.ndim != 2 or edge_logits.shape != harm.shape:
        raise ValueError("edge_logits and harm must be square matrices of equal shape")
    count = edge_logits.shape[0]
    valid = torch.ones_like(harm, dtype=torch.bool) if valid_mask is None else valid_mask.bool()
    if valid.shape != harm.shape:
        raise ValueError("valid_mask must match harm shape")
    valid = valid.clone()
    valid.fill_diagonal_(False)
    finite = torch.isfinite(harm)
    valid &= finite
    if not valid.any():
        raise ValueError("No valid off-diagonal harm labels are available")
    safe_harm = torch.where(finite, harm.float(), torch.zeros_like(harm.float()))
    safe_harm = safe_harm.clamp_min(0.0)
    row_max = safe_harm.masked_fill(~valid, 0.0).amax(dim=1, keepdim=True).clamp_min(1e-8)
    target = (safe_harm / row_max).clamp(0.0, 1.0)
    probabilities = torch.sigmoid(edge_logits)
    regression = F.smooth_l1_loss(probabilities[valid], target[valid])

    ranking_terms = []
    for source in range(count):
        source_valid = valid[source]
        positives = source_valid & (safe_harm[source] > positive_threshold)
        negatives = source_valid & ~positives
        if positives.any() and negatives.any():
            positive_logits = edge_logits[source][positives]
            negative_logits = edge_logits[source][negatives]
            ranking_terms.append(F.relu(
                ranking_margin - positive_logits[:, None] + negative_logits[None, :]
            ).mean())
    if not ranking_terms:
        return regression
    return regression + float(ranking_weight) * torch.stack(ranking_terms).mean()


def select_gnn_neighbors(class_ids: Sequence[int], edge_prob: torch.Tensor,
                         source_class: int, k: int = 5,
                         min_score: float = 0.0) -> List[Tuple[int, float]]:
    """Select top-k directed GNN neighbors for one source class."""
    if edge_prob.ndim != 2 or edge_prob.shape[0] != edge_prob.shape[1]:
        raise ValueError("edge_prob must be a square matrix")
    if len(class_ids) != edge_prob.shape[0]:
        raise ValueError("class_ids and edge_prob must have matching lengths")
    try:
        source_index = list(class_ids).index(int(source_class))
    except ValueError as exc:
        raise KeyError(f"Unknown source class {source_class}") from exc
    values = []
    for index, class_id in enumerate(class_ids):
        if index == source_index:
            continue
        score = float(edge_prob[source_index, index].detach().cpu())
        if score > min_score:
            values.append((int(class_id), score))
    values.sort(key=lambda item: (-item[1], item[0]))
    return values[:max(0, int(k))]


def fit_interference_gnn(model: ClassInterferenceGNN,
                         stages: Sequence[Mapping[str, torch.Tensor]],
                         epochs: int = 200, lr: float = 1e-3,
                         weight_decay: float = 1e-4, grad_clip: float = 1.0,
                         ranking_margin: float = 0.25,
                         ranking_weight: float = 1.0
                         ) -> List[float]:
    """Fit on historical stage artifacts and return mean loss per epoch."""
    if not stages:
        raise ValueError("At least one historical stage is required")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    device = next(model.parameters()).device
    history = []
    for _ in range(int(epochs)):
        losses = []
        for stage in stages:
            features = stage["features"].to(device)
            harm = stage["harm"].to(device)
            valid_mask = stage.get("valid_mask")
            if valid_mask is not None:
                valid_mask = valid_mask.to(device)
            output = model(features)
            loss = harm_prediction_loss(
                output["edge_logits"], harm, valid_mask,
                ranking_margin=ranking_margin, ranking_weight=ranking_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite GNN harm-prediction loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(sum(losses) / len(losses))
    return history


def save_gnn_checkpoint(model: ClassInterferenceGNN, path: str | Path,
                        extra: Mapping[str, object] | None = None) -> None:
    """Save a versioned GNN checkpoint with architecture metadata."""
    payload = {
        "format_version": ClassInterferenceGNN.format_version,
        "config": model.config(),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = dict(extra)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_gnn_checkpoint(path: str | Path, device: torch.device | str = "cpu"
                        ) -> Tuple[ClassInterferenceGNN, Dict[str, object]]:
    """Load a versioned GNN checkpoint and return ``(model, metadata)``."""
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if not isinstance(payload, Mapping) or "config" not in payload or "state_dict" not in payload:
        raise ValueError("Invalid GNN checkpoint: expected config and state_dict")
    model = ClassInterferenceGNN(**dict(payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, dict(payload.get("extra", {}))
