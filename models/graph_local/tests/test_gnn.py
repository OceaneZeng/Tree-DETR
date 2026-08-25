from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from models.graph_local.gnn import (
    ClassInterferenceGNN,
    fit_interference_gnn,
    load_gnn_checkpoint,
    save_gnn_checkpoint,
    select_gnn_neighbors,
)


def _synthetic_stage():
    features = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    harm = torch.tensor([
        [0.0, 0.9, 0.1],
        [0.1, 0.0, 0.9],
        [0.9, 0.1, 0.0],
    ])
    valid_mask = torch.ones(3, 3, dtype=torch.bool)
    valid_mask.fill_diagonal_(False)
    return {"features": features, "harm": harm, "valid_mask": valid_mask}


def test_gnn_masks_self_edges_and_returns_directed_scores():
    torch.manual_seed(4)
    model = ClassInterferenceGNN(3, hidden_dim=8, message_steps=1)
    output = model(_synthetic_stage()["features"])
    assert output["edge_logits"].shape == (3, 3)
    assert torch.isneginf(output["edge_logits"].diag()).all()
    assert torch.all(output["edge_prob"].diag() == 0)


def test_gnn_learns_directional_harm_ordering():
    torch.manual_seed(7)
    stage = _synthetic_stage()
    model = ClassInterferenceGNN(3, hidden_dim=16, message_steps=1)
    history = fit_interference_gnn(model, [stage], epochs=50, lr=1e-2)
    neighbors = select_gnn_neighbors(
        [0, 1, 2], model(stage["features"])["edge_prob"], source_class=2, k=1)
    assert history[-1] < history[0]
    assert neighbors[0][0] == 0


def test_gnn_checkpoint_round_trip():
    torch.manual_seed(9)
    model = ClassInterferenceGNN(3, hidden_dim=8, message_steps=1)
    features = _synthetic_stage()["features"]
    before = model(features)["edge_logits"]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "gnn.pt"
        save_gnn_checkpoint(model, path, extra={"source": "test"})
        restored, metadata = load_gnn_checkpoint(path)
    after = restored(features)["edge_logits"]
    assert metadata["source"] == "test"
    assert torch.equal(before, after)

