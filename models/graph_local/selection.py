"""Graph neighborhood selection, independent of experiment data settings."""
from __future__ import annotations
import random
import torch
from models.graph_local.gnn import compress_gradient_sketches, load_gnn_checkpoint

def rank_stage_old_classes(class_ids: list[int], edge_scores: torch.Tensor,
                           new_ids: list[int], old_ids: list[int], k: int,
                           min_score: float = 0.0, aggregation: str = "top_mean",
                           aggregation_top_n: int = 3) -> tuple[list[int], dict]:
    """Select a stage-level top-k over old classes only.

    A target's risk aggregates directed edges from current-stage classes. This
    treats ``k`` as the total high-risk neighborhood size, rather than taking k
    neighbors from every source and filtering afterwards.
    """
    if edge_scores.ndim != 2 or edge_scores.shape[0] != edge_scores.shape[1]:
        raise ValueError("edge_scores must be a square matrix")
    if len(class_ids) != edge_scores.shape[0]:
        raise ValueError("class_ids and edge_scores must have matching lengths")

    index = {int(class_id): position for position, class_id in enumerate(class_ids)}
    missing = sorted((set(new_ids) | set(old_ids)) - set(index))
    if missing:
        raise KeyError(f"Graph scores are missing classes: {missing}")

    target_ids = list(dict.fromkeys(int(class_id) for class_id in old_ids))
    source_scores: dict[str, list[list[float | int]]] = {}
    target_scores = {target: [] for target in target_ids}
    detached = edge_scores.detach().float().cpu()
    for source in new_ids:
        ranked = []
        for target in target_ids:
            score = float(detached[index[int(source)], index[target]])
            target_scores[target].append(score)
            ranked.append([target, score])
        ranked.sort(key=lambda item: (-item[1], item[0]))
        source_scores[str(int(source))] = ranked

    if aggregation not in {"max", "top_mean"}:
        raise ValueError(f"unsupported graph aggregation: {aggregation}")
    top_n = max(1, int(aggregation_top_n))
    aggregate = {}
    for target, values in target_scores.items():
        ordered_values = sorted(values, reverse=True)
        if aggregation == "max":
            aggregate[target] = ordered_values[0]
        else:
            used = ordered_values[:top_n]
            aggregate[target] = sum(used) / len(used)
    aggregate_ranking = sorted(
        ([target, score] for target, score in aggregate.items() if score > min_score),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [int(target) for target, _score in aggregate_ranking[:max(0, int(k))]]
    return selected, {
        "selection_scope": "stage_top_k_old_classes",
        "aggregation": ("max_over_new_classes" if aggregation == "max"
                        else f"mean_top_{top_n}_over_new_classes"),
        "requested_k": int(k),
        "selected_k": len(selected),
        "min_score": float(min_score),
        "candidate_old_classes": target_ids,
        "aggregated_scores": aggregate_ranking,
        "scores": source_scores,
    }

def select_neighbors(args, features: dict[int, torch.Tensor], new_ids: list[int],
                     old_ids: list[int]) -> tuple[list[int], dict]:
    if not old_ids or not new_ids or args.control == "global":
        return (list(old_ids) if args.control == "global" else []), {"estimator": "none"}
    gnn, metadata = load_gnn_checkpoint(args.gnn_checkpoint, device="cpu")
    if metadata.get("supervision") != "empirical_train_loss_increase":
        raise ValueError(
            "The GNN checkpoint is not empirically supervised. Run "
            "tools/owod/calibrate_interference_gnn.py on a completed earlier stage."
        )
    if not metadata.get("production_ready", False):
        raise ValueError(
            "The GNN checkpoint came from a partial smoke calibration. "
            "Run calibration without --source-limit before the detector experiment."
        )
    ordered, compressed = compress_gradient_sketches(features, output_dim=gnn.input_dim)
    with torch.no_grad():
        edge_prob = gnn(compressed)["edge_prob"]
    selected, details = rank_stage_old_classes(
        ordered, edge_prob, new_ids, old_ids, args.graph_k, args.gnn_min_score,
        getattr(args, "graph_aggregation", "top_mean"),
        getattr(args, "graph_aggregation_top_n", 3))
    return selected, {"estimator": "trainable_class_interference_gnn",
                      "checkpoint": str(args.gnn_checkpoint.resolve()),
                      "feature_source": "detector_decoder_ffn_gradient_sketch",
                      "score_semantics": "predicted_directed_train_loss_increase",
                      "metadata": metadata, **details}

def random_neighbors(old_ids: list[int], graph_k: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(old_ids, min(len(old_ids), max(0, graph_k))))
