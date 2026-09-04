import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from engine import _coco_ap50_for_categories
from models.graph_local.gnn import ClassInterferenceGNN, save_gnn_checkpoint
from models.graph_local.replay import build_increment_annotation
from tools.owod.calibrate_interference_gnn import resolve_gnn_ablation, source_masks
from tools.owod.run_graph_local_increment import (random_neighbors,
                                                   get_parser,
                                                   rank_stage_old_classes,
                                                   select_neighbors,
                                                   stage_paths)


def test_stage_ranking_only_selects_old_classes_and_treats_k_as_total():
    class_ids = [1, 2, 10, 11]
    scores = torch.tensor([
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.9, 0.2, 0.0, 1.0],
        [0.1, 0.8, 1.0, 0.0],
    ])

    selected, details = rank_stage_old_classes(
        class_ids, scores, new_ids=[10, 11], old_ids=[1, 2], k=2)

    assert selected == [1, 2]
    assert details["selection_scope"] == "stage_top_k_old_classes"
    assert details["selected_k"] == 2
    assert set(selected).issubset({1, 2})


def test_random_control_can_match_graph_neighborhood_size():
    selected = random_neighbors([1, 2, 3, 4], graph_k=3, seed=42)
    assert len(selected) == 3
    assert set(selected).issubset({1, 2, 3, 4})


def test_resume_defaults_to_none_instead_of_current_directory():
    parser = get_parser()
    args = parser.parse_args([
        "--coco-path", "coco",
        "--manifest", "manifest.json",
        "--stage", "1",
        "--checkpoint", "checkpoint.pth",
        "--output-dir", "output",
    ])
    assert args.resume is None
    assert args.gnn_checkpoint is None


def test_primary_ablations_disable_exactly_one_gnn_component():
    base = {
        "gnn_message_steps": 2,
        "gnn_ranking_margin": 0.25,
    }
    expected = {
        "full": (True, 2, 1.0),
        "no_node_encoder": (False, 2, 1.0),
        "no_message_passing": (True, 0, 1.0),
        "no_ranking_loss": (True, 2, 0.0),
    }
    for name, values in expected.items():
        config = resolve_gnn_ablation(SimpleNamespace(gnn_ablation=name, **base))
        assert (config["use_node_encoder"], config["message_steps"],
                config["ranking_weight"]) == values


def test_gnn_checkpoint_must_have_empirical_harm_supervision():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "legacy_gnn.pt"
        save_gnn_checkpoint(ClassInterferenceGNN(input_dim=2), checkpoint,
                            extra={"supervision": "cosine_proxy"})
        args = SimpleNamespace(
            control="graph", graph_k=1,
            gnn_checkpoint=checkpoint, gnn_min_score=0.0)
        try:
            select_neighbors(args, {
                1: torch.tensor([1.0, 0.0]),
                10: torch.tensor([0.0, 1.0]),
            }, new_ids=[10], old_ids=[1])
        except ValueError as error:
            assert "not empirically supervised" in str(error)
        else:
            raise AssertionError("legacy proxy-supervised GNN checkpoint was accepted")


def test_source_holdout_masks_entire_rows_without_label_leakage():
    class_ids = [1, 2, 3, 4, 5]
    valid = ~torch.eye(len(class_ids), dtype=torch.bool)
    train, validation, held_out = source_masks(class_ids, valid, fraction=0.4, seed=42)

    held_indices = [class_ids.index(class_id) for class_id in held_out]
    assert len(held_out) == 2
    assert not train[held_indices].any()
    assert torch.equal(validation[held_indices], valid[held_indices])
    assert not (train & validation).any()
    assert torch.equal(train | validation, valid)


def test_source_holdout_ignores_unprobed_rows():
    class_ids = [1, 2, 3, 4]
    valid = torch.zeros(4, 4, dtype=torch.bool)
    valid[0, 1:] = True
    valid[2, [0, 1, 3]] = True
    train, validation, held_out = source_masks(class_ids, valid, fraction=0.5, seed=1)

    assert len(held_out) == 1
    assert held_out[0] in {1, 3}
    assert not train[1].any() and not validation[1].any()
    assert not train[3].any() and not validation[3].any()
    assert torch.equal(train | validation, valid)


def test_category_ap50_uses_requested_category_subset():
    precision = np.full((10, 101, 3, 4, 3), -1.0)
    precision[0, :, 0, 0, 2] = 0.2
    precision[0, :, 1, 0, 2] = 0.6
    precision[0, :, 2, 0, 2] = 0.9
    evaluator = SimpleNamespace(
        eval={"precision": precision},
        params=SimpleNamespace(
            catIds=[1, 2, 3],
            iouThrs=np.linspace(0.5, 0.95, 10),
            areaRngLbl=["all", "small", "medium", "large"],
            maxDets=[1, 10, 100],
        ),
    )

    assert np.isclose(_coco_ap50_for_categories(evaluator, [1, 2]), 0.4)


def test_stage_one_replay_pool_comes_from_previous_stage():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        manifest = root / "split_manifest.json"
        (root / "stage_0").mkdir()
        (root / "stage_1").mkdir()
        previous = root / "stage_0" / "instances_train2017.json"
        current = root / "stage_1" / "instances_increment_train2017.json"
        validation = root / "stage_1" / "instances_val2017_full.json"
        stage0_increment = root / "stage_0" / "instances_increment_train2017.json"
        stage0_known = root / "stage_0" / "instances_val2017.json"
        stage0_full = root / "stage_0" / "instances_val2017_full.json"
        stage1_train = root / "stage_1" / "instances_train2017.json"
        stage1_known = root / "stage_1" / "instances_val2017.json"
        for path in (previous, current, validation, stage0_increment, stage0_known,
                     stage0_full, stage1_train, stage1_known):
            path.write_text("{}", encoding="utf-8")
        records = []
        for index, paths in enumerate((
            (stage0_increment, previous, stage0_known, stage0_full),
            (current, stage1_train, stage1_known, validation),
        )):
            records.append({"index": index, "files": dict(zip(
                ("increment_train", "train", "known_val", "full_val"),
                map(str, paths)))})
        manifest.write_text(json.dumps({
            "official_annotations": True, "stages": records,
        }), encoding="utf-8")

        _manifest, _current, replay_pool, _validation = stage_paths(
            manifest, stage=1, coco_path=root)

        assert replay_pool == previous


def test_replay_merge_deduplicates_annotations_and_unions_categories():
    categories = {
        1: {"id": 1, "name": "old"},
        2: {"id": 2, "name": "new"},
    }
    image = {"id": 7, "file_name": "7.jpg"}
    new = {
        "images": [image],
        "annotations": [{"id": 20, "image_id": 7, "category_id": 2, "bbox": [0, 0, 2, 2]}],
        "categories": [categories[2]],
    }
    base = {
        "images": [image],
        "annotations": [
            {"id": 10, "image_id": 7, "category_id": 1, "bbox": [2, 2, 2, 2]},
            {"id": 20, "image_id": 7, "category_id": 2, "bbox": [0, 0, 2, 2]},
        ],
        "categories": [categories[1], categories[2]],
    }
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "combined.json"
        build_increment_annotation(new, base, [1], 1, output)
        combined = json.loads(output.read_text(encoding="utf-8"))

    assert len(combined["annotations"]) == 2
    assert {category["id"] for category in combined["categories"]} == {1, 2}
