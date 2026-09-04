import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

from engine import _coco_ap50_for_categories
from models.graph_local.replay import build_increment_annotation
from tools.owod.run_graph_local_increment import (build_prototype_similarity_matrix,
                                                   random_neighbors,
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


def test_cosine_selection_ranks_old_targets_before_top_k():
    # The strongest per-source neighbor is another new class. The old bug took
    # top-k first and then discarded that new class, returning no replay class.
    features = {
        1: torch.tensor([0.8, 0.6]),
        2: torch.tensor([0.0, 1.0]),
        10: torch.tensor([1.0, 0.0]),
        11: torch.tensor([0.99, 0.1]),
    }
    args = SimpleNamespace(control="graph", graph_estimator="cosine", graph_k=1)

    selected, details = select_neighbors(args, features, new_ids=[10, 11], old_ids=[1, 2])

    assert selected == [1]
    assert details["requested_k"] == 1
    assert details["selected_k"] == 1


def test_prototype_graph_uses_positive_similarity_not_negative_gradient_conflict():
    class_ids, scores = build_prototype_similarity_matrix({
        1: torch.tensor([1.0, 0.0]),
        2: torch.tensor([0.8, 0.6]),
        3: torch.tensor([-1.0, 0.0]),
    })
    index = {class_id: position for position, class_id in enumerate(class_ids)}
    assert torch.isclose(scores[index[1], index[2]], torch.tensor(0.8))
    assert scores[index[1], index[3]] == 0.0


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
        manifest.write_text(json.dumps({"stages": [{}, {}]}), encoding="utf-8")
        (root / "stage_0").mkdir()
        (root / "stage_1").mkdir()
        previous = root / "stage_0" / "instances_train2017.json"
        current = root / "stage_1" / "instances_increment_train2017.json"
        validation = root / "stage_1" / "instances_val2017_full.json"
        for path in (previous, current, validation):
            path.write_text("{}", encoding="utf-8")

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
