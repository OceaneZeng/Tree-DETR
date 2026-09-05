import json
import tempfile
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from engine import _coco_ap50_for_categories
from datasets.samplers import ReplayBalancedSampler
from models.graph_local.distillation import old_class_distillation_losses
from models.graph_local.gnn import ClassInterferenceGNN, save_gnn_checkpoint
from models.graph_local.lora import (freeze_for_class_ids, inject_decoder_lora,
                                     merge_decoder_lora)
from models.graph_local.losses import local_margin_loss, projection_loss
from models.graph_local.protection import build_off_neighborhood_basis
from models.graph_local.pseudo_labels import select_teacher_pseudo_labels
from models.graph_local.replay import build_increment_annotation
from tools.owod.calibrate_interference_gnn import resolve_gnn_ablation, source_masks
from tools.owod.run_graph_local_increment import (random_neighbors,
                                                   build_main_command,
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
        class_ids, scores, new_ids=[10, 11], old_ids=[1, 2], k=2,
        aggregation="max")

    assert selected == [1, 2]
    assert details["selection_scope"] == "stage_top_k_old_classes"
    assert details["selected_k"] == 2
    assert set(selected).issubset({1, 2})


def test_top_mean_aggregation_is_robust_to_one_source_outlier():
    scores = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0, 0.0],
    ])
    selected, details = rank_stage_old_classes(
        [1, 2, 10, 11, 12], scores, new_ids=[10, 11, 12], old_ids=[1, 2],
        k=1, aggregation="top_mean", aggregation_top_n=3)
    assert selected == [2]
    assert details["aggregation"] == "mean_top_3_over_new_classes"


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


def test_retention_options_are_forwarded_to_detector_command():
    args = get_parser().parse_args([
        "--coco-path", "coco", "--manifest", "manifest.json", "--stage", "1",
        "--checkpoint", "base.pth", "--output-dir", "out",
        "--replay-sampling-fraction", "0.1", "--old-class-distillation",
        "--lr", "5e-5", "--lr-backbone", "5e-6",
    ])
    command = build_main_command(
        args, Path("train.json"), Path("val.json"), Path("arm"),
        active_ids=[1, 2, 10], old_ids=[1, 2], reset_classifier=False,
        selected_ids=[2], off_basis_path=Path("basis.pt"))
    joined = " ".join(command)
    assert "--replay-sampling-fraction 0.1" in joined
    assert "--old-class-distillation" in joined
    assert "--distill-old-class-ids 1 2" in joined
    assert "--lr 5e-05" in joined
    assert "--lr_backbone 5e-06" in joined
    assert "--neighbor-scoped-lora" in joined
    assert "--trainable-class-ids 10" in joined
    assert "--graph-local-class-ids 10 2" in joined
    assert "--off-neighborhood-basis basis.pt" in joined
    assert "--teacher-completion" in joined
    assert "--teacher-old-class-ids 1 2" in joined


def test_internal_gnn_ablations_disable_exactly_one_component():
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
            "official_annotations": True,
            "source_reference": "official-fixture-v1",
            "stages": records,
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


def test_adaptive_replay_uses_base_and_risk_quotas_and_tags_images():
    categories = [{"id": value, "name": str(value)} for value in (1, 2, 10)]
    old_images = [{"id": value, "file_name": f"{value}.jpg"} for value in range(1, 9)]
    base = {
        "images": old_images,
        "annotations": [
            {"id": value, "image_id": value, "category_id": 1 if value <= 4 else 2,
             "bbox": [0, 0, 2, 2]}
            for value in range(1, 9)
        ],
        "categories": categories,
    }
    new = {
        "images": [{"id": 20, "file_name": "20.jpg"}],
        "annotations": [{"id": 20, "image_id": 20, "category_id": 10,
                         "bbox": [0, 0, 2, 2]}],
        "categories": categories,
    }
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "adaptive.json"
        info = build_increment_annotation(
            new, base, [1, 2], 0, output, class_quotas={1: 3, 2: 1})
        combined = json.loads(output.read_text(encoding="utf-8"))

    assert info["class_quotas"] == {"1": 3, "2": 1}
    assert info["replay_images"] == 4
    assert sum(bool(image["owod_replay"]) for image in combined["images"]) == 4


def test_balanced_replay_sampler_keeps_fraction_on_each_rank():
    dataset = list(range(100))
    replay = list(range(10))
    rank0 = ReplayBalancedSampler(dataset, replay, 0.25, num_replicas=2, rank=0, seed=42)
    rank1 = ReplayBalancedSampler(dataset, replay, 0.25, num_replicas=2, rank=1, seed=42)
    for sampler in (rank0, rank1):
        indices = list(iter(sampler))
        assert len(indices) == 50
        assert sum(index in replay for index in indices) == 12


def test_old_class_distillation_is_zero_at_initialization_and_detects_drift():
    teacher = {
        "pred_logits": torch.tensor([[[2.0, -1.0], [0.5, -2.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]]),
    }
    student = {key: value.clone().requires_grad_(True) for key, value in teacher.items()}
    equal, count = old_class_distillation_losses(
        student, teacher, [0], score_threshold=0.0, max_queries_per_image=2)
    assert count == 2
    assert abs(float(equal["loss_distill_cls"])) < 1e-6
    assert abs(float(equal["loss_distill_bbox"])) < 1e-6

    shifted = {key: value.clone().requires_grad_(True) for key, value in teacher.items()}
    shifted["pred_logits"] = (shifted["pred_logits"] + 1.0).detach().requires_grad_(True)
    shifted["pred_boxes"] = (shifted["pred_boxes"] + 0.1).detach().requires_grad_(True)
    drift, _ = old_class_distillation_losses(
        shifted, teacher, [0], score_threshold=0.0, max_queries_per_image=2)
    total = drift["loss_distill_cls"] + drift["loss_distill_bbox"]
    assert float(total) > 0
    total.backward()
    assert shifted["pred_logits"].grad is not None


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 6)
        self.linear2 = nn.Linear(6, 4)

    def forward(self, value):
        return self.linear2(torch.relu(self.linear1(value)))


class _TinyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.decoder = nn.Module()
        self.transformer.decoder.layers = nn.ModuleList(
            [_TinyDecoderLayer(), _TinyDecoderLayer()])
        head = nn.Linear(4, 5)
        self.class_embed = nn.ModuleList([head, head])

    def forward(self, value):
        for layer in self.transformer.decoder.layers:
            value = layer(value)
        return self.class_embed[-1](value)


def test_local_margin_is_zero_when_margin_is_met_and_positive_otherwise():
    def matcher(_outputs, _targets):
        return [(torch.tensor([0]), torch.tensor([0]))]

    targets = [{"labels": torch.tensor([1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]])}]
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    good = {"pred_logits": torch.tensor([[[0.0, 3.0, 1.0]]]), "pred_boxes": boxes}
    bad = {"pred_logits": torch.tensor([[[0.0, 1.2, 1.0]]]), "pred_boxes": boxes}
    assert float(local_margin_loss(good, targets, matcher, [1, 2], margin=1.0)) == 0.0
    assert float(local_margin_loss(bad, targets, matcher, [1, 2], margin=1.0)) > 0.0


def test_off_neighborhood_basis_is_orthonormal_and_projection_has_gradients():
    sketches = {
        1: torch.arange(1, 9, dtype=torch.float32),
        2: torch.arange(8, 0, -1, dtype=torch.float32),
        10: torch.ones(8),
    }
    basis = build_off_neighborhood_basis(sketches, excluded_classes=[10], max_rank=2)
    assert basis.shape == (8, 2)
    assert torch.allclose(basis.t() @ basis, torch.eye(2), atol=1e-5)
    delta = torch.randn(8, requires_grad=True)
    loss = projection_loss(delta, basis)
    loss.backward()
    assert float(loss) >= 0.0
    assert delta.grad is not None and float(delta.grad.abs().sum()) > 0.0


def test_teacher_completion_filters_ground_truth_overlap_and_adds_old_boxes():
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -8.0, -8.0], [-8.0, 8.0, -8.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.1, 0.1, 0.1, 0.1]]]),
    }
    targets = [{
        "labels": torch.tensor([2]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        "area": torch.tensor([0.04]),
        "iscrowd": torch.tensor([0]),
        "size": torch.tensor([100, 100]),
    }]
    completed, counts = select_teacher_pseudo_labels(
        outputs, targets, old_class_ids=[0, 1], score_threshold=0.5,
        ground_truth_iou=0.5)
    assert counts == [1]
    assert completed[0]["labels"].tolist() == [2, 1]
    assert completed[0]["boxes"].shape[0] == 2


def test_teacher_completion_keeps_overlapping_boxes_from_different_classes():
    outputs = {
        "pred_logits": torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]]),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.5, 0.5, 0.2, 0.2]]]),
    }
    targets = [{"labels": torch.empty(0, dtype=torch.long),
                "boxes": torch.empty(0, 4)}]
    completed, counts = select_teacher_pseudo_labels(
        outputs, targets, old_class_ids=[0, 1], score_threshold=0.5)
    assert counts == [2]
    assert completed[0]["labels"].tolist() == [0, 1]


def test_only_lora_and_current_classifier_rows_update_and_merge_is_equivalent():
    torch.manual_seed(0)
    model = _TinyDetector()
    inject_decoder_lora(model, rank=2, last_n=1)
    handles, parameters = freeze_for_class_ids(model, [3, 4])
    old_head = model.class_embed[0].weight[:3].detach().clone()
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    optimizer = torch.optim.AdamW(parameters, lr=0.1, weight_decay=0.0)
    loss = model(torch.randn(3, 4)).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert torch.equal(model.class_embed[0].weight[:3], old_head)
    for name, before in frozen.items():
        assert torch.equal(dict(model.named_parameters())[name], before)

    for module in model.modules():
        if hasattr(module, "lora_b"):
            nn.init.normal_(module.lora_b)
    inputs = torch.randn(2, 4)
    before_merge = model(inputs).detach()
    assert merge_decoder_lora(model, last_n=1) == 2
    after_merge = model(inputs).detach()
    assert torch.allclose(before_merge, after_merge, atol=1e-5, rtol=1e-5)
    for handle in handles:
        handle.remove()
