from pathlib import Path
from types import SimpleNamespace

import pytest

from main import get_args_parser
from tools.owod.run_d2_continuation import build_command, main
from tools.owod.run_graph_local_increment import get_parser, build_main_command


def fixture():
    d2 = vars(get_args_parser().parse_args([]))
    d2.update(owod_stage=1, num_classes=91, epochs=20, lr=1e-4, lr_drop=15,
              teacher_completion=True, coco_path='/data/coco', owod_manifest='/data/manifest.json',
              replay_sampling_fraction=.1, eval_interval=5, seed=42)
    source = dict(exemplars_per_class=None, base_exemplars_per_class=10,
                  risk_extra_exemplars_per_class=40, graph_aggregation='top_mean',
                  graph_aggregation_top_n=3)
    graph = dict(graph=dict(checkpoint='/gnn/stage0.pt', requested_k=5, min_score=0),
                 max_images_per_class=12, last_decoder_layers=2)
    args = SimpleNamespace(stage=2, output_dir=Path('/output'), gpus='0,1', master_port=29583, resume=False)
    return args, d2, source, graph


def test_continuation_keeps_d2_training_and_gnn_recipe():
    args, d2, source, graph = fixture()
    command = build_command(args, d2, source, graph, Path('/d2/graph'))
    parsed = get_parser().parse_args(command[2:])
    assert parsed.checkpoint == Path('/d2/graph/checkpoint.pth')
    assert parsed.lr_backbone == 2e-5
    assert parsed.lr == 1e-4 and parsed.lr_drop == 15
    assert not parsed.neighbor_scoped_lora
    assert parsed.off_projection_coef == parsed.local_margin_coef == 0
    assert parsed.teacher_completion and not parsed.old_class_distillation
    assert parsed.base_exemplars_per_class == 10 and parsed.risk_extra_exemplars_per_class == 40
    detector_command = build_main_command(parsed, Path('train.json'), Path('val.json'),
                                         Path('out/graph'), [1, 2, 3], [1, 2], False, [1])
    assert '--teacher-old-class-ids' in detector_command
    assert '--neighbor-scoped-lora' not in detector_command
    assert '--reset-classifier' not in detector_command
    assert '--lr_backbone' in detector_command


def test_stage3_loads_stage2_and_resume_loads_current_optimizer():
    args, d2, source, graph = fixture()
    args.stage, args.resume = 3, True
    command = build_command(args, d2, source, graph, Path('/output/stage_2/graph'))
    parsed = get_parser().parse_args(command[2:])
    assert parsed.checkpoint == Path('/output/stage_2/graph/checkpoint.pth')
    assert parsed.resume == Path('/output/stage_3/graph/checkpoint.pth')
    d2['lr_backbone'] = 0
    with pytest.raises(ValueError, match='Source must be D2'):
        build_command(args, d2, source, graph, Path('/d4/graph'))


def test_summarize_does_not_require_remote_files(tmp_path, capsys):
    assert main(['--summarize', '--d2-dir', str(tmp_path / 'd2'), '--output-dir', str(tmp_path / 'out')]) == 0
    assert 'stage epoch previous current known H status' in capsys.readouterr().out
