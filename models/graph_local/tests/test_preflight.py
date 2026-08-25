from models.graph_local.preflight import run_preflight


def test_synthetic_preflight_exercises_every_module():
    report = run_preflight()
    assert report["all_passed"]
    assert set(report["gates"]) == {
        "G1_graph_predicts_harm",
        "G2_trainable_gnn_predicts_harm",
        "R1_balanced_local_replay",
        "P1_pseudo_label_completion",
        "M1_local_margin_signal",
        "O1_off_neighborhood_projection",
        "L1_low_rank_adaptation",
    }
