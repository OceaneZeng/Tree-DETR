# IOD data and diagnostics

The old IOD baseline launchers were removed. This directory now contains only
reusable data preparation and graph-diagnostic utilities:

- `coco_incremental.py` and `build_coco_protocols.sh` build reproducible
  disjoint-image COCO splits.
- `build_replay_annotation.py` creates a fixed-budget replay annotation.
- `estimate_conflict_risk.py` and `analyze_risk_drop.py` measure the graph
  hypothesis without claiming an OWOD result.
- `validate_split.py` checks split integrity.

Run the primary OWOD baselines from `tools/owod/run_baseline.py`. Each run
writes `train.log`, `log.txt`, `run_config.json`, and `command.txt` under its
output directory. The OWOD runner accepts `vanilla_d_detr`, `ore_star`,
`ow_detr`, `prob`, and `oracle`.
