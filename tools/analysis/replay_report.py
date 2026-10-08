"""Historical log validation only; no data resampling or experiment settings."""
import json
import math
from pathlib import Path
from tools.owod.protocol import file_sha256 as file_hash

METHODS = ('glca', 'uniform', 'uniform_teacher')
LABELS = {'glca': 'GLCA-DETR', 'uniform': 'Uniform Replay',
          'uniform_teacher': 'Uniform Replay + teacher supervision'}
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / 'exps/history/replay_comparison'

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8', newline='\n')
    temporary.replace(path)

def metric_rows(path):
    if not path.is_file():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                print(f"Pending partial final metric line: {path}")
                break
            raise
        if "test_owod_current_ap50" in row:
            rows[int(row["epoch"])] = row
    return rows

def final_metrics_valid(row, plan):
    keys = ['previous_ap50', 'current_ap50', 'known_ap50']
    if plan['unknown_gt'] > 0:
        keys += ['u_recall', 'h_score']
    return all(isinstance(row.get('test_owod_' + k), (int, float))
               and math.isfinite(row['test_owod_' + k]) and 0 <= row['test_owod_' + k] <= 1 for k in keys)

def validate_run_settings(directory, plan):
    path = directory / 'graph/run_config.json'
    if not path.is_file():
        raise ValueError(f'Missing training run_config: {path}')
    actual = read_json(path)
    for key, expected in plan['settings'].items():
        if actual.get(key) != expected:
            raise ValueError(f'Training setting differs from plan: {path}: {key}')
    # Use recorded absolute paths so copied logs can be reported on another OS.
    expected_annotation = plan['directory'].replace('\\', '/').rstrip('/') + '/train.json'
    if (actual['train_ann'].replace('\\', '/') != expected_annotation
            or actual.get('world_size', 1) != len(plan['gpus'].split(','))):
        raise ValueError(f'Training annotation/world_size differs from plan: {path}')
