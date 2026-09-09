#!/usr/bin/env bash
# Prepare COCO2017 and existing four-stage OWOD inputs for EW-DETR.
# This script validates an OWOD split; it never invents one.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
COCO_ROOT=""
OWOD_ROOT=""
MANIFEST=""
SYNC_FROM=""
DOWNLOAD_COCO=0
VALIDATE_ONLY=0

usage() {
  cat <<'EOF'
Prepare EW-DETR data.
  --download-coco       Download and unpack full COCO2017.
  --sync-from PATH      rsync data/coco and data/coco-owod from PATH.
  --project-root PATH   Tree-DETR checkout.
  --coco-root PATH      COCO root.
  --owod-root PATH      OWOD order root.
  --manifest PATH       Manifest path.
  --validate-only       Only validate existing inputs.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --download-coco) DOWNLOAD_COCO=1; shift ;;
    --sync-from) SYNC_FROM="$2"; shift 2 ;;
    --project-root) PROJECT_ROOT="$2"; shift 2 ;;
    --coco-root) COCO_ROOT="$2"; shift 2 ;;
    --owod-root) OWOD_ROOT="$2"; shift 2 ;;
    --manifest) MANIFEST="$2"; shift 2 ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PROJECT_ROOT="$(cd -- "$PROJECT_ROOT" && pwd -P)"
[[ -n "$COCO_ROOT" ]] || COCO_ROOT="$PROJECT_ROOT/data/coco"
[[ -n "$OWOD_ROOT" ]] || OWOD_ROOT="$PROJECT_ROOT/data/coco-owod/m-owodb/order0"
mkdir -p "$COCO_ROOT" "$OWOD_ROOT"
COCO_ROOT="$(cd -- "$COCO_ROOT" && pwd -P)"
OWOD_ROOT="$(cd -- "$OWOD_ROOT" && pwd -P)"
[[ -n "$MANIFEST" ]] || MANIFEST="$OWOD_ROOT/split_manifest_deus.invalid.json"
if [[ "$MANIFEST" != /* ]]; then MANIFEST="$PROJECT_ROOT/$MANIFEST"; fi

if [[ -n "$SYNC_FROM" && "$DOWNLOAD_COCO" -eq 1 ]]; then
  echo "ERROR: choose --sync-from or --download-coco, not both" >&2; exit 2
fi
if [[ "$VALIDATE_ONLY" -eq 0 && -z "$SYNC_FROM" && "$DOWNLOAD_COCO" -eq 0 ]]; then
  echo "ERROR: choose --sync-from, --download-coco, or --validate-only" >&2; exit 2
fi

if [[ "$DOWNLOAD_COCO" -eq 1 ]]; then
  echo "== Downloading full COCO2017 =="
  (cd "$PROJECT_ROOT" && bash tools/download_data.sh --root "$COCO_ROOT" --full --skip-ckpt)
fi

if [[ -n "$SYNC_FROM" ]]; then
  command -v rsync >/dev/null 2>&1 || { echo "ERROR: rsync is required" >&2; exit 1; }
  echo "== Synchronizing COCO data from $SYNC_FROM =="
  rsync -aP --info=progress2 "$SYNC_FROM/data/coco/" "$COCO_ROOT/"
  echo "== Synchronizing OWOD annotations from $SYNC_FROM =="
  rsync -aP --info=progress2 "$SYNC_FROM/data/coco-owod/m-owodb/order0/" "$OWOD_ROOT/"
fi

for required in "$COCO_ROOT/train2017" "$COCO_ROOT/val2017" \
  "$COCO_ROOT/annotations/instances_train2017.json" \
  "$COCO_ROOT/annotations/instances_val2017.json" "$MANIFEST"; do
  [[ -e "$required" ]] || { echo "ERROR: missing $required" >&2; exit 1; }
done

# Normalize copied manifest paths. Only path fields change; a backup is kept.
OWOD_ROOT="$OWOD_ROOT" MANIFEST="$MANIFEST" python - <<'PY'
import json, os
from pathlib import Path
manifest_path = Path(os.environ['MANIFEST']).resolve()
owod_root = Path(os.environ['OWOD_ROOT']).resolve()
payload = json.loads(manifest_path.read_text(encoding='utf-8'))
if len(payload.get('stages', [])) != 4:
    raise SystemExit('manifest must contain exactly four stages')
names = {'increment_train': ('instances_increment_train2017.json','instances_increment_only_train2017.json'),
         'train': ('instances_train2017.json',), 'known_val': ('instances_val2017.json',),
         'full_val': ('instances_val2017_full.json',)}
changed = payload.get('annotation_root') != str(owod_root)
payload['annotation_root'] = str(owod_root)
for index, stage in enumerate(payload['stages']):
    stage_dir = owod_root / f'stage_{index}'
    files = stage.setdefault('files', {})
    for key, candidates in names.items():
        found = next((stage_dir / n for n in candidates if (stage_dir / n).is_file()), None)
        if found is None: raise SystemExit(f'stage_{index}: missing {key} in {stage_dir}')
        value = str(found.resolve()); changed |= files.get(key) != value; files[key] = value
if changed:
    backup = manifest_path.with_suffix(manifest_path.suffix + '.before_path_rewrite')
    if not backup.exists(): backup.write_text(manifest_path.read_text(encoding='utf-8'), encoding='utf-8')
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Manifest paths normalized; backup: {backup}')
else: print('Manifest paths already point to this server')
PY

# Validate the four-stage protocol and every referenced image.
COCO_ROOT="$COCO_ROOT" MANIFEST="$MANIFEST" python - <<'PY'
import json, os
from pathlib import Path
from tools.owod.protocol import stage_files
coco = Path(os.environ['COCO_ROOT']).resolve(); manifest_path = Path(os.environ['MANIFEST']).resolve()
manifest = json.loads(manifest_path.read_text(encoding='utf-8')); previous = set()
for index in range(4):
    manifest, files = stage_files(manifest_path, index, allow_unverified=True)
    current = {int(v) for v in manifest['stages'][index]['classes']}
    if len(current) != 20 or current & previous: raise SystemExit(f'stage_{index}: expected 20 disjoint classes')
    increment = json.loads(files['increment_train'].read_text(encoding='utf-8'))
    train = json.loads(files['train'].read_text(encoding='utf-8'))
    full_val = json.loads(files['full_val'].read_text(encoding='utf-8'))
    if not {int(a['category_id']) for a in increment.get('annotations', [])}.issubset(current):
        raise SystemExit(f'stage_{index}: increment contains outside classes')
    train_items = {item['id']: item for item in train.get('images', [])}
    train_items.update({item['id']: item for item in increment.get('images', [])})
    for item in train_items.values():
        image = (coco / 'train2017' / item['file_name']).resolve(); image.relative_to((coco / 'train2017').resolve())
        if not image.is_file(): raise SystemExit(f'stage_{index}: missing train image {image}')
    for item in full_val.get('images', []):
        image = (coco / 'val2017' / item['file_name']).resolve(); image.relative_to((coco / 'val2017').resolve())
        if not image.is_file(): raise SystemExit(f'stage_{index}: missing val image {image}')
    print(f"stage_{index}: classes={len(current)}, train_images={len(train_items)}, val_images={len(full_val.get('images', []))}, OK")
    previous |= current
print(f"COCO train2017 JPG files: {sum(1 for _ in (coco/'train2017').glob('*.jpg'))}")
print(f"COCO val2017 JPG files: {sum(1 for _ in (coco/'val2017').glob('*.jpg'))}")
print(f"Prepared COCO root: {coco}"); print(f"Prepared OWOD manifest: {manifest_path}")
PY
