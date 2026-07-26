#!/usr/bin/env bash
# ------------------------------------------------------------------------
# Tree-DETR : download COCO 2017 + a pretrained Deformable-DETR checkpoint
# ------------------------------------------------------------------------
# Lays COCO out exactly as datasets/coco.py expects:
#
#   <root>/
#     train2017/                       *.jpg   (only with --full)
#     val2017/                         *.jpg
#     annotations/instances_train2017.json
#                 instances_val2017.json
#
# and pulls a Deformable-DETR checkpoint (Google Drive, via gdown) for the
# Stage-0 feature extraction / warm-start.
#
# Default is the LIGHTWEIGHT set: val2017 + annotations (~1.25 GB) only.
# Add --full to also fetch train2017 (~18 GB).
#
# Usage:
#   bash tools/download_data.sh                      # val2017 + ann + main ckpt
#   bash tools/download_data.sh --full               # + train2017 (18 GB)
#   bash tools/download_data.sh --root ./data/coco   # custom dataset root
#   bash tools/download_data.sh --ckpt refine        # a different checkpoint
#   bash tools/download_data.sh --skip-ckpt          # dataset only
#   bash tools/download_data.sh --skip-coco          # checkpoint only
# ------------------------------------------------------------------------
set -euo pipefail

# ---- defaults ----------------------------------------------------------
ROOT="./data/coco"          # dataset root (pass to main.py as --coco_path)
CKPT_DIR="./checkpoints"    # where the .pth lands
CKPT="main"                 # which pretrained model (see CKPT_IDS below)
WANT_FULL=0                 # 1 => also download train2017 (18 GB)
SKIP_COCO=0
SKIP_CKPT=0

# Deformable-DETR model zoo (Google Drive file ids, from README.md) -------
declare -A CKPT_IDS=(
  [single_scale]="1WEjQ9_FgfI5sw5OZZ4ix-OKk-IJ_-SDU"       # single scale,      AP 39.4
  [single_scale_dc5]="1m_TgMjzH7D44fbA-c_jiBZ-xf-odxGdk"   # single scale DC5,  AP 41.5
  [main]="1nDWZWHuRwtwGden77NLM9JoWe-YisJnA"               # multi-scale (DEFAULT), AP 44.5
  [refine]="1JYKyRYzUH7uo9eVfDaVCiaIGZb5YTCuI"             # + iter bbox refine, AP 46.2  (needs --with_box_refine)
  [two_stage]="15I03A7hNTpwuLNdfuEmW9_taZMNVssEp"          # ++ two-stage,       AP 46.9  (needs --with_box_refine --two_stage)
)

# ---- arg parsing -------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)      ROOT="$2"; shift 2 ;;
    --ckpt-dir)  CKPT_DIR="$2"; shift 2 ;;
    --ckpt)      CKPT="$2"; shift 2 ;;
    --full)      WANT_FULL=1; shift ;;
    --val-only)  WANT_FULL=0; shift ;;
    --skip-coco) SKIP_COCO=1; shift ;;
    --skip-ckpt) SKIP_CKPT=1; shift ;;
    -h|--help)
      sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---- helpers -----------------------------------------------------------
have() { command -v "$1" >/dev/null 2>&1; }

fetch() {   # fetch <url> <out.zip> ; resumable, skips if already unpacked
  local url="$1" out="$2"
  if [[ -f "$out" ]]; then
    echo "  [skip] $out already downloaded"
  elif have wget; then
    wget -c -O "$out" "$url"
  elif have curl; then
    curl -L -C - -o "$out" "$url"
  else
    echo "ERROR: need wget or curl" >&2; exit 1
  fi
}

unzip_to() {  # unzip_to <zip> <dest_dir> <sentinel_path>
  local zip="$1" dest="$2" sentinel="$3"
  if [[ -e "$sentinel" ]]; then
    echo "  [skip] $sentinel already present"
  else
    echo "  unzip $zip -> $dest"
    mkdir -p "$dest"
    unzip -q -o "$zip" -d "$dest"
  fi
}

# ========================================================================
# 1) COCO 2017
# ========================================================================
if [[ "$SKIP_COCO" -eq 0 ]]; then
  echo "== COCO 2017 -> $ROOT =="
  mkdir -p "$ROOT"
  TMP="$ROOT/_zips"; mkdir -p "$TMP"

  echo "-- val2017 images (~1 GB) --"
  fetch "http://images.cocodataset.org/zips/val2017.zip" "$TMP/val2017.zip"
  unzip_to "$TMP/val2017.zip" "$ROOT" "$ROOT/val2017"

  echo "-- annotations (~250 MB) --"
  fetch "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" "$TMP/ann.zip"
  unzip_to "$TMP/ann.zip" "$ROOT" "$ROOT/annotations/instances_val2017.json"

  if [[ "$WANT_FULL" -eq 1 ]]; then
    echo "-- train2017 images (~18 GB) --"
    fetch "http://images.cocodataset.org/zips/train2017.zip" "$TMP/train2017.zip"
    unzip_to "$TMP/train2017.zip" "$ROOT" "$ROOT/train2017"
  else
    echo "  [note] train2017 skipped (use --full to fetch it, ~18 GB)."
    echo "         For a training smoke test you can symlink val as train:"
    echo "           ln -sfn \"\$(readlink -f $ROOT/val2017)\" $ROOT/train2017"
    echo "           cp $ROOT/annotations/instances_val2017.json $ROOT/annotations/instances_train2017.json"
  fi
  echo "COCO ready under: $ROOT"
else
  echo "== COCO download skipped (--skip-coco) =="
fi

# ========================================================================
# 2) Deformable-DETR checkpoint (Google Drive via gdown)
# ========================================================================
if [[ "$SKIP_CKPT" -eq 0 ]]; then
  echo "== Deformable-DETR checkpoint ($CKPT) -> $CKPT_DIR =="
  ID="${CKPT_IDS[$CKPT]:-}"
  if [[ -z "$ID" ]]; then
    echo "ERROR: unknown --ckpt '$CKPT'. Options: ${!CKPT_IDS[*]}" >&2; exit 2
  fi
  mkdir -p "$CKPT_DIR"
  OUT="$CKPT_DIR/r50_deformable_detr_${CKPT}.pth"

  if [[ -f "$OUT" ]]; then
    echo "  [skip] $OUT already present"
  else
    if ! have gdown; then
      echo "  installing gdown (Google Drive downloader)..."
      pip install -q gdown
    fi
    # uc?id form handles the large-file confirm token across gdown versions
    gdown "https://drive.google.com/uc?id=${ID}" -O "$OUT"
  fi
  echo "checkpoint ready: $OUT"
  echo "  use with:  --resume $OUT"
  case "$CKPT" in
    refine)    echo "  NOTE: this model needs  --with_box_refine" ;;
    two_stage) echo "  NOTE: this model needs  --with_box_refine --two_stage" ;;
  esac
else
  echo "== checkpoint download skipped (--skip-ckpt) =="
fi

echo
echo "Done."
