#!/usr/bin/env python
"""Import and validate official M-OWODB or S-OWODB stage annotations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.owod.protocol import PROTOCOLS, build_official_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", required=True, type=Path,
                        help="directory containing official stage_0 ... stage_3 files")
    parser.add_argument("--protocol", required=True, choices=sorted(PROTOCOLS))
    parser.add_argument("--source-reference", required=True,
                        help="official release URL, repository commit, or archive identifier")
    parser.add_argument("--output", required=True, type=Path,
                        help="destination split_manifest.json")
    parser.add_argument("--allow-image-overlap", action="store_true",
                        help="diagnostic only; resulting manifest is not paper-comparable")
    args = parser.parse_args()
    manifest = build_official_manifest(
        args.annotation_root, args.protocol, args.source_reference,
        allow_image_overlap=args.allow_image_overlap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["integrity"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
