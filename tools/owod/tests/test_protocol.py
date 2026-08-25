import json
import tempfile
import unittest
from pathlib import Path

from tools.owod.build_protocol import build


class OWODProtocolTests(unittest.TestCase):
    def test_mowodb_manifest_and_stage_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "coco"
            ann = root / "annotations"
            ann.mkdir(parents=True)
            categories = [{"id": index, "name": f"c{index}", "supercategory": "g"}
                          for index in range(1, 81)]
            images = [{"id": index, "file_name": f"{index}.jpg"} for index in range(1, 9)]
            annotations = [{"id": index, "image_id": index, "category_id": index,
                           "bbox": [0, 0, 1, 1], "area": 1, "iscrowd": 0}
                          for index in range(1, 9)]
            payload = {"images": images, "annotations": annotations, "categories": categories}
            (ann / "instances_train2017.json").write_text(json.dumps(payload), encoding="utf-8")
            (ann / "instances_val2017.json").write_text(json.dumps(payload), encoding="utf-8")
            manifest = build(root, Path(tmp) / "out", "m-owodb", "id", 42, 0.1, None)
            self.assertEqual(manifest["benchmark"], "M-OWODB")
            self.assertEqual(len(manifest["stages"]), 4)
            self.assertTrue((Path(tmp) / "out/stage_0/instances_val2017_full.json").exists())


if __name__ == "__main__":
    unittest.main()
