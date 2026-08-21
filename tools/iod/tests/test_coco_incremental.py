import json
import tempfile
import unittest
from pathlib import Path

from tools.iod.coco_incremental import build_split


def make_coco(path: Path) -> None:
    categories = [{"id": i, "name": f"c{i}"} for i in range(80)]
    train_images = [{"id": i, "file_name": f"{i:012d}.jpg", "width": 10, "height": 10}
                    for i in range(80)]
    val_images = [{"id": i + 80, "file_name": f"{i + 80:012d}.jpg", "width": 10, "height": 10}
                  for i in range(80)]
    train_annotations = [{"id": i, "image_id": i, "category_id": i,
                    "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
                   for i in range(80)]
    val_annotations = [{"id": i + 80, "image_id": i + 80, "category_id": i,
                        "bbox": [0, 0, 5, 5], "area": 25, "iscrowd": 0}
                       for i in range(80)]
    (path / "annotations").mkdir(parents=True)
    (path / "annotations" / "instances_train2017.json").write_text(
        json.dumps({"images": train_images, "annotations": train_annotations,
                    "categories": categories}), encoding="utf-8")
    (path / "annotations" / "instances_val2017.json").write_text(
        json.dumps({"images": val_images, "annotations": val_annotations,
                    "categories": categories}), encoding="utf-8")


class SplitTest(unittest.TestCase):
    def test_protocol_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coco = root / "coco"
            output = root / "split"
            make_coco(coco)
            manifest = build_split(coco, output, "40+20x2", "id", 42, 0.10)
            self.assertTrue(manifest["integrity"]["passed"])
            self.assertEqual(len(manifest["stages"]), 3)
            self.assertEqual(len(set(manifest["category_order_source_ids"])), 80)
            self.assertEqual(sum(len(x["classes"]) for x in manifest["stages"]), 80)
            train_ids = [x for stage in manifest["stages"] for x in stage["train_image_ids"]]
            self.assertEqual(len(train_ids), len(set(train_ids)))
            self.assertTrue((output / "stage_2" / "instances_val2017.json").is_file())


if __name__ == "__main__":
    unittest.main()
