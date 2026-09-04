import json
import tempfile
import unittest
from pathlib import Path

from tools.owod.protocol import build_official_manifest, stage_files


def write_coco(path: Path, class_ids, image_ids):
    payload = {
        "images": [{"id": value, "file_name": f"{value}.jpg"} for value in image_ids],
        "annotations": [
            {"id": offset + 1, "image_id": image_id,
             "category_id": class_id, "bbox": [0, 0, 1, 1]}
            for offset, (class_id, image_id) in enumerate(zip(class_ids, image_ids))
        ],
        "categories": [{"id": value, "name": f"class-{value}"} for value in range(1, 81)],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_official_tree(root: Path, duplicate_image: bool = False):
    all_classes = list(range(1, 81))
    for stage in range(4):
        stage_dir = root / f"stage_{stage}"
        stage_dir.mkdir()
        current = all_classes[stage * 20:(stage + 1) * 20]
        current_images = list(range(stage * 20 + 1, (stage + 1) * 20 + 1))
        if duplicate_image and stage == 1:
            current_images[0] = 1
        active = all_classes[:(stage + 1) * 20]
        active_images = list(range(1, len(active) + 1))
        write_coco(stage_dir / "instances_increment_train2017.json", current, current_images)
        write_coco(stage_dir / "instances_train2017.json", active, active_images)
        write_coco(stage_dir / "instances_val2017.json", active, active_images)
        write_coco(stage_dir / "instances_val2017_full.json", all_classes, all_classes)


class OfficialProtocolTests(unittest.TestCase):
    def test_imports_four_disjoint_official_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_official_tree(root)
            manifest = build_official_manifest(root, "m-owodb", "official-fixture-v1")
            self.assertTrue(manifest["official_annotations"])
            self.assertTrue(manifest["paper_comparable"])
            self.assertEqual(manifest["integrity"]["category_count"], 80)
            manifest_path = root / "split_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded, files = stage_files(manifest_path, 2)
            self.assertEqual(loaded["stages"][2]["classes"], list(range(41, 61)))
            self.assertTrue(files["full_val"].is_file())

    def test_rejects_cross_task_image_duplication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_official_tree(root, duplicate_image=True)
            with self.assertRaisesRegex(ValueError, "reuse images"):
                build_official_manifest(root, "m-owodb", "official-fixture-v1")


if __name__ == "__main__":
    unittest.main()
