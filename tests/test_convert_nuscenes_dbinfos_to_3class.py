import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "tools" / "convert_nuscenes_dbinfos_to_3class.py"
SPEC = importlib.util.spec_from_file_location(
    "convert_nuscenes_dbinfos_to_3class", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ConvertNuScenesDbinfosTo3ClassTest(unittest.TestCase):
    def test_convert_dbinfos_merges_expected_classes(self):
        dbinfos = {
            "car": [{"name": "car", "id": 1}],
            "truck": [{"name": "truck", "id": 2}],
            "bicycle": [{"name": "bicycle", "id": 3}],
            "motorcycle": [{"name": "motorcycle", "id": 4}],
            "pedestrian": [{"name": "pedestrian", "id": 5}],
            "traffic_cone": [{"name": "traffic_cone", "id": 6}],
        }

        converted = MODULE.convert_dbinfos(dbinfos)

        self.assertEqual(set(converted.keys()), {"vehicle", "bicycle", "pedestrian"})
        self.assertEqual([item["id"] for item in converted["vehicle"]], [1, 2])
        self.assertEqual([item["name"] for item in converted["vehicle"]], ["vehicle", "vehicle"])
        self.assertEqual([item["id"] for item in converted["bicycle"]], [3, 4])
        self.assertEqual([item["id"] for item in converted["pedestrian"]], [5])

    def test_convert_dbinfos_rejects_unknown_classes_by_default(self):
        with self.assertRaises(KeyError):
            MODULE.convert_dbinfos({"unknown_class": [{"name": "unknown_class"}]})

    def test_convert_dbinfos_can_drop_unknown_classes(self):
        converted = MODULE.convert_dbinfos(
            {
                "car": [{"name": "car", "id": 1}],
                "unknown_class": [{"name": "unknown_class", "id": 2}],
            },
            allow_unknown=True,
        )

        self.assertEqual([item["id"] for item in converted["vehicle"]], [1])
        self.assertEqual(converted["bicycle"], [])
        self.assertEqual(converted["pedestrian"], [])


if __name__ == "__main__":
    unittest.main()
