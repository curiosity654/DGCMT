import argparse
import copy
import os
import pickle


TARGET_CLASS_NAMES = ("vehicle", "bicycle", "pedestrian")
RAW_TO_TARGET = {
    "car": "vehicle",
    "truck": "vehicle",
    "construction_vehicle": "vehicle",
    "bus": "vehicle",
    "trailer": "vehicle",
    "vehicle.emergency.ambulance": "vehicle",
    "vehicle.emergency.police": "vehicle",
    "bicycle": "bicycle",
    "motorcycle": "bicycle",
    "pedestrian": "pedestrian",
    "animal": None,
    "human.pedestrian.personal_mobility": None,
    "human.pedestrian.stroller": None,
    "human.pedestrian.wheelchair": None,
    "movable_object.debris": None,
    "movable_object.pushable_pullable": None,
    "static_object.bicycle_rack": None,
    "traffic_cone": None,
    "barrier": None,
}


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def load_pickle_compat(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as exc:
        if "numpy._core" not in str(exc):
            raise
        with open(path, "rb") as f:
            return _NumpyCompatUnpickler(f).load()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert nuScenes dbinfos from the default taxonomy to the fixed "
            "3-class CMT ontology."
        )
    )
    parser.add_argument(
        "input",
        help="Path to the source dbinfos pkl, e.g. data/nuscenes/nuscenes_dbinfos_train.pkl",
    )
    parser.add_argument(
        "output",
        help=(
            "Path to the output pkl, e.g. "
            "data/nuscenes/nuscenes_3class_dbinfos_train.pkl"
        ),
    )
    parser.add_argument(
        "--allow-unknown",
        action="store_true",
        help="Drop unknown raw classes instead of raising an error.",
    )
    return parser.parse_args()


def _validate_dbinfos(dbinfos):
    if not isinstance(dbinfos, dict):
        raise TypeError("Expected dbinfos to be dict[class_name, list[info]].")

    validated = {}
    for raw_name, infos in dbinfos.items():
        if not isinstance(infos, list):
            raise TypeError(
                "Expected dbinfos[{!r}] to be a list, got {}.".format(
                    raw_name, type(infos).__name__
                )
            )
        validated[str(raw_name)] = infos
    return validated


def convert_dbinfos(dbinfos, allow_unknown=False):
    converted = {class_name: [] for class_name in TARGET_CLASS_NAMES}

    for raw_name, infos in dbinfos.items():
        target_name = RAW_TO_TARGET.get(str(raw_name))
        if str(raw_name) not in RAW_TO_TARGET:
            if allow_unknown:
                continue
            raise KeyError(
                "Unknown raw class {!r} found in dbinfos. "
                "Pass --allow-unknown to drop it.".format(raw_name)
            )

        if target_name is None:
            continue

        for info in infos:
            if not isinstance(info, dict):
                raise TypeError(
                    "Expected each db info under {!r} to be a dict, got {}.".format(
                        raw_name, type(info).__name__
                    )
                )
            remapped_info = copy.deepcopy(info)
            remapped_info["name"] = target_name
            converted[target_name].append(remapped_info)

    return converted


def print_summary(source_dbinfos, converted_dbinfos):
    print("Source dbinfos:")
    for raw_name in sorted(source_dbinfos.keys()):
        print("  {}: {}".format(raw_name, len(source_dbinfos[raw_name])))

    print("Converted 3-class dbinfos:")
    for target_name in TARGET_CLASS_NAMES:
        print("  {}: {}".format(target_name, len(converted_dbinfos[target_name])))


def main():
    args = parse_args()

    dbinfos = load_pickle_compat(args.input)
    dbinfos = _validate_dbinfos(dbinfos)
    converted = convert_dbinfos(dbinfos, allow_unknown=args.allow_unknown)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "wb") as f:
        pickle.dump(converted, f)

    print_summary(dbinfos, converted)
    print("Saved converted 3-class dbinfos to: {}".format(args.output))


if __name__ == "__main__":
    main()
