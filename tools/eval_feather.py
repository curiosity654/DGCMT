import argparse
from os import path as osp
from pathlib import Path

import pandas as pd

import tri3d.datasets as tri3d_datasets

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, **kwargs):
        return iterable


TWO_WHEELER_CATEGORIES = {
    "BICYCLE",
    "BICYCLIST",
    "MOTORCYCLE",
    "MOTORCYCLIST",
    "WHEELED_RIDER",
}

ARGO_3CLASS_EVAL_CATEGORIES = ("VEHICLE", "CYCLIST", "PEDESTRIAN")

ARGO_3CLASS_VEHICLE_CATEGORIES = {
    "VEHICLE",
    "CAR",
    "TRUCK",
    "CONSTRUCTION_VEHICLE",
    "BUS",
    "TRAILER",
    "REGULAR_VEHICLE",
    "LARGE_VEHICLE",
    "BOX_TRUCK",
    "TRUCK_CAB",
    "ARTICULATED_BUS",
    "SCHOOL_BUS",
    "VEHICULAR_TRAILER",
}

ARGO_3CLASS_CYCLIST_CATEGORIES = {
    "CYCLIST",
    "BICYCLE",
    "MOTORCYCLE",
    "BICYCLIST",
    "MOTORCYCLIST",
    "WHEELED_RIDER",
}

ARGO_3CLASS_PEDESTRIAN_CATEGORIES = {
    "PEDESTRIAN",
    "OFFICIAL_SIGNALER",
}

NUSC_TO_WAYMO_MAPPING = {
    "car": "VEHICLE",
    "truck": "VEHICLE",
    "construction_vehicle": "VEHICLE",
    "bus": "VEHICLE",
    "trailer": "VEHICLE",
    "motorcycle": "CYCLIST",
    "bicycle": "CYCLIST",
    "pedestrian": "PEDESTRIAN",
    "barrier": "SIGN",
    "traffic_cone": "SIGN",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate saved feather predictions for Argo or Waymo."
    )
    parser.add_argument("--dataset", choices=["argo", "waymo"], default="argo")
    parser.add_argument("--path", required=True, help="results file in feather format")
    parser.add_argument("--argo2-root", default="./data/argo2/argo2_format/")
    parser.add_argument(
        "--argo-eval-mode",
        choices=["official", "3class"],
        default="official",
        help="AV2 evaluation ontology: official fine-grained classes or aggregated 3-class eval",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="optional category whitelist, e.g. REGULAR_VEHICLE PEDESTRIAN or VEHICLE CYCLIST PEDESTRIAN",
    )
    parser.add_argument(
        "--merge-two-wheelers",
        action="store_true",
        help="merge BICYCLE/BICYCLIST/MOTORCYCLE/MOTORCYCLIST into one category for compatibility eval",
    )
    parser.add_argument(
        "--merged-two-wheelers-name",
        default="CYCLIST",
        help="target category name used when --merge-two-wheelers is enabled",
    )
    parser.add_argument("--waymo-root", default="./data/waymo")
    parser.add_argument("--waymo-split", default="val")
    parser.add_argument("--waymo-subset", default=None)
    return parser.parse_args()


def _normalize_waymo_category(category):
    cat = str(category).upper()
    if cat in {"VEHICLE", "PEDESTRIAN", "CYCLIST", "SIGN"}:
        return cat
    return NUSC_TO_WAYMO_MAPPING.get(cat.lower())


def _normalize_argo_3class_category(category):
    cat = str(category).strip().upper()
    if cat in ARGO_3CLASS_VEHICLE_CATEGORIES:
        return "VEHICLE"
    if cat in ARGO_3CLASS_CYCLIST_CATEGORIES:
        return "CYCLIST"
    if cat in ARGO_3CLASS_PEDESTRIAN_CATEGORIES:
        return "PEDESTRIAN"
    return None


def run_argo_eval(args):
    from av2.utils.io import read_feather

    dts = read_feather(Path(args.path))
    dts["category"] = dts["category"].astype(str).str.upper()
    argo2_root = args.argo2_root
    val_anno_path = osp.join(argo2_root, "sensor/val_anno.feather")
    gts = read_feather(Path(val_anno_path))
    gts["category"] = gts["category"].astype(str).str.upper()

    if args.argo_eval_mode == "3class":
        from projects.mmdet3d_plugin.datasets.av2_evaluation import (
            DetectionCfg,
            evaluate,
        )

        dts["category"] = dts["category"].map(_normalize_argo_3class_category)
        gts["category"] = gts["category"].map(_normalize_argo_3class_category)
        dts = dts[dts["category"].notna()].copy()
        gts = gts[gts["category"].notna()].copy()
        eval_categories = set(ARGO_3CLASS_EVAL_CATEGORIES)
    else:
        from av2.evaluation.detection.constants import HIERARCHY
        from av2.evaluation.detection.eval import evaluate
        from av2.evaluation.detection.utils import DetectionCfg

        try:
            from av2.evaluation.detection.constants import CompetitionCategories

            eval_categories = set(x.value for x in CompetitionCategories)
        except ImportError:
            # Newer av2 versions remove CompetitionCategories.
            eval_categories = set(HIERARCHY["FINEGRAIN"])

        merged_two_wheelers_name = args.merged_two_wheelers_name.upper()
        if args.merge_two_wheelers:
            dts["category"] = dts["category"].replace(
                {k: merged_two_wheelers_name for k in TWO_WHEELER_CATEGORIES}
            )
            gts["category"] = gts["category"].replace(
                {k: merged_two_wheelers_name for k in TWO_WHEELER_CATEGORIES}
            )
        eval_categories = set(x.upper() for x in eval_categories)
        if args.merge_two_wheelers:
            eval_categories = eval_categories | {merged_two_wheelers_name}

    dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
    gts = gts.set_index(["log_id", "timestamp_ns"]).sort_values("category")
    requested_categories = None
    if args.categories:
        requested_categories = set(c.upper() for c in args.categories)
        unsupported = requested_categories - eval_categories
        if unsupported:
            raise ValueError(
                f"Unsupported categories for AV2 eval: {sorted(unsupported)}"
            )
        dts = dts[dts["category"].isin(requested_categories)]
        gts = gts[gts["category"].isin(requested_categories)]

    valid_uuids = set(gts.index.tolist()) & set(dts.index.tolist())
    gts = gts.loc[list(valid_uuids)].sort_index()
    dts = dts.loc[list(valid_uuids)].sort_index()

    categories = set(gts["category"].unique().tolist()) & eval_categories
    if requested_categories is not None:
        categories &= requested_categories
    if not categories:
        raise ValueError("No valid AV2 categories left after remapping/filtering.")

    split = "val"
    dataset_dir = Path(argo2_root) / "sensor" / split
    if args.argo_eval_mode == "3class":
        cfg = DetectionCfg(
            dataset_dir=dataset_dir,
            categories=tuple(sorted(categories)),
            eval_only_roi_instances=True,
            eval_range_m=(0.0, 150.0),
        )
    else:
        try:
            cfg = DetectionCfg(
                dataset_dir=dataset_dir,
                categories=tuple(sorted(categories)),
                split=split,
                max_range_m=200.0,
                eval_only_roi_instances=True,
            )
        except TypeError:
            cfg = DetectionCfg(
                dataset_dir=dataset_dir,
                categories=tuple(sorted(categories)),
                max_range_m=200.0,
                eval_only_roi_instances=True,
            )

    print(f"Start Argo evaluation ({args.argo_eval_mode}) ...")
    if args.argo_eval_mode == "3class":
        _, _, metrics, _ = evaluate(dts.reset_index(), gts.reset_index(), cfg)
    else:
        _, _, metrics = evaluate(dts.reset_index(), gts.reset_index(), cfg)
    valid_categories = sorted(categories) + ["AVERAGE_METRICS"]
    print(metrics.loc[valid_categories])


def _load_waymo_ground_truth(waymo_root, waymo_split, waymo_subset):
    type_map = {"VEHICLE", "PEDESTRIAN", "CYCLIST", "SIGN"}

    tri3d_kwargs = {"split": waymo_split}
    if waymo_subset:
        tri3d_kwargs["subset"] = waymo_subset
    dataset = tri3d_datasets.Waymo(waymo_root, **tri3d_kwargs)
    if hasattr(dataset, "pcl_sensors") and dataset.pcl_sensors:
        sensor = dataset.pcl_sensors[0]
    else:
        # Backward-compatible fallback for older Tri3D variants.
        alignment_obj = getattr(dataset, "alignment", None)
        if callable(alignment_obj):
            raise ValueError(
                "Cannot infer primary LiDAR sensor from dataset.alignment(). "
                "Please use a Tri3D Waymo dataset variant exposing pcl_sensors."
            )
        if alignment_obj is not None and hasattr(alignment_obj, "axes"):
            sensor = alignment_obj.axes[0]
        else:
            raise ValueError("Dataset must expose point cloud sensors (pcl_sensors).")

    sequences = list(dataset.sequences())
    total_frames = 0
    for seq in sequences:
        total_frames += len(dataset.timestamps(seq, sensor))
    print(
        f"Building Waymo GT cache from {len(sequences)} sequences, "
        f"~{total_frames} frames ..."
    )

    gt_by_key = {}
    frame_pbar = tqdm(total=total_frames, desc="Waymo GT frames", unit="frame")
    for seq in sequences:
        record = str(dataset.records[seq])
        timestamps = dataset.timestamps(seq, sensor)
        for frame in range(len(timestamps)):
            key = (record, int(timestamps[frame] * 1e6))
            boxes = dataset.boxes(seq, frame, coords=sensor)
            frame_gts = []
            for box in boxes:
                label = str(box.label).upper()
                if label not in type_map:
                    continue
                frame_gts.append(
                    {
                        "category": label,
                        "bbox": [
                            float(box.center[0]),
                            float(box.center[1]),
                            float(box.center[2]),
                            float(box.size[0]),
                            float(box.size[1]),
                            float(box.size[2]),
                            float(box.heading),
                        ],
                        "difficulty": 2
                        if getattr(box, "difficulty_level_det", 1) == 2
                        else 1,
                        "speed": (
                            [float(box.speed[0]), float(box.speed[1])]
                            if hasattr(box, "speed")
                            else None
                        ),
                    }
                )
            gt_by_key[key] = frame_gts
            frame_pbar.update(1)
    frame_pbar.close()
    print(f"Waymo GT cache ready: {len(gt_by_key)} frame keys.")
    return gt_by_key


def run_waymo_eval(args):
    try:
        import tensorflow as tf
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.metrics.python import config_util_py
        from waymo_open_dataset.metrics.python import wod_detection_evaluator
    except Exception as exc:
        raise ImportError(
            "Waymo eval requires tensorflow + waymo_open_dataset. "
            "Install waymo-open-dataset-tf-2-12-0==1.6.7"
        ) from exc

    required_cols = {
        "log_id",
        "timestamp_micros",
        "center_x",
        "center_y",
        "center_z",
        "length",
        "width",
        "height",
        "heading",
        "score",
        "category",
    }
    dts = pd.read_feather(args.path)
    missing = required_cols - set(dts.columns)
    if missing:
        raise ValueError(f"Waymo pred feather missing required columns: {sorted(missing)}")
    if len(dts) == 0:
        raise ValueError("Prediction feather is empty, nothing to evaluate.")

    dts["log_id"] = dts["log_id"].astype(str)
    dts["timestamp_micros"] = dts["timestamp_micros"].astype("int64")
    dts["category"] = dts["category"].map(_normalize_waymo_category)
    dts = dts[dts["category"].notna()].copy()
    if len(dts) == 0:
        raise ValueError("No valid Waymo categories left after category mapping.")

    pred_keys = set(zip(dts["log_id"], dts["timestamp_micros"]))
    gt_by_key = _load_waymo_ground_truth(args.waymo_root, args.waymo_split, args.waymo_subset)
    common_keys = sorted(pred_keys & set(gt_by_key.keys()))
    if not common_keys:
        raise ValueError("No overlapping (log_id, timestamp_micros) between prediction and GT.")

    dts = dts.set_index(["log_id", "timestamp_micros"]).sort_index()
    type_map = {
        "VEHICLE": label_pb2.Label.TYPE_VEHICLE,
        "PEDESTRIAN": label_pb2.Label.TYPE_PEDESTRIAN,
        "CYCLIST": label_pb2.Label.TYPE_CYCLIST,
        "SIGN": label_pb2.Label.TYPE_SIGN,
    }
    level_map = {
        1: label_pb2.Label.LEVEL_1,
        2: label_pb2.Label.LEVEL_2,
    }

    frame_id_map = {key: idx for idx, key in enumerate(common_keys)}
    pred_frame_ids = []
    pred_bboxes = []
    pred_types = []
    pred_scores = []
    pred_overlap_nlz = []

    gt_frame_ids = []
    gt_bboxes = []
    gt_types = []
    gt_difficulty = []
    gt_speed = []

    print(f"Packing predictions/GT for {len(common_keys)} aligned frames ...")
    for key in tqdm(common_keys, desc="Waymo eval packing", unit="frame"):
        frame_id = frame_id_map[key]

        frame_pred = dts.loc[[key]] if key in dts.index else pd.DataFrame()
        for row in frame_pred.itertuples(index=False):
            pred_frame_ids.append(frame_id)
            pred_bboxes.append(
                [
                    float(row.center_x),
                    float(row.center_y),
                    float(row.center_z),
                    float(row.length),
                    float(row.width),
                    float(row.height),
                    float(row.heading),
                ]
            )
            pred_types.append(type_map[row.category])
            pred_scores.append(float(row.score))
            pred_overlap_nlz.append(False)

        frame_gts = gt_by_key[key]
        for gt in frame_gts:
            gt_frame_ids.append(frame_id)
            gt_bboxes.append(gt["bbox"])
            gt_types.append(type_map[gt["category"]])
            gt_difficulty.append(level_map.get(gt["difficulty"], label_pb2.Label.LEVEL_1))
            if gt["speed"] is not None:
                gt_speed.append(gt["speed"])

    if not pred_bboxes:
        raise ValueError("No valid predictions after frame/category filtering.")
    if not gt_bboxes:
        raise ValueError("No GT boxes found on aligned frames.")

    predictions = dict(
        prediction_frame_id=tf.constant(pred_frame_ids, dtype=tf.int64),
        prediction_bbox=tf.constant(pred_bboxes, dtype=tf.float32),
        prediction_type=tf.constant(pred_types, dtype=tf.uint8),
        prediction_score=tf.constant(pred_scores, dtype=tf.float32),
        prediction_overlap_nlz=tf.constant(pred_overlap_nlz, dtype=tf.bool),
    )
    groundtruths = dict(
        ground_truth_frame_id=tf.constant(gt_frame_ids, dtype=tf.int64),
        ground_truth_bbox=tf.constant(gt_bboxes, dtype=tf.float32),
        ground_truth_type=tf.constant(gt_types, dtype=tf.uint8),
        ground_truth_difficulty=tf.constant(gt_difficulty, dtype=tf.uint8),
    )
    if len(gt_speed) == len(gt_bboxes):
        groundtruths["ground_truth_speed"] = tf.constant(gt_speed, dtype=tf.float32)

    print(
        f"Start Waymo evaluation on {len(common_keys)} aligned frames, "
        f"{len(pred_bboxes)} preds, {len(gt_bboxes)} gts ..."
    )
    evaluator = wod_detection_evaluator.WODDetectionEvaluator()
    evaluator.update_state(groundtruths, predictions)
    metrics = evaluator.result()

    breakdown_names = config_util_py.get_breakdown_names_from_config(evaluator._config)
    ap = metrics.average_precision.numpy()
    aph = metrics.average_precision_ha_weighted.numpy()
    detail = {}
    metric_prefix = "pts_bbox_Waymo"

    for name, ap_val, aph_val in zip(breakdown_names, ap, aph):
        detail[f"{metric_prefix}/{name}/mAP"] = float(ap_val)
        detail[f"{metric_prefix}/{name}/mAPH"] = float(aph_val)

    def get_metric(name, values):
        try:
            idx = breakdown_names.index(name)
        except ValueError:
            return None
        return float(values[idx])

    summary_types = ["VEHICLE", "PEDESTRIAN", "CYCLIST"]
    for level in ["LEVEL_1", "LEVEL_2"]:
        ap_vals = []
        aph_vals = []
        for obj_type in summary_types:
            key = f"OBJECT_TYPE_TYPE_{obj_type}_{level}"
            ap_val = get_metric(key, ap)
            aph_val = get_metric(key, aph)
            if ap_val is not None:
                ap_vals.append(ap_val)
            if aph_val is not None:
                aph_vals.append(aph_val)
        if ap_vals:
            detail[f"{metric_prefix}/Overall/{'L1' if level.endswith('1') else 'L2'} mAP"] = (
                float(sum(ap_vals) / len(ap_vals))
            )
        if aph_vals:
            detail[f"{metric_prefix}/Overall/{'L1' if level.endswith('1') else 'L2'} mAPH"] = (
                float(sum(aph_vals) / len(aph_vals))
            )

    print("\n" + "=" * 80)
    print("Waymo Evaluation Results (Official):")
    for key in [
        f"{metric_prefix}/Overall/L1 mAP",
        f"{metric_prefix}/Overall/L1 mAPH",
        f"{metric_prefix}/Overall/L2 mAP",
        f"{metric_prefix}/Overall/L2 mAPH",
    ]:
        if key in detail:
            print(f"{key}: {detail[key]:.4f}")
    print("-" * 80)
    for name in breakdown_names:
        map_key = f"{metric_prefix}/{name}/mAP"
        maph_key = f"{metric_prefix}/{name}/mAPH"
        print(f"{name}: mAP={detail[map_key]:.4f}, mAPH={detail[maph_key]:.4f}")
    print("=" * 80 + "\n")


def main():
    args = parse_args()
    if args.dataset == "argo":
        run_argo_eval(args)
        return
    run_waymo_eval(args)


if __name__ == "__main__":
    main()