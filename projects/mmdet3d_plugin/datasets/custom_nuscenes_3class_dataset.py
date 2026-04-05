# ------------------------------------------------------------------------
# Copyright (c) 2023 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import mmcv

from mmdet.datasets import DATASETS
from mmdet3d.core.bbox import LiDARInstance3DBoxes

from .custom_nuscenes_dataset import CustomNuScenesDataset


def _safe_nanmean(values):
    if len(values) == 0:
        return float("nan")
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _yaw_abs_diff(yaw_a, yaw_b):
    diff = (yaw_a - yaw_b + np.pi) % (2 * np.pi) - np.pi
    return float(np.abs(diff))


def _init_metrics_store(num_classes, thresholds):
    return {
        thr: {
            cls_id: {
                "scores": [],
                "tp": [],
                "fp": [],
                "ate": [],
                "aoe": [],
                "ase": [],
                "ave": [],
                "num_gt": 0,
            }
            for cls_id in range(num_classes)
        }
        for thr in thresholds
    }


def _boxes_to_pred_dict(det):
    boxes_3d = det["boxes_3d"]
    scores_3d = det["scores_3d"]
    labels_3d = det["labels_3d"]

    boxes = boxes_3d.tensor.detach().cpu().numpy()
    scores = scores_3d.detach().cpu().numpy()
    labels = labels_3d.detach().cpu().numpy().astype(np.int64)

    if boxes.shape[0] == 0:
        return dict(
            centers=np.zeros((0, 3), dtype=np.float32),
            sizes=np.zeros((0, 3), dtype=np.float32),
            yaws=np.zeros((0,), dtype=np.float32),
            velocities=np.zeros((0, 2), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )

    if boxes.shape[1] >= 9:
        velocities = boxes[:, 7:9].astype(np.float32)
    else:
        velocities = np.full((boxes.shape[0], 2), np.nan, dtype=np.float32)

    return dict(
        centers=boxes[:, :3].astype(np.float32),
        sizes=boxes[:, 3:6].astype(np.float32),
        yaws=boxes[:, 6].astype(np.float32),
        velocities=velocities,
        scores=scores.astype(np.float32),
        labels=labels,
    )


def _ann_to_gt_dict(ann_info):
    boxes = ann_info["gt_bboxes_3d"].tensor.numpy()
    labels = ann_info["gt_labels_3d"].astype(np.int64)

    if boxes.shape[0] == 0:
        return dict(
            centers=np.zeros((0, 3), dtype=np.float32),
            sizes=np.zeros((0, 3), dtype=np.float32),
            yaws=np.zeros((0,), dtype=np.float32),
            velocities=np.zeros((0, 2), dtype=np.float32),
            labels=np.zeros((0,), dtype=np.int64),
        )

    if boxes.shape[1] >= 9:
        velocities = boxes[:, 7:9].astype(np.float32)
    else:
        velocities = np.full((boxes.shape[0], 2), np.nan, dtype=np.float32)

    return dict(
        centers=boxes[:, :3].astype(np.float32),
        sizes=boxes[:, 3:6].astype(np.float32),
        yaws=boxes[:, 6].astype(np.float32),
        velocities=velocities,
        labels=labels,
    )


def _accumulate_sample(metrics_store, pred, gt, num_classes, thresholds):
    for cls_id in range(num_classes):
        pred_mask = pred["labels"] == cls_id
        gt_mask = gt["labels"] == cls_id

        pred_centers = pred["centers"][pred_mask]
        pred_sizes = pred["sizes"][pred_mask]
        pred_yaws = pred["yaws"][pred_mask]
        pred_scores = pred["scores"][pred_mask]
        pred_velocities = pred["velocities"][pred_mask]

        gt_centers = gt["centers"][gt_mask]
        gt_sizes = gt["sizes"][gt_mask]
        gt_yaws = gt["yaws"][gt_mask]
        gt_velocities = gt["velocities"][gt_mask]

        if pred_scores.shape[0] > 0:
            order = np.argsort(-pred_scores)
            pred_centers = pred_centers[order]
            pred_sizes = pred_sizes[order]
            pred_yaws = pred_yaws[order]
            pred_scores = pred_scores[order]
            pred_velocities = pred_velocities[order]

        for thr in thresholds:
            cls_store = metrics_store[thr][cls_id]
            cls_store["num_gt"] += int(gt_centers.shape[0])

            assigned_gt = np.zeros(gt_centers.shape[0], dtype=bool)
            for pred_idx in range(pred_centers.shape[0]):
                cls_store["scores"].append(float(pred_scores[pred_idx]))
                if gt_centers.shape[0] == 0:
                    cls_store["tp"].append(0.0)
                    cls_store["fp"].append(1.0)
                    continue

                dists = np.linalg.norm(gt_centers - pred_centers[pred_idx], axis=1)
                dists[assigned_gt] = np.inf
                matched_gt = int(np.argmin(dists))
                min_dist = float(dists[matched_gt])

                if np.isfinite(min_dist) and min_dist <= thr:
                    assigned_gt[matched_gt] = True
                    cls_store["tp"].append(1.0)
                    cls_store["fp"].append(0.0)
                    cls_store["ate"].append(min_dist)
                    cls_store["aoe"].append(
                        _yaw_abs_diff(gt_yaws[matched_gt], pred_yaws[pred_idx])
                    )

                    pred_dims = pred_sizes[pred_idx]
                    gt_dims = gt_sizes[matched_gt]
                    min_dims = np.minimum(pred_dims, gt_dims)
                    inter = float(np.prod(min_dims))
                    pred_vol = float(np.prod(pred_dims))
                    gt_vol = float(np.prod(gt_dims))
                    union = pred_vol + gt_vol - inter
                    cls_store["ase"].append(1.0 - inter / max(union, 1e-6))

                    pred_v = pred_velocities[pred_idx]
                    gt_v = gt_velocities[matched_gt]
                    if np.any(np.isnan(pred_v)) or np.any(np.isnan(gt_v)):
                        cls_store["ave"].append(float("nan"))
                    else:
                        cls_store["ave"].append(float(np.linalg.norm(pred_v - gt_v)))
                else:
                    cls_store["tp"].append(0.0)
                    cls_store["fp"].append(1.0)


def _finalize_metrics(metrics_store, class_names, thresholds, metric_prefix):
    detail = {}
    ap_cls_means = []
    ate_cls_means = []
    aoe_cls_means = []
    ase_cls_means = []
    ave_cls_means = []

    for cls_id, cls_name in enumerate(class_names):
        class_ap_vals = []
        class_ate_vals = []
        class_aoe_vals = []
        class_ase_vals = []
        class_ave_vals = []

        for thr in thresholds:
            cls_store = metrics_store[thr][cls_id]
            thr_str = "{:.1f}".format(thr)

            scores = np.asarray(cls_store["scores"], dtype=np.float32)
            tp = np.asarray(cls_store["tp"], dtype=np.float32)
            fp = np.asarray(cls_store["fp"], dtype=np.float32)
            num_gt = int(cls_store["num_gt"])

            if scores.shape[0] > 0:
                order = np.argsort(-scores)
                tp = tp[order]
                fp = fp[order]
                tp_cum = np.cumsum(tp)
                fp_cum = np.cumsum(fp)
            else:
                tp_cum = np.zeros((0,), dtype=np.float32)
                fp_cum = np.zeros((0,), dtype=np.float32)

            if num_gt > 0:
                recalls = tp_cum / max(num_gt, 1)
                precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
                recalls = np.concatenate(([0.0], recalls))
                precisions = np.concatenate(([1.0], precisions))
                ap = float(np.trapz(precisions, recalls))
            else:
                ap = float("nan")

            ate = _safe_nanmean(cls_store["ate"])
            aoe = _safe_nanmean(cls_store["aoe"])
            ase = _safe_nanmean(cls_store["ase"])
            ave = _safe_nanmean(cls_store["ave"])

            detail["{}/{}_AP_dist_{}".format(metric_prefix, cls_name, thr_str)] = round(
                ap, 4
            )
            detail["{}/{}_ATE_dist_{}".format(metric_prefix, cls_name, thr_str)] = round(
                ate, 4
            )
            detail["{}/{}_AOE_dist_{}".format(metric_prefix, cls_name, thr_str)] = round(
                aoe, 4
            )
            detail["{}/{}_ASE_dist_{}".format(metric_prefix, cls_name, thr_str)] = round(
                ase, 4
            )
            detail["{}/{}_AVE_dist_{}".format(metric_prefix, cls_name, thr_str)] = round(
                ave, 4
            )

            class_ap_vals.append(ap)
            class_ate_vals.append(ate)
            class_aoe_vals.append(aoe)
            class_ase_vals.append(ase)
            class_ave_vals.append(ave)

        ap_mean = _safe_nanmean(class_ap_vals)
        ate_mean = _safe_nanmean(class_ate_vals)
        aoe_mean = _safe_nanmean(class_aoe_vals)
        ase_mean = _safe_nanmean(class_ase_vals)
        ave_mean = _safe_nanmean(class_ave_vals)

        detail["{}/{}_AP_dist_mean".format(metric_prefix, cls_name)] = round(ap_mean, 4)
        detail["{}/{}_ATE_dist_mean".format(metric_prefix, cls_name)] = round(
            ate_mean, 4
        )
        detail["{}/{}_AOE_dist_mean".format(metric_prefix, cls_name)] = round(
            aoe_mean, 4
        )
        detail["{}/{}_ASE_dist_mean".format(metric_prefix, cls_name)] = round(
            ase_mean, 4
        )
        detail["{}/{}_AVE_dist_mean".format(metric_prefix, cls_name)] = round(
            ave_mean, 4
        )

        ap_cls_means.append(ap_mean)
        ate_cls_means.append(ate_mean)
        aoe_cls_means.append(aoe_mean)
        ase_cls_means.append(ase_mean)
        ave_cls_means.append(ave_mean)

    detail["{}/mAP".format(metric_prefix)] = round(_safe_nanmean(ap_cls_means), 4)
    detail["{}/mATE".format(metric_prefix)] = round(_safe_nanmean(ate_cls_means), 4)
    detail["{}/mAOE".format(metric_prefix)] = round(_safe_nanmean(aoe_cls_means), 4)
    detail["{}/mASE".format(metric_prefix)] = round(_safe_nanmean(ase_cls_means), 4)
    detail["{}/mAVE".format(metric_prefix)] = round(_safe_nanmean(ave_cls_means), 4)
    return detail


@DATASETS.register_module()
class CustomNuScenes3ClassDataset(CustomNuScenesDataset):
    CLASSES = ("vehicle", "bicycle", "pedestrian")
    RAW_TO_TARGET = {
        "car": "vehicle",
        "truck": "vehicle",
        "construction_vehicle": "vehicle",
        "bus": "vehicle",
        "trailer": "vehicle",
        "bicycle": "bicycle",
        "motorcycle": "bicycle",
        "pedestrian": "pedestrian",
        "barrier": None,
        "traffic_cone": None,
    }

    def __init__(self, *args, distance_thresholds=(0.5, 1.0, 2.0, 4.0), **kwargs):
        self.distance_thresholds = [float(x) for x in distance_thresholds]
        super(CustomNuScenes3ClassDataset, self).__init__(*args, **kwargs)

    def _get_keep_mask_and_names(self, raw_names):
        mapped_names = [self.RAW_TO_TARGET.get(name) for name in raw_names]
        keep_mask = np.array(
            [name is not None and name in self.CLASSES for name in mapped_names],
            dtype=bool,
        )
        kept_names = np.asarray(
            [mapped_names[i] for i in range(len(mapped_names)) if keep_mask[i]]
        )
        return keep_mask, kept_names

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            raw_names = info["gt_names"][info["valid_flag"]]
        else:
            raw_names = info["gt_names"]

        keep_mask, mapped_names = self._get_keep_mask_and_names(raw_names)
        del keep_mask

        cat_ids = []
        for name in set(mapped_names.tolist()):
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def get_ann_info(self, index):
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0

        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_raw = info["gt_names"][mask]
        keep_mask, gt_names_3d = self._get_keep_mask_and_names(gt_names_raw)

        gt_bboxes_3d = gt_bboxes_3d[keep_mask]
        gt_labels_3d = np.array(
            [self.CLASSES.index(cat) for cat in gt_names_3d], dtype=np.int64
        )

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask][keep_mask]
            if gt_velocity.shape[0] > 0:
                nan_mask = np.isnan(gt_velocity[:, 0])
                gt_velocity[nan_mask] = [0.0, 0.0]
                gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)
            else:
                gt_bboxes_3d = np.zeros((0, 9), dtype=np.float32)
        elif gt_bboxes_3d.shape[0] == 0:
            gt_bboxes_3d = np.zeros((0, 7), dtype=np.float32)

        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5),
        ).convert_to(self.box_mode_3d)

        return dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
        )

    def _evaluate_single(self, results, result_name):
        metrics_store = _init_metrics_store(len(self.CLASSES), self.distance_thresholds)
        metric_prefix = "{}_NuScenes3Class".format(result_name)

        print("Evaluating {} with custom nuScenes 3-class metric...".format(result_name))
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            pred = _boxes_to_pred_dict(det)
            valid_mask = (pred["labels"] >= 0) & (pred["labels"] < len(self.CLASSES))
            for key in ["centers", "sizes", "yaws", "velocities", "scores", "labels"]:
                pred[key] = pred[key][valid_mask]

            gt = _ann_to_gt_dict(self.get_ann_info(sample_id))
            _accumulate_sample(
                metrics_store,
                pred,
                gt,
                len(self.CLASSES),
                self.distance_thresholds,
            )

        return _finalize_metrics(
            metrics_store, list(self.CLASSES), self.distance_thresholds, metric_prefix
        )

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        del metric, logger, jsonfile_prefix
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), (
            "The length of results is not equal to the dataset len: {} != {}".format(
                len(results), len(self)
            )
        )

        if len(results) == 0:
            return {}

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            results_dict = self._evaluate_single(results, "pts_bbox")
        else:
            results_dict = {}
            for name in result_names:
                if name not in results[0]:
                    continue
                print("Evaluating bboxes of {}".format(name))
                results_dict.update(self._evaluate_single([out[name] for out in results], name))

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict
