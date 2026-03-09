#!/usr/bin/env python3
"""Single-frame dump + inference + visualization tool (MMCV 1.x)."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import os.path as osp
import pickle
from copy import deepcopy

import cv2
import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.parallel import DataContainer, collate, scatter

from mmdet3d.apis import init_model
from mmdet3d.datasets import build_dataset

# PyTorch 2.6 changed torch.load default to weights_only=True.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump and run single-frame inference with visualization.')
    parser.add_argument('--config', required=True, help='Config path')
    parser.add_argument('--checkpoint', required=True, help='Checkpoint path')
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split in cfg.data')
    parser.add_argument('--index', type=int, default=0, help='Sample index')
    parser.add_argument('--device', default='cuda:0', help='Inference device')
    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--out-dir', default='work_dirs/single_frame_vis')
    parser.add_argument('--dump', action='store_true', help='Dump pkl')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config fields, e.g. key=value')
    return parser.parse_args()


def import_plugin_modules(cfg, config_path):
    if not hasattr(cfg, 'plugin') or not cfg.plugin:
        return
    if isinstance(cfg.plugin, str):
        plugin_path = cfg.plugin.strip('/').replace('/', '.')
        if plugin_path:
            importlib.import_module(plugin_path)
            return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = os.path.dirname(cfg.plugin_dir).split('/')
    else:
        module_dir = os.path.dirname(config_path).split('/')
    module_path = module_dir[0]
    for m in module_dir[1:]:
        module_path = module_path + '.' + m
    importlib.import_module(module_path)


def build_dataset_from_cfg(cfg, split):
    if split not in cfg.data:
        raise KeyError(f'cfg.data.{split} is not defined')
    ds_cfg = deepcopy(cfg.data[split])
    if split != 'train':
        ds_cfg.test_mode = True
    return build_dataset(ds_cfg)


def unwrap_dc(value):
    if isinstance(value, DataContainer):
        return value.data
    if isinstance(value, list) and len(value) == 1 and isinstance(
            value[0], DataContainer):
        return value[0].data
    return value


def unwrap_for_dump(value):
    if isinstance(value, dict):
        return {k: unwrap_for_dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [unwrap_for_dump(v) for v in value]
    if isinstance(value, DataContainer):
        return unwrap_for_dump(value.data)
    if hasattr(value, 'tensor'):
        return unwrap_for_dump(value.tensor)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return value


def prepare_model_inputs(sample, model):
    data = collate([sample], samples_per_gpu=1)
    device = next(model.parameters()).device
    if device.type == 'cuda':
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        data = scatter(data, [device_index])[0]
    else:
        for k, v in list(data.items()):
            if isinstance(v, list) and len(v) > 0 and hasattr(v[0], 'data'):
                data[k] = v[0].data
    return data


def denorm_img_tensor_to_bgr_list(img_tensor, img_norm_cfg):
    if isinstance(img_tensor, torch.Tensor):
        imgs = img_tensor.detach().cpu().numpy()
    else:
        imgs = np.asarray(img_tensor)
    # [N, C, H, W] -> [N, H, W, C]
    if imgs.ndim == 4:
        imgs = imgs.transpose(0, 2, 3, 1)

    mean = np.array(img_norm_cfg.get('mean', [0, 0, 0]), dtype=np.float32)
    std = np.array(img_norm_cfg.get('std', [1, 1, 1]), dtype=np.float32)
    to_rgb = bool(img_norm_cfg.get('to_rgb', False))

    out = []
    for img in imgs:
        img = img.astype(np.float32) * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.append(np.ascontiguousarray(img))
    return out


def ensure_cv2_image(img):
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if not img.flags['C_CONTIGUOUS']:
        img = np.ascontiguousarray(img)
    if not img.flags['WRITEABLE']:
        img = img.copy()
    return img


def draw_projected_boxes(img, lidar2img, corners_3d, color, thickness=2):
    img = ensure_cv2_image(img)
    h, w = img.shape[:2]
    for corner in corners_3d:
        hom = np.concatenate([corner, np.ones((8, 1), dtype=np.float32)], axis=1)
        proj = hom @ lidar2img.T
        depth = proj[:, 2]
        if np.any(depth <= 1e-5):
            continue
        uv = proj[:, :2] / depth[:, None]
        if not np.all(np.isfinite(uv)):
            continue
        uv_int = uv.astype(np.int32)
        for s, e in BOX_EDGES:
            p1 = (int(uv_int[s, 0]), int(uv_int[s, 1]))
            p2 = (int(uv_int[e, 0]), int(uv_int[e, 1]))
            ok, cp1, cp2 = cv2.clipLine((0, 0, w, h), p1, p2)
            if ok:
                cv2.line(img, cp1, cp2, color, thickness)
    return img


def boxes3d_to_bev_polygons(boxes_tensor):
    if boxes_tensor is None or len(boxes_tensor) == 0:
        return np.zeros((0, 4, 2), dtype=np.float32)
    centers = boxes_tensor[:, 0:2].astype(np.float32)
    dx = boxes_tensor[:, 3].astype(np.float32)
    dy = boxes_tensor[:, 4].astype(np.float32)
    yaw = boxes_tensor[:, 6].astype(np.float32)

    local = np.stack([
        np.stack([dx / 2, dy / 2], axis=1),
        np.stack([dx / 2, -dy / 2], axis=1),
        np.stack([-dx / 2, -dy / 2], axis=1),
        np.stack([-dx / 2, dy / 2], axis=1),
    ],
                     axis=1)
    c = np.cos(yaw)
    s = np.sin(yaw)
    rot = np.stack([np.stack([c, -s], axis=1), np.stack([s, c], axis=1)],
                   axis=1)
    rotated = local @ rot.transpose(0, 2, 1)
    return rotated + centers[:, None, :]


def visualize_bev(points, pred_polys, gt_polys, out_path):
    plt.figure(figsize=(10, 10))
    if points.size > 0:
        plt.scatter(points[:, 0], points[:, 1], s=0.2, c='gray', alpha=0.6)
    if pred_polys is not None and len(pred_polys) > 0:
        for box in pred_polys:
            plt.plot(
                [box[0, 0], box[1, 0], box[2, 0], box[3, 0], box[0, 0]],
                [box[0, 1], box[1, 1], box[2, 1], box[3, 1], box[0, 1]],
                'r-',
                linewidth=1.0)
    if gt_polys is not None and len(gt_polys) > 0:
        for box in gt_polys:
            plt.plot(
                [box[0, 0], box[1, 0], box[2, 0], box[3, 0], box[0, 0]],
                [box[0, 1], box[1, 1], box[2, 1], box[3, 1], box[0, 1]],
                'g-',
                linewidth=1.0)
    plt.axis('equal')
    plt.title('BEV (red=pred, green=gt)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def extract_pred_instances(result):
    if not isinstance(result, dict):
        raise TypeError(f'Unexpected model result type: {type(result)}')
    pred = result['pts_bbox'] if 'pts_bbox' in result else result
    boxes = pred['boxes_3d'] if 'boxes_3d' in pred else pred['bboxes_3d']
    scores = pred['scores_3d']
    labels = pred['labels_3d']
    return boxes, scores, labels


def maybe_extract_gt(sample):
    gt_boxes = None
    gt_labels = None
    if 'gt_bboxes_3d' in sample:
        gt_boxes = unwrap_dc(sample['gt_bboxes_3d'])
    if 'gt_labels_3d' in sample:
        gt_labels = unwrap_dc(sample['gt_labels_3d'])
    if isinstance(gt_boxes, list) and len(gt_boxes) == 1:
        gt_boxes = gt_boxes[0]
    if isinstance(gt_labels, list) and len(gt_labels) == 1:
        gt_labels = gt_labels[0]
    return gt_boxes, gt_labels


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugin_modules(cfg, args.config)

    dataset = build_dataset_from_cfg(cfg, args.split)
    index = max(0, min(args.index, len(dataset) - 1))
    sample = dataset[index]

    img_metas = unwrap_dc(sample['img_metas'])
    points = unwrap_dc(sample['points'])
    img_tensor = unwrap_dc(sample['img'])
    if isinstance(img_metas, list) and len(img_metas) == 1 and isinstance(
            img_metas[0], dict):
        img_metas = img_metas[0]
    if isinstance(points, list) and len(points) == 1:
        points = points[0]
    if isinstance(img_tensor, list) and len(img_tensor) == 1 and isinstance(
            img_tensor[0], torch.Tensor):
        img_tensor = img_tensor[0]

    model = init_model(cfg, args.checkpoint, device=args.device)
    data = prepare_model_inputs(sample, model)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]

    pred_boxes, pred_scores, pred_labels = extract_pred_instances(result)
    pred_scores_np = pred_scores.detach().cpu().numpy()
    keep = pred_scores_np >= args.score_thr
    pred_labels_np = pred_labels.detach().cpu().numpy()[keep]
    pred_scores_np = pred_scores_np[keep]
    pred_boxes_tensor = pred_boxes.tensor.detach().cpu().numpy()[keep]
    pred_corners = pred_boxes.corners.detach().cpu().numpy()[keep]
    pred_bev_polys = boxes3d_to_bev_polygons(pred_boxes_tensor)

    gt_boxes, gt_labels = maybe_extract_gt(sample)
    gt_corners = None
    gt_bev_polys = None
    if gt_boxes is not None and hasattr(gt_boxes, 'tensor'):
        gt_boxes_tensor = gt_boxes.tensor.detach().cpu().numpy()
        gt_corners = gt_boxes.corners.detach().cpu().numpy()
        gt_bev_polys = boxes3d_to_bev_polygons(gt_boxes_tensor)

    img_norm_cfg = img_metas.get('img_norm_cfg', dict(mean=[0, 0, 0], std=[1, 1, 1]))
    imgs = denorm_img_tensor_to_bgr_list(img_tensor, img_norm_cfg)
    lidar2img = np.asarray(img_metas['lidar2img'])
    if lidar2img.ndim == 2:
        lidar2img = np.repeat(lidar2img[None, ...], len(imgs), axis=0)

    for vid, img in enumerate(imgs):
        img = draw_projected_boxes(
            img,
            lidar2img[vid],
            pred_corners,
            color=(0, 0, 255),
            thickness=2)
        if gt_corners is not None:
            img = draw_projected_boxes(
                img,
                lidar2img[vid],
                gt_corners,
                color=(0, 255, 0),
                thickness=2)
        out_name = osp.join(args.out_dir, f'frame_{index:06d}_cam{vid}.jpg')
        cv2.imwrite(out_name, img)

    if isinstance(points, torch.Tensor):
        points_np = points.detach().cpu().numpy()
    else:
        points_np = np.asarray(points)
    bev_path = osp.join(args.out_dir, f'frame_{index:06d}_bev.jpg')
    visualize_bev(points_np[:, :3], pred_bev_polys, gt_bev_polys, bev_path)

    classes = getattr(model, 'CLASSES', None)
    if classes is None and hasattr(dataset, 'CLASSES'):
        classes = dataset.CLASSES
    if classes is None:
        classes = []

    summary = {
        'index': int(index),
        'sample_idx': img_metas.get('sample_idx'),
        'num_pred': int(len(pred_scores_np)),
        'score_thr': float(args.score_thr),
        'classes': list(classes),
        'pred_labels': pred_labels_np.tolist(),
        'pred_scores': pred_scores_np.tolist(),
        'has_gt': gt_boxes is not None,
    }
    with open(
            osp.join(args.out_dir, f'frame_{index:06d}_summary.json'),
            'w',
            encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    if args.dump:
        dump_dict = {
            'img_metas': unwrap_for_dump(img_metas),
            'sample': unwrap_for_dump(sample),
            'result': unwrap_for_dump(result),
        }
        dump_path = osp.join(args.out_dir, f'frame_{index:06d}_dump.pkl')
        with open(dump_path, 'wb') as f:
            pickle.dump(dump_dict, f)

    print(f'[Done] split={args.split} index={index} saved to: {args.out_dir}')


if __name__ == '__main__':
    main()
