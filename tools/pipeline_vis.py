# tools/pipeline_vis.py
import os
import sys
import argparse
import mmcv
import numpy as np
import cv2
import torch
from typing import Any, Dict
from mmcv import Config
from mmdet3d.datasets import build_dataset

# 将项目根目录添加到 python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.abspath(os.path.join(script_dir, '..'))
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

# 导入插件以注册 CustomNuScenesDataset 和自定义 Pipeline
import projects.mmdet3d_plugin

def get_color_map(class_names):
    """为不同类别生成颜色"""
    color_map = {}
    for i, name in enumerate(class_names):
        hue = int(i * (180 / len(class_names)))
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color_map[i] = tuple(map(int, color_bgr))
    return color_map

def extract_result(dc):
    """从 DataContainer 中提取结果"""
    if hasattr(dc, 'data'):
        res = dc.data
        if isinstance(res, list):
            return res[0]
        return res
    return dc

def to_numpy(x):
    """通用的转换函数，处理 DataContainer, Tensor, BasePoints 等"""
    x = extract_result(x)
    
    # 处理 mmdet3d 的 Points 对象 (LiDARPoints, etc.)
    if hasattr(x, 'tensor'):
        x = x.tensor
        
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        return x.numpy()
    return np.array(x)

def build_dump_from_results(results: Dict[str, Any], extra_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Build a picklable dump dict from pipeline results.
    Only includes what is needed for model inference and visualization.
    """
    dump: Dict[str, Any] = {}
    # Core tensors
    dump['points'] = to_numpy(extract_result(results['points']))
    dump['img'] = to_numpy(extract_result(results['img']))

    # Extract img_metas first since some keys might be inside it
    metas = results.get('img_metas', None)
    if metas is not None:
        metas = extract_result(metas)
    
    # Meta used at inference/visualization
    # Try both top-level and img_metas for these keys
    for k in ['lidar2img', 'intrinsics', 'extrinsics', 'timestamp', 'img_timestamp', 'ego_pose', 'ego_pose_inv', 'prev_exists']:
        v = None
        # First try top-level
        if k in results:
            v = extract_result(results[k])
            # For batch data like lidar2img, intrinsics, extrinsics, extract first element
            if k in ['lidar2img', 'intrinsics', 'extrinsics'] and isinstance(v, (list, tuple, np.ndarray)) and len(v) > 0:
                v = v[0]
        # If not found in top-level, try img_metas
        elif metas is not None:
            if isinstance(metas, dict) and k in metas:
                v = metas[k]
            elif isinstance(metas, (list, tuple)) and len(metas) > 0 and isinstance(metas[0], dict) and k in metas[0]:
                v = metas[0][k]
        
        if v is not None:
            dump[k] = to_numpy(v)

    # img_metas: store a minimal subset to avoid non-picklable objects
    if metas is not None:
        # metas is a dict-like or list-like depending on pipeline; extract safe fields
        safe = {}
        for mk in ['pts_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'box_mode_3d', 'box_type_3d', 'scene_token']:
            if isinstance(metas, dict) and mk in metas:
                safe[mk] = metas[mk]
            elif isinstance(metas, (list, tuple)) and len(metas) > 0 and isinstance(metas[0], dict) and mk in metas[0]:
                safe[mk] = metas[0][mk]
        dump['img_metas_min'] = safe

    # Extra meta for de-normalization and class info
    dump['img_norm_cfg'] = extra_meta.get('img_norm_cfg')
    dump['img_scale'] = extra_meta.get('img_scale')
    dump['class_names'] = extra_meta.get('class_names')
    dump['point_cloud_range'] = extra_meta.get('point_cloud_range')

    # Optional: keep 2D annotations if present (for debugging/vis)
    for opt_k in ['gt_bboxes', 'gt_labels', 'centers2d', 'depths']:
        if opt_k in results:
            dump[opt_k] = to_numpy(extract_result(results[opt_k]))

    # 3D GT for single-scene evaluation
    if 'gt_bboxes_3d' in results:
        gt_bboxes_3d = extract_result(results['gt_bboxes_3d'])
        try:
            corners = gt_bboxes_3d.corners  # (M, 8, 3)
            dump['gt_bboxes_3d_corners'] = to_numpy(corners)
        except Exception:
            pass
    if 'gt_labels_3d' in results:
        dump['gt_labels_3d'] = to_numpy(results['gt_labels_3d'])

    return dump

def export_ply_with_boxes(points, boxes_3d, labels_3d, color_map, filename):
    """导出包含点云和 3D GT 框的 PLY 文件"""
    from tools.infer_vis import export_point_cloud_with_boxes
    
    # pts_np shape: [N, C]
    pts_np = to_numpy(points)
    
    # boxes_3d should be a LiDARInstance3DBoxes or similar
    # labels_3d should be a numpy array
    
    export_point_cloud_with_boxes(
        pts_np[:, :3],
        pts_np[:, 3] if pts_np.shape[1] > 3 else None,
        boxes_3d,
        to_numpy(labels_3d),
        color_map,
        filename=filename
    )

def main():
    parser = argparse.ArgumentParser(description='Simplified GT Visualization')
    parser.add_argument('--cfg', default='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py')
    parser.add_argument('--idx', type=int, default=100, help='Sample index')
    parser.add_argument('--split', default='vis', choices=['train', 'val', 'test', 'vis'], help='Dataset split to use')
    parser.add_argument('--out-dir', default='vis_output/gt')
    parser.add_argument('--gt', action='store_true', help='Whether to visualize GT boxes')
    parser.add_argument('--dump', action='store_true', help='Whether to dump data to pkl')
    parser.add_argument('--dump-path', type=str, default=None, help='Path to dump pkl')
    args = parser.parse_args()

    # 1. 加载配置并构建数据集
    cfg = Config.fromfile(args.cfg)
    
    # 根据 split 获取对应的配置
    if args.split == 'train':
        data_cfg = cfg.data.train
    elif args.split == 'val':
        data_cfg = cfg.data.val
    elif args.split == 'vis':
        data_cfg = cfg.data.vis
    else:
        data_cfg = cfg.data.test

    # 处理 CBGSDataset 包装
    target_cfg = data_cfg.dataset if hasattr(data_cfg, 'dataset') else data_cfg
    
    # 只有在 config 中没有定义 data_root 时才使用默认值
    if not hasattr(target_cfg, 'data_root') or target_cfg.data_root is None:
        target_cfg.data_root = 'data/nuscenes/'

    # 如果是 val/test 模式，通常需要处理 MultiScaleFlipAug3D 包装并设置 test_mode
    # 注意：'vis' 模式已经在 config 中设置好了 test_mode=False 和干净的 pipeline
    if args.split in ['val', 'test']:
        target_cfg.test_mode = True
        if hasattr(target_cfg, 'pipeline'):
            new_pipeline = []
            for p in target_cfg.pipeline:
                if p['type'] == 'MultiScaleFlipAug3D':
                    new_pipeline.extend(p['transforms'])
                else:
                    new_pipeline.append(p)
            target_cfg.pipeline = new_pipeline

    dataset = build_dataset(data_cfg)
    
    # 2. 获取样本数据
    print(f"Loading sample index {args.idx}...")
    data = dataset[args.idx]
    
    # 3. 准备输出目录
    real_dataset = dataset
    if hasattr(dataset, 'dataset'):
        real_dataset = dataset.dataset
        
    info = real_dataset.data_infos[args.idx % len(real_dataset)]
    sample_token = info.get('token', info.get('sample_idx', f'idx_{args.idx}'))
    output_dir = os.path.join(args.out_dir, f'sample_{args.idx}_{str(sample_token)[:8]}')
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. 提取数据
    points = data['points']
    imgs_tensor = extract_result(data['img']) # [num_views, C, H, W]
    img_metas = extract_result(data['img_metas'])
    
    gt_bboxes_3d = None
    gt_labels_3d = None
    if args.gt:
        if 'gt_bboxes_3d' in data:
            gt_bboxes_3d = extract_result(data['gt_bboxes_3d']) # LiDARInstance3DBoxes
            gt_labels_3d = data['gt_labels_3d']
        else:
            print("Warning: --gt is set but 'gt_bboxes_3d' not found in data. Skipping GT visualization.")
    
    # 注意：在 cmt 配置中，lidar2img 可能在 img_metas 里
    lidar2img = img_metas.get('lidar2img', None)
    
    class_names = cfg.class_names
    color_map = get_color_map(class_names)

    # 5. 3D 可视化
    ply_path = os.path.join(output_dir, 'points.ply')
    if gt_bboxes_3d is not None:
        export_ply_with_boxes(points, gt_bboxes_3d, gt_labels_3d, color_map, ply_path)
        print(f"3D GT saved to: {ply_path}")
    else:
        # 仅导出点云
        from tools.infer_vis import export_point_cloud_with_boxes
        pts_np = to_numpy(points)
        export_point_cloud_with_boxes(pts_np[:, :3], pts_np[:, 3] if pts_np.shape[1] > 3 else None, 
                                    None, None, color_map, filename=ply_path)
        print(f"3D points saved to: {ply_path}")

    # 6. 2D 投影可视化
    num_views = imgs_tensor.shape[0]
    img_norm_cfg = img_metas.get('img_norm_cfg', cfg.img_norm_cfg)
    
    for view_id in range(num_views):
        # 反归一化
        img = imgs_tensor[view_id].permute(1, 2, 0).cpu().numpy()
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        img = (img * std + mean).clip(0, 255).astype(np.uint8)
        img = img.copy() # Ensure contiguity for OpenCV
        
        if img_norm_cfg.get('to_rgb', False):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
        # 裁剪回原始尺寸（仅当使用了 PadMultiViewImage 且没有 Resize 时）
        if 'ori_shape' in img_metas and 'img_shape' in img_metas:
            ori_h, ori_w = img_metas['ori_shape'][view_id][:2]
            img_h, img_w = img_metas['img_shape'][view_id][:2]
            # 如果 img_shape 等于 ori_shape，说明没有 Resize，只有可能存在 Padding
            if img_h == ori_h and img_w == ori_w:
                # 检查实际 img 是否被 padding 了（例如被 PadMultiViewImage）
                curr_h, curr_w = img.shape[:2]
                if curr_h > ori_h or curr_w > ori_w:
                    img = img[:ori_h, :ori_w]

        # 绘制投影
        if lidar2img is not None and gt_bboxes_3d is not None:
            corners_3d = gt_bboxes_3d.corners.cpu().numpy()
            labels_3d_np = to_numpy(gt_labels_3d)
            
            for i, corners in enumerate(corners_3d):
                label = int(labels_3d_np[i])
                color = color_map.get(label, (0, 255, 0))
                
                corners_hom = np.concatenate([corners, np.ones((8, 1))], axis=1)
                pts_2d_hom = corners_hom @ lidar2img[view_id].T
                
                if np.all(pts_2d_hom[:, 2] > 0) and np.all(np.isfinite(pts_2d_hom)):
                    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
                    
                    # 确保坐标是有限的且可以转换为整数
                    if not np.all(np.isfinite(pts_2d)):
                        continue
                        
                    edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],
                        [4, 5], [5, 6], [6, 7], [7, 4],
                        [0, 4], [1, 5], [2, 6], [3, 7]
                    ]
                    for edge in edges:
                        p1 = pts_2d[edge[0]]
                        p2 = pts_2d[edge[1]]
                        
                        # OpenCV 要求必须是标准 python int 类型的 tuple
                        pt1 = (int(p1[0]), int(p1[1]))
                        pt2 = (int(p2[0]), int(p2[1]))
                        
                        cv2.line(img, pt1, pt2, color, 2)

        out_img_path = os.path.join(output_dir, f'view_{view_id}.jpg')
        cv2.imwrite(out_img_path, img)
        
    print(f"2D Projections saved to: {output_dir}")

    # 7. Dump data
    if args.dump:
        extra_meta = dict(
            img_norm_cfg=img_norm_cfg,
            img_scale=img_metas.get('img_shape', [None])[0][:2] if 'img_shape' in img_metas else None,
            class_names=class_names,
            point_cloud_range=cfg.get('point_cloud_range', None),
        )
        dump_dict = build_dump_from_results(data, extra_meta)
        folder_name = os.path.basename(output_dir)
        dump_path = args.dump_path or os.path.join(output_dir, f'{folder_name}.pkl')
        mmcv.dump(dump_dict, dump_path)
        print(f"Dumped pipeline outputs to {dump_path}")

if __name__ == '__main__':
    main()
