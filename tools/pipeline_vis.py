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

def draw_orientation_arrow(img, center_3d, yaw, lidar2img, color, arrow_length=2.0):
    """在 2D 图像中绘制 3D 物体朝向箭头
    
    Args:
        img: 图像数组
        center_3d: 3D box 中心点 [3]
        yaw: 偏航角 (radians)
        lidar2img: 投影矩阵 [4, 4]
        color: 箭头颜色
        arrow_length: 箭头长度（米），默认 2.0 米
    """
    # 根据 yaw 计算方向向量 (LiDAR 坐标系下，yaw=0 沿 X 轴)
    direction = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    
    # 箭头终点：从中心沿方向延伸
    arrow_end_3d = center_3d + direction * arrow_length
    
    # 投影到 2D
    center_hom = np.append(center_3d, 1.0)  # [4]
    arrow_end_hom = np.append(arrow_end_3d, 1.0)  # [4]
    
    center_2d_hom = lidar2img @ center_hom  # [4]
    arrow_end_2d_hom = lidar2img @ arrow_end_hom  # [4]
    
    # 检查深度是否为正
    if center_2d_hom[2] > 0 and arrow_end_2d_hom[2] > 0:
        # 透视除法
        center_2d = center_2d_hom[:2] / center_2d_hom[2]
        arrow_end_2d = arrow_end_2d_hom[:2] / arrow_end_2d_hom[2]
        
        # 检查坐标是否有效
        if np.all(np.isfinite(center_2d)) and np.all(np.isfinite(arrow_end_2d)):
            # 转换为整数坐标
            pt_start = (int(center_2d[0]), int(center_2d[1]))
            pt_end = (int(arrow_end_2d[0]), int(arrow_end_2d[1]))
            
            # 绘制箭头（线宽 3，箭头大小 0.3）
            cv2.arrowedLine(img, pt_start, pt_end, color, 3, tipLength=0.3)
            
            return True
    
    return False

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

def build_token_to_idx_map(dataset):
    """构建 token 到索引的映射"""
    token_to_idx = {}
    real_dataset = dataset
    if hasattr(dataset, 'dataset'):
        real_dataset = dataset.dataset
    
    for i in range(len(real_dataset)):
        info = real_dataset.data_infos[i]
        token = info.get('token', info.get('sample_idx', None))
        if token:
            token_to_idx[token] = i
    
    return token_to_idx


def load_tokens_from_file(token_file):
    """从文件加载 token 列表
    
    支持的格式:
    - 每行一个 token
    - 以 # 开头的行是注释
    - 空行会被忽略
    """
    tokens = []
    with open(token_file, 'r') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if line and not line.startswith('#'):
                tokens.append(line)
    return tokens


def main():
    parser = argparse.ArgumentParser(description='Simplified GT Visualization')
    parser.add_argument('--cfg', default='projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs.py')
    parser.add_argument('--idx', type=int, default=None, help='Sample index (deprecated, use --token instead)')
    parser.add_argument('--token', type=str, default=None, help='Sample token to visualize')
    parser.add_argument('--token-list', nargs='+', default=None, help='List of sample tokens to visualize')
    parser.add_argument('--token-file', type=str, default=None, help='File containing list of tokens (one per line)')
    parser.add_argument('--split', default='vis', choices=['train', 'val', 'test', 'vis'], help='Dataset split to use')
    parser.add_argument('--out-dir', default='vis_output/gt')
    parser.add_argument('--gt', action='store_true', help='Whether to visualize GT boxes')
    parser.add_argument('--show-orientation', action='store_true', help='Whether to show orientation arrows')
    parser.add_argument('--arrow-length', type=float, default=2.0, help='Length of orientation arrow in meters')
    parser.add_argument('--dump', action='store_true', help='Whether to dump data to pkl')
    parser.add_argument('--dump-path', type=str, default=None, help='Path to dump pkl')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    args = parser.parse_args()

    # 设置随机种子以确保可重复性
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

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
    
    # 2. 确定要可视化的样本列表
    tokens_to_visualize = []
    
    # 优先级: token-file > token-list > token > idx
    if args.token_file:
        print(f"Loading tokens from file: {args.token_file}")
        tokens_to_visualize = load_tokens_from_file(args.token_file)
        print(f"Loaded {len(tokens_to_visualize)} tokens from file")
    elif args.token_list:
        tokens_to_visualize = args.token_list
        print(f"Using {len(tokens_to_visualize)} tokens from command line")
    elif args.token:
        tokens_to_visualize = [args.token]
        print(f"Using single token: {args.token}")
    elif args.idx is not None:
        # 向后兼容：使用索引
        print(f"Using index mode (deprecated): idx={args.idx}")
        tokens_to_visualize = None
        sample_indices = [args.idx]
    else:
        # 默认使用索引 100
        print("No token or index specified, using default index 100")
        tokens_to_visualize = None
        sample_indices = [100]
    
    # 如果使用 token 模式，构建 token 到索引的映射
    if tokens_to_visualize:
        print("Building token to index mapping...")
        token_to_idx = build_token_to_idx_map(dataset)
        print(f"Found {len(token_to_idx)} tokens in dataset")
        
        # 将 tokens 转换为索引
        sample_indices = []
        for token in tokens_to_visualize:
            if token in token_to_idx:
                sample_indices.append(token_to_idx[token])
            else:
                print(f"Warning: Token '{token}' not found in dataset, skipping...")
        
        if len(sample_indices) == 0:
            print("Error: No valid tokens found!")
            return
        
        print(f"Will visualize {len(sample_indices)} samples")
    
    # 获取真实的底层数据集（绕过 CBGSDataset 等包装器）
    real_dataset = dataset
    if hasattr(dataset, 'dataset'):
        real_dataset = dataset.dataset
    
    # 3. 遍历所有要可视化的样本
    for sample_idx in sample_indices:
        print(f"\n{'='*80}")
        print(f"Processing sample index {sample_idx}...")
        print(f"{'='*80}")
        
        # 直接从底层数据集获取样本数据，避免 CBGSDataset 的随机采样
        data = real_dataset[sample_idx]
        
        # 准备输出目录
            
        info = real_dataset.data_infos[sample_idx % len(real_dataset)]
        sample_token = info.get('token', info.get('sample_idx', f'idx_{sample_idx}'))
        # 使用完整 token 作为目录名（避免不同 token 的前缀冲突）
        # NuScenes token 是32位字符串，其他数据集可能不同
        output_dir = os.path.join(args.out_dir, f'{sample_token}')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"  Sample index: {sample_idx}")
        print(f"  Token: {sample_token}")
        print(f"  Output: {output_dir}")
    
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
                corners_3d_all = gt_bboxes_3d.corners.cpu().numpy()
                boxes_3d_np = gt_bboxes_3d.tensor.cpu().numpy() # [M, 7/9] (x, y, z, l, w, h, yaw, ...)
                labels_3d_np = to_numpy(gt_labels_3d)
                
                for i, corners in enumerate(corners_3d_all):
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
                        
                        # 绘制朝向箭头
                        if args.show_orientation:
                            box_3d = boxes_3d_np[i]
                            center = box_3d[:3].copy()
                            # mmdet3d 的 boxes 中心通常在底面，需要移动到几何中心进行绘制
                            center[2] += box_3d[5] / 2.0 
                            yaw = box_3d[6]
                            draw_orientation_arrow(img, center, yaw, lidar2img[view_id], color, arrow_length=args.arrow_length)

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
    
    # 8. 输出总结
    print(f"\n{'='*80}")
    print(f"Visualization complete!")
    print(f"  Total samples processed: {len(sample_indices)}")
    print(f"  Output directory: {args.out_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
