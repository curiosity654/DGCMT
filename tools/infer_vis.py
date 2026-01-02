import os
import sys
import argparse
import json
import pickle
import traceback
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from copy import deepcopy

# 获取 playground/pipeline_vis.py 所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取工作区根目录 (playground 的上一级目录)
workspace_root = os.path.abspath(os.path.join(script_dir, '..'))

# 将工作区根目录添加到 sys.path
if workspace_root not in sys.path:
    sys.path.append(workspace_root)

from mmcv import Config
from mmdet.datasets import replace_ImageToTensor
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from mmdet3d.apis import init_model
from mmcv.runner import load_checkpoint
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.points import BasePoints, LiDARPoints
from projects.mmdet3d_plugin.datasets.custom_nuscenes_dataset import CustomNuScenesDataset
# Pipeline modules are unused here since we load preprocessed data from pkl


def export_point_cloud(points, filename='debug/point_cloud.ply', colors=None):
    """将点云导出为ply文件
    Args:
        points (torch.Tensor | np.ndarray): 点云数据，形状为(N, 3)，
            其中N是点的数量，3表示(x, y, z)坐标
        colors (torch.Tensor | np.ndarray): 点云颜色数据，形状为(N, 3)，
            其中N是点的数量，3表示(r, g, b)颜色值
        filename (str): 输出的ply文件名
    """
    # 如果输入是torch.Tensor，转换为numpy数组
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    
    # 确保points是float32类型
    points = points.astype(np.float32)
    if colors is not None:
        colors = colors.astype(np.uint8)
    
    # 创建ply文件头部
    if colors is not None:
        header = [
            'ply',
            'format ascii 1.0',
            f'element vertex {len(points)}',
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header'
        ]
    else:
        header = [
            'ply',
            'format ascii 1.0',
            f'element vertex {len(points)}',
            'property float x',
            'property float y',
            'property float z',
            'end_header'
        ]
    
    # 创建输出目录
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # 写入ply文件
    with open(filename, 'w') as f:
        f.write('\n'.join(header) + '\n')
        if colors is not None:
            for point, color in zip(points, colors):
                f.write(f'{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n')
        else:
            for point in points:
                f.write(f'{point[0]} {point[1]} {point[2]}\n')

def draw_3d_box_projection(img, corners_2d, color=(0, 255, 0), thickness=2):
    """在图像上绘制3D框的2D投影
    Args:
        img (np.ndarray): 图像
        corners_2d (np.ndarray): 3D框角点的2D投影坐标 (8, 2)
        color (tuple): 颜色
        thickness (int): 线条粗细
    """
    # 定义边的连接关系
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接底面和顶面的边
    ]
    
    # 绘制每条边
    for edge in edges:
        pt1 = tuple(map(int, corners_2d[edge[0]]))
        pt2 = tuple(map(int, corners_2d[edge[1]]))
        cv2.line(img, pt1, pt2, color, thickness)

def draw_2d_box(img, bbox, color=(0, 255, 0), thickness=2):
    """在图像上绘制2D边界框
    Args:
        img (np.ndarray): 图像
        bbox (np.ndarray): 2D边界框 [x1, y1, w, h]
        color (tuple): 颜色
        thickness (int): 线条粗细
    """
    x1, y1, w, h = map(int, bbox)
    # 计算边界框左上角坐标，假设[x1, y1]是中心点坐标
    # left = int(x1 - w/2)
    # top = int(y1 - h/2)
    left = int(x1)
    top = int(y1)
    # 绘制以中心点为基准的矩形
    cv2.rectangle(img, (left, top), (left + w, top + h), color, thickness)

def draw_center_point(img, center_2d, color=(255, 0, 0), radius=3):
    """在图像上绘制中心点
    Args:
        img (np.ndarray): 图像
        center_2d (np.ndarray): 中心点的2D坐标
        color (tuple): 颜色
        radius (int): 点的半径
    """
    center_2d = center_2d.copy()
    center = tuple(map(int, center_2d))
    cv2.circle(img, center, radius, color, -1)

def visualize_sparse_depth(img, sparse_depth, view_id, output_dir):
    """可视化稀疏深度图
    Args:
        img (np.ndarray): 原始图像
        sparse_depth (torch.Tensor): 稀疏深度图 [num_scale, 2, H, W]
        view_id (int): 视角ID
        output_dir (str): 输出目录
    """
    # 创建深度图可视化目录
    depth_dir = os.path.join(output_dir, 'sparse_depth')
    os.makedirs(depth_dir, exist_ok=True)
    
    # 获取第一个尺度的深度图
    depth_map = sparse_depth[0].detach().cpu().numpy()  # [H, W]
    depth_mask = sparse_depth[1].detach().cpu().numpy()  # [H, W]
    
    # 将深度值归一化到[0, 255]范围
    depth_map = depth_map * np.sqrt(156.89) + 14.41  # 反归一化
    depth_map = np.clip(depth_map, 0, 80)  # 限制深度范围
    depth_map = (depth_map / 80 * 255).astype(np.uint8)
    
    # 创建彩色深度图
    depth_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
    
    # 创建掩码
    mask = depth_mask > 0
    depth_colormap[~mask] = 0
    
    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 显示原始图像
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # 显示深度图
    axes[1].imshow(depth_colormap)
    axes[1].set_title('Sparse Depth Map')
    axes[1].axis('off')
    
    # 显示掩码
    axes[2].imshow(mask, cmap='gray')
    axes[2].set_title('Depth Mask')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(depth_dir, f'sparse_depth_view_{view_id}.png'))
    plt.close()
    
    # 保存单独的深度图
    cv2.imwrite(os.path.join(depth_dir, f'depth_map_view_{view_id}.png'), depth_colormap)
    cv2.imwrite(os.path.join(depth_dir, f'depth_mask_view_{view_id}.png'), (mask * 255).astype(np.uint8))

def export_point_cloud_with_boxes(points, points_intensity, boxes_3d, labels_3d, color_map, filename='debug/point_cloud_with_boxes.ply'):
    """将点云和3D边界框导出为同一个PLY文件 (使用边元素)
    Args:
        points (np.ndarray): 点云数据 [N, 3]
        points_intensity (np.ndarray): 点云强度值 [N]
        boxes_3d (object): 包含边界框信息的对象，需要有 .corners 属性 (Tensor/ndarray [M, 8, 3])
                         或本身就是一个 [M, 8, 3] 的 ndarray.
        labels_3d (torch.Tensor): 3D边界框标签 [M]
        color_map (dict): 类别到颜色的映射.
        filename (str): 输出文件名
    """
    # 准备点云顶点和颜色
    all_vertices = [points.astype(np.float32)]

    # 将强度值归一化到0-255并应用颜色图
    if points_intensity is not None and points_intensity.size > 0 and points_intensity.max() > points_intensity.min():
        normalized_intensity = ((points_intensity - points_intensity.min()) /
                              (points_intensity.max() - points_intensity.min()) * 255).astype(np.uint8)
    elif points_intensity is not None and points_intensity.size > 0: # Handle constant intensity
         normalized_intensity = np.full_like(points_intensity, 128, dtype=np.uint8) # Use mid-gray for constant intensity
    else: # Handle empty or None intensity
        normalized_intensity = np.zeros(points.shape[0], dtype=np.uint8) # Use black if no intensity

    colormap = cv2.applyColorMap(normalized_intensity[:, None], cv2.COLORMAP_JET)
    point_colors = colormap.squeeze(axis=1) # Squeeze potential extra dim
    if point_colors.ndim == 1: # Handle single point case
        point_colors = point_colors[np.newaxis, :]
    all_colors = [point_colors.astype(np.uint8)]

    # 准备边
    all_edges = []
    vertex_offset = len(points) # 顶点索引的起始偏移量

    # 定义单个框的边连接关系 (相对于该框的8个角点索引 0-7)
    box_edges_indices = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
        [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
        [0, 4], [1, 5], [2, 6], [3, 7]   # 连接边
    ]

    # 处理每个边界框
    box_corners_list = None
    num_boxes = 0
    if hasattr(boxes_3d, 'corners'):
        box_corners_tensor = boxes_3d.corners # Assuming shape [M, 8, 3]
        num_boxes = box_corners_tensor.shape[0]
        if hasattr(box_corners_tensor, 'cpu') and hasattr(box_corners_tensor, 'numpy'): # Check if it's a tensor (PyTorch/TF)
             box_corners_list = box_corners_tensor.cpu().numpy()
        elif isinstance(box_corners_tensor, np.ndarray): # Check if it's already numpy
             box_corners_list = box_corners_tensor
        else:
             print(f"Warning: boxes_3d.corners is of unexpected type: {type(box_corners_tensor)}. Trying to proceed.")
             try:
                 box_corners_list = np.array(box_corners_tensor) # Attempt conversion
                 if box_corners_list.shape[1:] != (8, 3): raise ValueError("Incorrect shape")
             except Exception as e:
                 print(f"Error: Could not convert boxes_3d.corners to numpy array: {e}")
                 box_corners_list = None

    elif isinstance(boxes_3d, np.ndarray) and boxes_3d.ndim == 3 and boxes_3d.shape[1:] == (8, 3):
        # Handle case where boxes_3d is directly a numpy array of corners [M, 8, 3]
        box_corners_list = boxes_3d
        num_boxes = boxes_3d.shape[0]
    elif boxes_3d is not None:
        print("Warning: Cannot determine box corners. 'boxes_3d' should have a '.corners' attribute or be an [M, 8, 3] numpy array.")

    if box_corners_list is not None:
        for i in range(num_boxes):
            corners = box_corners_list[i] # Shape [8, 3]
            label = labels_3d[i].item()
            color = color_map.get(label, (0, 0, 255))  # Default to red if not found
            box_color = np.array(color, dtype=np.uint8)

            all_vertices.append(corners.astype(np.float32))
            all_colors.append(np.tile(box_color, (8, 1)))

            # 添加当前框的边，注意索引要加上偏移量
            for edge_pair in box_edges_indices:
                # 边的两个顶点索引需要加上 vertex_offset
                all_edges.append([edge_pair[0] + vertex_offset, edge_pair[1] + vertex_offset])

            vertex_offset += 8 # 更新下一个框的起始索引
    elif boxes_3d is not None:
        print("Skipping box edge export due to issues with corner data.")


    # 合并所有顶点和颜色
    if not all_vertices: # Handle case with no points and no boxes
        print("Warning: No vertices to export.")
        return

    final_vertices = np.vstack(all_vertices)
    final_colors = np.vstack(all_colors)

    # 调用新的导出函数
    export_ply_with_edges(final_vertices, final_colors, all_edges, filename)

def export_ply_with_edges(vertices, colors, edges, filename):
    """导出包含顶点、颜色和边的PLY文件
    Args:
        vertices (np.ndarray): 顶点坐标 [N, 3] (float32)
        colors (np.ndarray): 顶点颜色 [N, 3] (uint8)
        edges (list of list): 边的顶点索引对 [[idx1, idx2], ...]
        filename (str): 输出文件名
    """
    num_vertices = vertices.shape[0]
    num_edges = len(edges)

    if num_vertices == 0:
        print(f"Warning: Skipping export to {filename} as there are no vertices.")
        return

    # 确保目录存在
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")


    # 确保数据类型正确
    vertices = vertices.astype(np.float32)
    colors = colors.astype(np.uint8)

    # 构造PLY文件头
    header_lines = [
        "ply",
        "format ascii 1.0",
        f"comment Generated by script",
        f"element vertex {num_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue"
    ]
    if num_edges > 0:
        header_lines.extend([
            f"element edge {num_edges}",
            "property int vertex1",
            "property int vertex2"
        ])
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"


    # 写入文件
    try:
        with open(filename, 'w') as f:
            f.write(header)
            # 写入顶点数据 (x, y, z, r, g, b)
            for i in range(num_vertices):
                f.write(f"{vertices[i, 0]:.6f} {vertices[i, 1]:.6f} {vertices[i, 2]:.6f} {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n")
            # 写入边数据 (vertex1, vertex2)
            if num_edges > 0:
                for edge in edges:
                    f.write(f"{edge[0]} {edge[1]}\n")
        print(f"Point cloud with {num_vertices} vertices and {num_edges} edges saved to {filename}")
    except Exception as e:
        print(f"Error writing PLY file {filename}: {e}")

def project_lidar_to_image(points_lidar, lidar2img_matrix, img_shape):
    """将LiDAR点投影到图像平面
    Args:
        points_lidar (np.ndarray): LiDAR点云 (N, 3+)
        lidar2img_matrix (np.ndarray): lidar2img变换矩阵 (4, 4)
        img_shape (tuple): 图像形状 (H, W)
    Returns:
        tuple: (投影点坐标 (M, 2), 投影点深度 (M,), 投影点索引 (M,))
    """
    points_xyz = points_lidar[:, :3]
    points_hom = np.hstack((points_xyz, np.ones((points_xyz.shape[0], 1)))) # N, 4

    # 投影到相机坐标系+齐次坐标
    points_cam_hom = points_hom @ lidar2img_matrix.T # (N, 4) @ (4, 4) -> (N, 4)

    # 深度值
    depth = points_cam_hom[:, 2]

    # 过滤掉深度小于等于0的点
    positive_depth_mask = depth > 1e-6
    points_cam_hom = points_cam_hom[positive_depth_mask]
    depth = depth[positive_depth_mask]
    original_indices = np.arange(len(points_lidar))[positive_depth_mask] # 跟踪原始索引

    if points_cam_hom.shape[0] == 0:
        return np.empty((0, 2)), np.empty((0,)), np.empty((0,), dtype=int)

    # 透视除法得到像素坐标 (u, v)
    points_img = points_cam_hom[:, :2] / points_cam_hom[:, 2:3]

    # 过滤掉图像边界外的点
    H, W = img_shape[:2]
    in_bounds_mask = (points_img[:, 0] >= 0) & (points_img[:, 0] < W) & \
                     (points_img[:, 1] >= 0) & (points_img[:, 1] < H)

    points_img_valid = points_img[in_bounds_mask]
    depth_valid = depth[in_bounds_mask]
    indices_valid = original_indices[in_bounds_mask]

    return points_img_valid, depth_valid, indices_valid

def get_color_map(class_names):
    """为类别创建颜色映射"""
    color_map = {}
    for i, name in enumerate(class_names):
        # 使用HSV颜色空间生成区分度高的颜色
        hue = int(i * (180 / len(class_names)))
        # 将HSV颜色转换为BGR
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        color_map[i] = tuple(map(int, color_bgr))
    return color_map

def build_data_from_dump(dump: dict, device: torch.device):
    """Build model input dict from dumped pkl content."""
    imgs_np = dump['img']  # (N, C, H, W)
    points_np = dump['points']  # (M, 5)
    lidar2img = dump.get('lidar2img')
    intrinsics = dump.get('intrinsics')
    extrinsics = dump.get('extrinsics')
    prev_exists = dump.get('prev_exists')
    img_timestamp = dump.get('img_timestamp')
    ego_pose = dump.get('ego_pose')
    ego_pose_inv = dump.get('ego_pose_inv')
    pts_filename = dump.get('pts_filename')

    # Tensors
    img_tensor = torch.from_numpy(imgs_np).float().to(device)
    pts_tensor = torch.from_numpy(points_np).float().to(device)

    data: dict = {}
    data['img'] = [img_tensor.unsqueeze(0)]
    data['points'] = [pts_tensor.unsqueeze(0)]

    # Build img_metas minimal dict
    metas_min = dump.get('img_metas_min', {})
    img_metas = {
        **metas_min,
        'lidar2img': lidar2img,
        'intrinsics': intrinsics,
        'extrinsics': extrinsics,
        'img_norm_cfg': dump.get('img_norm_cfg'),
        'img_shape': metas_min.get('img_shape', img_tensor.shape[-2:][::-1] if img_tensor.dim() == 4 else None),
    }
    data['img_metas'] = [[img_metas]]

    # Other fields expected by the model (keep device for tensors)
    for k, v in {
        'pts_filename': pts_filename,
        # 'lidar2img': lidar2img,
        # 'intrinsics': intrinsics,
        # 'extrinsics': extrinsics,
        # 'prev_exists': prev_exists,
        # 'img_timestamp': img_timestamp,
        # 'ego_pose': ego_pose,
        # 'ego_pose_inv': ego_pose_inv,
    }.items():
        if v is None:
            continue
        # Convert numpy to torch when appropriate; keep lists/tuples as-is
        if isinstance(v, np.ndarray):
            v_t = torch.from_numpy(v).to(device)
            data[k] = [[v_t]]
        else:
            data[k] = [[v]]

    return data

def _bev_poly_from_corners(corners_xyz: np.ndarray) -> np.ndarray:
    """Return BEV polygon (4 points) from 3D 8 corners: take XY of bottom face 0..3."""
    if corners_xyz.shape[0] != 8:
        return None
    # Assuming corners ordering matches bottom 0..3, top 4..7
    poly = corners_xyz[:4, :2].astype(np.float32)
    return poly

def _poly_iou_bev(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """Compute polygon IoU on BEV by rasterization (robust, dependency-free)."""
    if poly1 is None or poly2 is None:
        return 0.0
    # Translate to positive coords
    all_pts = np.vstack([poly1, poly2])
    min_xy = np.min(all_pts, axis=0)
    poly1_s = (poly1 - min_xy + 1.0) * 10.0
    poly2_s = (poly2 - min_xy + 1.0) * 10.0
    max_xy = np.max(np.vstack([poly1_s, poly2_s]), axis=0) + 10
    W = int(np.clip(max_xy[0], 32, 4096))
    H = int(np.clip(max_xy[1], 32, 4096))
    mask1 = np.zeros((H, W), dtype=np.uint8)
    mask2 = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask1, [poly1_s.astype(np.int32)], 1)
    cv2.fillPoly(mask2, [poly2_s.astype(np.int32)], 1)
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return float(inter) / float(union) if union > 0 else 0.0

def single_scene_eval_bev(gt_corners: np.ndarray, gt_labels: np.ndarray,
                          pred_boxes, pred_labels: torch.Tensor,
                          iou_thr: float = 0.5) -> dict:
    """Greedy match predictions to GT by class using BEV IoU.
    Returns TP, FP, FN and mAP-like stats per class (precision/recall at iou_thr).
    """
    pred_corners = pred_boxes.corners.detach().cpu().numpy()  # (P, 8, 3)
    pred_labels_np = pred_labels.detach().cpu().numpy() if hasattr(pred_labels, 'detach') else np.array(pred_labels)
    gt_labels_np = gt_labels.astype(np.int64).reshape(-1) if gt_labels is not None else np.zeros((0,), dtype=np.int64)

    if len(gt_labels_np) > 0:
        classes = np.unique(np.concatenate([gt_labels_np, pred_labels_np]))
    else:
        classes = np.unique(pred_labels_np)
    metrics = {}
    for c in classes:
        gt_idx = np.where(gt_labels_np == c)[0]
        pr_idx = np.where(pred_labels_np == c)[0]
        gt_used = np.zeros(len(gt_idx), dtype=bool)
        tp = 0
        for pi in pr_idx:
            pc = _bev_poly_from_corners(pred_corners[pi])
            best_iou = 0.0
            best_g = -1
            for gi_i, gi in enumerate(gt_idx):
                if gt_used[gi_i]:
                    continue
                gc = _bev_poly_from_corners(gt_corners[gi]) if gt_corners is not None and len(gt_idx)>0 else None
                iou = _poly_iou_bev(pc, gc)
                if iou > best_iou:
                    best_iou = iou
                    best_g = gi_i
            if best_iou >= iou_thr and best_g >= 0:
                tp += 1
                gt_used[best_g] = True
        fp = len(pr_idx) - tp
        fn = len(gt_idx) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics[int(c)] = dict(tp=tp, fp=fp, fn=fn, precision=prec, recall=rec)
    # Aggregate
    all_tp = sum(m['tp'] for m in metrics.values())
    all_fp = sum(m['fp'] for m in metrics.values())
    all_fn = sum(m['fn'] for m in metrics.values())
    metrics['overall'] = dict(
        tp=all_tp, fp=all_fp, fn=all_fn,
        precision=all_tp/(all_tp+all_fp) if (all_tp+all_fp)>0 else 0.0,
        recall=all_tp/(all_tp+all_fn) if (all_tp+all_fn)>0 else 0.0,
    )
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Run inference and visualization from dumped pkl')
    parser.add_argument('--pkl', type=str, required=True, help='path to dumped pkl from pipeline_vis.py')
    parser.add_argument('--cfg', type=str, default='projects/configs/nusc/split/decoupled_loss/gt_sample/mv2dfusion-isfusion_freeze-swint_single-no_mem_v2-decoder_decoupled_loss-gt_sample.py')
    parser.add_argument('--ckpt', type=str, default='checkpoints/isfusion_split_converted.pth')
    parser.add_argument('--out-dir', type=str, default='vis_output/infer_from_pkl')
    parser.add_argument('--iou-thr', type=float, default=0.5, help='IoU threshold for single-scene BEV evaluation')
    parser.add_argument('--score-thr', type=float, default=0.0, help='Score threshold for visualization (filter boxes with score < threshold)')
    args = parser.parse_args()

    # Load dump
    with open(args.pkl, 'rb') as f:
        dump = pickle.load(f)

    # Load model
    cfg = Config.fromfile(args.cfg)
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.ckpt, map_location='cpu')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Colors and de-norm
    class_names = dump.get('class_names', cfg.get('class_names', []))
    color_map = get_color_map(class_names)
    img_norm_cfg = dump.get('img_norm_cfg', dict(mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375], to_rgb=True))

    # Output dir
    sample_name = os.path.splitext(os.path.basename(args.pkl))[0]
    output_dir = os.path.join(args.out_dir, sample_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build data dict
    data = build_data_from_dump(dump, device)

    # Inference
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    # Per-view 2D visualization
    img_tensors = data['img'][0][0]
    num_views = img_tensors.shape[0]
    lidar2img_matrices = data['img_metas'][0][0]['lidar2img']

    # debug_attention_maps(output_dir, img_tensors, img_norm_cfg)

    results = results[0]['pts_bbox']
    pred_bboxes = results['boxes_3d']
    pred_scores = results['scores_3d']
    pred_labels = results['labels_3d']

    # Filter by score threshold
    if args.score_thr > 0:
        score_mask = pred_scores >= args.score_thr
        pred_bboxes = pred_bboxes[score_mask]
        pred_scores = pred_scores[score_mask]
        pred_labels = pred_labels[score_mask]
        print(f'Filtered predictions: {score_mask.sum().item()}/{len(score_mask)} boxes with score >= {args.score_thr}')

    # Save 3D PLY with boxes
    points_np = data['points'][0][0].detach().cpu().numpy()
    export_point_cloud_with_boxes(
        points_np[:, :3],
        points_np[:, 3] if points_np.shape[1] > 3 else None,
        pred_bboxes,
        pred_labels,
        color_map,
        filename=os.path.join(output_dir, 'pred_point_cloud_with_boxes.ply')
    )

    # Per-view 2D visualization
    img_tensors = data['img'][0]
    num_views = img_tensors.shape[0]
    lidar2img_matrices = data['img_metas'][0][0]['lidar2img']

    for view_id in range(num_views):
        img_tensor = img_tensors[view_id]
        lidar2img = lidar2img_matrices[view_id]

        # NOTE: lidar2img from dump file should already be correctly updated by the pipeline
        # (e.g., by ResizeCropFlipImage), so we use it directly without further adjustment

        # To numpy (C,H,W)->(H,W,C)
        img = img_tensor.permute(1, 2, 0).contiguous().detach().cpu().numpy()

        # De-normalize
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        if img_norm_cfg.get('to_rgb', True):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)

        # 智能裁剪：检查是否有 padding
        # 如果 img_shape > ori_shape，说明图片被 pad 了，需要裁剪回 ori_shape
        img_metas_dict = data['img_metas'][0][0]
        if 'ori_shape' in img_metas_dict and 'img_shape' in img_metas_dict:
            ori_shapes = img_metas_dict['ori_shape']
            img_shapes = img_metas_dict['img_shape']
            if isinstance(ori_shapes, (list, tuple)) and isinstance(img_shapes, (list, tuple)):
                if view_id < len(ori_shapes) and view_id < len(img_shapes):
                    ori_h, ori_w = ori_shapes[view_id][:2]
                    img_h, img_w = img_shapes[view_id][:2]
                    curr_h, curr_w = img.shape[:2]
                    # 如果 img_shape > ori_shape（有padding），裁剪到 ori_shape
                    if img_h > ori_h or img_w > ori_w:
                        # 裁剪到原始尺寸
                        img = img[:ori_h, :ori_w]

        vis_img = img.copy()

        corners_3d = pred_bboxes.corners.detach().cpu().numpy()
        scores_np = pred_scores.detach().cpu().numpy() if hasattr(pred_scores, 'detach') else pred_scores
        for i, corners in enumerate(corners_3d):
            label = int(pred_labels[i].item()) if hasattr(pred_labels[i], 'item') else int(pred_labels[i])
            score = float(scores_np[i])
            color = color_map.get(label, (0, 0, 255))

            corners_hom = np.concatenate([corners, np.ones((8, 1))], axis=1)
            corners_2d_hom = corners_hom @ np.array(lidar2img).T
            if np.all(corners_2d_hom[:, 2] > 0):
                corners_2d = corners_2d_hom[:, :2] / corners_2d_hom[:, 2:3]
                draw_3d_box_projection(vis_img, corners_2d, color=color, thickness=2)
                
                # Draw score text
                text = f'{score:.2f}'
                text_pos = tuple(map(int, corners_2d.mean(axis=0)))
                cv2.putText(vis_img, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out_path = os.path.join(output_dir, f'vis_view_{view_id}.png')
        cv2.imwrite(out_path, vis_img)
        print(f'Saved visualization to {out_path}')

    # Single-scene evaluation (BEV IoU greedy matching)
    gt_corners = dump.get('gt_bboxes_3d_corners', None)
    gt_labels_3d = dump.get('gt_labels_3d', None)
    if gt_corners is not None and gt_labels_3d is not None:
        try:
            gt_corners_np = np.array(gt_corners)
            gt_labels_np = np.array(gt_labels_3d).astype(np.int64)
            metrics = single_scene_eval_bev(gt_corners_np, gt_labels_np, pred_bboxes, pred_labels, iou_thr=args.iou_thr)
            metrics_path = os.path.join(output_dir, 'single_scene_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f'Saved single-scene eval metrics to {metrics_path}')
        except Exception as e:
            print(f'Failed single-scene eval: {e}')

def debug_attention_maps(output_dir, img_tensors, img_norm_cfg):
    debug_dir = f"{output_dir}/attention"
    os.makedirs(debug_dir, exist_ok=True)

    # debug attention
    from projects.mmdet3d_plugin.models.utils.attention import AttentionStore
    from skimage import exposure
    import numpy as np
    for i in range(6):
        attention_dir = os.path.join(debug_dir, f'attention_{i+1}')
        os.makedirs(attention_dir, exist_ok=True)
        attention = AttentionStore.get_store(f'attention_{i+1}')  # [1, 8, num_query, total_tokens]
        if attention is not None:
            # 计算所有head的平均attention
            bev_tokens = 180 * 180
            bev_attention = attention[..., :bev_tokens]  # [1, num_query, 180*180]
            bev_attention = bev_attention.mean(dim=1)  # [1, num_query, total_tokens]
            # 重塑BEV attention为网格形状
            bev_attention = bev_attention.view(1, -1, 180, 180)  # [1, num_query, 180, 180]
            # 对每个query取平均，得到整体的attention map
            # bev_attention = bev_attention.mean(dim=1)  # [1, 180, 180]
            bev_attention = bev_attention.max(dim=1)[0]
            
            # 转换为numpy数组
            bev_attention_np = bev_attention[0].cpu().numpy() if bev_attention[0].is_cuda else bev_attention[0].numpy()
            
            bev_attention_norm = bev_attention_np

            # p_min, p_max = np.percentile(bev_attention_np, [1, 99])  # ✅ 裁剪1%极端值
            # bev_attention_clipped = np.clip(bev_attention_np, p_min, p_max)
            # bev_attention_norm = bev_attention_clipped
            # bev_attention_norm = (bev_attention_clipped - p_min) / (p_max - p_min)

            # bev_attention_equalized = exposure.equalize_hist(bev_attention_np)

            # bev_attention_norm = bev_attention_equalized
            
            # 保存原始灰度图（用于对比）
            plt.imsave(os.path.join(attention_dir, 'bev_attention_gray.png'), bev_attention_norm, cmap='gray')
            
            # 转换为彩色热力图
            bev_heatmap = cv2.applyColorMap((bev_attention_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # OpenCV使用BGR，需要转换为RGB用于保存
            bev_heatmap_rgb = cv2.cvtColor(bev_heatmap, cv2.COLOR_BGR2RGB)
            
            # 保存彩色热力图
            cv2.imwrite(os.path.join(attention_dir, 'bev_attention_heatmap.png'), bev_heatmap)
            
            # 使用matplotlib保存更高质量的热力图（带colorbar）
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(bev_attention_norm, cmap='jet', aspect='auto', vmin=bev_attention_norm.min(), vmax=bev_attention_norm.max())
            # im = ax.imshow(bev_attention_norm, cmap='jet', aspect='auto')
            ax.set_title(f'BEV Attention Heatmap - Layer {i+1}')
            ax.set_xlabel('X (BEV)')
            ax.set_ylabel('Y (BEV)')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(os.path.join(attention_dir, 'bev_attention_heatmap_with_colorbar.png'), dpi=150)
            plt.close()
            
            print(f'Saved BEV attention heatmap to {attention_dir}')

        rv_attention_keypoints, rv_attention_weights = AttentionStore.get_store(f'deform_attention_keypoints_{i+1}'), AttentionStore.get_store(f'deform_attention_weights_{i+1}')
        # rv_attention_keypoints: [num_views, num_query, num_heads, num_levels, num_points, 2]
        # rv_attention_weights: [num_views, num_query, num_heads, num_levels * num_points]
        
        if rv_attention_keypoints is not None and rv_attention_weights is not None:
            # 可视化RV deformable attention
            visualize_rv_deformable_attention(
                rv_attention_keypoints, 
                rv_attention_weights, 
                img_tensors, 
                img_norm_cfg, 
                attention_dir
            )

def visualize_rv_deformable_attention(keypoints, weights, img_tensors, img_norm_cfg, output_dir):
    """可视化RV deformable attention的采样点和权重（批量化优化版本）
    Args:
        keypoints (torch.Tensor): 采样点坐标 [num_views, num_query, num_heads, num_levels, num_points, 2]
        weights (torch.Tensor): 采样点权重 [num_views, num_query, num_heads, num_levels * num_points]
        img_tensors (torch.Tensor): 图像tensor [num_views, C, H, W]
        img_norm_cfg (dict): 图像归一化配置
        output_dir (str): 输出目录
    """
    keypoints = keypoints.cpu().numpy()
    weights = weights.cpu().numpy()
    
    num_views, num_query, num_heads, num_levels, num_points, _ = keypoints.shape
    
    # 重塑weights以匹配keypoints的维度
    weights = weights.reshape(num_views, num_query, num_heads, num_levels, num_points)
    
    # 对每个视角进行可视化
    for view_id in range(num_views):
        img_tensor = img_tensors[view_id]
        
        # 转换图像为numpy (C,H,W)->(H,W,C)
        img = img_tensor.permute(1, 2, 0).contiguous().detach().cpu().numpy()
        
        # 反归一化
        mean = np.array(img_norm_cfg['mean'], dtype=np.float32)
        std = np.array(img_norm_cfg['std'], dtype=np.float32)
        if img_norm_cfg.get('to_rgb', True):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        H, W = img.shape[:2]
        
        # 获取当前视角的keypoints和weights
        view_keypoints = keypoints[view_id]  # [num_query, num_heads, num_levels, num_points, 2]
        view_weights = weights[view_id]      # [num_query, num_heads, num_levels, num_points]
        
        # 归一化权重到[0, 1]
        weight_min = view_weights.min()
        weight_max = view_weights.max()
        if weight_max > weight_min:
            view_weights_norm = (view_weights - weight_min) / (weight_max - weight_min)
        else:
            view_weights_norm = np.zeros_like(view_weights)
        
        # === 批量化处理：展平所有维度 ===
        # 形状: [num_query * num_heads * num_levels * num_points, 2]
        all_keypoints = view_keypoints.reshape(-1, 2)
        # 形状: [num_query * num_heads * num_levels * num_points]
        all_weights = view_weights_norm.reshape(-1)
        # 为每个点分配head_id用于颜色映射
        # 形状: [num_query * num_heads * num_levels * num_points]
        head_ids = np.repeat(np.arange(num_heads), num_query * num_levels * num_points)
        head_ids = np.tile(head_ids.reshape(num_heads, -1).T.reshape(-1), 1)
        # 重新计算正确的head_ids
        head_ids = np.tile(
            np.repeat(np.arange(num_heads), num_levels * num_points),
            num_query
        )
        
        # 转换为像素坐标
        x_pix = (all_keypoints[:, 0] * W).astype(np.int32)
        y_pix = (all_keypoints[:, 1] * H).astype(np.int32)
        
        # 过滤出图像范围内的点
        valid_mask = (x_pix >= 0) & (x_pix < W) & (y_pix >= 0) & (y_pix < H)
        x_pix_valid = x_pix[valid_mask]
        y_pix_valid = y_pix[valid_mask]
        weights_valid = all_weights[valid_mask]
        head_ids_valid = head_ids[valid_mask]
        
        # 为每个head使用不同的颜色
        head_colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, num_heads))[:, :3] * 255
        
        # === 批量绘制采样点（完全向量化版本）===
        vis_img = img.copy().astype(np.float32)
        
        # 按权重排序，先绘制权重小的点（避免被覆盖）
        sort_idx = np.argsort(weights_valid)
        x_pix_sorted = x_pix_valid[sort_idx]
        y_pix_sorted = y_pix_valid[sort_idx]
        weights_sorted = weights_valid[sort_idx]
        head_ids_sorted = head_ids_valid[sort_idx]
        
        # 使用更高效的批量绘制方法
        # 预先计算所有点的半径和alpha
        alphas = weights_sorted * 0.7
        radii = (2 + weights_sorted * 3).astype(np.int32)
        
        # 批量绘制所有点
        for i in range(len(x_pix_sorted)):
            x, y = x_pix_sorted[i], y_pix_sorted[i]
            weight = weights_sorted[i]
            head_id = head_ids_sorted[i]
            alpha = alphas[i]
            radius = radii[i]
            color = head_colors[head_id]
            
            # 计算圆的边界框
            x1, y1 = max(0, x - radius), max(0, y - radius)
            x2, y2 = min(W, x + radius + 1), min(H, y + radius + 1)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 创建局部mask和overlay（只在需要的区域）
            local_h, local_w = y2 - y1, x2 - x1
            local_overlay = np.zeros((local_h, local_w, 3), dtype=np.float32)
            
            # 在局部坐标系中绘制圆
            local_x, local_y = x - x1, y - y1
            cv2.circle(local_overlay, (local_x, local_y), radius, color, -1)
            
            # 创建mask
            local_mask = (local_overlay > 0).any(axis=2)
            
            # 只在圆的区域进行混合
            roi = vis_img[y1:y2, x1:x2]
            roi[local_mask] = roi[local_mask] * (1 - alpha) + local_overlay[local_mask] * alpha
            vis_img[y1:y2, x1:x2] = roi
        
        vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        
        # 保存可视化结果
        out_path = os.path.join(output_dir, f'rv_deform_attention_view_{view_id}.png')
        cv2.imwrite(out_path, vis_img)
        
        # === 批量化生成热力图（高效版本 - 使用局部更新）===
        heatmap = np.zeros((H, W), dtype=np.float32)
        
        sigma = 5
        radius_heat = int(3 * sigma)  # 3-sigma范围
        sigma_sq_2 = 2 * sigma * sigma
        
        # 预计算高斯核（可重用）
        kernel_size = 2 * radius_heat + 1
        kernel_center = radius_heat
        y_k, x_k = np.ogrid[-kernel_center:kernel_center+1, -kernel_center:kernel_center+1]
        gaussian_kernel = np.exp(-(x_k*x_k + y_k*y_k) / sigma_sq_2)
        
        # 批量处理所有有效点（使用局部更新而非全局网格）
        for i in range(len(x_pix_valid)):
            x, y = x_pix_valid[i], y_pix_valid[i]
            weight = weights_valid[i]
            
            # 计算热力图更新区域
            x1, y1 = max(0, x - radius_heat), max(0, y - radius_heat)
            x2, y2 = min(W, x + radius_heat + 1), min(H, y + radius_heat + 1)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 计算核在图像中的有效区域
            kx1, ky1 = kernel_center - (x - x1), kernel_center - (y - y1)
            kx2, ky2 = kx1 + (x2 - x1), ky1 + (y2 - y1)
            
            # 裁剪高斯核并应用
            kernel_crop = gaussian_kernel[ky1:ky2, kx1:kx2]
            heatmap[y1:y2, x1:x2] += kernel_crop * weight
        
        # 归一化热力图
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # 转换为彩色热力图
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # 与原图叠加
        heatmap_overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)
        
        # 保存热力图
        heatmap_path = os.path.join(output_dir, f'rv_deform_attention_heatmap_view_{view_id}.png')
        cv2.imwrite(heatmap_path, heatmap_overlay)
        
        print(f'Saved RV deformable attention visualization to {out_path}')
        print(f'Saved RV deformable attention heatmap to {heatmap_path}')

if __name__ == '__main__':
    main()