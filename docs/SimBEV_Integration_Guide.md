# SimBEV 数据集集成指南

本文档详细说明如何让 UnifiedMMDet3DDataset 支持 SimBEV 数据集的训练和测试。

## 概述

SimBEV 是一个基于 Carla 仿真器的合成多任务多传感器驾驶数据生成工具。本集成方案通过实现 Tri3D 数据集接口，使现有的 CMT Fusion 模型能够直接在 SimBEV 数据上进行训练和评估。

## 文件修改清单

### 1. 新建文件

#### `thirdparty/tri3d/tri3d/datasets/simbev.py`
SimBEV 数据集的核心实现，包含：
- `SimBEV` 类：继承自 Tri3D 的 `Dataset` 基类
- 解析 SimBEV JSON 元数据格式
- 加载点云数据 (`.npz` 格式)
- 加载多视角图像 (7个相机)
- 加载 3D 边界框标注
- 实现了所有 Tri3D `Dataset` 基类要求的接口方法

**关键特性：**
- 相机传感器：`CAM_FRONT_LEFT`, `CAM_FRONT`, `CAM_FRONT_RIGHT`, `CAM_BACK_LEFT`, `CAM_BACK`, `CAM_BACK_RIGHT`
- 点云传感器：`LIDAR`
- 支持坐标系变换：sensor → ego → world
- 类别映射：SimBEV 标签 → NuScenes 10类

#### `projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset-simbev.py`
完整的 CMT Fusion 模型配置文件，包含：
- 针对 SimBEV 数据集的特定配置
- 6类检测任务：`car`, `truck`, `bus`, `motorcycle`, `bicycle`, `pedestrian`
- 数据预处理 pipeline
- 模型架构配置
- 训练和测试配置

### 2. 修改文件

#### `thirdparty/tri3d/tri3d/datasets/__init__.py`
**修改内容：** 注册 SimBEV 数据集类

```python
from .simbev import SimBEV

__all__ = [
    # ... 其他类
    "SimBEV",
]
```

#### `projects/mmdet3d_plugin/datasets/unified_mmdet3d_dataset.py`
**修改内容：** 添加 SimBEV 支持

1. **新增类别映射字典：**
```python
SIMBEV_MAPPING = {
    'car': 'car',
    'truck': 'truck',
    'bus': 'bus',
    'motorcycle': 'motorcycle',
    'bicycle': 'bicycle',
    'pedestrian': 'pedestrian',
    'van': 'car',  # Map van to car
    'trailer': 'trailer',
}
```

2. **在 `__init__` 中添加 SimBEV 分支：**
```python
elif dataset_type == 'SimBEV':
    self.cat_mapping = self.SIMBEV_MAPPING
```

## 数据结构说明

### SimBEV 数据组织
```
data/simbev/original/
├── simbev_infos_train.json   # 训练集元数据
├── simbev_infos_val.json     # 验证集元数据
├── simbev_infos_test.json    # 测试集元数据
└── sweeps/                   # 传感器数据
    ├── LIDAR/               # 点云 (.npz)
    ├── RGB-CAM_FRONT/       # 图像 (.jpg)
    └── ... (共7个相机)
```

### 元数据格式 (JSON)
```json
{
    "metadata": {
        "camera_intrinsics": [[...]],
        "LIDAR": {
            "sensor2lidar_translation": [...],
            "sensor2lidar_rotation": [...],
            "sensor2ego_translation": [...],
            "sensor2ego_rotation": [...]
        },
        "CAM_FRONT": {...},
        ...
    },
    "data": {
        "scene_0000": {
            "scene_info": {...},
            "scene_data": [
                {
                    "ego2global_translation": [...],
                    "ego2global_rotation": [...],
                    "timestamp": ...,
                    "RGB-CAM_FRONT": "/path/to/image.jpg",
                    "LIDAR": "/path/to/points.npz",
                    "GT_DET": "/path/to/annotations.bin"
                },
                ...
            ]
        }
    }
}
```

## 使用方法

### 训练模型
```bash
python tools/train.py \
    projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset-simbev.py
```

### 测试模型
```bash
python tools/test.py \
    projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset-simbev.py \
    checkpoints/your_checkpoint.pth \
    --eval bbox
```

### 可视化
```bash
python tools/infer_vis.py \
    projects/configs/fusion/cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset-simbev.py \
    checkpoints/your_checkpoint.pth \
    --show-dir vis_output
```

## 注意事项

1. **类别映射**：SimBEV 的 `van` 类被映射到 `car`，以保持与 NuScenes 10类的一致性
2. **点云格式**：SimBEV 使用 `.npz` 格式存储点云，包含 `x, y, z, intensity` 等字段
3. **时间戳**：SimBEV 的时间戳以微秒为单位，需要与 NuScenes 的纳秒格式区分
4. **坐标系**：SimBEV 使用与 NuScenes 相似的坐标系，X 轴向前，Y 轴向左，Z 轴向上

## 故障排除

### 问题：无法加载 GT_DET 文件
**解决方案**：确保 `ground_truth/det` 目录中的 `.bin` 文件已正确解压

### 问题：点云加载失败
**解决方案**：检查 `sweeps/LIDAR/` 目录中的 `.npz` 文件是否存在且格式正确

### 问题：类别映射错误
**解决方案**：检查 `UnifiedMMDet3DDataset.SIMBEV_MAPPING` 字典是否包含所有需要的类别映射

## 参考

- [SimBEV GitHub Repository](https://github.com/GoodarzMehr/SimBEV)
- [Tri3D Documentation](../../thirdparty/tri3d/docs/)
- [CMT Fusion Paper](https://arxiv.org/abs/2206.00609)
