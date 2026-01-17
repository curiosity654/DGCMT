这是一份关于在 mmdet3d Codebase 中基于 `tri3d` 库实现 **`UnifiedMMDet3DDataset`** 的技术总结。

---

# 技术总结：基于 Tri3D 的统一多模态 3D 检测数据集框架

## 1. 总体框架设计 (Architecture Design)

为了实现多数据集（nuScenes, Waymo, Argo2 等）的 **Domain Generalization (DG)** 研究，我们设计并实现了一个统一的数据加载与处理框架。其核心思想是 **“接口统一，底层抽象”**。

### 核心组件
*   **`UnifiedMMDet3DDataset` (Dataset 层)**：作为 mmdet3d 的插件类，它不再依赖预处理的 `.pkl` 文件，而是直接通过 `tri3d` 库动态加载数据集。它负责管理索引、类别映射以及与评估 SDK 的对接。
*   **Tri3D 适配层**：利用 `tri3d` 库将不同数据集的原始存储格式抽象为统一的 `Sequence/Frame` 结构。
*   **自定义 Pipelines (变换层)**：实现了一套专用的数据读取 Pipeline（`LoadPointsFromTri3D`, `LoadMultiViewImageFromTri3D`, `LoadAnnotationsFromTri3D`），将 `tri3d` 对象转换为模型可用的 Tensor 格式。

---

## 2. 关键设计与重大改进历史

在实现过程中，我们解决了多个影响性能、对齐精度和训练收敛的深层问题：

### A. 性能优化：解决 Deepcopy 瓶颈
*   **问题**：在使用 `MultiScaleFlipAug3D` 等增强器时，评估速度从 9 fps 跌至 1.2 fps。
*   **原因**：`tri3d` 数据集对象非常庞大且嵌套复杂，`mmdet3d` 管道在每一帧都会对其进行 `deepcopy`，导致 CPU 产生巨大的递归拷贝开销。
*   **解决**：实现了 **`Tri3DObjectWrapper`**。通过重写 `__deepcopy__` 方法使其在深拷贝时仅返回自身引用，瞬间恢复了原始性能。

### B. 坐标系偏转修正 (The "Undo Z-90" Fix)
*   **问题**：初始训练时 Loss 极高（200+），梯度频繁出现 `NaN`。
*   **原因**：`tri3d` 为了统一坐标系，默认将 nuScenes 的 Lidar 坐标系绕 Z 轴旋转了 +90 度。这破坏了预训练模型学到的图像特征与 3D 空间的对齐关系，且导致 Heading 偏移。
*   **解决**：在所有数据加载管道中引入了 **`undo_tri3d_rot` (+90° 逆旋转补偿)**。将点云、GT 框和投影矩阵同步拨回 nuScenes 原生坐标系。
    *   **尺寸修正**：将尺寸顺序从 Tri3D 的 `[L, W, H]` 修正为 NuScenes 预期的 `[W, L, H]`。
    *   **航向修正**：修正了 Heading 偏移公式：`heading_native = box.heading + np.pi / 2`。

### C. 类别映射与数据对齐
*   **问题**：训练 Iter 数量不一致，mAP 低。
*   **原因**：
    1. 前缀模糊匹配（如 `human.pedestrian`）导致误收录了 `stroller`（婴儿车）等非标准类别。
    2. NuScenes 评估要求提交 Global 坐标系坐标，而初版提交的是本地 Lidar 坐标。
*   **解决**：
    *   **细化映射**：显式指定子类（`adult`, `child`, `police_officer` 等）以精确对齐官方 10 类。
    *   **缓存统计**：在 `load_annotations` 时预计算并缓存 `cat_ids`，加速 `CBGSDataset` 的初始化。

### D. 自动化评估集成 (Evaluation Support)
*   **Global 转换**：利用 `tri3d.alignment(..., 'global')` 结合坐标修正矩阵，实现了从 $Native\_Lidar \to Tri3D\_Lidar \to Global$ 的自动转换。
*   **中心点偏移**：修正了 $Z$ 轴坐标，将 mmdet3d 的 **Bottom Center** 自动转换为官方评估要求的 **Gravity Center**。
*   **SDK 兼容**：实现了 `format_results` 与 `evaluate` 接口，自动创建临时 JSON 并调用 `nuscenes-devkit` 计算 mAP 和 NDS。

---

## 3. 核心优势与未来扩展

1.  **极简配置**：现在只需在 Config 中切换 `dataset_type='Waymo'` 或 `'NuScenes'` 并指定 `split` 即可完成数据切换，无需再运行复杂的 `create_data.py`。
2.  **动态对齐**：利用 `tri3d` 的插值能力，实现了 Lidar 和 Camera 之间的高精度运动补偿，理论对齐精度优于传统的静态 `.pkl` 方式。
3.  **DG 研究友好**：所有数据集现在都输出相同的“NuScenes 视角”数据，模型可以在完全一致的几何空间内感知来自不同域的数据分布。

### 当前状态
*   **NuScenes**: 已完美适配，训练 Loss 与原始代码库对齐，评估指标正常。
*   **下一步**: 验证该框架在 Waymo 和 Argo2 上的 Zero-shot/Few-shot 迁移性能。

---

## 4. NuScenes Unified Dataset 修复记录 (2026-01)

### A. 相机顺序对齐
*   **问题**：Tri3D 的 `cam_sensors` 顺序来自原始 `sensor.json`，与 pkl 的固定顺序不一致，导致 `lidar2img` 视角错位。
*   **修复**：在 `thirdparty/tri3d/tri3d/datasets/nuscenes.py` 中强制排序为 `CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT`。

### B. `_format_bbox` 全局坐标变换
*   **问题**：修正点云坐标后，`_format_bbox` 仍使用 `Rot(+90)`，导致全局坐标偏转，评估 mAP 变为 0。
*   **修复**：`native2tri3d` 改为 `Rot(-90)`，与 pipeline 的 `undo_tri3d_rot=+90` 保持一致。

### C. `z` 重心修正
*   **问题**：mmdet3d 的 `LiDARInstance3DBoxes` 使用 bottom center，但 NuScenes 评估要求 gravity center，导致 `z` 偏差约 0.7~0.9m。
*   **修复**：在 `_format_bbox` 中使用 `z_center = z + h/2` 作为平移。

### D. 结果验证
*   **格式对齐**：`tools/compare_format_bbox.py` 对比 pkl GT 后，x/y/z/yaw 全部收敛到 1e-3 内。
*   **评估结果**：完整 val 集评估 mAP ≈ **70.26**，达到可接受范围。
