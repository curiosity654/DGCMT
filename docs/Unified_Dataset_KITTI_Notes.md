# Unified Dataset KITTI Notes

This document summarizes the KITTI support added for the unified Tri3D-based dataset workflow.

## Scope

- Add KITTI support to `tri3d` (data_object_* format under `data/KITTI`).
- Enable unified dataset loading, visualization, and evaluation.
- Provide minimal configs for visualization and evaluation.

## KITTI Dataset Layout

Expected directory layout under `data/KITTI`:

- `training/calib/*.txt`
- `training/image_2/*.png`
- `training/image_3/*.png`
- `training/label_2/*.txt`
- `training/velodyne/*.bin`
- `testing/` mirrors the same subfolders but without `label_2`.

## Tri3D KITTI Adapter

Location:
- `thirdparty/tri3d/tri3d/datasets/kitti.py`

Key behavior:
- Single sequence (`seq=0`), frame ids derived from velodyne filenames.
- Point cloud from `velodyne/*.bin` (x, y, z, intensity).
- Cameras: `CAM2`/`CAM3` with image planes `IMG2`/`IMG3`.
- Calibration:
  - Uses `Tr_velo_to_cam` and `R0_rect` for lidar->cam.
  - Uses `P2`/`P3` for projection into image plane.
- 3D boxes:
  - Parsed from `label_2/*.txt` (camera frame).
  - Converted to LiDAR frame using `inv(R0_rect * Tr_velo_to_cam)`.
  - Center converted from bottom center to gravity center (add `h/2`).
- Poses are identity (KITTI has no global pose sequence).

Registered in:
- `thirdparty/tri3d/tri3d/datasets/__init__.py`

## Unified Dataset Integration

Location:
- `projects/mmdet3d_plugin/datasets/unified_mmdet3d_dataset.py`

Additions:
- KITTI label mapping for unified 10-class training.
- KITTI evaluation mapping (NuScenes -> KITTI classes).
- KITTI evaluation path using `mmdet3d.core.evaluation.kitti_eval`.

## Pipeline Notes

Location:
- `projects/mmdet3d_plugin/datasets/pipelines/tri3d_pipelines.py`

For KITTI:
- `LoadMultiViewImageFromTri3D` uses full `P2/P3 * R0_rect * Tr_velo_to_cam` to build `lidar2img`.
- Uses `IMG2`/`IMG3` for image plane selection.

## Configs

Visualization:
- `projects/configs/fusion/vis_config_unified_kitti.py`

Evaluation:
- `projects/configs/fusion/eval_config_unified_kitti.py`

Both configs use the same padding strategy as the Waymo vis config:
- `PadMultiViewImage` with `size='same2max'` and `size_divisor=32`.

## Commands

Visualization:

```bash
conda run -n dgcmt-codex python tools/pipeline_vis.py \
  --cfg projects/configs/fusion/vis_config_unified_kitti.py \
  --idx 100 --split vis --out-dir vis_output/gt/unified_kitti_dataset --gt
```

Evaluation:

```bash
conda run -n dgcmt-codex python tools/test.py \
  projects/configs/fusion/eval_config_unified_kitti.py \
  /path/to/ckpt.pth --eval bbox
```
