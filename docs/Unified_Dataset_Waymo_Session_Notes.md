# Unified Dataset Waymo Session Notes

This document summarizes the Waymo-related changes and operational notes from this session.

## Code Changes

- Added Waymo label mapping and official metric evaluation in `projects/mmdet3d_plugin/datasets/unified_mmdet3d_dataset.py`.
  - Uses `waymo_open_dataset.metrics.python.wod_detection_evaluator.WODDetectionEvaluator`.
  - Maps Unified NuScenes classes to Waymo types (VEHICLE/PEDESTRIAN/CYCLIST/SIGN).
  - Produces per-breakdown mAP/mAPH and Overall L1/L2 metrics.
- Added Waymo visualization config `projects/configs/fusion/vis_config_unified_waymo.py`.
- Added Waymo evaluation config `projects/configs/fusion/eval_config_unified_waymo.py`.
  - Uses the same padding strategy as the vis config (`PadMultiViewImage` with `size='same2max'`).
- Updated `projects/mmdet3d_plugin/datasets/pipelines/tri3d_pipelines.py` to skip camera sensors with empty timelines (Waymo validation has 5 active cameras and 3 empty ones). This avoids `IndexError` during image loading.
- Updated Tri3D Waymo dataset init to accept `timeline_root` (original layout) while reading actual data from `data_root` (optimized layout).
- Added `tri3d_kwargs` passthrough in `UnifiedMMDet3DDataset` so configs can forward `timeline_root`.

## Dataset Layout Requirements (Waymo Parquet)

Tri3D Waymo expects the following directories under each split (e.g. `data/waymo/validation/`):

- `camera_box/`
- `camera_calibration/`
- `camera_image/`
- `camera_segmentation/`
- `lidar/`
- `lidar_box/`
- `lidar_calibration/`
- `lidar_pose/`
- `lidar_segmentation/`
- `vehicle_pose/`

### Metadata Files

PyArrow expects `_metadata.parquet` in parquet directories. If you only have `_metadata` (no suffix), create a symlink:

```bash
bash create_waymo_metadata_links.sh
```

Note: `camera_box/_metadata` must be moved or renamed, otherwise Tri3D will treat it as a record and crash. Example:

```bash
mkdir -p data/waymo/validation/camera_box/metadata_backup
mv data/waymo/validation/camera_box/_metadata data/waymo/validation/camera_box/metadata_backup/
```

## Environment Notes

Waymo official metrics require `waymo-open-dataset-tf-2-12-0==1.6.7`.
Installing it pulled in TensorFlow 2.13 and downgraded several packages (e.g. `typing-extensions`, `pandas`, `matplotlib`, `pillow`, `pyarrow`). If unrelated tooling breaks, the environment may need reconciliation.

## Known Performance Bottlenecks

Evaluation on Waymo is slow because Tri3D Waymo does heavy per-frame parquet reads and range-image decoding:

- `Waymo._points` reads parquet + decodes range images + applies lidar pose compensation.
- `sweeps_num=10` multiplies this cost by 10.

Optimizations discussed:

- Use `sweeps_num=0` for evaluation to reduce IO.
- Disable camera loading for lidar-only eval.
- Add caching for `_points`, `_poses`, `_calibration`, `_boxes` at the unified wrapper layer (without modifying `thirdparty/`).

## Timeline Init vs. Optimized Reads

Waymo parquet files optimized by `thirdparty/tri3d/tri3d/datasets/optimize_waymo.py` are faster for timestamp-filtered reads, but Tri3D init builds timelines by scanning per-sensor timestamps. This can make init slower when using the optimized layout directly. The current setup splits the responsibilities:

- `data_root = data/waymo_optimized` for actual point/image/label reads.
- `timeline_root = data/waymo_timeline` for fast init (original layout or slimmed parquet).

`data/waymo_timeline` is a minimal dataset containing only the files/columns needed to build timelines:

- `camera_box`, `lidar`, `lidar_segmentation`, `camera_image`, `camera_segmentation`
- For each parquet, only timestamp + sensor id columns are kept.

Create it with:

```bash
python create_waymo_timeline.py --src data/waymo --dst data/waymo_timeline --slim --strict
```

Notes:

- The script skips `_metadata.parquet` and other non-parquet files.
- If you already created `data/waymo_timeline`, remove it before re-running to avoid stale full copies.

## Commands Used

Visualization (Waymo):

```bash
conda run -n dgcmt-codex python tools/pipeline_vis.py \
  --cfg projects/configs/fusion/vis_config_unified_waymo.py \
  --idx 100 --split vis --out-dir vis_output/gt/unified_waymo_dataset --gt
```

Evaluation (Waymo):

```bash
conda run -n dgcmt-codex python tools/test.py \
  projects/configs/fusion/eval_config_unified_waymo.py \
  /path/to/ckpt.pth --eval bbox
```
