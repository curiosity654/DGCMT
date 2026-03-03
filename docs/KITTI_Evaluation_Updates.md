# KITTI Evaluation Updates (Unified Dataset)

This document records the recent KITTI evaluation-related updates in the unified dataset pipeline.

Last updated: 2026-03-02

## Summary

### 1) KITTI GT dimension order fix
- File: `projects/mmdet3d_plugin/datasets/unified_mmdet3d_dataset.py`
- In `_load_kitti_gts()`, KITTI `label_2` raw size is `h,w,l`.
- Internal KITTI eval path expects dimensions aligned with camera-box convention (`l,h,w`).
- GT dimension order is now converted before evaluation to avoid systematic BEV/3D IoU mismatch.

### 2) KITTI prediction FOV filter (implemented first stage)
- File: `projects/mmdet3d_plugin/datasets/unified_mmdet3d_dataset.py`
- In `_format_results_kitti()`, prediction boxes are now filtered by center-point FOV before formatting:
  - positive camera depth (`depth > 0`)
  - projected center inside image range (`0 <= u < img_w`, `0 <= v < img_h`)
- This aligns prediction scope better with KITTI front-view annotation coverage.

### 3) KITTI config point-dimension alignment
- Files:
  - `projects/configs/fusion/eval_config_unified_kitti.py`
  - `projects/configs/fusion/vis_config_unified_kitti.py`
- KITTI point loading is aligned to 5-dim usage in current model pipeline:
  - `load_dim=5`
  - `use_dim=[0, 1, 2, 3, 4]`

### 4) Eval config supports a train-style pipeline for dump/vis
- File: `projects/configs/fusion/eval_config_unified_kitti.py`
- Added `train_pipeline` variant while keeping test data behavior:
  - same test-like logic
  - only difference is annotation loading (`LoadAnnotationsFromTri3D`)
  - includes `gt_bboxes_3d` and `gt_labels_3d` in `Collect3D`
- Added `data.train` using this pipeline with `test_mode=True` to keep inference/eval behavior stable.

### 5) Visualization tooling compatibility fixes (for KITTI debug workflow)
- File: `tools/pipeline_vis.py`
  - Added robust unwrapping for nested `DataContainer`/list outputs (especially with `MultiScaleFlipAug3D`).
  - Added `img_metas` normalization helper to support both dict/list styles.
  - Prevented crashes from unexpected points container shape during PLY export.
- File: `tools/infer_vis.py`
  - `--show-gt` now warns explicitly if dump does not contain GT keys.
  - Default GT color remains `255,255,255` (B,G,R).

## Recommended Commands

### KITTI evaluation
```bash
bash tools/dist_test.sh \
  projects/configs/fusion/eval_config_unified_kitti.py \
  checkpoints/cmt_voxel0075_vov_1600x640_epoch20.pth \
  1 \
  --eval bbox > logs/kitti_eval.log
```

### Dump one KITTI sample with GT (train split path)
```bash
python tools/pipeline_vis.py \
  --cfg projects/configs/fusion/eval_config_unified_kitti.py \
  --split train \
  --out-dir vis_output/debug/unified_kitti_dataset/train \
  --dump \
  --idx 100 \
  --gt
```

### Inference visualization with GT overlay
```bash
python tools/infer_vis.py \
  --pkl vis_output/debug/unified_kitti_dataset/train/0_100/0_100.pkl \
  --cfg projects/configs/fusion/eval_config_unified_kitti.py \
  --ckpt checkpoints/cmt_voxel0075_vov_1600x640_epoch20.pth \
  --show-gt \
  --gt-color 0,255,255
```

## Notes
- KITTI metric lines such as `AP40@0.70,0.70,0.70` and `AP40@0.70,0.50,0.50` correspond to IoU thresholds for `bbox, bev, 3d`.
- Better single-frame visualization does not necessarily imply high dataset-level AP.
- Current FOV filter is center-based. Additional strict filters (for example projected 2D area validity or range constraints) can be added later if needed.
