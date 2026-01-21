# Unified Dataset ISFusion NuScenes Debug Summary

## Current Symptom
- Unified NuScenes evaluation for ISFusion drops from ~72 mAP (official config) to ~65–70 mAP when using the unified dataset pipeline.

## Data Dump Comparison (Unified vs Original ISFusion)
Artifacts compared under:
- `vis_output/debug/unified_nusc-isfusion/*/*.pkl`
- `vis_output/debug/isfusion-test/*/*.pkl`

Key differences observed:
- **Image normalization mismatch**
  - Unified `img` values are not normalized (max ~1136) vs original normalized range (~2.6).
  - Suggests missing/incorrect `ImageNormalize` or different `img_norm_cfg`/order.
- **Camera order permutation**
  - `lidar2img` matrices match only after a permutation (example mapping A→B: `[0, 3, 4, 2, 1, 5]`).
  - If `img`, `lidar2img`, and `img_aug_matrix` are aligned, model should be mostly order-invariant; however any view-specific masking/embedding can break this.
- **Metadata gaps**
  - Unified `img_metas_min` contains only `box_type_3d`; original contains `ori_shape`, `img_shape`, `pad_shape`, `box_mode_3d`.
  - Unified `img_scale` is `None` vs original `(3, 384)`.
  - Original includes `timestamp`, unified does not.
- **Points**
  - `points` match numerically (max diff ~3.8e-06), so point coordinates are aligned.

## Likely Root Causes
1) **Image pipeline mismatch**: normalization/resize/crop sequence not identical to original ISFusion.
2) **Per-view augmentation mismatch**: current ImageAug3D may apply per-camera random aug instead of shared aug across all views.
3) **Metadata missing**: `img_metas` fields used by model (e.g., `input_shape`) may diverge.
4) **Camera order mismatch**: should be confirmed against original camera ordering even if theoretically order-invariant.

## Next Checks
- Ensure unified pipeline uses the exact original ISFusion image augmentation + normalization logic and ordering.
- Align `img_metas` fields with original dataset outputs.
- Verify that `img`, `lidar2img`, and `img_aug_matrix` are strictly aligned and share the same camera ordering as original configs.
