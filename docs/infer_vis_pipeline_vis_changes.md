# Infer/Pipeline Vis Updates (OpenCode + Codex)

This document records the recent changes to visualization tooling implemented by OpenCode + Codex.

Last updated: 2026-01-21

## Summary

### tools/pipeline_vis.py
- Dump now includes raw 3D GT boxes in addition to corners.
- Added `gt_bboxes_3d` to the dump payload as the raw box tensor (numpy) when available.
- Existing `gt_bboxes_3d_corners` and `gt_labels_3d` dumps are preserved.

### tools/infer_vis.py
- Added GT overlay options for inference visualization:
  - `--show-gt` toggles GT drawing if present in the dump.
  - `--gt-color` controls GT color (B,G,R), default `255,255,255`.
- GT box reconstruction now respects box dimension by passing `box_dim` to `LiDARInstance3DBoxes`.
- Added BEV visualization output (`bev_boxes.png`) with optional GT overlay when `--show-gt` is enabled.
- BEV polygon extraction changed to use a convex hull over 8 corners for correct width visualization.
- Added BEV range override for consistent scaling:
  - `--bev-range xmin,ymin,xmax,ymax` (default `-54,-54,54,54`).

## Usage

### Dump pipeline outputs with GT
```
python tools/pipeline_vis.py --cfg <cfg> --token <token> --split <train|val|vis> --dump
```

### Infer and visualize with GT + BEV
```
python tools/infer_vis.py --pkl <dump.pkl> --cfg <cfg> --ckpt <ckpt> --show-gt --bev-range -54,-54,54,54
```

## Notes
- BEV rendering is now consistent with 3D visualization width after switching to convex hull polygons.
- These updates were implemented by OpenCode + Codex as requested.
