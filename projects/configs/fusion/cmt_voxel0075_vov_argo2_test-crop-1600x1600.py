"""
Argoverse2 Zero-shot Test Configuration - 1600x1600 with Crop

This config uses ResizeCropFlipImage to crop Argoverse2 images to 1600x1600.
Strategy: Resize to cover 1600x1600 (using max()), then center crop to 1600x1600.
This may crop some image content but maintains consistent 1600x1600 output.

Original AV2 image sizes:
- Horizontal cams: 2048x1550 (W x H)
- Vertical cams: 1550x2048 (W x H)

Processing:
- Horizontal: 2048x1550 -> resize to cover 1600x1600 -> 843x1600 -> crop to 1600x1600
- Vertical: 1550x2048 -> resize to cover 1600x1600 -> 1600x843 -> crop to 1600x1600
"""

_base_ = ['./cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset.py']

# Override data root for Argoverse2
data_root = 'data/argo2'

# NuScenes 10 classes (same as base config)
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

# Argoverse2 ida_aug_conf for 1600x1600 crop mode
ida_aug_conf_1600_crop = {
    "resize_lim": (0.31, 0.42),  # Adaptive resize range
    "final_dim": (1600, 1600),  # Square target
    "bot_pct_lim": (0.0, 0.0),  # Center crop (no bottom crop)
    "rot_lim": (0.0, 0.0),
    "H": 2048,  # Max original dimension
    "W": 2048,  # Max original dimension
    "rand_flip": False,  # No flip for testing
}

# Test pipeline with Argoverse2-specific settings
test_pipeline = [
    dict(
        type='LoadPointsFromTri3D',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        sweeps_num=10,
        undo_z_rotation=False,  # AV2 doesn't need rotation undo
        timestamp_unit=1e9,     # AV2 uses nanoseconds
    ),
    dict(
        type='LoadMultiViewImageFromTri3D',
        undo_z_rotation=False,  # AV2 doesn't need rotation undo
    ),
    # CRITICAL: PointsRangeFilter is required for correct detection
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            # ResizeCropFlipImage: resize to cover 1600x1600, then center crop
            # Default behavior: uses max(fH/H, fW/W) to cover final_dim, then crops excess
            dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf_1600_crop, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size='same2max', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

# Override data config
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        _delete_=True
    ),  # No training on AV2
    test=dict(
        _delete_=True,  # Delete inherited test config to avoid NuScenes-specific params
        type='UnifiedMMDet3DDataset',
        dataset_type='Argoverse2',
        data_root=data_root,
        split='val',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
    )
)

# Evaluation settings for Argoverse2
evaluation = dict(
    interval=1,
    eval_range_m=[0.0, 150.0],  # AV2 evaluates up to 150m
)
