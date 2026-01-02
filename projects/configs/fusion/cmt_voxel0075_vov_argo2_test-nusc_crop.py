"""
Argoverse2 Zero-shot Test Configuration

This config is for testing a NuScenes-trained CMT model on Argoverse2 dataset.
Key differences from NuScenes config:
1. undo_z_rotation=False in all Pipeline classes (AV2 uses native Tri3D coords)
2. timestamp_unit=1e9 for AV2 nanoseconds (vs 1e6 for NuScenes microseconds)
3. Uses Argoverse2 dataset type and evaluation
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

# Argoverse2 image size is different - 1550x2048 (7 ring cameras)
# Adjust ida_aug_conf for Argoverse2
ida_aug_conf_av2 = {
    "resize_lim": (0.47, 0.625),  # Adjusted for 2048x1550 -> ~1600x640
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 1550,
    "W": 2048,
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
            dict(type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf_av2, training=False),
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
