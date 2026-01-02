# projects/configs/fusion/vis_config_unified_argo2.py
_base_ = ['./cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset.py']

# Override base data_root
data_root = 'data/argo2'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

# Argoverse2 image size: 2048x1550 (7 ring cameras)
ida_aug_conf_av2 = dict(
    resize_lim=(0.47, 0.625),  # Adjusted for 2048x1550 -> ~1600x640
    final_dim=(640, 1600),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=1550,
    W=2048,
    rand_flip=False,  # No flip for visualization
)

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

vis_pipeline = [
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
    dict(
        type='LoadAnnotationsFromTri3D',
        undo_z_rotation=False,  # AV2 doesn't need rotation undo
    ),
    # Range filter
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    # For visualization, we skip ResizeCropFlipImage to keep original aspect ratio
    # and avoid distortion. The projection will use original intrinsics.
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size='same2max', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d'))
]

data = dict(
    vis=dict(
        _delete_=True,  # Delete inherited config
        type='UnifiedMMDet3DDataset',
        dataset_type='Argoverse2',
        data_root='data/argo2',
        split='val',
        pipeline=vis_pipeline,
        classes=class_names,
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
    )
)
