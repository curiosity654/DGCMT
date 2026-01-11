# projects/configs/fusion/eval_config_unified_kitti.py
_base_ = ['./cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset.py']

data_root = 'data/KITTI'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

test_pipeline = [
    dict(
        type='LoadPointsFromTri3D',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=[0, 1, 2, 3],
        sweeps_num=0,
        undo_z_rotation=False,
        timestamp_unit=1.0,
    ),
    dict(
        type='LoadMultiViewImageFromTri3D',
        undo_z_rotation=False,
    ),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
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
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size='same2max', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]

data = dict(
    test=dict(
        _delete_=True,
        type='UnifiedMMDet3DDataset',
        dataset_type='KITTI',
        data_root='data/KITTI',
        split='training',
        pipeline=test_pipeline,
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
