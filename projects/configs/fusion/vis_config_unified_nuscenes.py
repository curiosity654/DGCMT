# projects/configs/fusion/vis_config_unified.py
_base_ = ['./cmt_voxel0075_vov_1600x640_cbgs-wo_gtsample-unified_dataset.py']

# 这里的变量在继承时会自动合并，我们直接引用即可
vis_pipeline = [
    dict(
        type='LoadPointsFromTri3D',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        sweeps_num=10,
    ),
    dict(type='LoadMultiViewImageFromTri3D'),
    dict(type='LoadAnnotationsFromTri3D'),
    # 移除 ModalMask3D (避免背景变黑)
    dict(type='PointsRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='ObjectRangeFilter', point_cloud_range=[-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]),
    dict(type='ObjectNameFilter', classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]),
    dict(type='PointShuffle'),
    # 保持基础变换但关闭随机性 (training=False)
    dict(type='ResizeCropFlipImage', 
         data_aug_conf = dict(
            resize_lim=(0.94, 1.25),
            final_dim=(640, 1600),
            bot_pct_lim=(0.0, 0.0),
            rot_lim=(0.0, 0.0),
            H=900,
            W=1600,
            rand_flip=True), 
         training=False),
    dict(type='NormalizeMultiviewImage', 
         mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ]),
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
        type='UnifiedMMDet3DDataset',
        dataset_type='NuScenes',
        data_root='data/nuscenes/',
        subset='v1.0-trainval',
        split='train',
        pipeline=vis_pipeline,
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=True,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR')
)
