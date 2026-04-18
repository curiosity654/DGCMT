_base_ = ['../cmt-cbgs-nuscenes-3_class.py']

dataset_type = 'CustomNuScenes3ClassDataset'
class_names = ['vehicle', 'bicycle', 'pedestrian']
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False,
)
ida_aug_conf = dict(
    resize_lim=(0.94, 1.25),
    final_dim=(640, 1600),
    bot_pct_lim=(0.0, 0.0),
    rot_lim=(0.0, 0.0),
    H=900,
    W=1600,
    rand_flip=True,
)

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
        test_mode=True,
    ),
    dict(type='LoadMultiViewImageFromFiles'),
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
                translation_std=[0, 0, 0],
            ),
            dict(type='RandomFlip3D'),
            dict(
                type='ResizeCropFlipImage',
                data_aug_conf=ida_aug_conf,
                training=False,
            ),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False,
            ),
            dict(type='Collect3D', keys=['points', 'img']),
        ],
    ),
]

data = dict(
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        filter_empty_gt=False,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
        filter_empty_gt=False,
    ),
)

evaluation = dict(
    interval=1,
    export_unieval_package=True,
    format_only=True,
    export_dir='evaluation/unieval/cmt_nuscenes_3class',
    export_split='val',
    label_space='unified:3class',
    coord_system='lidar',
    box_origin='bottom_center',
    source_codebase='dgcmt',
    task='detection',
)
