_base_ = ["./cmt_voxel0075_vov_1600x640_cbgs-xyz.py"]

class_names = ["vehicle", "bicycle", "pedestrian"]
dataset_type = "CustomNuScenes3ClassDataset"
data_root = "data/nuscenes/"
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[57.375, 57.120, 58.395],
    to_rgb=False,
)
ida_aug_conf = {
    "resize_lim": (0.94, 1.25),
    "final_dim": (640, 1600),
    "bot_pct_lim": (0.0, 0.0),
    "rot_lim": (0.0, 0.0),
    "H": 900,
    "W": 1600,
    "rand_flip": True,
}

train_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2],
    ),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(type="LoadAnnotations3D", with_bbox_3d=True, with_label_3d=True),
    dict(
        type="UnifiedObjectSample",
        sample_2d=True,
        mixup_rate=0.5,
        db_sampler=dict(
            type="UnifiedDataBaseSampler",
            data_root=data_root,
            info_path=data_root + "nuscenes_3class_dbinfos_train.pkl",
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(vehicle=5, bicycle=5, pedestrian=5),
            ),
            classes=class_names,
            sample_groups=dict(vehicle=7, bicycle=6, pedestrian=2),
            points_loader=dict(
                type="LoadPointsFromFile",
                coord_type="LIDAR",
                load_dim=5,
                use_dim=[0, 1, 2],
            ),
        ),
    ),
    dict(type="ModalMask3D", mode="train"),
    dict(
        type="GlobalRotScaleTransAll",
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5],
    ),
    dict(
        type="CustomRandomFlip3D",
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
    ),
    dict(type="PointsRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectRangeFilter", point_cloud_range=point_cloud_range),
    dict(type="ObjectNameFilter", classes=class_names),
    dict(type="PointShuffle"),
    dict(type="ResizeCropFlipImage", data_aug_conf=ida_aug_conf, training=True),
    dict(type="NormalizeMultiviewImage", **img_norm_cfg),
    dict(type="PadMultiViewImage", size_divisor=32),
    dict(type="DefaultFormatBundle3D", class_names=class_names),
    dict(
        type="Collect3D",
        keys=["points", "img", "gt_bboxes_3d", "gt_labels_3d"],
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "lidar2img",
            "depth2img",
            "cam2img",
            "pad_shape",
            "scale_factor",
            "flip",
            "pcd_horizontal_flip",
            "pcd_vertical_flip",
            "box_mode_3d",
            "box_type_3d",
            "img_norm_cfg",
            "pcd_trans",
            "sample_idx",
            "pcd_scale_factor",
            "pcd_rotation",
            "pts_filename",
            "transformation_3d_flow",
            "rot_degree",
            "gt_bboxes_3d",
            "gt_labels_3d",
        ),
    ),
]

test_pipeline = [
    dict(
        type="LoadPointsFromFile",
        coord_type="LIDAR",
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type="LoadPointsFromMultiSweeps",
        sweeps_num=10,
        use_dim=[0, 1, 2],
    ),
    dict(type="LoadMultiViewImageFromFiles"),
    dict(
        type="MultiScaleFlipAug3D",
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type="GlobalRotScaleTrans",
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0],
            ),
            dict(type="RandomFlip3D"),
            dict(
                type="ResizeCropFlipImage",
                data_aug_conf=ida_aug_conf,
                training=False,
            ),
            dict(type="NormalizeMultiviewImage", **img_norm_cfg),
            dict(type="PadMultiViewImage", size_divisor=32),
            dict(
                type="DefaultFormatBundle3D",
                class_names=class_names,
                with_label=False,
            ),
            dict(type="Collect3D", keys=["points", "img"]),
        ],
    ),
]

data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            pipeline=train_pipeline,
            classes=class_names,
        )
    ),
    val=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
    ),
    test=dict(
        type=dataset_type,
        pipeline=test_pipeline,
        classes=class_names,
    ),
)

model = dict(
    pts_bbox_head=dict(
        tasks=[dict(num_class=3, class_names=class_names)],
        bbox_coder=dict(num_classes=3),
    )
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="TensorboardLoggerHook"),
        dict(
            type="WandbLoggerHook",
            init_kwargs=dict(
                project="Dataset_Generalize_Fusion",
                name="cmt-cbgs-nuscenes-xyz-3_class",
            ),
        ),
    ],
)

custom_hooks = [
    dict(
        type="DropAugmentationHook",
        drop_epoch=15,
        pipeline_name="UnifiedObjectSample",
    ),
]

resume_from = None