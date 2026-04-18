_base_ = ['../nuscenes/cmt_voxel0075_vov_argo2-3class-test.py']

evaluation = dict(
    interval=1,
    eval_range_m=[0.0, 150.0],
    export_unieval_package=True,
    format_only=True,
    export_dir='evaluation/unieval/cmt_argo2_3class',
    export_split='val',
    label_space='unified:3class',
    coord_system='lidar',
    box_origin='gravity_center',
    source_codebase='dgcmt',
    task='detection',
)
