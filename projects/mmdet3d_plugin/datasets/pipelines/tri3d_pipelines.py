import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES
from mmdet3d.core.points import LiDARPoints
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from tri3d.geometry import as_matrix, Pipeline, Rotation

@PIPELINES.register_module()
class LoadPointsFromTri3D:
    def __init__(self, coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4], sweeps_num=0):
        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.sweeps_num = sweeps_num
        # Inverse of the Z-90 rotation that Tri3D applies to NuScenes
        self.undo_tri3d_rot = Rotation.from_euler("Z", -np.pi / 2)

    def __call__(self, results):
        dataset = results['tri3d_dataset']
        seq = int(results['seq'])
        frame = int(results['frame'])
        sensor = results['primary_sensor']
        
        current_ts = dataset.timestamps(seq, sensor)[frame]

        def get_rotated_points(f_idx):
            pts = dataset.points(seq, f_idx, sensor)
            # UNDO Tri3D's internal 90-deg rotation to return to native NuScenes coords
            pts[:, :3] = self.undo_tri3d_rot.apply(pts[:, :3])
            return pts

        points = get_rotated_points(frame)
        
        if points.shape[1] >= 4:
            points = points[:, :4]
        else:
            padding = np.zeros((points.shape[0], 4 - points.shape[1]))
            points = np.hstack([points, padding])
            
        points = np.hstack([points, np.zeros((points.shape[0], 1))])
        sweep_points_list = [points]

        if self.sweeps_num > 0:
            all_timestamps = dataset.timestamps(seq, sensor)
            for i in range(1, self.sweeps_num + 1):
                sweep_idx = frame - i
                if sweep_idx < 0: break
                
                sweep_ts = all_timestamps[sweep_idx]
                time_lag = (current_ts - sweep_ts) / 1e6 # NuScenes timestamps are in microseconds
                
                try:
                    # alignment in Tri3D is coordinate-system aware. 
                    # transform aligns sweep_pts (in Tri3D coords) to frame (in Tri3D coords)
                    # We want native_sweep_pts aligned to native_frame_pts
                    # Goal: native_frame_pts = R_inv @ transform @ tri3d_sweep_pts
                    transform = dataset.alignment(seq, (sweep_idx, frame), (sensor, sensor))
                    
                    # R_inv is self.undo_tri3d_rot
                    combined_mat = as_matrix(self.undo_tri3d_rot) @ as_matrix(transform)
                    
                    sweep_pts = dataset.points(seq, sweep_idx, sensor)
                    sweep_pts_xyz = sweep_pts[:, :3]
                    # Apply the transformation that moves points from Tri3D-sweep to Native-frame
                    sweep_pts_xyz_hom = np.hstack([sweep_pts_xyz, np.ones((sweep_pts_xyz.shape[0], 1))])
                    sweep_pts[:, :3] = (sweep_pts_xyz_hom @ combined_mat.T)[:, :3]
                    
                    if sweep_pts.shape[1] >= 4:
                        sweep_pts = sweep_pts[:, :4]
                    else:
                        padding = np.zeros((sweep_pts.shape[0], 4 - sweep_pts.shape[1]))
                        sweep_pts = np.hstack([sweep_pts, padding])
                        
                    sweep_pts = np.hstack([sweep_pts, np.full((sweep_pts.shape[0], 1), time_lag)])
                    sweep_points_list.append(sweep_pts)
                except:
                    continue

        points = np.concatenate(sweep_points_list, axis=0)
        if points.shape[1] < self.load_dim:
            padding = np.zeros((points.shape[0], self.load_dim - points.shape[1]))
            points = np.hstack([points, padding])
            
        points = points[:, self.use_dim]
        results['points'] = LiDARPoints(points, points_dim=points.shape[1])
        return results

@PIPELINES.register_module()
class LoadMultiViewImageFromTri3D:
    def __init__(self):
        self.undo_tri3d_rot = Rotation.from_euler("Z", -np.pi / 2)

    def __call__(self, results):
        dataset = results['tri3d_dataset']
        seq = int(results['seq'])
        lidar_frame = int(results['frame'])
        lidar_sensor = results['primary_sensor']
        
        imgs, lidar2img_rts, lidar2cam_rts, cam_intrinsics = [], [], [], []
        target_timestamp = dataset.timestamps(seq, lidar_sensor)[lidar_frame]
        
        for cam_sensor in dataset.cam_sensors:
            cam_timestamps = dataset.timestamps(seq, cam_sensor)
            idx = np.searchsorted(cam_timestamps, target_timestamp)
            cam_frame = int(max(0, min(len(cam_timestamps)-1, idx)))
            
            # Image Loading (RGB -> BGR)
            img = np.array(dataset.image(seq, cam_frame, cam_sensor))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgs.append(img)
            
            img_plane_sensor = cam_sensor.replace('CAM', 'IMG')
            if img_plane_sensor not in dataset.img_sensors:
                for s in dataset.img_sensors:
                    if cam_sensor[4:] in s: 
                        img_plane_sensor = s
                        break

            try:
                # IMPORTANT: UNDO the LIDAR rotation for lidar2cam and lidar2img
                # Tri3D: l2c = Cam_pose.inv() @ Lidar_pose_tri3d
                # We want: l2c_native = Cam_pose.inv() @ Lidar_pose_native
                # Lidar_pose_tri3d = Lidar_pose_native @ R
                # So: l2c_native = l2c @ R
                l2c_transform = dataset.alignment(seq, (lidar_frame, cam_frame), (lidar_sensor, cam_sensor))
                lidar2cam = as_matrix(l2c_transform) @ np.linalg.inv(self.undo_tri3d_rot.as_matrix())
                    
                c2i_transform = dataset.alignment(seq, cam_frame, (cam_sensor, img_plane_sensor))
                cam_intrinsic = np.eye(4)
                target = c2i_transform
                if isinstance(target, Pipeline):
                    for op in target.operations:
                        if hasattr(op, 'intrinsics'):
                            target = op; break
                
                if hasattr(target, 'intrinsics'):
                    fx, fy, cx, cy = target.intrinsics[:4]
                    cam_intrinsic[0, 0], cam_intrinsic[1, 1] = fx, fy
                    cam_intrinsic[0, 2], cam_intrinsic[1, 2] = cx, cy
                
                lidar2img = cam_intrinsic @ lidar2cam
            except:
                lidar2cam = lidar2img = cam_intrinsic = np.eye(4)

            lidar2cam_rts.append(lidar2cam)
            lidar2img_rts.append(lidar2img)
            cam_intrinsics.append(cam_intrinsic)

        results.update(dict(
            img=imgs, lidar2img=lidar2img_rts, lidar2cam=lidar2cam_rts,
            cam_intrinsic=cam_intrinsics, img_shape=[i.shape for i in imgs],
            ori_shape=[i.shape for i in imgs], pad_shape=[i.shape for i in imgs],
            scale_factor=1.0
        ))
        return results

@PIPELINES.register_module()
class LoadAnnotationsFromTri3D:
    def __init__(self):
        self.undo_tri3d_rot = Rotation.from_euler("Z", -np.pi / 2)
        # nuScenes official attributes mapping
        self.attr_table = [
            "cycle.with_rider",
            "cycle.without_rider",
            "pedestrian.moving",
            "pedestrian.standing",
            "pedestrian.sitting_lying_down",
            "vehicle.moving",
            "vehicle.parked",
            "vehicle.stopped",
        ]

    def __call__(self, results):
        dataset = results["tri3d_dataset"]
        seq, frame, sensor = (
            int(results["seq"]),
            int(results["frame"]),
            results["primary_sensor"],
        )
        cat_mapping, classes = results.get("cat_mapping", {}), results.get("CLASSES", [])

        boxes_tri3d = dataset.boxes(seq, frame, coords=sensor)
        gt_bboxes_3d, gt_labels_3d, gt_attributes_3d = [], [], []

        # Matrix to undo the lidar rotation for box centers and headings
        undo_mat = self.undo_tri3d_rot.as_matrix()

        for box in boxes_tri3d:
            # 1. UNDO Lidar rotation for center and heading
            center_native = undo_mat[:3, :3] @ box.center

            # NuScenes native heading:
            # In Tri3D, 0 is along X-axis.
            # After -90 deg rotation, Tri3D's X becomes native Y.
            # In native NuScenes, heading 0 is along X-axis, pi/2 is along Y-axis.
            # So Tri3D heading 0 should become native heading pi/2.
            heading_native = box.heading + np.pi / 2

            # 2. Dimensions:
            # Tri3D size is [Length, Width, Height]
            l_tri3d, w_tri3d, h_tri3d = box.size

            # In LiDARInstance3DBoxes for NuScenes, the order is [x, y, z, l, w, h, yaw]
            # After rotation, Tri3D's "Length" is still the dimension along the heading.
            l, w, h = l_tri3d, w_tri3d, h_tri3d

            # 3. Velocity:
            # box.velocity is in Tri3D LiDAR frame, rotate to Native LiDAR frame
            if hasattr(box, "velocity"):
                vel_native = undo_mat[:3, :3] @ box.velocity
                vx, vy = vel_native[0], vel_native[1]
            else:
                vx, vy = 0.0, 0.0

            raw_label = box.label
            mapped_label = None
            for k, v in cat_mapping.items():
                if raw_label.startswith(k):
                    mapped_label = v
                    break

            if mapped_label and mapped_label in classes:
                gt_labels_3d.append(classes.index(mapped_label))
                # [x, y, z, l, w, h, yaw, vx, vy]
                gt_bboxes_3d.append(
                    [
                        center_native[0],
                        center_native[1],
                        center_native[2],
                        l,
                        w,
                        h,
                        heading_native,
                        vx,
                        vy,
                    ]
                )

                # 4. Attributes:
                attr_idx = 0  # Default to 0 if not found, or use a specific "ignore" value
                if hasattr(box, "attributes") and box.attributes:
                    for attr_name in box.attributes:
                        if attr_name in self.attr_table:
                            attr_idx = self.attr_table.index(attr_name)
                            break
                gt_attributes_3d.append(attr_idx)

        if gt_bboxes_3d:
            gt_bboxes_3d = np.array(gt_bboxes_3d, dtype=np.float32)
            gt_labels_3d = np.array(gt_labels_3d)
            gt_attributes_3d = np.array(gt_attributes_3d)
        else:
            gt_bboxes_3d = np.zeros((0, 9), dtype=np.float32)
            gt_labels_3d = np.zeros((0,), dtype=int)
            gt_attributes_3d = np.zeros((0,), dtype=int)

        results["gt_bboxes_3d"] = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5)
        )
        results["gt_labels_3d"] = gt_labels_3d
        results["gt_attributes_3d"] = gt_attributes_3d
        return results
