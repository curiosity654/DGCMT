import numpy as np
import mmcv
import os.path as osp
import tempfile
import torch
import pandas as pd
from pathlib import Path
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
import tri3d.datasets as tri3d_datasets
from pyquaternion import Quaternion
from tri3d.geometry import RigidTransform, Rotation
from tqdm import tqdm

class Tri3DObjectWrapper:
    """A wrapper to prevent deepcopy of the large Tri3D dataset object.
    
    When deepcopy is called on this object, it returns itself instead of
    performing a recursive copy of its internal state. This is crucial
    for performance when large objects are passed through pipelines.
    """
    def __init__(self, obj):
        # Use object.__setattr__ to avoid recursion since we override __setattr__
        object.__setattr__(self, 'obj', obj)
        
    def __getattr__(self, name):
        return getattr(self.obj, name)
    
    def __setattr__(self, name, value):
        setattr(self.obj, name, value)
    
    def __deepcopy__(self, memo):
        # Return the same wrapper instance to avoid deepcopy of the internal obj
        return self

@DATASETS.register_module()
class UnifiedMMDet3DDataset(Custom3DDataset):
    
    # Default NuScenes mapping using prefixes to support hierarchical labels
    # e.g., 'vehicle.bus.rigid' will match 'vehicle.bus'
    NUSC_MAPPING = {
        'vehicle.car': 'car',
        'vehicle.truck': 'truck',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.bus': 'bus',
        'vehicle.trailer': 'trailer',
        'movable_object.barrier': 'barrier',
        'vehicle.motorcycle': 'motorcycle',
        'vehicle.bicycle': 'bicycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
    }

    # Argoverse2 to NuScenes 10-class mapping (for training/inference)
    ARGO2_MAPPING = {
        'REGULAR_VEHICLE': 'car',
        'LARGE_VEHICLE': 'truck',
        'BOX_TRUCK': 'truck',
        'TRUCK': 'truck',
        'TRUCK_CAB': 'truck',
        'ARTICULATED_BUS': 'bus',
        'BUS': 'bus',
        'SCHOOL_BUS': 'bus',
        'VEHICULAR_TRAILER': 'trailer',
        'CONSTRUCTION_BARREL': 'barrier',
        'MOTORCYCLE': 'motorcycle',
        'MOTORCYCLIST': 'motorcycle',
        'BICYCLE': 'bicycle',
        'BICYCLIST': 'bicycle',
        'WHEELED_RIDER': 'bicycle',
        'PEDESTRIAN': 'pedestrian',
        'OFFICIAL_SIGNALER': 'pedestrian',
        'CONSTRUCTION_CONE': 'traffic_cone',
        'BOLLARD': 'barrier',
        # Ignored categories (not in NuScenes 10 classes):
        # 'ANIMAL', 'DOG', 'STROLLER', 'WHEELCHAIR', 'WHEELED_DEVICE',
        # 'SIGN', 'STOP_SIGN', 'MESSAGE_BOARD_TRAILER', 'TRAFFIC_LIGHT_TRAILER',
        # 'MOBILE_PEDESTRIAN_CROSSING_SIGN', 'RAILED_VEHICLE'
    }

    # NuScenes 10-class to Argoverse2 mapping (for evaluation)
    NUSC_TO_AV2_MAPPING = {
        'car': 'REGULAR_VEHICLE',
        'truck': 'TRUCK',
        'construction_vehicle': 'LARGE_VEHICLE',
        'bus': 'BUS',
        'trailer': 'VEHICULAR_TRAILER',
        'barrier': 'BOLLARD',
        'motorcycle': 'MOTORCYCLE',
        'bicycle': 'BICYCLE',
        'pedestrian': 'PEDESTRIAN',
        'traffic_cone': 'CONSTRUCTION_CONE',
    }

    # KITTI to NuScenes 10-class mapping (for training/inference)
    KITTI_MAPPING = {
        'Car': 'car',
        'Van': 'car',
        'Truck': 'truck',
        'Pedestrian': 'pedestrian',
        'Person_sitting': 'pedestrian',
        'Cyclist': 'bicycle',
        'Tram': 'bus',
    }

    # NuScenes 10-class to KITTI mapping (for evaluation)
    NUSC_TO_KITTI_MAPPING = {
        'car': 'Car',
        'truck': 'Van',
        'construction_vehicle': 'Van',
        'bus': 'Van',
        'trailer': 'Van',
        'pedestrian': 'Pedestrian',
        'bicycle': 'Cyclist',
        'motorcycle': 'Cyclist',
    }

    # Waymo to NuScenes 10-class mapping (for training/inference)
    WAYMO_MAPPING = {
        'VEHICLE': 'car',
        'PEDESTRIAN': 'pedestrian',
        'CYCLIST': 'bicycle',
        'SIGN': 'barrier',
    }

    # NuScenes 10-class to Waymo mapping (for evaluation)
    NUSC_TO_WAYMO_MAPPING = {
        'car': 'VEHICLE',
        'truck': 'VEHICLE',
        'construction_vehicle': 'VEHICLE',
        'bus': 'VEHICLE',
        'trailer': 'VEHICLE',
        'motorcycle': 'CYCLIST',
        'bicycle': 'CYCLIST',
        'pedestrian': 'PEDESTRIAN',
        'barrier': 'SIGN',
        'traffic_cone': 'SIGN',
    }

    # nuScenes official attributes mapping
    ATTR_TABLE = [
        "cycle.with_rider",
        "cycle.without_rider",
        "pedestrian.moving",
        "pedestrian.standing",
        "pedestrian.sitting_lying_down",
        "vehicle.moving",
        "vehicle.parked",
        "vehicle.stopped",
    ]

    # Map categories to a default attribute for fallback
    DEFAULT_ATTR = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }

    def __init__(self, 
                 dataset_type, 
                 data_root, 
                 subset=None,
                 split=None,
                 pipeline=None, 
                 classes=None, 
                 modality=None, 
                 box_type_3d='LiDAR', 
                 filter_empty_gt=True, 
                 test_mode=False, 
                 file_client_args=dict(backend='disk'),
                 cat_mapping=None,
                 **kwargs):
        # 1. Initialize Tri3D dataset
        self.dataset_type_name = dataset_type  # Save for branching in evaluate/format
        self.tri3d_cls = getattr(tri3d_datasets, dataset_type)
        
        tri3d_kwargs = {}
        if subset is not None:
            tri3d_kwargs['subset'] = subset
        if split is not None:
            tri3d_kwargs['split'] = split
        tri3d_kwargs.update(kwargs.pop('tri3d_kwargs', {}))
        # Initialize Tri3D dataset and wrap it to prevent deepcopy overhead
        tri3d_dataset = self.tri3d_cls(data_root, **tri3d_kwargs)
        self.tri3d_dataset = Tri3DObjectWrapper(tri3d_dataset)
        
        # Initialize category mapping based on dataset type
        if cat_mapping is not None:
            self.cat_mapping = cat_mapping
        elif dataset_type == 'NuScenes':
            self.cat_mapping = self.NUSC_MAPPING
        elif dataset_type == 'Argoverse2':
            self.cat_mapping = self.ARGO2_MAPPING
        elif dataset_type == 'Waymo':
            self.cat_mapping = self.WAYMO_MAPPING
        elif dataset_type == 'KITTI':
            self.cat_mapping = self.KITTI_MAPPING
        else:
            self.cat_mapping = {}

        # 2. Initialize standard Dataset attributes (mimicking Custom3DDataset)
        self.data_root = data_root
        self.ann_file = None 
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # 3. Load annotations using our custom logic
        self.data_infos = self.load_annotations(None)

        # 4. Build pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # 5. Set group flag
        if not self.test_mode:
            self._set_group_flag()

    def _map_label(self, raw_label):
        """Helper to map detailed labels to common classes using prefixes."""
        for k, v in self.cat_mapping.items():
            if raw_label.startswith(k):
                return v
        return None

    def load_annotations(self, ann_file):
        """Rebuild data index from Tri3D."""
        data_infos = []
        if hasattr(self.tri3d_dataset, 'pcl_sensors') and self.tri3d_dataset.pcl_sensors:
            sensor = self.tri3d_dataset.pcl_sensors[0]
        else:
            raise ValueError("Dataset must have point cloud sensors")

        try:
            sequences = self.tri3d_dataset.sequences()
        except NotImplementedError:
             sequences = []

        print(f"Indexing {len(sequences)} sequences from {self.tri3d_dataset.__class__.__name__}...")
        
        for seq in tqdm(sequences, desc="Indexing sequences"):
            try:
                frames = self.tri3d_dataset.keyframes(seq, sensor)
                is_keyframes = True
            except (AttributeError, NotImplementedError):
                frames = self.tri3d_dataset.frames(seq, sensor)
                is_keyframes = False
                
            for i, frame in enumerate(frames):
                cat_ids = []
                
                # Only compute cat_ids for training (used by CBGS and empty frame filtering)
                if not self.test_mode:
                    boxes = self.tri3d_dataset.boxes(seq, frame, coords=sensor)
                    for box in boxes:
                        mapped_label = self._map_label(box.label)
                        if mapped_label and mapped_label in self.CLASSES:
                            cat_ids.append(self.CLASSES.index(mapped_label))
                    
                    # Skip empty frames in training
                    if self.filter_empty_gt and len(cat_ids) == 0:
                        continue

                # Build data_info dict with dataset-specific fields
                data_info = dict(
                    seq=seq,
                    frame=frame,
                    sensor=sensor,
                    sample_idx=f"{seq}_{frame}",
                    cat_ids=list(set(cat_ids))
                )

                # NuScenes-specific: Get sample token for evaluation
                if self.dataset_type_name == 'NuScenes':
                    token = None
                    if hasattr(self.tri3d_dataset.obj, 'scenes'):
                        try:
                            if is_keyframes:
                                token = self.tri3d_dataset.obj.scenes[seq].sample_tokens[i]
                        except (AttributeError, IndexError):
                            pass
                    data_info['token'] = token

                # Argoverse2-specific: Get log_id and timestamp_ns for evaluation
                elif self.dataset_type_name == 'Argoverse2':
                    # log_id is the sequence directory name
                    log_id = self.tri3d_dataset.records[seq].name
                    # timestamp_ns from tri3d timeline
                    timestamp_ns = int(self.tri3d_dataset.timestamps(seq, sensor)[frame])
                    data_info['log_id'] = log_id
                    data_info['timestamp_ns'] = timestamp_ns
                elif self.dataset_type_name == 'Waymo':
                    log_id = self.tri3d_dataset.records[seq]
                    timestamp_micros = int(
                        self.tri3d_dataset.timestamps(seq, sensor)[frame] * 1e6
                    )
                    data_info['log_id'] = log_id
                    data_info['timestamp_micros'] = timestamp_micros

                data_infos.append(data_info)
        
        print(f"Loaded {len(data_infos)} frames (filtered: {self.filter_empty_gt})")
        return data_infos

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            tri3d_dataset=self.tri3d_dataset,
            seq=info['seq'],
            frame=info['frame'],
            token=info.get('token'),
            primary_sensor=info['sensor'],
            sample_idx=info['sample_idx'],
            cat_mapping=self.cat_mapping, 
            CLASSES=self.CLASSES
        )
        return input_dict

    def get_cat_ids(self, idx):
        """Used by CBGSDataset."""
        return self.data_infos[idx].get('cat_ids', [])

    def _get_heuristic_attr(self, cls_name, velocity):
        """Heuristic to assign attributes based on velocity, matching NuScenesDataset."""
        vel_norm = np.sqrt(velocity[0]**2 + velocity[1]**2)
        if vel_norm > 0.2:
            if cls_name in [
                    'car', 'construction_vehicle', 'bus', 'truck', 'trailer'
            ]:
                return 'vehicle.moving'
            elif cls_name in ['bicycle', 'motorcycle']:
                return 'cycle.with_rider'
            else:
                return self.DEFAULT_ATTR.get(cls_name, '')
        else:
            if cls_name in ['pedestrian']:
                return 'pedestrian.standing'
            elif cls_name in ['bus']:
                return 'vehicle.stopped'
            else:
                return self.DEFAULT_ATTR.get(cls_name, '')

    def _format_bbox(self, results, sample_token, seq=None, frame=None, sensor=None):
        """Convert predictions to NuScenes format."""
        nusc_annos = []
        if "pts_bbox" in results:
            bboxes = results["pts_bbox"]["boxes_3d"]
            scores = results["pts_bbox"]["scores_3d"]
            labels = results["pts_bbox"]["labels_3d"]
            attr_labels = results["pts_bbox"].get("attr_labels_3d")
        else:
            bboxes = results["boxes_3d"]
            scores = results["scores_3d"]
            labels = results["labels_3d"]
            attr_labels = results.get("attr_labels_3d")

        # LiDARInstance3DBoxes to numpy
        bboxes_tensor = bboxes.tensor.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        if attr_labels is not None:
            attr_labels = attr_labels.cpu().numpy()

        # Get sensor to world transform
        if seq is not None and frame is not None and sensor is not None:
            # tri3d_dataset.poses(seq, sensor) returns a batched transform
            # which is Tri3D_LIDAR to World
            sensor2world = self.tri3d_dataset.poses(seq, sensor)[frame]

            # The model was trained on native NuScenes LIDAR frame (via LoadPointsFromTri3D)
            # native_LIDAR = Rot(-90) @ Tri3D_LIDAR  => Tri3D_LIDAR = Rot(90) @ native_LIDAR
            # So native_LIDAR to World = Tri3D_LIDAR to World @ Rot(90)
            native2tri3d = RigidTransform(Rotation.from_euler("Z", np.pi / 2), [0, 0, 0])
            native2world = sensor2world @ native2tri3d
        else:
            native2world = None

        for i in range(len(bboxes_tensor)):
            # LiDARInstance3DBoxes for NuScenes: [x, y, z, l, w, h, yaw, vx, vy]
            # NuScenes expects: [x, y, z] for translation, [w, l, h] for size
            x, y, z, l, w, h, yaw = bboxes_tensor[i, :7]

            if native2world is not None:
                # Create box to native LIDAR transform
                # Translation is [x, y, z], Rotation is around Z
                box2native = RigidTransform(Rotation.from_euler("Z", yaw), [x, y, z])

                # Box to world
                box2world = native2world @ box2native

                # Get global translation and rotation
                trans = box2world.translation.vec
                quat = box2world.rotation.quat  # [w, x, y, z]

                # 3. Velocity rotation:
                if bboxes_tensor.shape[1] >= 9:
                    vx, vy = bboxes_tensor[i, 7:9]
                    vel_native = np.array([vx, vy, 0.0])
                    # Rotate velocity to world frame (vector rotation only)
                    vel_world = native2world.apply(vel_native) - native2world.apply(
                        np.zeros(3)
                    )
                    trans_vel = [float(vel_world[0]), float(vel_world[1])]
                else:
                    trans_vel = [0.0, 0.0]
            else:
                # Fallback
                trans = [x, y, z]
                quat = Quaternion(axis=[0, 0, 1], radians=yaw).elements
                trans_vel = [0.0, 0.0]

            # 4. Attribute mapping
            cls_name = self.CLASSES[labels[i]]
            if attr_labels is not None:
                attr_idx = attr_labels[i]
                if 0 <= attr_idx < len(self.ATTR_TABLE):
                    attr_name = self.ATTR_TABLE[attr_idx]
                else:
                    attr_name = self._get_heuristic_attr(cls_name, trans_vel)
            else:
                attr_name = self._get_heuristic_attr(cls_name, trans_vel)

            nusc_anno = dict(
                sample_token=sample_token,
                translation=[float(trans[0]), float(trans[1]), float(trans[2])],
                size=[float(w), float(l), float(h)],  # NuScenes results.json expects [w, l, h]
                rotation=[
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                ],
                velocity=trans_vel,
                detection_name=cls_name,
                detection_score=float(scores[i]),
                attribute_name=attr_name,
            )
            nusc_annos.append(nusc_anno)
        return nusc_annos

    def format_results(self, outputs, jsonfile_prefix=None):
        """Format the results based on dataset type."""
        if self.dataset_type_name == 'Argoverse2':
            return self._format_results_av2(outputs, jsonfile_prefix)
        if self.dataset_type_name == 'Waymo':
            return self._format_results_waymo(outputs, jsonfile_prefix)
        if self.dataset_type_name == 'KITTI':
            return self._format_results_kitti(outputs, jsonfile_prefix)
        else:
            return self._format_results_nusc(outputs, jsonfile_prefix)

    def _format_results_nusc(self, outputs, jsonfile_prefix=None):
        """Format the results to NuScenes json format."""
        nusc_annos = {}
        print(f"Formatting {len(outputs)} results for NuScenes...")
        
        for i, out in enumerate(outputs):
            info = self.data_infos[i]
            token = info.get('token')
            if token is None:
                continue
            nusc_annos[token] = self._format_bbox(
                out, token, seq=info['seq'], frame=info['frame'], sensor=info['sensor'])
            
        modality = self.modality if self.modality is not None else {}
        meta = dict(
            use_camera=modality.get('use_camera', True),
            use_lidar=modality.get('use_lidar', True),
            use_radar=modality.get('use_radar', False),
            use_map=modality.get('use_map', False),
            use_external=modality.get('use_external', False)
        )
        results = dict(results=nusc_annos, meta=meta)
        
        if jsonfile_prefix is not None:
            mmcv.dump(results, f"{jsonfile_prefix}.submission.json")
            
        return results

    def evaluate(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluation based on dataset type."""
        if self.dataset_type_name == 'Argoverse2':
            return self._evaluate_av2(results, logger, jsonfile_prefix, **kwargs)
        if self.dataset_type_name == 'Waymo':
            return self._evaluate_waymo(results, logger, jsonfile_prefix, **kwargs)
        if self.dataset_type_name == 'KITTI':
            return self._evaluate_kitti(results, logger, jsonfile_prefix, **kwargs)
        else:
            return self._evaluate_nusc(results, logger, jsonfile_prefix, **kwargs)

    def _evaluate_nusc(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluation in NuScenes protocol."""
        print(f"Evaluating {len(results)} results (NuScenes)...")
        
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
            
        results_json = self.format_results(results, jsonfile_prefix)
        nusc_annos = results_json['results']
        
        # Determine evaluation set
        eval_set = 'val'
        if self.tri3d_dataset.obj.subset == 'v1.0-mini':
            eval_set = 'mini_val'
        elif hasattr(self.tri3d_dataset.obj, 'split'):
            if self.tri3d_dataset.obj.split in ['train', 'val', 'test']:
                eval_set = self.tri3d_dataset.obj.split
            elif self.tri3d_dataset.obj.split == 'val_mini':
                # Map our custom val_mini to 'val' split for SDK to load annotations
                eval_set = 'val'

        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval
        from nuscenes.eval.detection.config import config_factory
        
        nusc = NuScenes(version=self.tri3d_dataset.obj.subset, dataroot=self.data_root, verbose=False)
        cfg = config_factory('detection_cvpr_2019')
        
        nusc_eval = DetectionEval(
            nusc,
            config=cfg,
            result_path=f"{jsonfile_prefix}.submission.json",
            eval_set=eval_set,
            output_dir=osp.dirname(jsonfile_prefix),
            verbose=True
        )
        
        # If we are using a custom split, force the SDK to only evaluate on those tokens
        if self.tri3d_dataset.obj.split == 'val_mini':
            nusc_eval.sample_tokens = list(nusc_annos.keys())
            print(f"Forcing evaluation on {len(nusc_eval.sample_tokens)} mini split samples")

        nusc_eval.main(plot_examples=0)
        
        # Load metrics from the saved JSON file (this ensures proper serialization)
        metrics = mmcv.load(osp.join(osp.dirname(jsonfile_prefix), 'metrics_summary.json'))
        
        # Build a flat dictionary of scalar values for logging
        detail = dict()
        metric_prefix = 'pts_bbox_NuScenes'
        
        # Per-class AP at different distances
        if 'label_aps' in metrics:
            for name in self.CLASSES:
                if name in metrics['label_aps']:
                    for k, v in metrics['label_aps'][name].items():
                        val = float('{:.4f}'.format(v))
                        detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
        
        # Per-class true positive errors
        if 'label_tp_errors' in metrics:
            for name in self.CLASSES:
                if name in metrics['label_tp_errors']:
                    for k, v in metrics['label_tp_errors'][name].items():
                        val = float('{:.4f}'.format(v))
                        detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
        
        # Mean true positive errors
        ErrNameMapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        if 'tp_errors' in metrics:
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix, ErrNameMapping.get(k, k))] = val
        
        # Overall metrics
        if 'nd_score' in metrics:
            detail['{}/NDS'.format(metric_prefix)] = float('{:.4f}'.format(metrics['nd_score']))
        if 'mean_ap' in metrics:
            detail['{}/mAP'.format(metric_prefix)] = float('{:.4f}'.format(metrics['mean_ap']))
            
        # Print summary
        print(f"\n{'='*50}")
        print(f"Evaluation Results:")
        print(f"  NDS: {detail.get('NuScenes/NDS', 'N/A')}")
        print(f"  mAP: {detail.get('NuScenes/mAP', 'N/A')}")
        print(f"{'='*50}\n")
            
        # Clean up
        if tmp_dir is not None:
            tmp_dir.cleanup()
            
        return detail

    # ==================== Waymo Evaluation Methods ====================

    def _format_bbox_waymo(self, results, log_id, timestamp_micros):
        """Convert predictions to Waymo-like format for a single frame."""
        if "pts_bbox" in results:
            bboxes = results["pts_bbox"]["boxes_3d"]
            scores = results["pts_bbox"]["scores_3d"]
            labels = results["pts_bbox"]["labels_3d"]
        else:
            bboxes = results["boxes_3d"]
            scores = results["scores_3d"]
            labels = results["labels_3d"]

        bboxes_tensor = bboxes.tensor.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        waymo_boxes = []
        for i in range(len(bboxes_tensor)):
            x, y, z, l, w, h, yaw = bboxes_tensor[i, :7]
            z_center = z + h / 2.0
            cls_name = self.CLASSES[labels[i]]
            waymo_cls = self.NUSC_TO_WAYMO_MAPPING.get(cls_name, 'UNKNOWN')

            waymo_boxes.append({
                'center_x': float(x),
                'center_y': float(y),
                'center_z': float(z_center),
                'length': float(l),
                'width': float(w),
                'height': float(h),
                'heading': float(yaw),
                'score': float(scores[i]),
                'category': waymo_cls,
                'log_id': log_id,
                'timestamp_micros': int(timestamp_micros),
            })

        return waymo_boxes

    def _format_results_waymo(self, outputs, jsonfile_prefix=None):
        """Format the results to a simple Waymo DataFrame format."""
        print(f"Formatting {len(outputs)} results for Waymo...")

        all_boxes = []
        for i, out in enumerate(outputs):
            info = self.data_infos[i]
            log_id = info.get('log_id')
            timestamp_micros = info.get('timestamp_micros')
            if log_id is None or timestamp_micros is None:
                continue
            frame_boxes = self._format_bbox_waymo(out, log_id, timestamp_micros)
            all_boxes.extend(frame_boxes)

        if len(all_boxes) == 0:
            print("Warning: No boxes to format!")
            return pd.DataFrame()

        dts = pd.DataFrame(all_boxes).sort_values("score", ascending=False)

        if jsonfile_prefix is not None:
            feather_path = f"{jsonfile_prefix}_waymo_dts.feather"
            dts.to_feather(feather_path)
            print(f"Results saved to {feather_path}")

        persistent_path = osp.join(self.data_root, f'{self.tri3d_dataset.split}_dts.feather')
        dts.to_feather(persistent_path)
        print(f"Results also saved to {persistent_path}")

        return dts.set_index(["log_id", "timestamp_micros"]).sort_index()

    def _load_waymo_annotations(self):
        """Load Waymo ground truth annotations."""
        anno_path = osp.join(self.data_root, f'{self.tri3d_dataset.split}_anno.feather')

        if osp.exists(anno_path):
            print(f"Loading cached annotations from {anno_path}")
            gts = pd.read_feather(anno_path)
            return gts.set_index(["log_id", "timestamp_micros"]).sort_index()

        print("Building annotations from Tri3D dataset...")
        all_gts = []

        for info in self.data_infos:
            seq = info['seq']
            frame = info['frame']
            sensor = info['sensor']
            log_id = info.get('log_id')
            timestamp_micros = info.get('timestamp_micros')
            if log_id is None or timestamp_micros is None:
                continue

            boxes = self.tri3d_dataset.boxes(seq, frame, coords=sensor)
            for box in boxes:
                num_pts = getattr(box, 'num_lidar_points_in_box', 0)
                all_gts.append({
                    'center_x': float(box.center[0]),
                    'center_y': float(box.center[1]),
                    'center_z': float(box.center[2]),
                    'length': float(box.size[0]),
                    'width': float(box.size[1]),
                    'height': float(box.size[2]),
                    'heading': float(box.heading),
                    'num_lidar_points_in_box': int(num_pts),
                    'category': box.label,
                    'log_id': log_id,
                    'timestamp_micros': int(timestamp_micros),
                })

        gts = pd.DataFrame(all_gts)
        gts.to_feather(anno_path)
        print(f"Cached annotations to {anno_path}")

        return gts.set_index(["log_id", "timestamp_micros"]).sort_index()

    def _evaluate_waymo(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluation in Waymo protocol (official metrics)."""
        print(f"Evaluating {len(results)} results (Waymo)...")

        try:
            import tensorflow as tf
            from waymo_open_dataset import label_pb2
            from waymo_open_dataset.metrics.python import config_util_py
            from waymo_open_dataset.metrics.python import wod_detection_evaluator
        except Exception as exc:
            raise ImportError(
                "waymo_open_dataset is required for official Waymo metrics. "
                "Install via: pip install waymo-open-dataset-tf-2-12-0==1.6.7"
            ) from exc

        pred_frame_ids = []
        pred_bboxes = []
        pred_types = []
        pred_scores = []
        pred_overlap_nlz = []

        type_map = {
            'VEHICLE': label_pb2.Label.TYPE_VEHICLE,
            'PEDESTRIAN': label_pb2.Label.TYPE_PEDESTRIAN,
            'CYCLIST': label_pb2.Label.TYPE_CYCLIST,
            'SIGN': label_pb2.Label.TYPE_SIGN,
        }

        for i, out in enumerate(results):
            frame_id = i
            if "pts_bbox" in out:
                bboxes = out["pts_bbox"]["boxes_3d"]
                scores = out["pts_bbox"]["scores_3d"]
                labels = out["pts_bbox"]["labels_3d"]
            else:
                bboxes = out["boxes_3d"]
                scores = out["scores_3d"]
                labels = out["labels_3d"]

            bboxes_tensor = bboxes.tensor.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

            for j in range(len(bboxes_tensor)):
                x, y, z, l, w, h, yaw = bboxes_tensor[j, :7]
                z_center = z + h / 2.0
                cls_name = self.CLASSES[labels[j]]
                waymo_cls = self.NUSC_TO_WAYMO_MAPPING.get(cls_name)
                if waymo_cls is None or waymo_cls not in type_map:
                    continue
                pred_frame_ids.append(frame_id)
                pred_bboxes.append([x, y, z_center, l, w, h, yaw])
                pred_types.append(type_map[waymo_cls])
                pred_scores.append(float(scores[j]))
                pred_overlap_nlz.append(False)

        gt_frame_ids = []
        gt_bboxes = []
        gt_types = []
        gt_difficulty = []
        gt_speed = []

        for i, info in enumerate(self.data_infos):
            seq = info['seq']
            frame = info['frame']
            sensor = info['sensor']
            boxes = self.tri3d_dataset.boxes(seq, frame, coords=sensor)

            for box in boxes:
                if box.label not in type_map:
                    continue
                gt_frame_ids.append(i)
                gt_bboxes.append([
                    float(box.center[0]),
                    float(box.center[1]),
                    float(box.center[2]),
                    float(box.size[0]),
                    float(box.size[1]),
                    float(box.size[2]),
                    float(box.heading),
                ])
                gt_types.append(type_map[box.label])
                diff = getattr(box, 'difficulty_level_det', 1)
                gt_difficulty.append(label_pb2.Label.LEVEL_2 if diff == 2 else label_pb2.Label.LEVEL_1)
                if hasattr(box, 'speed'):
                    gt_speed.append([float(box.speed[0]), float(box.speed[1])])

        if len(pred_bboxes) == 0:
            print("No predictions to evaluate!")
            return {}

        if len(gt_bboxes) == 0:
            print("No ground truth boxes to evaluate!")
            return {}

        predictions = dict(
            prediction_frame_id=tf.constant(pred_frame_ids, dtype=tf.int64),
            prediction_bbox=tf.constant(pred_bboxes, dtype=tf.float32),
            prediction_type=tf.constant(pred_types, dtype=tf.uint8),
            prediction_score=tf.constant(pred_scores, dtype=tf.float32),
            prediction_overlap_nlz=tf.constant(pred_overlap_nlz, dtype=tf.bool),
        )

        groundtruths = dict(
            ground_truth_frame_id=tf.constant(gt_frame_ids, dtype=tf.int64),
            ground_truth_bbox=tf.constant(gt_bboxes, dtype=tf.float32),
            ground_truth_type=tf.constant(gt_types, dtype=tf.uint8),
            ground_truth_difficulty=tf.constant(gt_difficulty, dtype=tf.uint8),
        )

        if len(gt_speed) == len(gt_bboxes):
            groundtruths['ground_truth_speed'] = tf.constant(gt_speed, dtype=tf.float32)

        evaluator = wod_detection_evaluator.WODDetectionEvaluator()
        evaluator.update_state(groundtruths, predictions)
        metrics = evaluator.result()

        breakdown_names = config_util_py.get_breakdown_names_from_config(evaluator._config)
        ap = metrics.average_precision.numpy()
        aph = metrics.average_precision_ha_weighted.numpy()

        detail = {}
        metric_prefix = 'pts_bbox_Waymo'
        for name, ap_val, aph_val in zip(breakdown_names, ap, aph):
            detail[f'{metric_prefix}/{name}/mAP'] = float(ap_val)
            detail[f'{metric_prefix}/{name}/mAPH'] = float(aph_val)

        def get_metric(name, values):
            try:
                idx = breakdown_names.index(name)
            except ValueError:
                return None
            return float(values[idx])

        summary_types = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        for obj_type in ['VEHICLE', 'PEDESTRIAN', 'CYCLIST', 'SIGN']:
            for level in ['LEVEL_1', 'LEVEL_2']:
                key = f'OBJECT_TYPE_TYPE_{obj_type}_{level}'
                ap_val = get_metric(key, ap)
                aph_val = get_metric(key, aph)
                if ap_val is not None:
                    detail[f'{metric_prefix}/{obj_type}/{"L1" if level.endswith("1") else "L2"} mAP'] = ap_val
                if aph_val is not None:
                    detail[f'{metric_prefix}/{obj_type}/{"L1" if level.endswith("1") else "L2"} mAPH'] = aph_val

        for level in ['LEVEL_1', 'LEVEL_2']:
            ap_vals = []
            aph_vals = []
            for obj_type in summary_types:
                key = f'OBJECT_TYPE_TYPE_{obj_type}_{level}'
                ap_val = get_metric(key, ap)
                aph_val = get_metric(key, aph)
                if ap_val is not None:
                    ap_vals.append(ap_val)
                if aph_val is not None:
                    aph_vals.append(aph_val)
            if ap_vals:
                detail[f'{metric_prefix}/Overall/{"L1" if level.endswith("1") else "L2"} mAP'] = float(sum(ap_vals) / len(ap_vals))
            if aph_vals:
                detail[f'{metric_prefix}/Overall/{"L1" if level.endswith("1") else "L2"} mAPH'] = float(sum(aph_vals) / len(aph_vals))

        print("\n" + "=" * 80)
        print("Waymo Evaluation Results (Official):")
        for key in [
            f'{metric_prefix}/Overall/L1 mAP',
            f'{metric_prefix}/Overall/L1 mAPH',
            f'{metric_prefix}/Overall/L2 mAP',
            f'{metric_prefix}/Overall/L2 mAPH',
        ]:
            if key in detail:
                print(f"{key}: {detail[key]:.4f}")
        print("=" * 80 + "\n")

        return detail

    def _basic_waymo_metrics(self, dts, gts):
        """Compute basic metrics when Waymo SDK is not available."""
        print("Computing basic metrics (Waymo)...")

        detail = {}
        detail['num_predictions'] = len(dts)
        detail['num_ground_truth'] = len(gts)

        if 'category' in dts.columns:
            for cat in dts['category'].unique():
                detail[f'num_pred_{cat}'] = int((dts['category'] == cat).sum())

        if 'category' in gts.columns:
            for cat in gts['category'].unique():
                detail[f'num_gt_{cat}'] = int((gts['category'] == cat).sum())

        return detail

    # ==================== KITTI Evaluation Methods ====================

    def _load_kitti_gts(self):
        """Load KITTI ground truth annotations from label_2 files."""
        gt_annos = []
        for info in self.data_infos:
            frame = info['frame']
            frame_id = self.tri3d_dataset._frame_id(frame)
            label_path = (
                Path(self.data_root) / self.tri3d_dataset.split / "label_2" / f"{frame_id}.txt"
            )

            names, truncated, occluded, alpha = [], [], [], []
            bboxes, dimensions, locations, rotation_y = [], [], [], []

            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        fields = line.strip().split()
                        if len(fields) < 15:
                            continue
                        names.append(fields[0])
                        truncated.append(float(fields[1]))
                        occluded.append(int(fields[2]))
                        alpha.append(float(fields[3]))
                        bbox = [float(x) for x in fields[4:8]]
                        dims = [float(fields[8]), float(fields[9]), float(fields[10])]
                        loc = [float(fields[11]), float(fields[12]), float(fields[13])]
                        ry = float(fields[14])
                        bboxes.append(bbox)
                        dimensions.append(dims)
                        locations.append(loc)
                        rotation_y.append(ry)

            anno = {
                'name': np.array(names),
                'truncated': np.array(truncated, dtype=np.float32),
                'occluded': np.array(occluded, dtype=np.int64),
                'alpha': np.array(alpha, dtype=np.float32),
                'bbox': np.array(bboxes, dtype=np.float32).reshape(-1, 4),
                'dimensions': np.array(dimensions, dtype=np.float32).reshape(-1, 3),
                'location': np.array(locations, dtype=np.float32).reshape(-1, 3),
                'rotation_y': np.array(rotation_y, dtype=np.float32),
            }
            gt_annos.append(anno)

        return gt_annos

    def _format_results_kitti(self, outputs, jsonfile_prefix=None):
        """Format the results to KITTI format for evaluation."""
        from mmdet3d.core.bbox import Box3DMode

        det_annos = []
        print(f"Formatting {len(outputs)} results for KITTI...")

        for i, out in enumerate(outputs):
            info = self.data_infos[i]
            frame = info['frame']
            calib = self.tri3d_dataset._load_calib(frame)
            lidar2cam = calib["R0_rect"] @ calib["Tr_velo_to_cam"]
            p2 = calib["P2"]

            if "pts_bbox" in out:
                bboxes = out["pts_bbox"]["boxes_3d"]
                scores = out["pts_bbox"]["scores_3d"]
                labels = out["pts_bbox"]["labels_3d"]
            else:
                bboxes = out["boxes_3d"]
                scores = out["scores_3d"]
                labels = out["labels_3d"]

            if bboxes.tensor.numel() == 0:
                det_annos.append(
                    dict(
                        name=np.array([]),
                        truncated=np.array([]),
                        occluded=np.array([]),
                        alpha=np.array([]),
                        bbox=np.zeros([0, 4]),
                        dimensions=np.zeros([0, 3]),
                        location=np.zeros([0, 3]),
                        rotation_y=np.array([]),
                        score=np.array([]),
                    )
                )
                continue

            cam_boxes = bboxes.convert_to(Box3DMode.CAM, rt_mat=lidar2cam)
            cam_boxes_np = cam_boxes.tensor.cpu().numpy()
            lidar_boxes_np = bboxes.tensor.cpu().numpy()
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()

            names, truncated, occluded, alpha = [], [], [], []
            bboxes2d, dimensions, locations, rotation_y, score_list = [], [], [], [], []

            corners = cam_boxes.corners.cpu().numpy()
            img_w, img_h = self.tri3d_dataset._get_image_size(frame, "IMG2")

            for j in range(len(cam_boxes_np)):
                cls_name = self.CLASSES[labels[j]]
                kitti_cls = self.NUSC_TO_KITTI_MAPPING.get(cls_name)
                if kitti_cls is None:
                    continue

                corners_h = np.concatenate(
                    [corners[j], np.ones((8, 1), dtype=np.float32)], axis=1
                )
                proj = corners_h @ p2.T
                proj[:, 0] /= np.clip(proj[:, 2], 1e-6, None)
                proj[:, 1] /= np.clip(proj[:, 2], 1e-6, None)
                x_min, y_min = proj[:, 0].min(), proj[:, 1].min()
                x_max, y_max = proj[:, 0].max(), proj[:, 1].max()
                x_min = float(np.clip(x_min, 0, img_w - 1))
                y_min = float(np.clip(y_min, 0, img_h - 1))
                x_max = float(np.clip(x_max, 0, img_w - 1))
                y_max = float(np.clip(y_max, 0, img_h - 1))

                names.append(kitti_cls)
                truncated.append(0.0)
                occluded.append(0)
                rotation = float(cam_boxes_np[j, 6])
                alpha_val = -np.arctan2(-lidar_boxes_np[j, 1], lidar_boxes_np[j, 0]) + rotation
                alpha.append(float(alpha_val))
                bboxes2d.append([x_min, y_min, x_max, y_max])
                dimensions.append(cam_boxes_np[j, 3:6].tolist())
                locations.append(cam_boxes_np[j, 0:3].tolist())
                rotation_y.append(rotation)
                score_list.append(float(scores[j]))

            det_annos.append(
                dict(
                    name=np.array(names),
                    truncated=np.array(truncated, dtype=np.float32),
                    occluded=np.array(occluded, dtype=np.int64),
                    alpha=np.array(alpha, dtype=np.float32),
                    bbox=np.array(bboxes2d, dtype=np.float32).reshape(-1, 4),
                    dimensions=np.array(dimensions, dtype=np.float32).reshape(-1, 3),
                    location=np.array(locations, dtype=np.float32).reshape(-1, 3),
                    rotation_y=np.array(rotation_y, dtype=np.float32),
                    score=np.array(score_list, dtype=np.float32),
                )
            )

        return det_annos

    def _evaluate_kitti(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluation in KITTI protocol."""
        from mmdet3d.core.evaluation import kitti_eval

        print(f"Evaluating {len(results)} results (KITTI)...")
        det_annos = self._format_results_kitti(results, jsonfile_prefix)
        gt_annos = self._load_kitti_gts()

        ap_result_str, ap_dict = kitti_eval(
            gt_annos,
            det_annos,
            ['Car', 'Pedestrian', 'Cyclist'],
            eval_types=['bbox', 'bev', '3d'],
        )
        print(ap_result_str)

        detail = {}
        for ap_type, ap in ap_dict.items():
            detail[f'pts_bbox_KITTI/{ap_type}'] = float('{:.4f}'.format(ap))

        return detail
    # ==================== Argoverse2 Evaluation Methods ====================

    def _yaw_to_quat(self, yaw):
        """Convert yaw angle to quaternion [w, x, y, z]."""
        # yaw is rotation about z-axis
        half_yaw = yaw / 2.0
        qw = np.cos(half_yaw)
        qx = 0.0
        qy = 0.0
        qz = np.sin(half_yaw)
        return np.array([qw, qx, qy, qz])

    def _format_bbox_av2(self, results, log_id, timestamp_ns):
        """Convert predictions to Argoverse2 format for a single frame."""
        if "pts_bbox" in results:
            bboxes = results["pts_bbox"]["boxes_3d"]
            scores = results["pts_bbox"]["scores_3d"]
            labels = results["pts_bbox"]["labels_3d"]
        else:
            bboxes = results["boxes_3d"]
            scores = results["scores_3d"]
            labels = results["labels_3d"]

        # Convert to numpy
        bboxes_tensor = bboxes.tensor.cpu().numpy()
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()

        av2_boxes = []
        for i in range(len(bboxes_tensor)):
            # LiDARInstance3DBoxes: [x, y, z, l, w, h, yaw, vx, vy]
            x, y, z, l, w, h, yaw = bboxes_tensor[i, :7]
            
            # Get gravity center (mmdet3d boxes have center at bottom)
            # For AV2, the center should be at gravity center
            z_center = z + h / 2.0  # Move from bottom to center
            
            # Convert yaw to quaternion
            quat = self._yaw_to_quat(yaw)
            
            # Map NuScenes class to AV2 class
            cls_name = self.CLASSES[labels[i]]
            av2_cls = self.NUSC_TO_AV2_MAPPING.get(cls_name, cls_name.upper())
            
            av2_boxes.append({
                'tx_m': x,
                'ty_m': y,
                'tz_m': z_center,
                'length_m': l,
                'width_m': w,
                'height_m': h,
                'qw': quat[0],
                'qx': quat[1],
                'qy': quat[2],
                'qz': quat[3],
                'score': float(scores[i]),
                'log_id': log_id,
                'timestamp_ns': int(timestamp_ns),
                'category': av2_cls,
            })
        
        return av2_boxes

    def _format_results_av2(self, outputs, jsonfile_prefix=None):
        """Format the results to Argoverse2 DataFrame format."""
        print(f"Formatting {len(outputs)} results for Argoverse2...")
        
        all_boxes = []
        for i, out in enumerate(outputs):
            info = self.data_infos[i]
            log_id = info.get('log_id')
            timestamp_ns = info.get('timestamp_ns')
            
            if log_id is None or timestamp_ns is None:
                continue
            
            frame_boxes = self._format_bbox_av2(out, log_id, timestamp_ns)
            all_boxes.extend(frame_boxes)
        
        if len(all_boxes) == 0:
            print("Warning: No boxes to format!")
            return pd.DataFrame()
        
        # Create DataFrame
        dts = pd.DataFrame(all_boxes)
        
        # Sort by score descending
        dts = dts.sort_values("score", ascending=False)
        
        # Save if prefix provided
        if jsonfile_prefix is not None:
            feather_path = f"{jsonfile_prefix}_av2_dts.feather"
            dts.to_feather(feather_path)
            print(f"Results saved to {feather_path}")
        
        # Also save to data_root for persistent cache
        persistent_path = osp.join(self.data_root, f'{self.tri3d_dataset.split}_dts.feather')
        dts.to_feather(persistent_path)
        print(f"Results also saved to {persistent_path}")
        
        # Set index for evaluation
        dts = dts.set_index(["log_id", "timestamp_ns"]).sort_index()
        
        return dts

    def _load_av2_annotations(self):
        """Load Argoverse2 ground truth annotations."""
        # Try to load cached annotations first
        anno_path = osp.join(self.data_root, f'{self.tri3d_dataset.split}_anno.feather')
        
        if osp.exists(anno_path):
            print(f"Loading cached annotations from {anno_path}")
            gts = pd.read_feather(anno_path)
            return gts.set_index(["log_id", "timestamp_ns"]).sort_index()
        
        # Build annotations from tri3d dataset
        print("Building annotations from Tri3D dataset...")
        all_gts = []
        
        for info in self.data_infos:
            seq = info['seq']
            frame = info['frame']
            sensor = info['sensor']
            log_id = info.get('log_id')
            timestamp_ns = info.get('timestamp_ns')
            
            if log_id is None or timestamp_ns is None:
                continue
            
            boxes = self.tri3d_dataset.boxes(seq, frame, coords=sensor)
            
            for box in boxes:
                # Keep original AV2 category to avoid losing information
                # from many-to-one-to-one mapping (e.g., BOX_TRUCK->truck->TRUCK)
                av2_cls = box.label
                
                quat = self._yaw_to_quat(box.heading)
                z_center = box.center[2]  # Tri3D boxes are already at gravity center
                
                # Get num_interior_pts if available
                num_pts = getattr(box, 'num_interior_pts', 1)
                
                all_gts.append({
                    'tx_m': box.center[0],
                    'ty_m': box.center[1],
                    'tz_m': z_center,
                    'length_m': box.size[0],
                    'width_m': box.size[1],
                    'height_m': box.size[2],
                    'qw': quat[0],
                    'qx': quat[1],
                    'qy': quat[2],
                    'qz': quat[3],
                    'num_interior_pts': num_pts,
                    'log_id': log_id,
                    'timestamp_ns': int(timestamp_ns),
                    'category': av2_cls,
                })
        
        gts = pd.DataFrame(all_gts)
        
        # Cache for future use
        gts.to_feather(anno_path)
        print(f"Cached annotations to {anno_path}")
        
        return gts.set_index(["log_id", "timestamp_ns"]).sort_index()

    def _evaluate_av2(self, results, logger=None, jsonfile_prefix=None, eval_range_m=None, **kwargs):
        """Evaluation in Argoverse2 protocol."""
        print(f"Evaluating {len(results)} results (Argoverse2)...")
        
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        
        # Format predictions
        dts = self._format_results_av2(results, jsonfile_prefix)
        
        if dts.empty:
            print("No predictions to evaluate!")
            return {}
        
        # Load ground truth
        gts = self._load_av2_annotations()
        
        # Filter to only evaluate on frames we have predictions for
        valid_uuids = set(dts.index.tolist()) & set(gts.index.tolist())
        print(f"Evaluating on {len(valid_uuids)} frames with both predictions and GT")
        
        if len(valid_uuids) == 0:
            print("No matching frames between predictions and GT!")
            return {}
        
        gts = gts.loc[list(valid_uuids)].sort_index()
        
        # Import AV2 evaluation utilities
        try:
            from av2.evaluation import SensorCompetitionCategories
            from projects.mmdet3d_plugin.datasets.av2_evaluation import DetectionCfg, evaluate as av2_evaluate
        except ImportError as e:
            print(f"Warning: Could not import AV2 evaluation modules: {e}")
            print("Falling back to basic metrics...")
            return self._basic_av2_metrics(dts, gts)
        
        # Determine evaluation categories (intersection of available categories)
        available_categories = set(gts["category"].unique().tolist())
        competition_categories = set(x.value for x in SensorCompetitionCategories)
        eval_categories = available_categories & competition_categories
        
        # Also include categories we predict (mapped from NuScenes)
        pred_categories = set(dts["category"].unique().tolist())
        eval_categories = eval_categories | (pred_categories & competition_categories)
        
        print(f"Evaluating on categories: {sorted(eval_categories)}")
        
        # Setup evaluation config
        split_dir = Path(self.data_root) / self.tri3d_dataset.split
        cfg = DetectionCfg(
            dataset_dir=split_dir if split_dir.exists() else None,
            categories=tuple(sorted(eval_categories)),
            eval_range_m=(0.0, 150.0) if eval_range_m is None else tuple(eval_range_m),
            eval_only_roi_instances=split_dir.exists(),  # Only if we have maps
        )
        
        # Run evaluation
        try:
            eval_dts, eval_gts, metrics, recall3d = av2_evaluate(
                dts.reset_index(), gts.reset_index(), cfg
            )
        except Exception as e:
            print(f"AV2 evaluation failed: {e}")
            return self._basic_av2_metrics(dts.reset_index(), gts.reset_index())
        
        # Print results
        valid_categories = sorted(eval_categories) + ["AVERAGE_METRICS"]
        print("\n" + "=" * 80)
        print("Argoverse2 Evaluation Results:")
        print(metrics.loc[valid_categories])
        print("=" * 80 + "\n")
        
        # Build result dict
        detail = {}
        metric_prefix = 'pts_bbox_AV2'
        
        for category in metrics.index:
            for col in metrics.columns:
                val = float(metrics.loc[category, col])
                detail[f'{metric_prefix}/{category}/{col}'] = val
        
        # Add overall metrics
        if 'AVERAGE_METRICS' in metrics.index:
            for col in metrics.columns:
                val = float(metrics.loc['AVERAGE_METRICS', col])
                detail[f'{metric_prefix}/mean_{col}'] = val
        
        # Cleanup
        if tmp_dir is not None:
            tmp_dir.cleanup()
        
        return detail

    def _basic_av2_metrics(self, dts, gts):
        """Compute basic metrics when full AV2 evaluation is not available."""
        print("Computing basic metrics...")
        
        detail = {}
        detail['num_predictions'] = len(dts)
        detail['num_ground_truth'] = len(gts)
        
        # Count predictions per category
        if 'category' in dts.columns:
            for cat in dts['category'].unique():
                detail[f'num_pred_{cat}'] = int((dts['category'] == cat).sum())
        
        return detail
