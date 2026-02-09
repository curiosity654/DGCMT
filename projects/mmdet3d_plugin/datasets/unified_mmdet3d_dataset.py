import hashlib
import json
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

try:
    from numba import njit
except Exception:  # noqa: BLE001
    njit = None


if njit is not None:
    @njit(cache=True)
    def _simbev_match_numba(pred_centers, gt_centers, threshold):
        """Greedy nearest-neighbor matching for one class/threshold."""
        num_pred = pred_centers.shape[0]
        num_gt = gt_centers.shape[0]

        tp = np.zeros(num_pred, dtype=np.float32)
        fp = np.zeros(num_pred, dtype=np.float32)
        matched_gt_idx = np.full(num_pred, -1, dtype=np.int64)
        min_dists = np.full(num_pred, np.nan, dtype=np.float32)

        if num_gt == 0:
            for i in range(num_pred):
                fp[i] = 1.0
            return tp, fp, matched_gt_idx, min_dists

        assigned = np.zeros(num_gt, dtype=np.uint8)
        for i in range(num_pred):
            best_idx = -1
            best_dist = 1e20
            px = pred_centers[i, 0]
            py = pred_centers[i, 1]
            pz = pred_centers[i, 2]
            for j in range(num_gt):
                if assigned[j] != 0:
                    continue
                dx = gt_centers[j, 0] - px
                dy = gt_centers[j, 1] - py
                dz = gt_centers[j, 2] - pz
                d = (dx * dx + dy * dy + dz * dz) ** 0.5
                if d < best_dist:
                    best_dist = d
                    best_idx = j

            if best_idx >= 0 and best_dist <= threshold:
                tp[i] = 1.0
                assigned[best_idx] = 1
                matched_gt_idx[i] = best_idx
                min_dists[i] = best_dist
            else:
                fp[i] = 1.0

        return tp, fp, matched_gt_idx, min_dists
else:
    def _simbev_match_numba(pred_centers, gt_centers, threshold):
        """Python fallback if numba is unavailable."""
        num_pred = pred_centers.shape[0]
        num_gt = gt_centers.shape[0]

        tp = np.zeros(num_pred, dtype=np.float32)
        fp = np.zeros(num_pred, dtype=np.float32)
        matched_gt_idx = np.full(num_pred, -1, dtype=np.int64)
        min_dists = np.full(num_pred, np.nan, dtype=np.float32)

        if num_gt == 0:
            fp[:] = 1.0
            return tp, fp, matched_gt_idx, min_dists

        assigned = np.zeros(num_gt, dtype=bool)
        for i in range(num_pred):
            dists = np.linalg.norm(gt_centers - pred_centers[i], axis=1)
            dists[assigned] = np.inf
            j = int(np.argmin(dists))
            d = float(dists[j])
            if np.isfinite(d) and d <= threshold:
                tp[i] = 1.0
                assigned[j] = True
                matched_gt_idx[i] = j
                min_dists[i] = d
            else:
                fp[i] = 1.0

        return tp, fp, matched_gt_idx, min_dists

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
    CACHE_VERSION = 1
    
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

    # SimBEV label mapping to NuScenes 10 classes
    SIMBEV_MAPPING = {
        'car': 'car',
        'truck': 'truck',
        'bus': 'bus',
        'motorcycle': 'motorcycle',
        'bicycle': 'bicycle',
        'pedestrian': 'pedestrian',
        'van': 'car',  # Map van to car
        'trailer': 'trailer',
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
                 load_interval=1,
                 use_cache=True,
                 cache_dir=None,
                 cache_refresh=False,
                 **kwargs):
        # 1. Initialize Tri3D dataset

        self.dataset_type_name = dataset_type  # Save for branching in evaluate/format
        self.tri3d_cls = getattr(tri3d_datasets, dataset_type)
        
        tri3d_kwargs = {}
        if subset is not None:
            tri3d_kwargs['subset'] = subset
        if split is not None:
            tri3d_kwargs['split'] = split
        
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
        elif dataset_type == 'SimBEV':
            self.cat_mapping = self.SIMBEV_MAPPING
        else:
            self.cat_mapping = {}

        # 2. Initialize standard Dataset attributes (mimicking Custom3DDataset)
        self.data_root = data_root
        self.subset = subset
        self.split = split
        self.ann_file = None 
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.load_interval = load_interval
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.cache_refresh = cache_refresh
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

    def _get_cache_meta(self, sensor):
        return {
            'cache_version': self.CACHE_VERSION,
            'dataset_type': self.dataset_type_name,
            'tri3d_cls': self.tri3d_cls.__name__,
            'subset': self.subset,
            'split': self.split,
            'sensor': sensor,
            'test_mode': self.test_mode,
            'filter_empty_gt': self.filter_empty_gt,
            'load_interval': self.load_interval,
            'classes': list(self.CLASSES),
            'cat_mapping': dict(self.cat_mapping),
        }

    def _get_cache_path(self, sensor):
        base_dir = self.cache_dir or osp.join(self.data_root, '.cache', 'unified_mmdet3d')
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        meta = self._get_cache_meta(sensor)
        meta_str = json.dumps(meta, sort_keys=True, ensure_ascii=True)
        meta_hash = hashlib.md5(meta_str.encode('utf-8')).hexdigest()
        split_tag = self.split or 'nosplit'
        subset_tag = self.subset or 'nosubset'
        filename = f"{self.dataset_type_name}_{subset_tag}_{split_tag}_{meta_hash}.pkl"
        return osp.join(base_dir, filename), meta

    def load_annotations(self, ann_file):
        """Rebuild data index from Tri3D."""
        data_infos = []
        if hasattr(self.tri3d_dataset, 'pcl_sensors') and self.tri3d_dataset.pcl_sensors:
            sensor = self.tri3d_dataset.pcl_sensors[0]
        else:
            raise ValueError("Dataset must have point cloud sensors")

        cache_path = None
        cache_meta = None
        if self.use_cache:
            cache_path, cache_meta = self._get_cache_path(sensor)
            if not self.cache_refresh and osp.exists(cache_path):
                cached = mmcv.load(cache_path)
                if cached.get('meta') == cache_meta and 'data_infos' in cached:
                    print(f"Loaded cached data_infos from {cache_path}")
                    return cached['data_infos']

        try:
            sequences = self.tri3d_dataset.sequences()
        except NotImplementedError:
             sequences = []

        print(f"Indexing {len(sequences)} sequences from {self.tri3d_dataset.__class__.__name__}...")
        
        indexed_frames = []
        for seq in tqdm(sequences, desc="Indexing sequences"):
            try:
                frames = self.tri3d_dataset.keyframes(seq, sensor)
                is_keyframes = True
            except (AttributeError, NotImplementedError):
                frames = self.tri3d_dataset.frames(seq, sensor)
                is_keyframes = False

            for frame_idx, frame in enumerate(frames):
                indexed_frames.append(
                    dict(
                        seq=seq,
                        frame=frame,
                        frame_idx=frame_idx,
                        is_keyframes=is_keyframes,
                    )
                )

        indexed_frames = indexed_frames[::self.load_interval]

        for item in tqdm(indexed_frames, desc="Building data infos"):
            seq = item["seq"]
            frame = item["frame"]
            frame_idx = item["frame_idx"]
            is_keyframes = item["is_keyframes"]
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
                if is_keyframes:
                    try:
                        token = self.tri3d_dataset.sample_tokens(seq)[frame_idx]
                    except Exception:
                        if hasattr(self.tri3d_dataset.obj, 'scenes'):
                            try:
                                token = self.tri3d_dataset.obj.scenes[seq].sample_tokens[frame_idx]
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

            data_infos.append(data_info)
        
        print(f"Loaded {len(data_infos)} frames (filtered: {self.filter_empty_gt})")
        if self.use_cache and cache_path is not None:
            mmcv.dump({'meta': cache_meta, 'data_infos': data_infos}, cache_path)
            print(f"Saved data_infos cache to {cache_path}")
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
            native2tri3d = RigidTransform(Rotation.from_euler("Z", -np.pi / 2), [0, 0, 0])
            native2world = sensor2world @ native2tri3d
        else:
            native2world = None

        for i in range(len(bboxes_tensor)):
            # LiDARInstance3DBoxes for NuScenes: [x, y, z, l, w, h, yaw, vx, vy]
            # NuScenes expects: [x, y, z] for translation, [w, l, h] for size
            x, y, z, l, w, h, yaw = bboxes_tensor[i, :7]
            z_center = z + h / 2.0

            if native2world is not None:
                # Create box to native LIDAR transform
                # Translation is gravity center [x, y, z_center]
                box2native = RigidTransform(Rotation.from_euler("Z", yaw), [x, y, z_center])

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
        elif self.dataset_type_name == 'SimBEV':
            return self._evaluate_simbev(results, logger, jsonfile_prefix, **kwargs)
        else:
            return self._evaluate_nusc(results, logger, jsonfile_prefix, **kwargs)

    def _extract_pred_arrays(self, result):
        """Extract prediction arrays from model output for a single sample."""
        if "pts_bbox" in result:
            pred = result["pts_bbox"]
        else:
            pred = result

        boxes_3d = pred["boxes_3d"]
        scores_3d = pred["scores_3d"]
        labels_3d = pred["labels_3d"]

        boxes = boxes_3d.tensor.detach().cpu().numpy()
        scores = scores_3d.detach().cpu().numpy()
        labels = labels_3d.detach().cpu().numpy().astype(np.int64)

        centers = boxes[:, :3] if boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
        yaws = boxes[:, 6] if boxes.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
        sizes = boxes[:, 3:6] if boxes.shape[0] > 0 else np.zeros((0, 3), dtype=np.float32)
        if boxes.shape[1] >= 9:
            velocities = boxes[:, 7:9]
        else:
            velocities = None

        return {
            "centers": centers,
            "yaws": yaws,
            "sizes": sizes,
            "velocities": velocities,
            "scores": scores,
            "labels": labels,
        }

    def _extract_gt_arrays(self, seq, frame, sensor):
        """Extract GT arrays from Tri3D boxes for a single sample."""
        boxes = self.tri3d_dataset.boxes(seq, frame, coords=sensor)

        gt_centers = []
        gt_yaws = []
        gt_sizes = []
        gt_velocities = []
        gt_label_ids = []

        for box in boxes:
            mapped_label = self._map_label(box.label)
            if mapped_label is None or mapped_label not in self.CLASSES:
                continue

            cls_id = self.CLASSES.index(mapped_label)
            gt_label_ids.append(cls_id)
            gt_centers.append(np.asarray(box.center, dtype=np.float32))
            gt_yaws.append(float(box.heading))
            gt_sizes.append(np.asarray(box.size, dtype=np.float32))

            vel = getattr(box, "velocity", None)
            if vel is None:
                gt_velocities.append(np.array([np.nan, np.nan], dtype=np.float32))
            else:
                vel = np.asarray(vel, dtype=np.float32)
                if vel.shape[0] >= 2:
                    gt_velocities.append(vel[:2])
                else:
                    gt_velocities.append(np.array([np.nan, np.nan], dtype=np.float32))

        if len(gt_label_ids) == 0:
            return {
                "centers": np.zeros((0, 3), dtype=np.float32),
                "yaws": np.zeros((0,), dtype=np.float32),
                "sizes": np.zeros((0, 3), dtype=np.float32),
                "velocities": np.zeros((0, 2), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
            }

        return {
            "centers": np.stack(gt_centers, axis=0),
            "yaws": np.asarray(gt_yaws, dtype=np.float32),
            "sizes": np.stack(gt_sizes, axis=0),
            "velocities": np.stack(gt_velocities, axis=0),
            "labels": np.asarray(gt_label_ids, dtype=np.int64),
        }

    @staticmethod
    def _yaw_abs_diff(yaw_a, yaw_b):
        """Absolute wrapped yaw difference in radians."""
        diff = (yaw_a - yaw_b + np.pi) % (2 * np.pi) - np.pi
        return float(np.abs(diff))

    @staticmethod
    def _safe_nanmean(vals):
        if len(vals) == 0:
            return np.nan
        arr = np.asarray(vals, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.mean(arr))

    @staticmethod
    def _safe_nanmax(vals):
        if len(vals) == 0:
            return np.nan
        arr = np.asarray(vals, dtype=np.float32)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.nan
        return float(np.max(arr))

    def _init_simbev_metrics_store(self, thresholds):
        return {
            thr: {
                cls_id: {
                    "scores": [],
                    "tp": [],
                    "fp": [],
                    "ate": [],
                    "aoe": [],
                    "ase": [],
                    "ave": [],
                    "num_gt": 0,
                }
                for cls_id in range(len(self.CLASSES))
            }
            for thr in thresholds
        }

    def _finalize_simbev_metrics(self, metrics_store, thresholds, metric_prefix):
        """Aggregate per-sample matching stats into final scalar metrics."""
        detail = {}
        ap_cls_means = []
        ate_cls_means = []
        aoe_cls_means = []
        ase_cls_means = []
        ave_cls_means = []

        for cls_id, cls_name in enumerate(self.CLASSES):
            class_ap_vals = []
            class_ate_vals = []
            class_aoe_vals = []
            class_ase_vals = []
            class_ave_vals = []

            for thr in thresholds:
                cls_store = metrics_store[thr][cls_id]
                thr_str = f"{thr:.1f}"

                scores = np.asarray(cls_store["scores"], dtype=np.float32)
                tp = np.asarray(cls_store["tp"], dtype=np.float32)
                fp = np.asarray(cls_store["fp"], dtype=np.float32)
                num_gt = int(cls_store["num_gt"])

                if scores.shape[0] > 0:
                    order = np.argsort(-scores)
                    tp = tp[order]
                    fp = fp[order]
                    tp_cum = np.cumsum(tp)
                    fp_cum = np.cumsum(fp)
                else:
                    tp_cum = np.zeros((0,), dtype=np.float32)
                    fp_cum = np.zeros((0,), dtype=np.float32)

                if num_gt > 0:
                    recalls = tp_cum / max(num_gt, 1)
                    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
                    recalls = np.concatenate(([0.0], recalls))
                    precisions = np.concatenate(([1.0], precisions))
                    ap = float(np.trapz(precisions, recalls))
                else:
                    ap = np.nan

                ate = self._safe_nanmean(cls_store["ate"])
                aoe = self._safe_nanmean(cls_store["aoe"])
                ase = self._safe_nanmean(cls_store["ase"])
                ave = self._safe_nanmean(cls_store["ave"])

                detail[f"{metric_prefix}/{cls_name}_AP_dist_{thr_str}"] = ap
                detail[f"{metric_prefix}/{cls_name}_ATE_dist_{thr_str}"] = ate
                detail[f"{metric_prefix}/{cls_name}_AOE_dist_{thr_str}"] = aoe
                detail[f"{metric_prefix}/{cls_name}_ASE_dist_{thr_str}"] = ase
                detail[f"{metric_prefix}/{cls_name}_AVE_dist_{thr_str}"] = ave

                class_ap_vals.append(ap)
                class_ate_vals.append(ate)
                class_aoe_vals.append(aoe)
                class_ase_vals.append(ase)
                class_ave_vals.append(ave)

            ap_mean = self._safe_nanmean(class_ap_vals)
            ate_mean = self._safe_nanmean(class_ate_vals)
            aoe_mean = self._safe_nanmean(class_aoe_vals)
            ase_mean = self._safe_nanmean(class_ase_vals)
            ave_mean = self._safe_nanmean(class_ave_vals)

            detail[f"{metric_prefix}/{cls_name}_AP_dist_mean"] = ap_mean
            detail[f"{metric_prefix}/{cls_name}_ATE_dist_mean"] = ate_mean
            detail[f"{metric_prefix}/{cls_name}_AOE_dist_mean"] = aoe_mean
            detail[f"{metric_prefix}/{cls_name}_ASE_dist_mean"] = ase_mean
            detail[f"{metric_prefix}/{cls_name}_AVE_dist_mean"] = ave_mean

            detail[f"{metric_prefix}/{cls_name}_AP_dist_max"] = self._safe_nanmax(class_ap_vals)
            detail[f"{metric_prefix}/{cls_name}_ATE_dist_max"] = self._safe_nanmax(class_ate_vals)
            detail[f"{metric_prefix}/{cls_name}_AOE_dist_max"] = self._safe_nanmax(class_aoe_vals)
            detail[f"{metric_prefix}/{cls_name}_ASE_dist_max"] = self._safe_nanmax(class_ase_vals)
            detail[f"{metric_prefix}/{cls_name}_AVE_dist_max"] = self._safe_nanmax(class_ave_vals)

            ap_cls_means.append(ap_mean)
            ate_cls_means.append(ate_mean)
            aoe_cls_means.append(aoe_mean)
            ase_cls_means.append(ase_mean)
            ave_cls_means.append(ave_mean)

        mAP = self._safe_nanmean(ap_cls_means)
        mATE = self._safe_nanmean(ate_cls_means)
        mAOE = self._safe_nanmean(aoe_cls_means)
        mASE = self._safe_nanmean(ase_cls_means)
        mAVE = self._safe_nanmean(ave_cls_means)

        detail[f"{metric_prefix}/mAP"] = mAP
        detail[f"{metric_prefix}/mATE"] = mATE
        detail[f"{metric_prefix}/mAOE"] = mAOE
        detail[f"{metric_prefix}/mASE"] = mASE
        detail[f"{metric_prefix}/mAVE"] = mAVE

        s_mAP = 0.0 if np.isnan(mAP) else mAP
        s_mATE = 0.0 if np.isnan(mATE) else max(0.0, 1.0 - mATE)
        s_mAOE = 0.0 if np.isnan(mAOE) else max(0.0, 1.0 - mAOE)
        s_mASE = 0.0 if np.isnan(mASE) else max(0.0, 1.0 - mASE)
        s_mAVE = 0.0 if np.isnan(mAVE) else max(0.0, 1.0 - mAVE)
        sds = (4.0 * s_mAP + s_mATE + s_mAOE + s_mASE + s_mAVE) / 8.0
        detail[f"{metric_prefix}/SDS"] = float(sds)

        print("\n" + "=" * 50)
        print("SimBEV Evaluation Results:")
        print(f"  mAP: {mAP}")
        print(f"  mATE: {mATE}")
        print(f"  mAOE: {mAOE}")
        print(f"  mASE: {mASE}")
        print(f"  mAVE: {mAVE}")
        print(f"  SDS: {sds}")
        print("=" * 50 + "\n")
        return detail

    def _evaluate_simbev(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluate SimBEV predictions with selectable backend."""
        use_numba_eval = kwargs.get("use_numba_eval", False)
        if use_numba_eval:
            return self._evaluate_simbev_numba(results, logger, jsonfile_prefix, **kwargs)
        return self._evaluate_simbev_python(results, logger, jsonfile_prefix, **kwargs)

    def _evaluate_simbev_python(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluate SimBEV predictions with distance-based matching (Python backend)."""
        print(f"Evaluating {len(results)} results (SimBEV, backend=python)...")
        thresholds = [0.5, 1.0, 2.0, 4.0]
        metric_prefix = "pts_bbox_SimBEV"
        metrics_store = self._init_simbev_metrics_store(thresholds)

        for i, result in enumerate(results):
            info = self.data_infos[i]
            pred = self._extract_pred_arrays(result)
            gt = self._extract_gt_arrays(info["seq"], info["frame"], info["sensor"])

            for cls_id in range(len(self.CLASSES)):
                pred_mask = pred["labels"] == cls_id
                gt_mask = gt["labels"] == cls_id

                pred_centers = pred["centers"][pred_mask]
                pred_sizes = pred["sizes"][pred_mask]
                pred_yaws = pred["yaws"][pred_mask]
                pred_scores = pred["scores"][pred_mask]
                pred_velocities = None if pred["velocities"] is None else pred["velocities"][pred_mask]

                gt_centers = gt["centers"][gt_mask]
                gt_sizes = gt["sizes"][gt_mask]
                gt_yaws = gt["yaws"][gt_mask]
                gt_velocities = gt["velocities"][gt_mask]

                if pred_scores.shape[0] > 0:
                    order = np.argsort(-pred_scores)
                    pred_centers = pred_centers[order]
                    pred_sizes = pred_sizes[order]
                    pred_yaws = pred_yaws[order]
                    pred_scores = pred_scores[order]
                    if pred_velocities is not None:
                        pred_velocities = pred_velocities[order]

                for thr in thresholds:
                    cls_store = metrics_store[thr][cls_id]
                    cls_store["num_gt"] += int(gt_centers.shape[0])

                    assigned_gt = np.zeros(gt_centers.shape[0], dtype=bool)
                    for pred_idx in range(pred_centers.shape[0]):
                        cls_store["scores"].append(float(pred_scores[pred_idx]))
                        if gt_centers.shape[0] == 0:
                            cls_store["tp"].append(0.0)
                            cls_store["fp"].append(1.0)
                            continue

                        dists = np.linalg.norm(gt_centers - pred_centers[pred_idx], axis=1)
                        dists[assigned_gt] = np.inf
                        matched_gt = int(np.argmin(dists))
                        min_dist = float(dists[matched_gt])

                        if np.isfinite(min_dist) and min_dist <= thr:
                            assigned_gt[matched_gt] = True
                            cls_store["tp"].append(1.0)
                            cls_store["fp"].append(0.0)

                            cls_store["ate"].append(min_dist)
                            cls_store["aoe"].append(
                                self._yaw_abs_diff(gt_yaws[matched_gt], pred_yaws[pred_idx])
                            )

                            # 1 - IoU with aligned center/yaw (size-only IoU approximation).
                            pred_dims = pred_sizes[pred_idx]
                            gt_dims = gt_sizes[matched_gt]
                            min_dims = np.minimum(pred_dims, gt_dims)
                            inter = float(np.prod(min_dims))
                            pred_vol = float(np.prod(pred_dims))
                            gt_vol = float(np.prod(gt_dims))
                            union = pred_vol + gt_vol - inter
                            ase = 1.0 - inter / max(union, 1e-6)
                            cls_store["ase"].append(float(ase))

                            if pred_velocities is None:
                                cls_store["ave"].append(np.nan)
                            else:
                                pred_v = pred_velocities[pred_idx]
                                gt_v = gt_velocities[matched_gt]
                                if np.any(np.isnan(gt_v)):
                                    cls_store["ave"].append(np.nan)
                                else:
                                    cls_store["ave"].append(float(np.linalg.norm(pred_v - gt_v)))
                        else:
                            cls_store["tp"].append(0.0)
                            cls_store["fp"].append(1.0)

        return self._finalize_simbev_metrics(metrics_store, thresholds, metric_prefix)

    def _evaluate_simbev_numba(self, results, logger=None, jsonfile_prefix=None, **kwargs):
        """Evaluate SimBEV predictions with numba-accelerated matching backend."""
        if njit is None:
            print("Numba is not available, falling back to python SimBEV evaluator.")
            return self._evaluate_simbev_python(results, logger, jsonfile_prefix, **kwargs)

        print(f"Evaluating {len(results)} results (SimBEV, backend=numba)...")
        thresholds = [0.5, 1.0, 2.0, 4.0]
        metric_prefix = "pts_bbox_SimBEV"
        metrics_store = self._init_simbev_metrics_store(thresholds)

        for i, result in enumerate(results):
            info = self.data_infos[i]
            pred = self._extract_pred_arrays(result)
            gt = self._extract_gt_arrays(info["seq"], info["frame"], info["sensor"])

            for cls_id in range(len(self.CLASSES)):
                pred_mask = pred["labels"] == cls_id
                gt_mask = gt["labels"] == cls_id

                pred_centers = pred["centers"][pred_mask]
                pred_sizes = pred["sizes"][pred_mask]
                pred_yaws = pred["yaws"][pred_mask]
                pred_scores = pred["scores"][pred_mask]
                pred_velocities = None if pred["velocities"] is None else pred["velocities"][pred_mask]

                gt_centers = gt["centers"][gt_mask]
                gt_sizes = gt["sizes"][gt_mask]
                gt_yaws = gt["yaws"][gt_mask]
                gt_velocities = gt["velocities"][gt_mask]

                if pred_scores.shape[0] > 0:
                    order = np.argsort(-pred_scores)
                    pred_centers = pred_centers[order]
                    pred_sizes = pred_sizes[order]
                    pred_yaws = pred_yaws[order]
                    pred_scores = pred_scores[order]
                    if pred_velocities is not None:
                        pred_velocities = pred_velocities[order]

                pred_centers_numba = np.asarray(pred_centers, dtype=np.float32)
                gt_centers_numba = np.asarray(gt_centers, dtype=np.float32)

                for thr in thresholds:
                    cls_store = metrics_store[thr][cls_id]
                    cls_store["num_gt"] += int(gt_centers.shape[0])

                    tp, fp, matched_gt_idx, min_dists = _simbev_match_numba(
                        pred_centers_numba, gt_centers_numba, float(thr)
                    )
                    cls_store["scores"].extend(pred_scores.astype(np.float32).tolist())
                    cls_store["tp"].extend(tp.tolist())
                    cls_store["fp"].extend(fp.tolist())

                    for pred_idx in range(pred_centers.shape[0]):
                        gt_idx = int(matched_gt_idx[pred_idx])
                        if gt_idx < 0:
                            continue

                        cls_store["ate"].append(float(min_dists[pred_idx]))
                        cls_store["aoe"].append(
                            self._yaw_abs_diff(gt_yaws[gt_idx], pred_yaws[pred_idx])
                        )

                        pred_dims = pred_sizes[pred_idx]
                        gt_dims = gt_sizes[gt_idx]
                        min_dims = np.minimum(pred_dims, gt_dims)
                        inter = float(np.prod(min_dims))
                        pred_vol = float(np.prod(pred_dims))
                        gt_vol = float(np.prod(gt_dims))
                        union = pred_vol + gt_vol - inter
                        ase = 1.0 - inter / max(union, 1e-6)
                        cls_store["ase"].append(float(ase))

                        if pred_velocities is None:
                            cls_store["ave"].append(np.nan)
                        else:
                            pred_v = pred_velocities[pred_idx]
                            gt_v = gt_velocities[gt_idx]
                            if np.any(np.isnan(gt_v)):
                                cls_store["ave"].append(np.nan)
                            else:
                                cls_store["ave"].append(float(np.linalg.norm(pred_v - gt_v)))

        return self._finalize_simbev_metrics(metrics_store, thresholds, metric_prefix)

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
        custom_scenes = None
        if self.tri3d_dataset.obj.subset == 'v1.0-mini':
            eval_set = 'mini_val'
        elif hasattr(self.tri3d_dataset.obj, 'split'):
            split = self.tri3d_dataset.obj.split
            if isinstance(split, (list, tuple, set)):
                custom_scenes = list(split)
                eval_set = 'val'
            elif split in ['train', 'val', 'test']:
                eval_set = split
            elif split == 'val_mini':
                # Map our custom val_mini to 'val' split for SDK to load annotations
                eval_set = 'val'

        from nuscenes.nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import DetectionEval, remove_extra_samples
        from nuscenes.eval.detection.config import config_factory
        from nuscenes.eval.common.loaders import load_gt
        
        nusc = NuScenes(version=self.tri3d_dataset.obj.subset, dataroot=self.data_root, verbose=False)
        cfg = config_factory('detection_cvpr_2019')
        if hasattr(self, 'CLASSES') and self.CLASSES is not None:
            cfg.class_names = list(self.CLASSES)
            cfg.class_range = {k: v for k, v in cfg.class_range.items() if k in cfg.class_names}
        
        nusc_eval = DetectionEval(
            nusc,
            config=cfg,
            result_path=f"{jsonfile_prefix}.submission.json",
            eval_set=eval_set,
            output_dir=osp.dirname(jsonfile_prefix),
            verbose=True,
            custom_scenes=custom_scenes,
        )
        
        # If we are using a custom split, force the SDK to only evaluate on those tokens
        if custom_scenes is not None:
            nusc_eval.sample_tokens = list(nusc_eval.gt_boxes.sample_tokens)
            print(f"Forcing evaluation on {len(nusc_eval.sample_tokens)} custom split samples")
        elif self.tri3d_dataset.obj.split == 'val_mini':
            nusc_eval.sample_tokens = list(nusc_annos.keys())
            print(f"Forcing evaluation on {len(nusc_eval.sample_tokens)} mini split samples")

        nusc_eval.main(plot_examples=0, render_curves=False)
        
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
