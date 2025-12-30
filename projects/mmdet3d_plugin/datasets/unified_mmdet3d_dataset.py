import numpy as np
import mmcv
import os.path as osp
import tempfile
from mmdet.datasets import DATASETS
from mmdet3d.datasets import Custom3DDataset
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
import tri3d.datasets as tri3d_datasets
from pyquaternion import Quaternion
from tri3d.geometry import RigidTransform, Rotation

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
        self.tri3d_cls = getattr(tri3d_datasets, dataset_type)
        
        tri3d_kwargs = {}
        if subset is not None:
            tri3d_kwargs['subset'] = subset
        if split is not None:
            tri3d_kwargs['split'] = split
        
        # Initialize Tri3D dataset and wrap it to prevent deepcopy overhead
        tri3d_dataset = self.tri3d_cls(data_root, **tri3d_kwargs)
        self.tri3d_dataset = Tri3DObjectWrapper(tri3d_dataset)
        
        # Initialize category mapping
        if cat_mapping is not None:
            self.cat_mapping = cat_mapping
        elif dataset_type == 'NuScenes':
            self.cat_mapping = self.NUSC_MAPPING
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
        
        for seq in sequences:
            try:
                frames = self.tri3d_dataset.keyframes(seq, sensor)
                is_keyframes = True
            except (AttributeError, NotImplementedError):
                frames = self.tri3d_dataset.frames(seq, sensor)
                is_keyframes = False
                
            for i, frame in enumerate(frames):
                # Pre-calculate category IDs for CBGS and filtering
                boxes = self.tri3d_dataset.boxes(seq, frame, coords=sensor)
                cat_ids = []
                for box in boxes:
                    mapped_label = self._map_label(box.label)
                    if mapped_label and mapped_label in self.CLASSES:
                        cat_ids.append(self.CLASSES.index(mapped_label))
                
                # In training mode, we might want to skip empty frames
                if not self.test_mode and self.filter_empty_gt and len(cat_ids) == 0:
                    continue

                # Get token for NuScenes evaluation if available
                token = None
                if hasattr(self.tri3d_dataset.obj, 'scenes'):
                    try:
                        # In Tri3D, NuScenes.sample_tokens(seq) returns tokens for keyframes.
                        # If we are iterating over keyframes, the index i corresponds to the sample token.
                        if is_keyframes:
                            token = self.tri3d_dataset.obj.scenes[seq].sample_tokens[i]
                        else:
                            # For non-keyframes, we'd need to find the nearest keyframe token
                            # or leave it as None. NuScenes evaluation only works on keyframes.
                            pass
                    except (AttributeError, IndexError):
                        pass

                data_infos.append(dict(
                    token=token,
                    seq=seq,
                    frame=frame,
                    sensor=sensor,
                    sample_idx=f"{seq}_{frame}",
                    cat_ids=list(set(cat_ids))
                ))
        
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
        """Format the results to json."""
        nusc_annos = {}
        print(f"Formatting {len(outputs)} results...")
        
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
        """Evaluation in NuScenes protocol."""
        print(f"Evaluating {len(results)} results...")
        
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
