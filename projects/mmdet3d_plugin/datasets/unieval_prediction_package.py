from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import pandas as pd

SCHEMA_VERSION = 'unieval.v1'
PAYLOAD_RELPATH = 'predictions.feather'

ARGO2_REQUIRED_COLUMNS = (
    'tx_m',
    'ty_m',
    'tz_m',
    'length_m',
    'width_m',
    'height_m',
    'qw',
    'qx',
    'qy',
    'qz',
    'score',
    'log_id',
    'timestamp_ns',
    'category',
)

WAYMO_REQUIRED_COLUMNS = (
    'log_id',
    'timestamp_micros',
    'center_x',
    'center_y',
    'center_z',
    'length',
    'width',
    'height',
    'heading',
    'score',
    'category',
)

KITTI_REQUIRED_COLUMNS = (
    'frame_id',
    'center_x',
    'center_y',
    'center_z',
    'length',
    'width',
    'height',
    'yaw',
    'score',
    'category',
)

NUSC_REQUIRED_COLUMNS = (
    'sample_token',
    'center_x',
    'center_y',
    'center_z',
    'length',
    'width',
    'height',
    'qw',
    'qx',
    'qy',
    'qz',
    'vx',
    'vy',
    'score',
    'category',
)

DATASET_REQUIRED_COLUMNS = {
    'argo2': ARGO2_REQUIRED_COLUMNS,
    'waymo': WAYMO_REQUIRED_COLUMNS,
    'kitti': KITTI_REQUIRED_COLUMNS,
    'nuscenes': NUSC_REQUIRED_COLUMNS,
}

NUSC_TO_3CLASS = {
    'car': 'VEHICLE',
    'truck': 'VEHICLE',
    'construction_vehicle': 'VEHICLE',
    'bus': 'VEHICLE',
    'trailer': 'VEHICLE',
    'vehicle': 'VEHICLE',
    'bicycle': 'BICYCLE',
    'motorcycle': 'BICYCLE',
    'cyclist': 'BICYCLE',
    'pedestrian': 'PEDESTRIAN',
}

ARGO_3CLASS_VEHICLE_CATEGORIES = {
    'VEHICLE',
    'CAR',
    'TRUCK',
    'CONSTRUCTION_VEHICLE',
    'BUS',
    'TRAILER',
    'REGULAR_VEHICLE',
    'LARGE_VEHICLE',
    'BOX_TRUCK',
    'TRUCK_CAB',
    'ARTICULATED_BUS',
    'SCHOOL_BUS',
    'VEHICULAR_TRAILER',
}

ARGO_3CLASS_BICYCLE_CATEGORIES = {
    'BICYCLE',
    'CYCLIST',
    'MOTORCYCLE',
    'BICYCLIST',
    'MOTORCYCLIST',
    'WHEELED_RIDER',
}

ARGO_3CLASS_PEDESTRIAN_CATEGORIES = {
    'PEDESTRIAN',
    'OFFICIAL_SIGNALER',
}


def ensure_required_columns(columns: Iterable[str],
                            required: Sequence[str]) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')


def ensure_manifest_fields(manifest: Mapping[str, object]) -> None:
    required = {
        'schema_version',
        'dataset',
        'task',
        'split',
        'source_codebase',
        'label_space',
        'coord_system',
        'box_origin',
        'payload_relpath',
    }
    missing = sorted(required - set(manifest))
    if missing:
        raise ValueError(f'manifest.json is missing required fields: {missing}')

    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError(
            f'Unsupported schema_version {manifest["schema_version"]!r}; '
            f'expected {SCHEMA_VERSION!r}.')
    if manifest['payload_relpath'] != PAYLOAD_RELPATH:
        raise ValueError(
            'payload_relpath must be '
            f'{PAYLOAD_RELPATH!r}, got {manifest["payload_relpath"]!r}.')
    if manifest['box_origin'] not in {'bottom_center', 'gravity_center'}:
        raise ValueError(
            'box_origin must be "bottom_center" or "gravity_center", '
            f'got {manifest["box_origin"]!r}.')

    dataset = str(manifest['dataset'])
    if dataset not in DATASET_REQUIRED_COLUMNS:
        raise ValueError(
            f'dataset must be one of {sorted(DATASET_REQUIRED_COLUMNS)}, '
            f'got {dataset!r}.')


def build_prediction_manifest(dataset: str,
                              task: str,
                              split: str,
                              source_codebase: str,
                              label_space: str,
                              coord_system: str,
                              box_origin: str) -> dict:
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        dataset=dataset,
        task=task,
        split=split,
        source_codebase=source_codebase,
        label_space=label_space,
        coord_system=coord_system,
        box_origin=box_origin,
        payload_relpath=PAYLOAD_RELPATH,
    )
    ensure_manifest_fields(manifest)
    return manifest


def yaw_to_quaternion(yaw: float) -> Tuple[float, float, float, float]:
    half = float(yaw) / 2.0
    return (float(math.cos(half)), 0.0, 0.0, float(math.sin(half)))


def normalize_argo_3class_category(category: Optional[str]) -> Optional[str]:
    if category is None:
        return None
    cat = str(category).strip().upper()
    if cat in ARGO_3CLASS_VEHICLE_CATEGORIES:
        return 'VEHICLE'
    if cat in ARGO_3CLASS_BICYCLE_CATEGORIES:
        return 'BICYCLE'
    if cat in ARGO_3CLASS_PEDESTRIAN_CATEGORIES:
        return 'PEDESTRIAN'
    return None


def normalize_nuscenes_3class_category(category: Optional[str]) -> Optional[str]:
    if category is None:
        return None
    return NUSC_TO_3CLASS.get(str(category).strip().lower())


def _ordered_columns(dataset: str, columns: Sequence[str]) -> List[str]:
    required = list(DATASET_REQUIRED_COLUMNS[dataset])
    extras = [col for col in columns if col not in required]
    return required + sorted(extras)


def _coerce_types(dataset: str, payload: pd.DataFrame) -> pd.DataFrame:
    payload = payload.copy()
    if dataset == 'argo2':
        if 'log_id' in payload.columns:
            payload['log_id'] = payload['log_id'].astype(str)
        if 'timestamp_ns' in payload.columns:
            payload['timestamp_ns'] = payload['timestamp_ns'].astype('int64')
    elif dataset == 'waymo':
        if 'log_id' in payload.columns:
            payload['log_id'] = payload['log_id'].astype(str)
        if 'timestamp_micros' in payload.columns:
            payload['timestamp_micros'] = payload['timestamp_micros'].astype(
                'int64')
    elif dataset == 'kitti':
        if 'frame_id' in payload.columns:
            payload['frame_id'] = payload['frame_id'].astype(str)
    elif dataset == 'nuscenes':
        if 'sample_token' in payload.columns:
            payload['sample_token'] = payload['sample_token'].astype(str)
    if 'category' in payload.columns:
        payload['category'] = payload['category'].astype(str)
    return payload


def _sort_payload(dataset: str, payload: pd.DataFrame) -> pd.DataFrame:
    if payload.empty:
        return payload.reset_index(drop=True)

    if dataset == 'argo2':
        keys = ['log_id', 'timestamp_ns', 'score', 'category', 'tx_m', 'ty_m']
        ascending = [True, True, False, True, True, True]
    elif dataset == 'waymo':
        keys = [
            'log_id', 'timestamp_micros', 'score', 'category', 'center_x',
            'center_y'
        ]
        ascending = [True, True, False, True, True, True]
    elif dataset == 'kitti':
        keys = ['frame_id', 'score', 'category', 'center_x', 'center_y']
        ascending = [True, False, True, True, True]
    elif dataset == 'nuscenes':
        keys = ['sample_token', 'score', 'category', 'center_x', 'center_y']
        ascending = [True, False, True, True, True]
    else:
        return payload.reset_index(drop=True)

    valid_keys = [key for key in keys if key in payload.columns]
    valid_ascending = ascending[:len(valid_keys)]
    return payload.sort_values(
        valid_keys, ascending=valid_ascending,
        kind='mergesort').reset_index(drop=True)


def write_prediction_package(package_dir: Union[str, Path],
                             manifest: Mapping[str, object],
                             payload: pd.DataFrame) -> dict:
    package_dir = Path(package_dir)
    manifest_dict = dict(manifest)
    ensure_manifest_fields(manifest_dict)

    dataset = str(manifest_dict['dataset'])
    ensure_required_columns(payload.columns, DATASET_REQUIRED_COLUMNS[dataset])

    package_dir.mkdir(parents=True, exist_ok=True)
    payload = _sort_payload(dataset, _coerce_types(dataset, payload))
    payload = payload.loc[:, _ordered_columns(dataset, payload.columns)]

    manifest_path = package_dir / 'manifest.json'
    payload_path = package_dir / PAYLOAD_RELPATH
    manifest_path.write_text(
        json.dumps(manifest_dict, indent=2, sort_keys=True) + '\n',
        encoding='utf-8')
    payload.to_feather(payload_path)

    return dict(
        package_dir=str(package_dir),
        manifest_path=str(manifest_path),
        payload_path=str(payload_path),
    )
