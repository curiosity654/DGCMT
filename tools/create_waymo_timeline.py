#!/usr/bin/env python3
import argparse
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

NEEDED_DIRS = [
    "camera_box",
    "lidar",
    "lidar_segmentation",
    "camera_image",
    "camera_segmentation",
]

SLIM_COLUMNS = {
    "lidar": ["key.frame_timestamp_micros", "key.laser_name"],
    "lidar_segmentation": ["key.frame_timestamp_micros", "key.laser_name"],
    "camera_image": ["[CameraImageComponent].pose_timestamp", "key.camera_name"],
    "camera_segmentation": ["key.frame_timestamp_micros", "key.camera_name"],
    "camera_box": ["key.frame_timestamp_micros"],
}


def copy_tree(
    src: Path,
    dst: Path,
    hardlink: bool,
    slim: bool,
    strict: bool,
    error_log: Path | None,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.iterdir():
        if path.is_dir():
            copy_tree(path, dst / path.name, hardlink, slim, strict, error_log)
        else:
            if path.suffix != ".parquet" or path.name.endswith("_metadata.parquet"):
                # Skip non-parquet files (e.g. metadata without suffix).
                continue
            target = dst / path.name
            if target.exists():
                continue
            if slim and src.name in SLIM_COLUMNS:
                try:
                    parquet_file = pq.ParquetFile(path)
                    available = set(parquet_file.schema.names)
                    columns = [c for c in SLIM_COLUMNS[src.name] if c in available]
                    table = parquet_file.read(columns=columns)
                    pq.write_table(table, target)
                except Exception as exc:  # noqa: BLE001 - fallback to full copy
                    if strict:
                        if error_log is not None:
                            error_log.parent.mkdir(parents=True, exist_ok=True)
                            with error_log.open("a", encoding="utf-8") as handle:
                                handle.write(f"{path}\t{type(exc).__name__}: {exc}\n")
                        raise
                    print(
                        f"[warn] slim failed for {path}: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    if hardlink:
                        target.hardlink_to(path)
                    else:
                        shutil.copy2(path, target)
            elif hardlink:
                target.hardlink_to(path)
            else:
                shutil.copy2(path, target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a minimal Waymo timeline dataset for tri3d init."
    )
    parser.add_argument("--src", required=True, help="Source Waymo root")
    parser.add_argument("--dst", required=True, help="Destination root")
    parser.add_argument(
        "--hardlink",
        action="store_true",
        help="Use hardlinks instead of copying files",
    )
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Write minimal parquet files for timeline init",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on slim parquet read errors instead of falling back",
    )
    parser.add_argument(
        "--error-log",
        type=Path,
        help="Optional path to write slim failures (path, error)",
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)

    for split in ["training", "validation", "testing"]:
        for name in NEEDED_DIRS:
            src_dir = src_root / split / name
            if not src_dir.exists():
                continue
            dst_dir = dst_root / split / name
            copy_tree(
                src_dir,
                dst_dir,
                args.hardlink,
                args.slim,
                args.strict,
                args.error_log,
            )


if __name__ == "__main__":
    main()
