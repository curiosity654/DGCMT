# Repository Guidelines

## Project Structure & Module Organization
- `projects/` holds training code extensions and configuration files. Use `projects/configs/` for experiment configs and `projects/mmdet3d_plugin/` for model/data pipeline additions.
- `tools/` provides entry points for training, testing, data prep, and visualization (e.g., `tools/train.py`, `tools/test.py`, `tools/dist_train.sh`).
- `data/` is the expected dataset root (e.g., `data/nuscenes`).
- `figs/` and `vis_output/` store figures and generated visualizations.
- `checkpoints/` is the typical location for saved model weights.

## Build, Test, and Development Commands
- `bash tools/dist_train.sh /path_to_config 8` runs distributed training with 8 GPUs.
- `bash tools/dist_test.sh /path_to_config /path_to_checkpoint 8 --eval bbox` evaluates a checkpoint.
- `python tools/train.py /path_to_config` runs single-GPU training.
- `python tools/test.py /path_to_config /path_to_checkpoint --eval bbox` runs single-GPU evaluation.
- `bash tools/create_data.sh <partition> <job_name> <dataset>` prepares datasets via SLURM; see `tools/create_data.py` for options.

## Coding Style & Naming Conventions
- Use 4-space indentation and standard Python style (PEP 8). Keep line lengths reasonable for readability.
- Name modules and functions in `snake_case`, classes in `CamelCase`, and constants in `UPPER_SNAKE_CASE`.
- Config filenames should be descriptive, mirroring dataset/modality (e.g., `cmt_voxel0075_vov_1600x640_cbgs.py`).

## Testing Guidelines
- Testing is primarily via training/evaluation scripts in `tools/` rather than unit tests.
- Use config-specific test scripts or evaluation runs to validate changes (example: `python tools/test.py ...`).
- For speed checks, `python tools/test_speed.py /path_to_config` is available.

## Commit & Pull Request Guidelines
- Recent commit history favors short, imperative messages (e.g., "fix argo2 label mapping"). Follow that pattern.
- PRs should include a clear summary, the config(s) used, and any relevant metrics or screenshots for visual outputs.
- Link related issues and note dataset/version assumptions (e.g., nuScenes split or Argoverse2 variant).

## Configuration & Data Notes
- Follow `README.md` for environment versions and dataset preparation steps.
- Keep dataset paths under `data/` and avoid hard-coding absolute paths in configs.
