# UniDrive Debug Log

## 背景
- 现象：使用 CARLA→nuScenes 数据生成与可视化时，曾发现部分车辆/行人方向与标注方向相反。
- 目标：定位方向反转原因并验证修复有效。

## 本次生成指令
```bash
PORTS=2000,2002 WORKERS=2 UNIDRIVE_ROT_MOD=2pi UNIDRIVE_ORI_DEBUG=1 \
HYPERPARAMS=../hyperparams/det_nusc.toml LIDAR_PARAMS=../hyperparams/det_nusc.toml \
bash ./routes_baselines.sh
```

## 推测原因
- `rotation_y` 在生成阶段使用 `% pi`，会将角度折叠到 $[0, \pi)$，导致 $180^\circ$ 方向信息丢失，从而出现正反混淆。

## 修改内容
### 1) 生成阶段支持可切换角度取模策略
- 环境变量 `UNIDRIVE_ROT_MOD`：
  - `2pi`（默认，保留完整方向）
  - `pi`（兼容旧行为）
  - `none`（不取模）
- 修改位置：
  - thirdparty/UniDrive/create/carla_nus/scenario_runner/scenario_runner.py
  - thirdparty/UniDrive/create/carla_nus/scenario_runner/scenario_runner_sweeps.py

### 2) 方向链路调试日志
- `UNIDRIVE_ORI_DEBUG=1` 时写出关键方向值（`car_yaw/agent_yaw/rotation_y` 等）。
- `UNIDRIVE_ORI_DEBUG_FILE` 可指定输出路径，默认 `orientation_debug.log`。
- 修改位置：
  - thirdparty/UniDrive/create/carla_nus/scenario_runner/scenario_runner.py
  - thirdparty/UniDrive/create/carla_nus/scenario_runner/scenario_runner_sweeps.py
  - thirdparty/UniDrive/create/nus_tool/position_3d.py

### 3) 角度归一化防断言
- 在写入 KITTI 标签前对 `alpha` 与 `rotation_y` 归一化到 $[-\pi, \pi]$，避免越界断言。
- 修改位置：
  - thirdparty/UniDrive/create/carla_nus/scenario_runner/scenario_runner.py
  - thirdparty/UniDrive/create/carla_nus/scenario_runner/scenario_runner_sweeps.py

## 结果
- 使用 `UNIDRIVE_ROT_MOD=2pi` 重新生成后，抽样可视化未发现方向错误样本。

## 相关日志
- 进度日志：
  - thirdparty/UniDrive/create/carla_nus/logs/det_nusc.progress
- 运行输出样例：
  - thirdparty/UniDrive/create/carla_nus/logs/det_nusc_route_0.out
  - thirdparty/UniDrive/create/carla_nus/logs/det_nusc_route_1.out

## 备注
- 如需进一步定位单个实例的方向链路，请开启 `UNIDRIVE_ORI_DEBUG=1` 并检查 `orientation_debug.log`。
