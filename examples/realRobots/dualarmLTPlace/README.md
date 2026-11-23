# Place 左臂训练：QwenPI v3 V2

完整训练流程、实际参数、原始任务标签差异和权重位置见 [CS 五动作训练说明](../dualarmLT/TRAINING_CS_LT.md)。

本目录提供数据准备、检查、归一化映射与QwenPI 训练入口。本文的 Place 动作使用已完成的 QwenPI v3 V2 / 20k。

| 项目 | 配置 |
|---|---|
| 原始数据 / 训练视图 | `place_lt` / `place_lt_v2` |
| 数据量 | 203 条，46,102 帧，30 FPS |
| 相机顺序 | `left_wrist`, `right_side` |
| State | 左臂 arm_1 pose 7 + 全部 DI 17 = 24 维 |
| Action | 左臂绝对目标 pose 7 + DO12、13、15、16 = 11 维 |
| Pose 顺序 | xyz + qx,qy,qz,qw，原始 `[7:14]` |
| DO 索引 | 原始 DO 向量 `[6,7,9,10]`，不是连续 `[6:10]` |
| 归一化 | pose q99，DI/DO binary，不输入 FT |

实际完成的 QwenPI 为两进程、每卡 BS8 / 20k。QwenPI 脚本默认每卡 BS1，复现时必须显式传入 BS8。

## 数据准备与样本检查

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLTPlace/train_files/train_qwenpi_v3_place_v2_h50.sh --sample
```

脚本创建轻量 `place_lt_v2` 视图并保留原始数据，样本预期为 `state=(1,24)`、`action=(50,11)`、两张图片。Place 的 DO14、DO17 也存在高电平记录，但不作为本任务预测输出。

## 新实验启动

下面使用两张空闲 GPU，另建新实验目录：

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0,1 examples/realRobots/dualarmLTPlace/train_files/run_qwenpi_v3_place_v2_h50_nohup.sh --num-processes 2 --batch-size 8 --steps 20000 --run-id dualarm_lt_place_qwenpi_v3_v2_h50_repro
```

通用参数有 `--steps`、`--batch-size`、`--num-processes`、`--run-id`，模式包括 `--check`、`--sample`、`--dry-run`、`--run`。当前 Place 包装脚本不支持 `--main-process-port`，使用 Accelerate 默认端口 29500。

任务语言直接读取 `meta/tasks.parquet`。训练时日志记录为 `Use the robot's left arm to slowly push in the charger box drawer.`；2026-11-23 当前数据标签为 `Lower the robot's left arm to place the charger into the charger box drawer.`，详见总说明中的版本差异。QwenPI 使用语言条件，复现时要注意这一标签版本差异。

```bash
tail -F /root/data/logs/starvla/place_lt_v2/qwenpi_latest.log
```

实际输出位于 `/root/data/outputs/starVLA/<RUN_ID>/<RUN_ID>/`，包含配置、统计、训练步数记录和 `final_model/pytorch_model.pt`。脚本拒绝覆盖已有 checkpoint，新实验使用新 run_id。字段映射由 `robot_mapping_v2.py` 和 `train_files/data_registry/data_config.py` 定义。
