# Place 左臂训练：ACT V2 与 QwenPI v3 V2

完整训练流程、实际参数、原始任务标签差异和权重位置见 [CS 五模型训练说明](../dualarmLT/TRAINING_CS_LT.md)。

本目录提供数据准备、检查、归一化映射与两类训练入口。CS 已完成 Place ACT 80k 和 Place QwenPI 20k。

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

实际完成的 ACT 为单进程 BS16 / 80k；QwenPI 为两进程、每卡 BS8 / 20k。QwenPI 脚本默认每卡 BS1，复现时必须显式传入 BS8。

## 数据准备与样本检查

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLTPlace/train_files/train_act_place_v2_h50.sh --sample
examples/realRobots/dualarmLTPlace/train_files/train_qwenpi_v3_place_v2_h50.sh --sample
```

脚本创建轻量 `place_lt_v2` 视图并保留原始数据，样本预期为 `state=(1,24)`、`action=(50,11)`、两张图片。Place 的 DO14、DO17 也存在高电平记录，但不作为本任务预测输出。

## 新实验启动

下面两个训练各自独立启动，不同时占用同一批 GPU：

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0 examples/realRobots/dualarmLTPlace/train_files/run_act_place_v2_h50_nohup.sh --num-processes 1 --batch-size 16 --steps 80000 --run-id dualarm_lt_place_act_h50_v2_repro

CUDA_VISIBLE_DEVICES=0,1 examples/realRobots/dualarmLTPlace/train_files/run_qwenpi_v3_place_v2_h50_nohup.sh --num-processes 2 --batch-size 8 --steps 20000 --run-id dualarm_lt_place_qwenpi_v3_v2_h50_repro
```

通用参数有 `--steps`、`--batch-size`、`--num-processes`、`--run-id`，模式包括 `--check`、`--sample`、`--dry-run`、`--run`。当前 Place 包装脚本不支持 `--main-process-port`，使用 Accelerate 默认端口 29500。

任务语言直接读取 `meta/tasks.parquet`。训练时日志记录为 `Use the robot's left arm to slowly push in the charger box drawer.`；2026-11-23 当前数据标签为 `Lower the robot's left arm to place the charger into the charger box drawer.`，详见总说明中的版本差异。QwenPI 使用语言条件，ACT 本配方不使用语言条件。

```bash
tail -F /root/data/logs/starvla/place_lt_v2/act_latest.log
tail -F /root/data/logs/starvla/place_lt_v2/qwenpi_latest.log
```

实际输出位于 `/root/data/outputs/starVLA/<RUN_ID>/<RUN_ID>/`，包含配置、统计、训练步数记录和 `final_model/pytorch_model.pt`。脚本拒绝覆盖已有 checkpoint，新实验使用新 run_id。字段映射由 `robot_mapping_v2.py` 和 `train_files/data_registry/data_config.py` 定义。
