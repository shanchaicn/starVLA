# Push/Pull 左臂 QwenPI v3 V2 H50

完整说明见 [CS 五动作训练说明](../dualarmLT/TRAINING_CS_LT.md)。CS 上最终已完成 **两进程、每卡 BS8、20,000 步** 的模型；脚本默认 BS1 是通用起始设置，早期 BS2 命令也不是最终模型参数。

## 数据与模型映射

- 原始数据 `push_pull_lt`，训练视图 `push_pull_lt_v2`：377 条示教，128,271 帧，30 FPS。
- 相机顺序：`left_wrist`、`left_side`，输入图像 224×224。
- 左臂 `arm_1` pose 为原始 `[7:14]`，xyz + xyzw；State 24 = pose 7 + 全部 DI 17。
- Action 9 = 左臂绝对目标 pose 7 + DO14、DO17；原始 DO 索引 `[8,11]`。
- DO14 控制左臂中间吸嘴，DO17 控制左臂中间升降气缸；对应真空反馈是 DI22。
- DO14 有 49,337 帧为 1、78,934 帧为 0；DO17 有 103,614 帧为 1、24,657 帧为 0。
- 同一个模型保留两条语言任务，不输入 FT：

```text
Use the robot's left arm to slowly pull out the charger box drawer.
Use the robot's left arm to slowly push in the charger box drawer.
```

## 数据检查

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLTPushPull/train_files/train_qwenpi_v3_push_pull_v2_h50.sh --sample
```

脚本创建轻量视图并检查 Parquet、二值 IO、四元数、基座和预训练路径。实际 StarVLA 样本应为 `state=(1,24)`、`action=(50,9)`、两张图片。原始数据保持不变。

## 复现训练配方

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0,1 examples/realRobots/dualarmLTPushPull/train_files/run_qwenpi_v3_push_pull_v2_h50_nohup.sh --num-processes 2 --batch-size 8 --steps 20000 --main-process-port 29605 --run-id dualarm_lt_push_pull_qwenpi_v3_v2_h50_repro
```

Qwen3-VL-4B-Instruct-Action 基座，加载 VLAct Qwen3 100k 预训练权重，随机初始化新的 9 维动作头并全参数微调。VLM 学习率 1e-5，动作头、投影层 1e-4，Warmup 500，梯度累积 1，启用 gradient checkpointing，每 5,000 步保存；默认推理采样 4 步。

配置：`train_files/qwenpi_v3_push_pull_v2_h50.yaml`。脚本支持 `--check`、`--sample`、`--dry-run`、`--run`，以及 `--steps`、`--batch-size`、`--num-processes`、`--run-id`、`--main-process-port`。

两张可见 GPU 对应 `--num-processes 2`，不能设为 4。BS8 为已完成大显存训练环境的参数，启动前检查当前空闲显存。全量视频缓存约 36 GiB，配置目录为 `/root/data/cache/starvla/push_pull_lt_v2_left_wrist_left_side_224_mmap`；需要按需解码时将 `load_all_data_for_training` 改为 false。

```bash
tail -F /root/data/logs/starvla/push_pull_lt_v2/qwenpi_latest.log
```

最终模型：`/root/data/outputs/starVLA/dualarm_lt_push_pull_qwenpi_v3_v2_h50/dualarm_lt_push_pull_qwenpi_v3_v2_h50/final_model/pytorch_model.pt`。新实验用新 run_id，保存的 `config.full.yaml` 是实际超参数依据。总说明附有完成实验的配置快照、启动命令和日志记录。
