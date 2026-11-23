# Snap / Insert 左臂 ACT V2 H50

完整训练说明见 [CS 五模型训练说明](../dualarmLT/TRAINING_CS_LT.md)。最终发布模型是 **从头训练完成的 BS16 / 80k 新版**，不是早期 BS2 / 20k 或 50k 续训目录。

| 任务 | 原始目录 / 训练视图 | 示教条数 | 帧数 | 相机顺序 |
|---|---|---:|---:|---|
| Snap | `snap_lt` / `snap_lt_v2` | 111 | 15,188 | `left_wrist`, `right_side` |
| Insert | `insert_lt` / `insert_lt_v2` | 106 | 14,254 | `left_wrist`, `right_side` |

共同映射：左臂 arm_1；pose 为 xyz + xyzw，原始切片 `[7:14]`；State 24 = pose 7 + DI 17；Action 11 = 绝对目标 pose 7 + DO12、DO13、DO15、DO16，DO 原始索引为 `[6,7,9,10]`。H50，224×224 图像，原始 30 FPS，不输入 FT。

**两份数据中 DO12、DO13、DO15、DO16 全程为 1，没有这些输出的开关切换示教。** ACT 本配方不使用任务语言，Snap 与 Insert 分别训练各自模型。

## 下载、准备与检查

数据源为有权限访问的 ModelScope 数据集 `chaishan/huashu`，子目录为：

```text
newsmy_usb_h264_640x400_30fps_20261120_4cam/Snap-c
newsmy_usb_h264_640x400_30fps_20261120_4cam/Insert-c
```

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLTSnapInsert/train_files/download_snap_insert.sh
examples/realRobots/dualarmLTSnapInsert/train_files/train_act_snap_v2_h50.sh --sample
examples/realRobots/dualarmLTSnapInsert/train_files/train_act_insert_v2_h50.sh --sample
```

下载器逐文件验证 SHA-256，完整文件跳过；凭据由 SDK 登录管理，不写入脚本。准备程序创建带独立 modality 的轻量训练视图，复用原始视频。样本均应为 `state=(1,24)`、`action=(50,11)`、两张图片。

## 从头训练 BS16 / 80k

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=1 examples/realRobots/dualarmLTSnapInsert/train_files/run_act_snap_fresh80k_nohup.sh
CUDA_VISIBLE_DEVICES=0 examples/realRobots/dualarmLTSnapInsert/train_files/run_act_insert_fresh80k_nohup.sh
```

各用一张空闲 GPU。ACT 使用 ImageNet ResNet18，学习率 1e-4、Warmup 5,000、每 10,000 步保存。`fresh80k` 自动创建带时间和 PID 的新 run_id，只支持可选 `--dry-run`，不会恢复早期权重。

需要修改参数时，改用通用入口：

```bash
CUDA_VISIBLE_DEVICES=1 examples/realRobots/dualarmLTSnapInsert/train_files/run_act_snap_v2_h50_nohup.sh --num-processes 1 --batch-size 16 --steps 80000 --main-process-port 29603 --run-id dualarm_lt_snap_act_h50_v2_r2
CUDA_VISIBLE_DEVICES=0 examples/realRobots/dualarmLTSnapInsert/train_files/run_act_insert_v2_h50_nohup.sh --num-processes 1 --batch-size 16 --steps 80000 --main-process-port 29604 --run-id dualarm_lt_insert_act_h50_v2_r2
```

默认端口 Snap 29603、Insert 29604。GPU 不同也需要分别使用通信端口。配置在 `train_files/act_snap_v2_h50.yaml`、`act_insert_v2_h50.yaml`。

```bash
tail -F /root/data/logs/starvla/snap_lt_v2/act_latest.log
tail -F /root/data/logs/starvla/insert_lt_v2/act_latest.log
```

输出为 `/root/data/outputs/starVLA/<RUN_ID>/<RUN_ID>/final_model/pytorch_model.pt`，训练配置和数据统计在同一运行目录。

## 早期续训脚本

`run_act_*_resume50k_nohup.sh` 与 `act_*_resume50k.yaml` 是早期 BS2 实验的工具，目标是总 50k，不能用作 BS16 / 80k 从头训练入口。checkpoint 只含模型权重，续训重新初始化优化器，并重算、推进学习率计划。详情和最终 RUN_ID 见总说明。
