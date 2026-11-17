# dualarmLT ACT 训练说明

数据集：[shanchai/suck_lt](https://huggingface.co/datasets/shanchai/suck_lt)

## 当前正式映射：V2 suck 单臂

完整数据统计显示，本任务中 `arm_0` 持续运动，`arm_1` 基本保持不动。
当前正式配置只建模运动臂和吸取所需数字 I/O，不使用力觉：

- 状态 24D：`arm_0_pose` 7D + 全部 `digital_input` 17D；
- 动作 11D：`arm_0_pose` 7D + `DO8..DO11` 4D；
- 未建模的 `arm_1`、其他数字输出和外部轴在真机控制端保持当前安全值；
- H=50，ACT 使用 batch size 16、80000 steps；
- QwenPI v3 使用每卡 batch size 1、20000 steps。

四路 H.264 相机按以下顺序输入模型：

1. `left_side`
2. `left_wrist`
3. `right_wrist`
4. `right_side`

数据集中的 `observation.images.base` 文件保持不变，仅不送入模型。这样可减少
约 20% 的视频解码、mmap 缓存容量和视觉骨干计算量；旧的 `base` mmap 文件可保留，
新训练不会打开它。四路 mmap 缓存可供 ACT V2 和 QwenPI v3 V2 共同复用。

## 数据准备与检查

训练数据默认路径：

```text
/root/data/datasets/suck_lt
```

运行 ACT V2 配置检查：

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLT/train_files/train_act_v2_h50.sh --check
```

检查通过时会输出：

```text
PREFLIGHT_OK recipe=v2_suck episodes=196 frames=82827 H=50 state=24 action=11 cameras=4 force=false stationary_arm=false base_ignored=true
```

## 启动训练

### V2：吸取任务专用单臂映射

V2 专门用于当前吸取任务，只建模实际运动的 `arm_0` 并去掉力觉输入：

- 状态 24D：`arm_0_pose` 7D + 全部 `digital_input` 17D；
- 动作 11D：`arm_0_pose` 7D + 实际变化的 `DO8..DO11` 4D；
- 不输入 `arm_1_pose`、`force_torque_compensated` 或原始力觉；
- 未建模的 `arm_1`、其他数字输出和外部轴在真机控制端保持当前安全值；
- 使用 `left_side`、`left_wrist`、`right_wrist`、`right_side` 四路相机，忽略 `base`；
- H=50、batch size 16、80000 steps，并复用四路 mmap 缓存。

检查 V2 配置：

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLT/train_files/train_act_v2_h50.sh --check
```

后台启动 V2 单卡训练：

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0 \
  examples/realRobots/dualarmLT/train_files/run_act_v2_h50_nohup.sh \
  --num-processes 1 --batch-size 16
```

V2 日志路径：

```bash
tail -f /root/data/outputs/starVLA/dualarm_lt_suck_act_h50_v2/logs/act_v2_h50_latest.log
```

### V2 的 StarVLA QwenPI v3 配置

`qwenpi_v3_v2_h50.yaml` 使用与 ACT V2 完全相同的 24D 状态、11D 动作、
四路相机和 H=50。VLM 从 `Qwen3-VL-4B-Instruct-Action` 加载，并使用
`VLAct_Qwen3_Pretrain/checkpoints/steps_100000_pytorch_model.pt` 做迁移；由于
V2 的状态/动作维度不同，action model 重新初始化，VLM 及其接口保持全量微调。

默认参数为每卡 batch 1、20000 steps、5000 steps 保存一次。检查配置：

```bash
cd /root/data/starVLA
examples/realRobots/dualarmLT/train_files/train_qwenpi_v3_v2_h50.sh --check
```

两卡启动（推荐 A800 80GB）：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
  examples/realRobots/dualarmLT/train_files/run_qwenpi_v3_v2_h50_nohup.sh \
  --num-processes 2 --batch-size 1
```

查看日志：

```bash
tail -f /root/data/outputs/starVLA/dualarm_lt_suck_qwenpi_v3_v2_h50/logs/qwenpi_v3_v2_h50_latest.log
```

## 主要文件

- `robot_mapping_v2.py`：V2 真机原始字段与策略向量之间的打包、解包映射。
- `train_files/modality.json`：LeRobot 字段切片及相机映射。
- `train_files/data_registry/data_config.py`：StarVLA 数据注册与归一化配置。
- `train_files/act_v2_h50.yaml`：ACT V2 训练超参数。
- `train_files/qwenpi_v3_v2_h50.yaml`：QwenPI v3 V2 训练超参数。
- `train_files/train_*_v2_h50.sh`：配置检查、dry-run 和前台训练入口。
- `train_files/run_*_v2_h50_nohup.sh`：后台训练及日志管理入口。
