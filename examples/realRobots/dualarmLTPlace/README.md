# place_lt：左臂 V2，ACT 与 StarVLA QwenPI v3

训练机根目录：`/root/data/starVLA`。原始数据：`/root/data/datasets/place_lt`。
2026-11-19 检查数据包含 203 个 episode、46,102 帧、30 FPS。
任务文本直接读取 `meta/tasks.parquet`：

> Use the robot's left arm to slowly push in the charger box drawer.

## 映射合同

| 项目 | 顺序 |
|---|---|
| 相机 | `left_wrist`, `right_side` |
| 状态 24D | `arm_1 xyz + qx,qy,qz,qw` 7D，然后全部 17D DI |
| 动作 11D | `arm_1 xyz + qx,qy,qz,qw` 7D，然后 DO12、DO13、DO15、DO16 |
| 时间窗口 | H=50，训练索引 0..49；原始数据 30 FPS |
| 归一化 | 位姿 q99，DI/DO binary；绝对位姿 abs |

原始位姿取 `[7:14]`；原始 DO 向量顺序为
`[DO4,DO6,DO8,DO9,DO10,DO11,DO12,DO13,DO14,DO15,DO16,DO17]`。
目标 DO 的索引是 `[6,7,9,10]`，不能取连续 `[6:10]`。
StarVLA action keys 按位姿、`left_suction_do[6:8]`、`left_cylinder_do[9:11]` 拼接。

DI 顺序：`DI4,DI6,DI8,DI9,DI10,DI11,DI12,DI13,DI14,DI15,DI16,DI17,DI18,DI19,DI20,DI21,DI22`。
不输入力觉。数据中 DO14 和 DO17 也有高电平记录，按本次要求不作为预测动作；
真机适配时需明确这两个输出的处理方式，不能误发或隐式切换。
`robot_mapping_v2.py` 仅做打包、解包，解包保留其他臂和其他 DO，不是完整真机执行器。

## 准备及验证

```bash
cd /root/data/starVLA
HERE=examples/realRobots/dualarmLTPlace/train_files

# 创建轻量训练视图 /root/data/datasets/place_lt_v2，复用原始数据和视频。
# 原始 place_lt 不修改；独立 modality.json 写入视图。
.venv/bin/python "$HERE/prepare_place_lt.py"

bash "$HERE/train_act_place_v2_h50.sh" --check
bash "$HERE/train_qwenpi_v3_place_v2_h50.sh" --check

# CPU 也可执行：通过真正 StarVLA loader 解码一个双相机样本，核对归一化后维度。
bash "$HERE/train_act_place_v2_h50.sh" --sample
bash "$HERE/train_qwenpi_v3_place_v2_h50.sh" --sample
```

数据视图通过符号链接复用原始文件，原始数据目录必须保留。
两套训练共用 `/root/data/cache/starvla/place_lt_v2_224_mmap`，约需 12.9 GiB 的
224×224 RGB 视频缓存。完整缓存首次在实际训练加载时建立；样本检查只解码样本。

## 方式一：ACT V2

默认：H=50、每卡 batch 16、80,000 steps、每 10,000 steps 保存。
视觉骨干使用 ImageNet ResNet18 权重；骨干权重不在本地缓存时需要联网下载。
ACT 不以语言为条件，任务文本主要随数据保存。

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0 \
  examples/realRobots/dualarmLTPlace/train_files/run_act_place_v2_h50_nohup.sh \
  --num-processes 1 --batch-size 16
```

## 方式二：StarVLA QwenPI v3（沿用 V2 I/O）

默认：H=50、每卡 batch 1、20,000 steps、每 5,000 steps 保存，4 步推理采样。
沿用原 V2 的 Qwen3-VL-4B-Instruct-Action 和 VLAct 100k 预训练权重；
action model 随机初始化，VLM 全量微调。训练时使用数据中的任务语言。

```bash
cd /root/data/starVLA
CUDA_VISIBLE_DEVICES=0,1 \
  examples/realRobots/dualarmLTPlace/train_files/run_qwenpi_v3_place_v2_h50_nohup.sh \
  --num-processes 2 --batch-size 1
```

GPU 未挂载时两种后台入口会明确报 `GPU_NOT_READY`，不会改为 CPU 训练。
GPU 是否可用须检查 `.venv/bin/python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'`。

## 参数、日志与产物

两种训练入口均支持 `--check`、`--sample`、`--dry-run`、`--run`。
后台入口自动使用 `--run`。共同参数：

| 参数 | 含义 |
|---|---|
| `--steps N` | 优化器训练步数 |
| `--batch-size N` | 每个 GPU 进程的 batch size |
| `--num-processes N` | GPU 进程数，不得超过可见 GPU 数 |
| `--run-id NAME` | 新实验名称；输出改到 `/root/data/outputs/starVLA/NAME` |
| `DATA_ROOT_DIR` 环境变量 | 原始数据及训练视图的父目录 |

YAML：

```text
/root/data/starVLA/examples/realRobots/dualarmLTPlace/train_files/act_place_v2_h50.yaml
/root/data/starVLA/examples/realRobots/dualarmLTPlace/train_files/qwenpi_v3_place_v2_h50.yaml
```

日志：

```bash
tail -f /root/data/logs/starvla/place_lt_v2/act_latest.log
tail -f /root/data/logs/starvla/place_lt_v2/qwenpi_latest.log
```

默认模型输出：

```text
/root/data/outputs/starVLA/dualarm_lt_place_act_h50_v2/
/root/data/outputs/starVLA/dualarm_lt_place_qwenpi_v3_v2_h50/
```

后台启动后以日志确认是否真正进入训练。入口打印启动 PID 和对应进程组停止命令；
只停止该训练进程组，不使用 killall/python 全局停止。
启动时若默认输出已包含 checkpoint，会要求用新 `--run-id`，避免混写实验。

## 验证范围

`--check` 扫描全部 Parquet，检查帧数、向量维度、DI/DO 二值、左臂四元数、
非连续 DO 索引、解包保留其他臂/DO、相机文件和配置。
`--sample` 再经实际 StarVLA registry、统计归一化及视频解码生成
`state=(1,24)`、`action=(50,11)`、2 张图片的样本。
这些检查不等同于 GPU 训练通过；训练启动和损失收敛必须在加载 GPU 后验证。
