# Push/Pull QwenPI v3 V2 离线评估

使用原训练数据比较 H50 检查点。输入为左腕和左侧图像及 State24，输出为左臂位姿和 DO14、DO17。三个步骤分别准备固定样本、执行 GPU 推理和在 CPU 上汇总误差。评估使用本地数据与权重，无需连接机器人。训练入口见[CS 训练说明](../TRAINING_CS_LT.md)。

## 1. 环境与目录

使用训练项目现有环境，不在系统 Python 中安装包：

```bash
cd /root/data/starVLA
export LT_ROOT=/root/data/starVLA
export LT_PY="$LT_ROOT/.venv/bin/python"
export LT_TOOLS="$LT_ROOT/examples/realRobots/dualarmLT/offline_eval"
export TMPDIR=/root/data/tmp
export XDG_CACHE_HOME=/root/data/.cache
export HF_HOME=/root/data/huggingface
export MPLCONFIGDIR=/root/data/.cache/matplotlib
```

项目环境已包含 PyTorch、Transformers、OmegaConf、NumPy、Pandas、PyArrow、PyAV、OpenCV。`prepare` / `analyze`不初始化策略模型；只有 `infer` 使用 CUDA。

现有 40k 运行目录如下，内含 20k seed 和 25k、30k、35k、40k 检查点：

```bash
export LT_RUN=/root/data/outputs/starVLA/dualarm_lt_push_pull_qwenpi_v3_v2_h50_resume40k_20261124_195355/dualarm_lt_push_pull_qwenpi_v3_v2_h50_resume40k_20261124_195355
export LT_EVAL=/root/data/outputs/starVLA/evaluations/push_pull_compare_repro_01
```

`--run-dir` 指向包含 `config.full.yaml`、`dataset_statistics.json`、`checkpoints/` 的内层运行目录。下载包如果只有 `final_model`，需要额外取得要比较的中间检查点；不能仅凭 40k 最终模型重建 20k/25k/30k/35k。

## 2. 固定观测样本：CPU

```bash
CUDA_VISIBLE_DEVICES= "$LT_PY" "$LT_TOOLS/evaluate_push_pull.py" prepare \
  --repo-root "$LT_ROOT" --run-dir "$LT_RUN" \
  --dataset-path /root/data/datasets/push_pull_lt_v2 \
  --output-dir "$LT_EVAL/corpus" \
  --per-task 16 --video-source cache
```

默认从训练配置的 `video_cache_dir` 读取已有 224×224 RGB `.npy` 缓存；换位置可加 `--cache-dir /你的缓存目录`。没有缓存时改用 `--video-source video`：按训练同样的 PyAV RGB 解码和 OpenCV 224×224 缩放，逐 episode 读取原始 MP4，不在磁盘生成一份 36GiB 的新缓存。

准备步骤选择 Pull、Push 各 16 条轨迹，每条早、中、晚各一个观测，存在 IO 切换时再补一个附近观测；本批实际为 32 条轨迹、128 个观测。每个目标包含 50 个有效动作，避开末端补齐。数据映射、相机顺序、二值 IO、归一化和训练集规模均会核对。

输出 `protocol.json`、`samples.npz`、`dataset_statistics.json`；已存在的 corpus 不覆盖。比较多个检查点时一直复用这份 corpus。该命令按**原训练数据**复现，未实现独立验证集划分或新数据集统计替换。

## 3. 多检查点推理：GPU

先只检查路径与参数，不初始化模型：

```bash
"$LT_PY" "$LT_TOOLS/evaluate_push_pull.py" infer \
  --repo-root "$LT_ROOT" --run-dir "$LT_RUN" \
  --corpus-dir "$LT_EVAL/corpus" --output-dir "$LT_EVAL/predictions" \
  --steps 20000 25000 30000 35000 40000 --repeats 2 --dry-run
```

正式离线推理：

```bash
CUDA_VISIBLE_DEVICES=0 "$LT_PY" "$LT_TOOLS/evaluate_push_pull.py" infer \
  --repo-root "$LT_ROOT" --run-dir "$LT_RUN" \
  --corpus-dir "$LT_EVAL/corpus" --output-dir "$LT_EVAL/predictions" \
  --steps 20000 25000 30000 35000 40000 --repeats 2
```

模型用 meta 初始化、严格加载全部 state_dict、重建非持久 RoPE 缓冲，再以原始参数 dtype 放入 GPU。CPU 权重使用 `mmap=True`，避免在 7GiB 系统内存容器中额外分配整份 10.9GB 权重；图像快照只解压一次。每个检查点依次占用同一张 GPU，不同时放五份模型，不量化。

已验证 A16 15,356MiB 环境可以执行本配置，峰值 PyTorch 分配约 10.43GiB。网络盘冷加载可能明显慢于模型推理；当前硬件快照不代表任何环境都具有相同吞吐或显存余量。

| 参数 | 含义 |
|---|---|
| `--steps` | 要比较的检查点步数；不是去噪步数 |
| `--repeats` | 每个观测的噪声组数量，默认 2；相同观测和噪声组在不同检查点共享 seed |
| `--device cuda:0` | 当前进程可见 GPU 中的索引；可配合 `CUDA_VISIBLE_DEVICES` |
| `--base-vlm` | 覆盖本机 Qwen3-VL-4B-Instruct-Action 的配置、tokenizer 路径 |
| `--skip-existing` | 复用配置、检查点、corpus、重复次数完全匹配的已完成结果 |
| `--limit 2` | 只跑两个观测的接口调试；不能拿此不完整输出生成正式比较 |

只跑一个检查点时，例如 `--steps 40000`。检查点目录仍须保留对应 `steps_40000_pytorch_model.pt`。网络访问默认关闭；不会启动模型 HTTP 服务。使用标准 `predict_action`、4 次去噪，没有 RTC 历史块引导，不能据此报告 RTC 闭环成功率。

每个完成结果为 `predictions_<step>.npz` 和同名 JSON 元数据；未完成文件使用 `partial_` 前缀，最终 NPZ 原子替换。已有最终结果默认拒绝覆盖，重新计算请用新目录。

## 4. 汇总误差：CPU

```bash
CUDA_VISIBLE_DEVICES= "$LT_PY" "$LT_TOOLS/analyze_push_pull.py" \
  --corpus-dir "$LT_EVAL/corpus" \
  --prediction-dir "$LT_EVAL/predictions" \
  --output-dir "$LT_EVAL/metrics"
```

输出：`summary.csv`、`metrics.json`、`per_episode.json`、`horizon_errors.json`。位置为三维欧氏距离 mm，旋转为四元数最短夹角 °，处理 `q/-q` 等价；DO 按 `>0.5` 二值化，计算准确率、平衡准确率、F1、TP/TN/FP/FN。分别给出第 1 步、前 10 步和 H50、Pull / Push、IO 切换附近结果。

位姿误差按轨迹等权平均。相对 20k 和 40k 相对 35k 的差值，使用按任务分层、按 episode 聚类的 5,000 次 bootstrap。少于每任务两条独立轨迹时不计算配对区间。区间是本批轨迹的探索性比较，不能替代独立验证集。

### 直接重算已经完成的 2026-11-25 结果

保留在 CS 的原预测文件可以直接重算，**不需要 GPU，不再加载权重**：

```bash
export LT_RECORDED=/root/data/outputs/starVLA/evaluations/push_pull_qwen_20k_40k_20261125
CUDA_VISIBLE_DEVICES= "$LT_PY" "$LT_TOOLS/analyze_push_pull.py" \
  --corpus-dir "$LT_RECORDED" --prediction-dir "$LT_RECORDED" \
  --output-dir /root/data/tmp/push_pull_metrics_replay
```

已对原始结果执行此重算，五个检查点的数值和配对区间完全一致。
