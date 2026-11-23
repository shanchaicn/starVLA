# StarVLA LT 2.16 Docker 部署与测试说明

## 1. 文档范围

本文用于汇报和复现 StarVLA LT 在 `192.168.2.16`（ARM64 NVIDIA Thor）上的 Docker 部署，内容包括 Suck、Place、Snap、Insert、Push/Pull 五个任务模型的服务端启动与合成推理、Suck Docker client，以及 `fake-dry-run`、`dry-run`、真机测试的步骤和验收标准。

本文不包含宿主机 Python 虚拟环境启动方式，不将模型权重、运行日志或现场密码纳入代码仓库。

## 2. 系统组成

```text
/home/test/Public/cs/starvla_lt/
├── docker/
│   ├── compose.yaml
│   ├── Dockerfile.server
│   ├── Dockerfile.client
│   ├── compose.sh
│   ├── run_server.sh
│   ├── run_client.sh
│   ├── test_fake_dry_run.sh
│   └── test_running_model.sh
├── models/                       # 模型权重，只读挂载到 /app/models
├── server/server_16.yaml         # 服务端 Profile 和模型路径
├── client/dualarm_lt_rtc_16.yaml # RTC、执行频率和安全参数
└── logs/                         # client 运行日志
```

Docker 镜像：

```text
starvla-lt-server:thor-20261120
starvla-lt-client:arm64-20261120
```

镜像面向 ARM64 NVIDIA Thor，不能直接用于 x86_64 主机。模型权重通过只读卷挂载，不打入镜像，也不提交到 GitHub。

### 2.1 五个任务主模型

| 任务 | Profile | Compose 服务 | 端口 | State / Action | RTC | 相机 | DO 输出 |
|---|---|---|---:|---|---|---|---|
| Suck | `qwenpi_v3_v2` | `qwen-v2` | 10199 | 24 / `50×11` | guided | left_side、right_wrist、right_side | DO8、9、10、11 |
| Place | `place_qwenpi_v3_v2` | `place-qwen-v2` | 10202 | 24 / `50×11` | guided | left_wrist、right_side | DO12、13、15、16 |
| Snap | `snap_act_v2` | `snap-act` | 10203 | 24 / `50×11` | fresh-replan | left_wrist、right_side | DO12、13、15、16 |
| Insert | `insert_act_v2` | `insert-act` | 10204 | 24 / `50×11` | fresh-replan | left_wrist、right_side | DO12、13、15、16 |
| Push/Pull | `push_pull_qwenpi_v3_v2` | `push-pull-qwen-v2` | 10205 | 24 / `50×9` | guided | left_wrist、left_side | DO14、17 |

五个任务按任务数统计；Push 和 Pull 共用一个语言条件 QwenPI 模型。Suck 另外保留两个对照版本，不额外计入五个任务：

| Suck 对照版本 | Compose 服务 | 端口 | State / Action | RTC | 相机 |
|---|---|---:|---|---|---|
| `qwenpi_v3_v3` | `qwen-v3` | 10200 | 7 / `50×11` | guided | left_side、right_wrist、right_side |
| `act_v2` | `act` | 10201 | 24 / `50×11` | fresh-replan | left_side、left_wrist、right_wrist、right_side |

服务端 Profile 中记录的任务指令如下。ACT 的 Snap/Insert 不使用语言作为模型条件，但仍用任务名标识 Profile：

| 任务 | 推理指令 |
|---|---|
| Suck | `Use the suction cup on the robot's right arm to pick up the phone support plate.` |
| Place | `Use the robot's left arm to slowly push in the charger box drawer.` |
| Snap | `Snap` |
| Insert | `Insert` |
| Push | `Use the robot's left arm to slowly push in the charger box drawer.` |
| Pull | `Use the robot's left arm to slowly pull out the charger box drawer.` |

Place 的名称和现有训练标签语义并不完全一致，部署时以该模型训练保存的任务文本为准，不应擅自替换为新的自然语言描述。

模型服务只负责预测 Action，不直接连接机器人运动 RPC。健康接口中的以下字段是正常设计：

```json
{
  "motion_authorized": false,
  "actions_executed": 0
}
```

只有 client 的 `--real` 路径可以发送机器人 RPC。

## 3. 部署准备

### 3.1 主机要求

- ARM64 NVIDIA Thor；
- Docker Engine、Docker Compose plugin 和 NVIDIA Container Runtime；
- 模型 checkpoint 已按 `server/server_16.yaml` 指定路径放入 `models/`；
- 真实数据测试需要 Topic `19091` 和相机 HTTP `6888`；
- 真机测试还需要机器人 RPC `5868`。

### 3.2 五个动作模型目录

五个动作模型均属于本次 Docker 部署范围，应分别放在以下目录：

```text
models/dualarm_lt_suck_qwenpi_v3_v2_h50/
models/dualarm_lt_place_qwenpi_v3_v2_h50/
models/dualarm_lt_snap_act_h50_v2_bs16_80k_20261122_065928_14701/
models/dualarm_lt_insert_act_h50_v2_bs16_80k_20261122_065931_14779/
models/dualarm_lt_push_pull_qwenpi_v3_v2_h50/
```

每个目录至少应包含与 `server/server_16.yaml` 对应的：

```text
config.full.yaml
dataset_statistics.json
final_model/pytorch_model.pt
```

模型权重不打入 Docker 镜像。Compose 将宿主机整个 `models/` 目录只读挂载为容器内 `/app/models`，因此复制或更新权重后无需重新构建镜像，但需要重启对应服务进程。

### 3.3 基础检查

```bash
ssh test@192.168.2.16
cd /home/test/Public/cs/starvla_lt/docker

./compose.sh version
./compose.sh ps -a
sudo docker image ls | grep starvla-lt
find ../models -path '*/final_model/pytorch_model.pt' -printf '%p %s\n' | sort
```

上述 `find` 至少应列出五个任务主模型的最终权重。模型目录名、checkpoint 大小、State/Action 维度必须与 `server/server_16.yaml` 一致；任一模型缺失都表示五动作部署不完整。

`test` 用户如无权直接访问 `/var/run/docker.sock`，`compose.sh` 会在交互终端调用 `sudo`。密码只在终端输入，不写入脚本、YAML 或代码仓库。

## 4. 启动模型服务

进入 Docker 部署目录。五个任务主模型的启动命令如下；每条命令会启动对应容器并等待健康检查通过：

```bash
cd /home/test/Public/cs/starvla_lt/docker

# Suck
./run_server.sh qwenpi_v3_v2

# Place
./run_server.sh place_qwenpi_v3_v2

# Snap
./run_server.sh snap_act_v2

# Insert
./run_server.sh insert_act_v2

# Push/Pull
./run_server.sh push_pull_qwenpi_v3_v2
```

Suck 的两个对照版本按需启动：

```bash
./run_server.sh qwenpi_v3_v3
./run_server.sh act_v2
```

`run_server.sh` 会等待 `/health` 成功后打印 `MODEL_READY`。容器显示 `Started` 只代表进程已启动，不代表模型已加载完成。

检查健康接口：

```bash
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10199/health
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10202/health
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10203/health
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10204/health
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10205/health

# Suck 对照版本
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10200/health
curl --noproxy '*' --max-time 5 -fsS http://127.0.0.1:10201/health
```

查看容器和日志：

```bash
./compose.sh ps
./compose.sh logs --tail 100 qwen-v2
./compose.sh logs --tail 100 place-qwen-v2
./compose.sh logs --tail 100 snap-act
./compose.sh logs --tail 100 insert-act
./compose.sh logs --tail 100 push-pull-qwen-v2

# Suck 对照版本
./compose.sh logs --tail 100 qwen-v3
./compose.sh logs --tail 100 act
```

Qwen 模型占用显存/统一内存较多。默认按“启动一个、验证一个、停止后再换下一个”的顺序测试，不应假定五个模型可以同时常驻。ACT 是否与一个 Qwen 同时保留由实际可用资源决定。

### 4.1 五动作部署验收

五动作部署完成的判据不是“五个容器同时运行”，而是五个 Profile 均能独立完成以下检查：

1. `run_server.sh <profile>` 输出对应的 `MODEL_READY`；
2. 对应端口 `/health` 返回正确的模型身份、State/Action 维度和 RTC 模式；
3. `test_running_model.sh <profile>` 返回两次有限 Action；
4. 输出始终为 `motion_authorized=false`、`actions_executed=0`；
5. 停止当前服务后能够切换到下一个动作模型。

建议按 Suck → Place → Snap → Insert → Push/Pull 的顺序逐个验收。这样既覆盖全部五个动作模型，也避免多个 Qwen 同时加载造成统一内存不足。

### 4.2 五动作逐个启动与推理

下面是一套可直接执行的顺序验收命令。每个模型启动后做两次合成输入推理，再停止该服务并切换到下一个模型：

```bash
cd /home/test/Public/cs/starvla_lt/docker

# 1. Suck：50×11，guided
./run_server.sh qwenpi_v3_v2
./test_running_model.sh qwenpi_v3_v2
./compose.sh stop qwen-v2

# 2. Place：50×11，guided
./run_server.sh place_qwenpi_v3_v2
./test_running_model.sh place_qwenpi_v3_v2
./compose.sh stop place-qwen-v2

# 3. Snap：50×11，fresh-replan
./run_server.sh snap_act_v2
./test_running_model.sh snap_act_v2
./compose.sh stop snap-act

# 4. Insert：50×11，fresh-replan
./run_server.sh insert_act_v2
./test_running_model.sh insert_act_v2
./compose.sh stop insert-act

# 5. Push/Pull：50×9，guided；默认测试 Pull 指令
./run_server.sh push_pull_qwenpi_v3_v2
./test_running_model.sh push_pull_qwenpi_v3_v2

# 同一模型改测 Push 指令
./test_running_model.sh push_pull_qwenpi_v3_v2 \
  --instruction "Use the robot's left arm to slowly push in the charger box drawer."
./compose.sh stop push-pull-qwen-v2
```

这组命令只做模型 HTTP 推理，不读取机器人 Topic/相机，也不打开机器人 RPC。若希望脚本自动创建并清理临时容器，使用第 6 节的 `test_fake_dry_run.sh`。

## 5. 测试分级与安全边界

| 测试级别 | 状态与图像来源 | Topic 19091 | 相机 6888 | 模型 HTTP | 机器人 RPC 5868 | 机器人动作 |
|---|---|---:|---:|---:|---:|---:|
| `fake-dry-run` | 合成状态、合成图像 | 否 | 否 | 是 | 否 | 否 |
| `dry-run` | 真实 RT/NRT 状态、真实相机 | 是 | 是 | 是 | 否 | 否 |
| 真机 `--real` | 真实 RT/NRT 状态、真实相机 | 是 | 是 | 是 | 是 | 是 |

测试顺序必须由低到高：先完成 `fake-dry-run`，再完成 `dry-run`，最后进入真机小步数测试。`fake-dry-run` 和 `dry-run` 均不会发送运动指令，但二者验证的链路不同，不能互相替代。

## 6. fake-dry-run：纯模型与容器测试

### 6.1 测试目的

`fake-dry-run` 使用合成状态和 `224×224` 合成图片，不订阅 Topic、不读取相机、不打开机器人 RPC。它验证：

- Docker 镜像、GPU 和依赖可用；
- checkpoint 与 Profile 配置能够加载；
- HTTP 请求和响应格式正确；
- Qwen 的 initial/guided 或 ACT 的 fresh-replan 推理可执行；
- 返回 Action 为有限数值；Suck、Place、Snap、Insert 为 `50×11`，Push/Pull 为 `50×9`。

### 6.2 操作命令

脚本可自行启动对应临时服务并执行两次推理：

```bash
cd /home/test/Public/cs/starvla_lt/docker

./test_fake_dry_run.sh qwenpi_v3_v2
./test_fake_dry_run.sh place_qwenpi_v3_v2
./test_fake_dry_run.sh snap_act_v2
./test_fake_dry_run.sh insert_act_v2
./test_fake_dry_run.sh push_pull_qwenpi_v3_v2
```

上述 Push/Pull 临时测试使用 Profile 中默认的 Pull 指令。Suck 对照版本可另外执行：

```bash
./test_fake_dry_run.sh qwenpi_v3_v3
./test_fake_dry_run.sh act_v2
```

如果服务已经在固定端口运行，可直接测试：

```bash
./test_running_model.sh qwenpi_v3_v2
./test_running_model.sh place_qwenpi_v3_v2
./test_running_model.sh snap_act_v2
./test_running_model.sh insert_act_v2
./test_running_model.sh push_pull_qwenpi_v3_v2
```

Push/Pull 的默认指令是 Pull。测试同一个模型的 Push 语言条件时，先启动服务，再覆盖测试指令：

```bash
./run_server.sh push_pull_qwenpi_v3_v2
./test_running_model.sh push_pull_qwenpi_v3_v2 \
  --instruction "Use the robot's left arm to slowly push in the charger box drawer."
```

Docker client 本身的无网络自检：

```bash
sudo docker compose -f compose.yaml run --rm -T --no-deps client \
  --profile qwenpi_v3_v2 --self-test
```

### 6.3 通过判据

- Suck、Place、Snap、Insert 两次推理均返回 `50×11` Action，Push/Pull 返回 `50×9`；
- Action 不包含 `NaN` 或 `Inf`；
- 输出包含 `PASS: two inference responses; no robot commands`；
- client 自检输出 `DUALARM_LT_RTC_EXECUTOR_SELF_TEST_PASS; NO NETWORK/ROBOT`；
- 测试过程没有连接 Topic、相机和机器人 RPC，没有产生机器人动作。

### 6.4 QwenPI V3 V2 连续测试记录

#### Thor Docker

连续测试命令：

```bash
cd /home/test/Public/cs/starvla_lt/docker
./test_fake_dry_run.sh qwenpi_v3_v2 --continuous --interval-s 1
```

2026-11-20 的测试输出显示，循环持续返回 `action_shape=[50,11]`，每轮依次完成 initial 和 guided replacement，且 `actions_executed=0`。日志片段中第 35–45 轮均出现 PASS；其中第 36、37、38、39、41、43 轮包含完整的两次延迟记录，可用于统计：

| 阶段 | 完整样本数 | 平均值 | 中位数 | 最小值 | 最大值 | 标准差 |
|---|---:|---:|---:|---:|---:|---:|
| initial，`rtc_guided=false` | 6 | 0.2435 s | 0.2428 s | 0.2347 s | 0.2512 s | 0.0056 s |
| guided replacement，`rtc_guided=true` | 6 | 0.2938 s | 0.2974 s | 0.2759 s | 0.3050 s | 0.0108 s |

代表性完整循环：

```text
[FAKE DRY-RUN] cycle=39 started=2026-11-20T17:30:25+08:00
{"action_shape":[50,11],"rtc_mode":"guided","rtc_guided":false,
 "inference_elapsed_s":0.24261945601028856,"actions_executed":0}
{"action_shape":[50,11],"rtc_mode":"guided","rtc_guided":true,
 "inference_elapsed_s":0.30500593400211073,"actions_executed":0}
[FAKE DRY-RUN] cycle=39 PASS: two 50x11 inference responses; no robot commands
```

日志中的 `Ctrl+C` 是人工终止连续测试。第 40、42、44、45 轮的控制台片段不完整，因此不计入延迟统计；已完整输出的循环结果不受影响。

#### RTX 5090D 对照测试

5090D 使用原生 252 服务包执行相同的合成输入 initial/replacement 测试：

```bash
cd /home/codeit/Public/zpf/StarVLA/inference/packages/dualarm_lt_rtc_20261118/server
./run_fake_test_252.sh v2 --continuous --interval-s 0
```

截图中的第 175 轮返回两次 `50×11` Action，`actions_executed=0`：initial 的 client round-trip / server inference 约为 `1.0169 / 0.9990 s`，replacement 约为 `1.3095 / 1.2941 s`，两者差值分别约为 17.9 ms 和 15.4 ms。该差值包含本机序列化、HTTP 和客户端处理开销。

Thor 与 5090D 记录来自不同运行环境和服务封装，只能证明各自链路能够连续完成 initial/guided 推理；不能仅凭这些数字得出硬件性能排名。严格性能比较还需要统一模型权重、软件版本、输入、预热轮数和计时口径。

## 7. Suck dry-run：真实数据闭环测试（不发指令）

> **当前 client 适配范围：仅 Suck。** `client/dualarm_lt_rtc_16.yaml`、右臂 Pose frame、工作空间和 DO8–DO11 映射只覆盖 Suck 的 `qwenpi_v3_v2`、`qwenpi_v3_v3`、`act_v2`。Place、Snap、Insert、Push/Pull 已支持服务启动与合成推理，但尚未加入左臂 client Profile、DO 映射和对应安全空间，因此不能使用本节的 `run_client.sh --dry-run`，更不能进入 `--real`。

### 7.1 测试目的

`dry-run` 从真实设备读取 RT/NRT Topic、末端 Pose、DI/DO 和相机图像，并向模型服务请求预测；执行端只打印本应发送的命令，日志标记为 `WOULD_SEND`，不打开或发送机器人运动 RPC。它验证：

- Topic 与相机数据能够持续采集；
- Pose、IO、相机名称和模型 state/action 映射正确；
- initial、guided 或 fresh-replan 调度能够连续运行；
- Action 后处理、裁剪、安全检查和日志记录符合配置；
- 真机执行前的完整数据链路可用。

### 7.2 前置检查

```bash
ss -ltn | grep -E ':19091|:6888'
curl --max-time 5 -I http://192.168.2.16:6888/ObCameraSensor/3/0/color
```

### 7.3 操作命令

先启动对应模型服务，再运行同一 Profile 的 client：

```bash
cd /home/test/Public/cs/starvla_lt/docker

./run_server.sh qwenpi_v3_v2
./run_client.sh qwenpi_v3_v2 --dry-run --max-actions 1
./run_client.sh qwenpi_v3_v2 --dry-run --max-actions 50
```

Suck 的其余两个 Profile：

```bash
./run_server.sh qwenpi_v3_v3
./run_client.sh qwenpi_v3_v3 --dry-run --max-actions 50

./run_server.sh act_v2
./run_client.sh act_v2 --dry-run --max-actions 50
```

仅在排查 dry-run 坐标系自动识别时，可显式指定 Topic Pose frame：

```bash
./run_client.sh qwenpi_v3_v2 --dry-run --max-actions 1 \
  --topic-pose-frame-for-dry-run controller
```

该参数只允许用于 dry-run，不能据此绕过真机坐标系核对。

### 7.4 通过判据

- RT/NRT Topic、Pose、DI/DO 和要求的相机均被读取；
- 模型返回 `50×11` Action，调度完成指定 `max-actions`；
- 待执行命令以 `WOULD_SEND` 打印，输出包含对应 Pose 和 DO 数值；
- 日志写入宿主机 `logs/`，无 `FAIL_CLOSED`、Topic timeout、camera read failure 或模型 HTTP 错误；
- 机器人 RPC 未建立、未发送，机械臂没有动作。

## 8. Suck 真机测试：真实数据与 RPC 执行

> 本节真机 client 命令只适用于 Suck。Place、Snap、Insert、Push/Pull 在完成左臂状态打包、坐标标定、DO 映射、安全工作空间和 client Profile 适配前，不得套用 Suck 命令发送真机 RPC。

### 8.1 测试目的和边界

真机模式在 dry-run 相同的数据和模型链路上，增加机器人 RPC 指令发送。Pose Action 通过 `JogAnyC`/`robottarget_value` 执行，DO Action 按配置映射为数字输出指令。真机测试会产生物理动作。

进入真机测试前必须同时满足：

- 对应 Profile 的 `fake-dry-run` 和 `dry-run` 已通过；
- 模型服务 `/health` 正常；
- Topic `19091`、相机 `6888`、RPC `5868` 正常；
- Pose frame、机器人侧、工具/工件坐标系、Action 维度和 DO 映射已人工核对；
- YAML 中执行频率、工作空间和单步平移/旋转限制已审核；
- 机器人处于可控初始位，路径清空，现场人员和独立急停已就位。

### 8.2 分阶段操作命令

先用最小动作数验证单条链路：

```bash
cd /home/test/Public/cs/starvla_lt/docker

./run_server.sh act_v2
./run_client.sh act_v2 --real --max-actions 1
```

确认 Pose、DO、RPC ACK 和机械臂响应一致后，再增加到 5 步：

```bash
./run_client.sh act_v2 --real --max-actions 5
```

Qwen Profile 使用相同的递增方法：

```bash
./run_server.sh qwenpi_v3_v2
./run_client.sh qwenpi_v3_v2 --real --max-actions 1
./run_client.sh qwenpi_v3_v2 --real --max-actions 5
```

只有在 1 步和 5 步测试、现场轨迹复核均通过后，才按批准的测试方案继续增加 `--max-actions`。不要直接从未验证配置启动长序列。

### 8.3 通过判据

- client 明确进入 `REAL`，并打印每次采集的 Pose、IO、模型 Action 和实际 RPC 指令；
- 指令提交数与批准的 `max-actions` 一致，RPC 通信无断开或超时；
- 机械臂轨迹、吸嘴/数字输出和预期任务一致；
- 需要吸取反馈的任务，按任务配置检查对应 DI（例如右臂真空反馈 DI18/DI19）；
- 全程无 `FAIL_CLOSED`、工作空间越界、单步限制触发、Topic/camera timeout；
- 完成后保存 `logs/*.jsonl`，用于复核模型延迟、控制周期、状态、预测 Action、实际发送值和 IO 变化。

`SDK True` 或“command submitted”只表示客户端已提交请求，不应单独视为控制器到位或动作完成；验收应结合 RPC 响应、实时状态和现场结果。

## 9. 关键参数位置

模型、端口和 Profile：

```text
/home/test/Public/cs/starvla_lt/server/server_16.yaml
```

现有 Suck client 的 RTC 调度、控制频率、DO 阈值、工作空间和单步限制：

```text
/home/test/Public/cs/starvla_lt/client/dualarm_lt_rtc_16.yaml
```

修改 YAML 后应依次重跑 fake-dry-run、dry-run 和小步数真机测试。安全参数变更需要现场复核，不应以关闭保护来代替数据映射或坐标系问题排查。

## 10. 停止与清理

停止单个服务：

```bash
./compose.sh stop qwen-v2
./compose.sh stop place-qwen-v2
./compose.sh stop snap-act
./compose.sh stop insert-act
./compose.sh stop push-pull-qwen-v2

# Suck 对照版本
./compose.sh stop qwen-v3 act
```

关闭本项目全部 Compose 服务：

```bash
./compose.sh down
```

查看测试残留容器：

```bash
./compose.sh ps -a
sudo docker ps -a --filter 'name=starvla-lt-fake-'
```

## 11. 构建、导出与迁移

GitHub 包不保存模型和离线依赖快照。首次构建或依赖变更时，先在部署机生成 Docker build context：

```bash
cd /home/test/Public/cs/starvla_lt/docker
python3 snapshot_dependencies.py
python3 refresh_context.py
./build.sh
```

导出镜像：

```bash
sudo docker save \
  starvla-lt-server:thor-20261120 \
  starvla-lt-client:arm64-20261120 \
  -o starvla-lt-thor-images.tar
```

目标 Thor 主机导入：

```bash
sudo docker load -i starvla-lt-thor-images.tar
```

模型目录另行通过受控渠道同步。运行部署还需要以下宿主机内容：

```text
models/
server/server_16.yaml
client/dualarm_lt_rtc_16.yaml
logs/（可新建空目录）
docker/compose.yaml 及运行脚本
```

## 12. 常见故障定位

- `permission denied ... /var/run/docker.sock`：使用 `./compose.sh` 的交互式 sudo，或显式执行 `sudo docker ...`。
- `/health Connection refused`：对应容器未启动、仍在加载或已经退出；检查 `compose ps` 和服务日志。
- `camera HTTP read failed ... 6888`：相机 HTTP 服务未监听或取流失败，不属于模型推理错误。
- `Topic timeout`：检查 `19091`、RT/NRT 发布线程和订阅连接。
- `peer closed connection`：检查机器人控制服务是否退出、重启或主动断开 RPC。
- `checkpoint size mismatch`：模型权重不完整或与 Profile 配置不匹配。
- `RTC schedule infeasible`：推理延迟、请求触发点和 Action horizon 不匹配，应调整 RTC 调度参数后重新完成三级测试。
- 单步平移/旋转或工作空间限制触发：先核对 Action 映射、坐标系和当前 Pose，再评估参数；不要直接取消保护。
- `motion_authorized: false`、`actions_executed: 0`：模型服务的正常安全边界，只有 client `--real` 负责动作执行。

## 13. 交付范围

代码仓库交付 Dockerfile、Compose 配置、服务端和客户端源码、YAML 示例、启动/测试脚本及本文档。以下内容不进入 GitHub：

- `models/` 和各类 checkpoint；
- `logs/`、JSONL 和现场运行状态文件；
- Python 虚拟环境、缓存和 Docker 离线依赖快照；
- Docker 镜像 tar 包；
- 账号密码、私钥和访问令牌。
