# StarVLA LT Thor Docker

正式部署、启动和测试说明见 [`../StarVLA_2.16_Docker部署说明.md`](../StarVLA_2.16_Docker部署说明.md)。

本目录保存 Dockerfile、Compose 配置和运行脚本。GitHub 不保存模型权重、日志、镜像归档和 `context/` 离线依赖快照。

首次构建或更新依赖时，在部署机重新生成 build context：

```bash
cd /home/test/Public/cs/starvla_lt/docker
python3 snapshot_dependencies.py
python3 refresh_context.py
./build.sh
```

三级测试入口：

```bash
# 五个任务模型：启动服务
./run_server.sh qwenpi_v3_v2
./run_server.sh place_qwenpi_v3_v2
./run_server.sh snap_act_v2
./run_server.sh insert_act_v2
./run_server.sh push_pull_qwenpi_v3_v2

# 五个任务模型：合成数据推理，不连接 Topic、相机或机器人 RPC
./test_fake_dry_run.sh qwenpi_v3_v2
./test_fake_dry_run.sh place_qwenpi_v3_v2
./test_fake_dry_run.sh snap_act_v2
./test_fake_dry_run.sh insert_act_v2
./test_fake_dry_run.sh push_pull_qwenpi_v3_v2

# 五动作真实 Topic/相机 RTC dry-run；只打印 WOULD_SEND，不发送机器人 RPC
./run_client.sh qwenpi_v3_v2 --dry-run --max-actions 50
./run_client.sh place_qwenpi_v3_v2 --dry-run --max-actions 50
./run_client.sh snap_act_v2 --dry-run --max-actions 50
./run_client.sh insert_act_v2 --dry-run --max-actions 50
./run_client.sh push_pull_qwenpi_v3_v2 --dry-run --max-actions 50

# 非 RTC dry-run：完整消费每个新 50 步 chunk
./run_client.sh place_qwenpi_v3_v2 --dry-run --no-rtc --max-actions 50

# 真机执行：左臂 Profile 默认 fail-closed，需先完成 YAML 中三项现场核验
./run_client.sh qwenpi_v3_v2 --real --max-actions 1
./run_client.sh qwenpi_v3_v2 --real --no-rtc --max-actions 1
```

五个动作均已配置 RTC 和非 RTC client。Place、Snap、Insert、Push/Pull 的左臂真机 RPC 默认关闭，必须先验证 wobj0 坐标、现场工作空间和 DO，再在对应 Profile 中开启三项 `real` 布尔安全门。
