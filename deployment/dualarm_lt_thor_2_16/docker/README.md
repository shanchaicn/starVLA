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

# 当前真实 Topic/相机 client 只适配 Suck；只打印 WOULD_SEND，不发送机器人 RPC
./run_client.sh qwenpi_v3_v2 --dry-run --max-actions 50

# 当前真机 client 只适配 Suck；仅在前两级通过且现场安全条件满足后使用
./run_client.sh qwenpi_v3_v2 --real --max-actions 1
```

Place、Snap、Insert、Push/Pull 已支持 Docker 服务和合成推理；尚未提供左臂真机 client Profile，不能套用 Suck 的 dry-run/real 命令。
