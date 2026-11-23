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
# 合成数据，不连接 Topic、相机或机器人 RPC
./test_fake_dry_run.sh qwenpi_v3_v2

# 真实 Topic/相机，只打印 WOULD_SEND，不发送机器人 RPC
./run_client.sh qwenpi_v3_v2 --dry-run --max-actions 50

# 真机执行：仅在前两级通过且现场安全条件满足后使用
./run_client.sh qwenpi_v3_v2 --real --max-actions 1
```
