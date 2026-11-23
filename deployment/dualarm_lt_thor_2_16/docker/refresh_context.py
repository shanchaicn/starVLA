#!/usr/bin/env python3
"""Snapshot code for rebuilding; excludes venvs, model weights and logs."""
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent.parent
dst = root / "docker/context"
ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.log")
for path in ["server/runtime/source", "server/runtime/eval_deps", "server/252_original/vendor_act"]:
    shutil.copytree(root / path, dst / path, dirs_exist_ok=True, ignore=ignore)
(dst / "client/vendor").mkdir(parents=True, exist_ok=True)
(dst / "sdk").mkdir(parents=True, exist_ok=True)
for name in ["launch_server_16.py", "dualarm_lt_rtc_server_16.py", "dualarm_lt_act_server_16.py", "rtc_contract.py", "server_16.yaml", "smoke_predict_16.py"]:
    shutil.copy2(root / "server" / name, dst / "server" / name)
for name in ["dualarm_lt_rtc_execute_16.py", "dualarm_lt_rtc_client_16.py", "direct_robot_rpc.py", "dualarm_lt_rtc_16.yaml"]:
    shutil.copy2(root / "client" / name, dst / "client" / name)
shutil.copy2(root / "client/vendor/rpc.so", dst / "client/vendor/rpc.so")
shutil.copy2("/home/test/Public/cs/get_status_sdk/dist/tb6r5_get_status-1.0.0-py3-none-any.whl", dst / "sdk")
print("Docker build context refreshed")
