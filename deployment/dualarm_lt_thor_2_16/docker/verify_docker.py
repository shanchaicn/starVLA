#!/usr/bin/env python3
"""Verify images with synthetic input and client self-tests; sends no robot RPC."""
import json
from pathlib import Path
import subprocess
import time
import urllib.request
import uuid

here = Path(__file__).resolve().parent
compose = ["docker", "compose", "--file", str(here / "compose.yaml")]
results = {"client_self_tests": {}, "server_self_tests": {}, "gpu_tests": {}, "motion_commands_sent": 0}


def run(args, timeout=180):
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout


def save():
    (here / "verification.json").write_text(json.dumps(results, indent=2) + "\n")


try:
    results["client_native_imports"] = run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "python", "starvla-lt-client:arm64-20261120", "-c",
        'import sys; sys.path.insert(0,"/app/client/vendor"); import cv2,numpy,get_status,rpc; print("SDK/CV2/RPC IMPORT PASS; NO NETWORK")'])
    for profile in ("act_v2", "qwenpi_v3_v2", "qwenpi_v3_v3"):
        results["client_self_tests"][profile] = run(compose + ["run", "--rm", "-T", "--no-deps", "client", "--profile", profile, "--self-test"])
        results["server_self_tests"][profile] = run(compose + ["run", "--rm", "-T", "--no-deps", "act", "--profile", profile, "--self-test"])
        save()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for index, (profile, folder, size) in enumerate([
        ("act_v2", "dualarm_lt_suck_act_h50_v2", 103360178),
        ("qwenpi_v3_v2", "dualarm_lt_suck_qwenpi_v3_v2_h50", 10941594936),
        ("qwenpi_v3_v3", "dualarm_lt_suck_qwenpi_v3_v3_h50", 10941560120),
    ]):
        checkpoint = here.parent / "models" / folder / "final_model/pytorch_model.pt"
        if not checkpoint.is_file() or checkpoint.stat().st_size != size:
            results["gpu_tests"][profile] = {"status": "pending_model_transfer"}
            save()
            continue
        port = 10381 + index
        name = "starvla-lt-verify-" + uuid.uuid4().hex[:8]
        run(compose + ["run", "-d", "--no-deps", "--name", name, "act", "--profile", profile, "--port", str(port)])
        try:
            deadline = time.monotonic() + 240
            while True:
                try:
                    with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except OSError:
                    running = subprocess.run(["docker", "inspect", "--format", "{{.State.Running}}", name], capture_output=True, text=True)
                    if running.returncode or running.stdout.strip() != "true" or time.monotonic() > deadline:
                        raise RuntimeError(run(["docker", "logs", name]))
                    time.sleep(2)
            prediction = run(["docker", "exec", name, "python", "/app/server/smoke_predict_16.py", "--profile", profile, "--port", str(port)], timeout=360)
            results["gpu_tests"][profile] = {"status": "passed", "health": health, "synthetic_prediction": prediction}
        finally:
            subprocess.run(["docker", "stop", "--time", "10", name], capture_output=True, timeout=30)
            subprocess.run(["docker", "container", "rm", name], capture_output=True, timeout=30)
        save()
    results["passed_for_available_models"] = True
except Exception as exc:
    results["passed_for_available_models"] = False
    results["error"] = str(exc)
    raise
finally:
    save()
