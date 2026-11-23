#!/usr/bin/env python3
"""Finish transfer verification and synthetic GPU checks, with no robot I/O."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-pid", type=int)
    args = parser.parse_args()
    status = {"stage": "waiting_for_transfer", "profiles": {}, "motion_commands_sent": 0}

    def save():
        status["updated_at_unix"] = time.time()
        target = here / "deployment_status.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(target)

    save()
    try:
        while not (here / "transfer.complete").is_file():
            if args.transfer_pid:
                try:
                    os.kill(args.transfer_pid, 0)
                except ProcessLookupError:
                    raise RuntimeError("Model transfer stopped before completion; inspect transfer.log")
            time.sleep(10)
        status["stage"] = "verifying_sha256"
        save()
        subprocess.run([sys.executable, str(here / "verify_models_16.py")], check=True)
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        for index, profile in enumerate(("act_v2", "qwenpi_v3_v2", "qwenpi_v3_v3")):
            status["stage"] = "gpu_check_" + profile
            save()
            launch = [sys.executable, str(here / "launch_server_16.py"), "--profile", profile]
            subprocess.run(launch + ["--check-config"], check=True)
            subprocess.run(launch + ["--self-test"], check=True)
            port = 10391 + index
            log = (here / (profile + "_gpu_check.log")).open("w")
            proc = subprocess.Popen(launch + ["--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
            try:
                deadline = time.monotonic() + 240
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"{profile} server exited with {proc.returncode}; inspect its GPU check log")
                    try:
                        with opener.open(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                            health = json.load(response)
                        break
                    except OSError:
                        if time.monotonic() > deadline:
                            raise TimeoutError(f"{profile} model startup exceeded 240 seconds")
                        time.sleep(2)
                result = subprocess.run([sys.executable, str(here / "smoke_predict_16.py"), "--profile", profile, "--port", str(port)], capture_output=True, text=True, timeout=360)
                (here / (profile + "_smoke_result.log")).write_text(result.stdout + result.stderr)
                if result.returncode:
                    raise RuntimeError(f"{profile} synthetic prediction failed; inspect smoke result log")
                status["profiles"][profile] = {"passed": True, "health": health, "synthetic_result": result.stdout}
                save()
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                log.close()
        status["stage"] = "ready"
        status["services_left_running"] = False
        save()
    except Exception as exc:
        status["stage"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


if __name__ == "__main__":
    main()
