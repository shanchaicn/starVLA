#!/usr/bin/env python3
"""Start, stop, switch and inspect this package's three inference servers."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Model:
    name: str
    script: str
    launcher: str
    port: int
    profile: str | None = None


MODELS = {
    "v2": Model("v2", "dualarm_lt_rtc_server_252.py", "run_server_252.sh", 10199, "qwenpi_v3_v2"),
    "v3": Model("v3", "dualarm_lt_rtc_server_252.py", "run_server_qwenpi_v3_v3_252.sh", 10200, "qwenpi_v3_v3"),
    "act": Model("act", "dualarm_lt_act_server_252.py", "run_server_act_v2_252.sh", 10201),
}


def pids_for(model: Model) -> list[int]:
    result = []
    script = str(ROOT / model.script)
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = [value.decode(errors="replace") for value in
                    (entry / "cmdline").read_bytes().split(b"\0") if value]
        except (OSError, PermissionError):
            continue
        if len(args) < 2 or args[1] != script:
            continue
        if "--serve" not in args or "--port" not in args:
            continue
        if args.index("--port") + 1 >= len(args) or args[args.index("--port") + 1] != str(model.port):
            continue
        if model.profile is not None and (
            "--profile" not in args or args.index("--profile") + 1 >= len(args)
            or args[args.index("--profile") + 1] != model.profile
        ):
            continue
        result.append(int(entry.name))
    return sorted(result)


def health(model: Model) -> dict | None:
    try:
        with urlopen(f"http://127.0.0.1:{model.port}/health", timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def status() -> None:
    for model in MODELS.values():
        pids = pids_for(model)
        data = health(model) if pids else None
        label = "healthy" if data else ("starting/unhealthy" if pids else "stopped")
        identity = f" model={data.get('model')}" if data else ""
        print(f"{model.name}: {label} port={model.port} pids={pids}{identity}")


def stop(model: Model) -> None:
    pids = pids_for(model)
    if not pids:
        print(f"{model.name}: already stopped")
        return
    for pid in pids:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if not pids_for(model):
            print(f"{model.name}: stopped pids={pids}")
            return
        time.sleep(0.25)
    raise RuntimeError(f"{model.name}: still running after SIGTERM; inspect pids={pids}")


def start(model: Model) -> None:
    if pids_for(model):
        print(f"{model.name}: already running pids={pids_for(model)}")
        return
    if model.name in ("v2", "v3"):
        other = MODELS["v3" if model.name == "v2" else "v2"]
        if pids_for(other):
            raise RuntimeError(f"{other.name} is running; use 'switch {model.name}'")
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{model.name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            [str(ROOT / model.launcher)], cwd=ROOT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
    print(f"{model.name}: starting pid={process.pid} log={log_path}", flush=True)
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{model.name}: exited {process.returncode}; inspect {log_path}")
        data = health(model)
        if data and data.get("status") == "DEV_SHADOW_ONLY":
            print(f"{model.name}: ready port={model.port} model={data.get('model')}")
            return
        time.sleep(1)
    raise RuntimeError(f"{model.name}: startup exceeded 120 s; inspect {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "start", "stop", "switch"))
    parser.add_argument("model", nargs="?", choices=(*MODELS, "all"))
    args = parser.parse_args()
    if args.action == "status":
        status()
        return
    if args.model is None:
        parser.error("start/stop/switch needs a model")
    if args.action == "stop" and args.model == "all":
        for model in MODELS.values():
            stop(model)
        return
    if args.model == "all":
        parser.error("'all' is only valid with stop")
    model = MODELS[args.model]
    if args.action == "start":
        start(model)
    elif args.action == "stop":
        stop(model)
    else:
        if model.name not in ("v2", "v3"):
            parser.error("switch supports v2 or v3")
        stop(MODELS["v3" if model.name == "v2" else "v2"])
        start(model)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as exc:
        print(f"model service error: {exc}", file=sys.stderr)
        raise SystemExit(2)
