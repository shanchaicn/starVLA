#!/usr/bin/env python3
"""Read-only persistent Topic monitor; never opens the robot RPC channel."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time

from get_status import StatusReader


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot(directory):
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    path = directory / ("topic_threads_" + stamp + ".txt")
    with path.open("w") as out:
        for command in (
            ["systemctl", "show", "codeit-deploy.service", "-p", "MainPID", "-p", "NRestarts", "-p", "ActiveEnterTimestamp"],
            ["ps", "-eLo", "pid,tid,cls,rtprio,psr,stat,pcpu,wchan:28,comm"],
            ["ss", "-ltnp"],
        ):
            out.write("\n" + " ".join(command) + "\n")
            out.flush()
            subprocess.run(command, stdout=out, stderr=out, timeout=5, check=False)
    return str(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.2.16")
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--stale-s", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent.parent / "logs"))
    args = parser.parse_args()
    if args.seconds <= 0 or args.stale_s <= 0:
        parser.error("seconds and stale-s must be positive")
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("topic_watch_" + datetime.now().strftime("%Y%m%dT%H%M%S_%f") + ".jsonl")
    names = ("SystemRtState", "SystemNrtState")
    start = time.monotonic()
    states = {name: {"received": 0, "frame_id": None, "timestamp": None,
                     "last_rx": None, "last_frame_change": None, "last_timestamp_change": None}
              for name in names}
    last_flags = None
    next_report = start + 1
    print("TOPIC_WATCH_LOG", path, "NO ROBOT RPC", flush=True)
    with path.open("x") as log, StatusReader(args.ip, args.port) as reader:
        while time.monotonic() - start < args.seconds:
            message = reader.read(50)
            tick = time.monotonic()
            if message is not None and type(message).__name__ in states:
                state = states[type(message).__name__]
                state["received"] += 1
                state["last_rx"] = tick
                for key in ("frame_id", "timestamp"):
                    value = int(getattr(message.head, key))
                    if state[key] is None or state[key] != value:
                        change_key = "last_frame_change" if key == "frame_id" else "last_timestamp_change"
                        state[change_key] = tick
                    state[key] = value
            if tick < next_report:
                continue
            next_report = tick + 1
            row = {"utc": now(), "elapsed_s": round(tick - start, 3), "topics": {}}
            flags = []
            for name, state in states.items():
                item = {key: state[key] for key in ("received", "frame_id", "timestamp")}
                for key in ("last_rx", "last_frame_change", "last_timestamp_change"):
                    item[key + "_age_s"] = None if state[key] is None else round(tick - state[key], 3)
                if state["last_rx"] is None or tick - state["last_rx"] > args.stale_s:
                    flag = "NO_MESSAGES"
                elif state["last_frame_change"] is None or tick - state["last_frame_change"] > args.stale_s:
                    flag = "REPEATED_FRAME"
                elif state["last_timestamp_change"] is None or tick - state["last_timestamp_change"] > args.stale_s:
                    flag = "TIMESTAMP_FROZEN"
                else:
                    flag = "UPDATING"
                item["status"] = flag
                flags.append(flag)
                row["topics"][name] = item
            if tuple(flags) != last_flags and any(flag != "UPDATING" for flag in flags):
                row["thread_snapshot"] = snapshot(directory)
            last_flags = tuple(flags)
            line = json.dumps(row, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("TOPIC_WATCH_OPERATOR_STOP", flush=True)
