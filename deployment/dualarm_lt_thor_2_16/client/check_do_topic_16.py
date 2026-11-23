#!/usr/bin/env python3
"""Read-only check that CodeIT NRT Topic publishes and updates DO8..DO11."""

import argparse
import time

from get_status import StatusReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.2.16")
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--io-model", default="HRC-dh-L-tool")
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    wanted = ("DO8", "DO9", "DO10", "DO11")
    deadline = time.monotonic() + args.seconds
    received = 0
    frames = set()
    timestamps = set()
    found = set()
    last_values = None
    last_print = 0.0
    models_seen = set()
    print("READ_ONLY: subscribing to SystemNrtState; no robot RPC", flush=True)
    with StatusReader(args.ip, args.port) as reader:
        while time.monotonic() < deadline:
            message = reader.read(250)
            if message is None or type(message).__name__ != "SystemNrtState":
                continue
            received += 1
            frames.add(int(message.head.frame_id))
            timestamps.add(int(message.head.timestamp))
            models = {model.model_name: model for model in message.model}
            models_seen.update(models)
            io_model = models.get(args.io_model)
            values = {} if io_model is None else {
                item.io_name.rsplit(".", 1)[-1]: float(item.io_data)
                for item in io_model.io
            }
            found.update(set(values).intersection(wanted))
            current = {name: values.get(name) for name in wanted}
            now = time.monotonic()
            if current != last_values or now - last_print >= 1.0:
                print(
                    f"NRT frame={message.head.frame_id} timestamp={message.head.timestamp} "
                    f"model={args.io_model} DO={current}",
                    flush=True,
                )
                last_values = current
                last_print = now

    missing = sorted(set(wanted) - found)
    updating = len(frames) > 1 and len(timestamps) > 1
    print(
        f"SUMMARY nrt_messages={received} unique_frames={len(frames)} "
        f"unique_timestamps={len(timestamps)} missing={missing} "
        f"models_seen={sorted(models_seen)}",
        flush=True,
    )
    if received == 0:
        raise SystemExit("FAIL: no SystemNrtState messages")
    if missing:
        raise SystemExit(f"FAIL: DO fields missing from {args.io_model}: {missing}")
    if not updating:
        raise SystemExit("FAIL: NRT messages received, but frame/timestamp did not advance")
    print("PASS: DO8..DO11 present in updating NRT Topic", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("OPERATOR_STOP: no robot RPC", flush=True)
