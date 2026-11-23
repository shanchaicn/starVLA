#!/usr/bin/env python3
"""Read-only check that RT Topic publishes finite FTValues in advancing frames."""

import argparse
import math
import time

from get_status import StatusReader, SystemRtState


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.2.16")
    parser.add_argument("--port", type=int, default=19091)
    parser.add_argument("--seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    deadline = time.monotonic() + args.seconds
    frames = set()
    timestamps = set()
    received = 0
    with_values = 0
    invalid = 0
    last_print = 0.0
    print("READ_ONLY: subscribing to SystemRtState.controller.ftvalues; no robot RPC", flush=True)
    with StatusReader(args.ip, args.port) as reader:
        while time.monotonic() < deadline:
            state = reader.read(250)
            if not isinstance(state, SystemRtState):
                continue
            received += 1
            frames.add(int(state.head.frame_id))
            timestamps.add(int(state.head.timestamp))
            values = [
                {name: float(getattr(ft, name)) for name in ("fx", "fy", "fz", "mx", "my", "mz")}
                for ft in state.controller.ftvalues
            ]
            if values:
                with_values += 1
                if any(not math.isfinite(value) for ft in values for value in ft.values()):
                    invalid += 1
            now = time.monotonic()
            if now - last_print >= 1.0:
                print(
                    f"RT frame={state.head.frame_id} timestamp={state.head.timestamp} "
                    f"controller={state.controller.controller_name!r} ftvalues={values}",
                    flush=True,
                )
                last_print = now

    print(
        f"SUMMARY rt_messages={received} with_ftvalues={with_values} "
        f"invalid={invalid} unique_frames={len(frames)} unique_timestamps={len(timestamps)}",
        flush=True,
    )
    if received == 0:
        raise SystemExit("FAIL: no SystemRtState messages")
    if with_values == 0:
        raise SystemExit("FAIL: controller.ftvalues is empty")
    if invalid:
        raise SystemExit("FAIL: controller.ftvalues contains NaN or infinity")
    if len(frames) < 2 or len(timestamps) < 2:
        raise SystemExit("FAIL: RT frame/timestamp did not advance")
    print("PASS: finite FTValues present in updating RT Topic", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("OPERATOR_STOP: no robot RPC", flush=True)
