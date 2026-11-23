#!/usr/bin/env python3
"""Capture one JPEG from each ObCameraSensor HTTP stream."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from urllib.request import ProxyHandler, build_opener

NO_PROXY_OPENER = build_opener(ProxyHandler({}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensors", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default=str(Path(__file__).with_name("camera_probe")))
    args = parser.parse_args()
    output = Path(args.output) / datetime.now().strftime("%Y%m%dT%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    failures = 0
    for sensor in args.sensors:
        url = f"http://192.168.2.16:6888/ObCameraSensor/{sensor}/0/color"
        data = bytearray()
        jpeg = None
        try:
            with NO_PROXY_OPENER.open(url, timeout=5.0) as stream:
                while len(data) < 8_000_000:
                    data.extend(stream.read(64 * 1024))
                    start = data.find(b"\xff\xd8")
                    end = data.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if start >= 0 and end >= 0:
                        jpeg = bytes(data[start:end + 2])
                        break
        except OSError:
            jpeg = None
        target = output / f"sensor_{sensor}.jpg"
        if jpeg is None:
            print(f"FAIL sensor={sensor} {url}")
            failures += 1
        else:
            target.write_bytes(jpeg)
            print(f"OK sensor={sensor} -> {target} bytes={len(jpeg)}")
    print(f"CAMERA_PROBE_DONE output={output} failures={failures}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()

