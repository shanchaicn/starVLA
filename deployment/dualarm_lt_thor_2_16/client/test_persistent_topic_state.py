"""Offline tests for the single-connection RT/NRT Topic cache."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

import dualarm_lt_rtc_client_16 as client


def message(name, frame_id):
    value = type(name, (), {})()
    value.head = types.SimpleNamespace(frame_id=frame_id, timestamp=frame_id * 2)
    return value


def config(**topic_overrides):
    topic = {
        "initial_timeout_s": 0.5,
        "rt_stale_s": 0.10,
        "nrt_stale_s": 0.10,
        "frame_stale_s": 0.20,
        "max_rt_nrt_rx_skew_s": 0.10,
    }
    topic.update(topic_overrides)
    return {"robot": {"topic_ip": "127.0.0.1", "topic_port": 19091, "topic": topic}}


class UpdatingReader:
    instances = 0

    def __init__(self, _ip, _port):
        type(self).instances += 1
        self.count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, _timeout_ms):
        time.sleep(0.001)
        self.count += 1
        name = "SystemRtState" if self.count % 2 else "SystemNrtState"
        return message(name, self.count)


class FailingReader(UpdatingReader):
    def read(self, timeout_ms):
        if self.count >= 20:
            raise RuntimeError("synthetic transport loss")
        return super().read(timeout_ms)


class FrozenReader(UpdatingReader):
    def read(self, _timeout_ms):
        if self.count < 2:
            self.count += 1
            name = "SystemRtState" if self.count == 1 else "SystemNrtState"
            return message(name, self.count)
        time.sleep(0.005)
        return None


class PersistentTopicStateTests(unittest.TestCase):
    def setUp(self):
        UpdatingReader.instances = 0
        self.module = types.SimpleNamespace(StatusReader=UpdatingReader)

    def test_many_concurrent_snapshots_share_one_reader(self):
        with patch.dict(sys.modules, {"get_status": self.module}):
            with client.PersistentTopicState(config()) as state:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    results = list(pool.map(lambda _index: state.snapshot(), range(200)))
                self.assertEqual(UpdatingReader.instances, 1)
                self.assertTrue(all(len(result) == 3 for result in results))
                self.assertEqual(state.diagnostics()["connection_generation"], 1)

    def test_transport_loss_is_sticky_and_fail_closed(self):
        self.module.StatusReader = FailingReader
        with patch.dict(sys.modules, {"get_status": self.module}):
            with client.PersistentTopicState(config()) as state:
                deadline = time.monotonic() + 0.5
                while not state.diagnostics()["faulted"] and time.monotonic() < deadline:
                    time.sleep(0.005)
                with self.assertRaisesRegex(client.ClientError, "restart client"):
                    state.snapshot()

    def test_stale_cached_frames_are_rejected(self):
        self.module.StatusReader = FrozenReader
        with patch.dict(sys.modules, {"get_status": self.module}):
            with client.PersistentTopicState(
                config(rt_stale_s=0.02, nrt_stale_s=0.02)
            ) as state:
                time.sleep(0.04)
                with self.assertRaisesRegex(client.ClientError, "stale"):
                    state.snapshot()

    def test_compensated_force_mapping_uses_ftvalue_zero_and_two(self):
        values = [
            types.SimpleNamespace(fx=i + 0.1, fy=i + 0.2, fz=i + 0.3,
                                  mx=i + 0.4, my=i + 0.5, mz=i + 0.6)
            for i in range(4)
        ]
        rt = types.SimpleNamespace(controller=types.SimpleNamespace(ftvalues=values))
        result = client.extract_compensated_ft(rt)
        self.assertEqual(result["components"], ["fx", "fy", "fz", "mx", "my", "mz"])
        self.assertEqual(result["left_arm1"]["index"], 0)
        self.assertEqual(result["left_arm1"]["values"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(result["right_arm0"]["index"], 2)
        self.assertEqual(result["right_arm0"]["values"], [2.1, 2.2, 2.3, 2.4, 2.5, 2.6])


if __name__ == "__main__":
    unittest.main()
