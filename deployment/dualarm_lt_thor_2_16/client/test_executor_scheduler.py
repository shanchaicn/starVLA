"""Offline timing tests for the real executor's 50-step buffer."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
from itertools import cycle
import json
from pathlib import Path
import tempfile
import time
import unittest
import sys
from unittest.mock import patch

import numpy as np

import dualarm_lt_rtc_execute_16 as executor


class SchedulerTests(unittest.TestCase):
    def run_simulated(
        self, mode, latency_s, max_actions=120, delay_steps=None,
        buffer_wait_s=0.5, simulated_hz=100.0,
    ):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {
                "inference": {
                    "active_profile": {"name": "test", "rtc_mode": mode},
                    "instruction": "offline timing test",
                },
                "execution": {"command_hz": 100.0, "max_buffer_wait_s": buffer_wait_s},
                "logging": {"directory": directory},
                "robot": {"digital_inputs": []},
                "frames": {"command_pose_frame_with_setwobj": "dataset"},
            }
            calls = []
            latencies = cycle(latency_s if isinstance(latency_s, (list, tuple)) else [latency_s])

            def fake_inference(_cfg, _dry_run, _rpc, previous=None, executed=None, delay=None):
                calls.append((previous is not None, executed, delay))
                current_latency = next(latencies)
                time.sleep(current_latency)
                chunk = np.zeros((50, 11), dtype=np.float32)
                chunk[:, 6] = 1.0
                return chunk, {
                    "inference_elapsed_s": current_latency,
                    "_client_timing": {"e2e_s": current_latency},
                }

            def fake_action(action, _cfg, _dry_run):
                state = np.r_[np.zeros(3), [0, 0, 0, 1]]
                status = {"topic_pose_xyzw": state.tolist(), "topic_pose_frame": "dataset", "io_values": {}}
                return state, status, action[:7], action[:7], [0, 0, 0, 0], {}, []

            with (
                patch.object(executor.base, "check_server_health", return_value={
                    "model": "offline", "rtc_mode": mode, "state_dim": 7, "camera_order": [],
                }),
                patch.object(
                    executor, "execution_schedule",
                    return_value=(simulated_hz, delay_steps or (20 if mode == "guided" else 28)),
                ),
                patch.object(executor, "inference_request", side_effect=fake_inference),
                patch.object(executor, "prepare_action", side_effect=fake_action),
                patch.object(executor, "print_observation"),
                redirect_stdout(io.StringIO()),
            ):
                error = None
                try:
                    executor.run(cfg, dry_run=True, max_actions=max_actions, actions_per_inference=20)
                except executor.base.ClientError as exc:
                    error = str(exc)

            action_log = next(
                path for path in Path(directory).glob("dualarm_lt_rtc_dry_run_*.jsonl")
                if not path.name.endswith("_events.jsonl")
            )
            rows = [json.loads(line) for line in action_log.read_text().splitlines()]
            event_log = next(Path(directory).glob("dualarm_lt_rtc_dry_run_*_events.jsonl"))
            events = [json.loads(line) for line in event_log.read_text().splitlines()]
            return calls, rows, events, error

    def test_guided_replans_without_blocking_at_twenty_actions(self):
        calls, rows, events, error = self.run_simulated("guided", 0.12)
        self.assertIsNone(error)
        self.assertEqual(len(rows), 120)
        self.assertGreaterEqual(sum(e["kind"] == "replacement" for e in events), 4)
        self.assertTrue(all(c[0] and c[2] >= 1 for c in calls[1:]))
        self.assertTrue(all(e["next_index"] >= e["delay_steps"] for e in events if e["kind"] == "replacement"))
        times = [__import__("datetime").datetime.fromisoformat(row["utc"]).timestamp() for row in rows]
        self.assertLess(max(b - a for a, b in zip(times, times[1:])), 0.06)

    def test_act_replans_without_guided_history(self):
        calls, rows, events, error = self.run_simulated("fresh_replan", 0.04)
        self.assertIsNone(error)
        self.assertEqual(len(rows), 120)
        self.assertGreaterEqual(sum(e["kind"] == "replacement" for e in events), 4)
        self.assertTrue(all(c == (False, None, None) for c in calls))

    def test_exhausted_buffer_stops_before_action_fifty_one(self):
        _, rows, events, error = self.run_simulated(
            "guided", 0.70, max_actions=60, buffer_wait_s=0.05,
        )
        self.assertIn("buffer exhausted", error)
        self.assertEqual(len(rows), 50)
        self.assertEqual(events[-2]["kind"], "buffer_exhausted")
        self.assertEqual(events[-1]["kind"], "client_failure")

    def test_short_buffer_wait_recovers_without_extra_motion_command(self):
        _, rows, events, error = self.run_simulated(
            "guided", 0.24, max_actions=70, delay_steps=28, buffer_wait_s=0.10,
        )
        self.assertIsNone(error)
        self.assertEqual(len(rows), 70)
        self.assertTrue(any(e["kind"] == "buffer_wait" for e in events))
        self.assertFalse(any(e["kind"] == "buffer_exhausted" for e in events))

    def test_measured_three_point_four_second_outlier_recovers_without_motion_during_wait(self):
        # Scale time by 10: 100 Hz test ticks and a 0.34 s request represent
        # the observed 10 Hz, 3.4 s ACT request after a 20-step plan switch.
        _, rows, events, error = self.run_simulated(
            "fresh_replan", [0.19, 0.19, 0.34, 0.19],
            max_actions=100, buffer_wait_s=0.2,
        )
        self.assertIsNone(error)
        self.assertEqual(len(rows), 100)
        waits = [e for e in events if e["kind"] == "buffer_wait"]
        self.assertTrue(waits)
        self.assertTrue(all(0 < e["waited_s"] < 0.2 for e in waits))
        self.assertFalse(any(e["kind"] == "buffer_exhausted" for e in events))

    def test_v3_profile_keeps_configured_eight_hz_and_sixteen_step_delay(self):
        path = Path(__file__).with_name("dualarm_lt_rtc_16.yaml")
        v2 = executor.base.load_config(str(path), "qwenpi_v3_v2")
        self.assertEqual(v2["inference"]["active_profile"]["upload_image_size"], [224, 224])
        cfg = executor.base.load_config(str(path), "qwenpi_v3_v3")
        self.assertEqual(executor.execution_schedule(cfg, cfg["inference"]["active_profile"]), (8.0, 16))
        act = executor.base.load_config(str(path), "act_v2")
        self.assertEqual(executor.execution_schedule(act, act["inference"]["active_profile"]), (10.0, 23))
        self.assertEqual(act["inference"]["active_profile"]["upload_image_size"], [224, 224])

    def test_ten_hz_replays_resized_image_latency_pattern(self):
        # Ten times faster wall clock, with the same action/latency ratio as
        # 10 Hz and measured 1.1-1.4 s resized-image requests.
        latencies = [0.10, 0.106, 0.121, 0.139, 0.143, 0.180, 0.120, 0.110]
        _, rows, events, error = self.run_simulated(
            "guided", latencies, max_actions=200, delay_steps=19,
            buffer_wait_s=0.05, simulated_hz=100.0,
        )
        self.assertIsNone(error)
        self.assertEqual(len(rows), 200)
        self.assertGreaterEqual(sum(e["kind"] == "replacement" for e in events), 5)

    def test_nonrtc_consumes_full_fresh_chunks_without_guided_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = {
                "inference": {
                    "active_profile": {"name": "test", "rtc_mode": "guided"},
                    "instruction": "offline timing test",
                },
                "execution": {"command_hz": 10.0},
                "logging": {"directory": directory},
                "robot": {"digital_inputs": []},
                "frames": {"command_pose_frame_with_setwobj": "dataset"},
            }
            calls = []

            def fake_inference(_cfg, _dry_run, _rpc, previous=None, executed=None, delay=None):
                calls.append((previous, executed, delay))
                chunk = np.zeros((50, 11), dtype=np.float32)
                chunk[:, 6] = 1.0
                return chunk, {
                    "rtc_guided": False,
                    "inference_elapsed_s": 0.1,
                    "_client_timing": {"e2e_s": 0.1},
                }

            def fake_action(action, _cfg, _dry_run):
                state = np.r_[np.zeros(3), [0, 0, 0, 1]]
                status = {"topic_pose_xyzw": state.tolist(), "topic_pose_frame": "dataset", "io_values": {}}
                return state, status, action[:7], action[:7], [0, 0, 0, 0], {}, []

            with (
                patch.object(executor.base, "check_server_health", return_value={
                    "model": "offline", "state_dim": 7, "camera_order": [],
                }),
                patch.object(executor, "inference_request", side_effect=fake_inference),
                patch.object(executor, "prepare_action", side_effect=fake_action),
                patch.object(executor, "print_observation"),
                patch.object(executor.time, "sleep"),
                redirect_stdout(io.StringIO()),
            ):
                executor.run_nonrtc(cfg, dry_run=True, max_actions=105)
            path = next(
                p for p in Path(directory).glob("dualarm_lt_nonrtc_dry_run_*.jsonl")
                if not p.name.endswith("_events.jsonl")
            )
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(rows), 105)
            self.assertEqual([row["chunk_index"] for row in rows],
                             list(range(50)) + list(range(50)) + list(range(5)))
            self.assertEqual(calls, [(None, None, None)] * 3)

    def test_real_entry_does_not_ask_for_keyboard_confirmation(self):
        with (
            patch.object(sys, "argv", ["executor", "--real", "--max-actions", "1"]),
            patch.object(executor, "run") as fake_run,
            patch.object(executor.base, "PersistentTopicState") as fake_topic,
            patch("builtins.input", side_effect=AssertionError("unexpected prompt")),
        ):
            fake_topic.return_value.__enter__.return_value = object()
            executor.main()
        self.assertTrue(fake_run.called)


if __name__ == "__main__":
    unittest.main()
