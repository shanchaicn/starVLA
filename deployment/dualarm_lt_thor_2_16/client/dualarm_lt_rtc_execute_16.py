#!/usr/bin/env python3
"""RTC and full-chunk RPC executor for the dual-arm LT right suction task."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

import dualarm_lt_rtc_client_16 as base
from direct_robot_rpc import DirectRobotRpc

def quat_conjugate(q):
    q = np.asarray(q, dtype=np.float64)
    return np.r_[-q[:3], q[3]]


def dataset_to_controller(pose, cfg):
    pose = np.asarray(pose, dtype=np.float64)
    tf = cfg["frames"]["dataset_from_controller"]
    t = np.asarray(tf["translation_m"], dtype=np.float64)
    q = np.asarray(tf["quaternion_xyzw"], dtype=np.float64)
    q /= np.linalg.norm(q)
    qi = quat_conjugate(q)
    p = base.quat_rotate(qi, pose[:3] - t)
    out_q = base.quat_mul(qi, pose[3:] / np.linalg.norm(pose[3:]))
    out_q /= np.linalg.norm(out_q)
    return np.r_[p, out_q]


def dataset_to_command_pose(pose, cfg):
    pose = np.asarray(pose, dtype=np.float64).copy()
    frame = cfg["frames"]["command_pose_frame_with_setwobj"]
    if frame == "dataset":
        pose[3:] /= np.linalg.norm(pose[3:])
        return pose
    return dataset_to_controller(pose, cfg)


def quat_angle(a, b):
    a = np.asarray(a, dtype=np.float64) / np.linalg.norm(a)
    b = np.asarray(b, dtype=np.float64) / np.linalg.norm(b)
    return 2.0 * math.acos(float(np.clip(abs(np.dot(a, b)), 0.0, 1.0)))


def slerp_limited(current, target, max_angle):
    current = np.asarray(current, dtype=np.float64) / np.linalg.norm(current)
    target = np.asarray(target, dtype=np.float64) / np.linalg.norm(target)
    dot = float(np.dot(current, target))
    if dot < 0:
        target, dot = -target, -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    angle = 2.0 * math.acos(dot)
    if angle <= max_angle:
        return target, False
    fraction = max_angle / angle
    theta = math.acos(dot)
    if theta < 1e-8:
        return current, False
    out = (math.sin((1 - fraction) * theta) / math.sin(theta)) * current
    out += (math.sin(fraction * theta) / math.sin(theta)) * target
    return out / np.linalg.norm(out), True


def guarded_action(action, live_state, cfg):
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (11,) or not np.isfinite(action).all():
        raise base.ClientError("action is not finite 11D")
    live_pose = np.asarray(live_state[:7], dtype=np.float64)
    target = action[:7].copy()
    target[3:] /= np.linalg.norm(target[3:])
    safety = cfg["execution"]["safety"]
    xyz_min = np.asarray(safety["dataset_xyz_min"], dtype=np.float64)
    xyz_max = np.asarray(safety["dataset_xyz_max"], dtype=np.float64)
    unclipped_xyz = target[:3].copy()
    target[:3] = np.clip(target[:3], xyz_min, xyz_max)
    workspace_clipped = not np.allclose(unclipped_xyz, target[:3])
    delta = target[:3] - live_pose[:3]
    distance = float(np.linalg.norm(delta))
    translation_clipped = False
    max_distance = float(safety["max_live_translation_m"])
    if distance > max_distance:
        if not safety.get("clip_translation_and_rotation", False):
            raise base.ClientError(f"live translation {distance:.4f}m exceeds {max_distance:.4f}m")
        target[:3] = live_pose[:3] + delta * (max_distance / distance)
        translation_clipped = True
    rotation_before = quat_angle(live_pose[3:], target[3:])
    target[3:], rotation_clipped = slerp_limited(
        live_pose[3:], target[3:], float(safety["max_live_rotation_rad"])
    )
    do_values = [int(value >= float(cfg["execution"]["do_threshold"])) for value in action[7:]]
    return target, do_values, {
        "workspace_clipped": workspace_clipped,
        "translation_clipped": translation_clipped,
        "rotation_clipped": rotation_clipped,
        "raw_live_translation_m": distance,
        "raw_live_rotation_rad": rotation_before,
    }


def robot_commands(controller_pose, do_values, cfg):
    result = []
    for do_num, value in zip(cfg["execution"]["action_do_nums"], do_values):
        result.append(f"{{SetHrcsiDout --do_num={do_num} --value={value}||NotRunExecute||NotRunExecute}}")
    pose = ",".join(f"{float(value):.10g}" for value in controller_pose)
    jog = cfg["execution"]["joganyc"]
    vec = lambda values: "{" + ",".join(f"{float(v):.10g}" for v in values) + "}"
    result.append(
        "{NotRunExecute||JogAnyC "
        f"--robottarget_value={{{pose}}} "
        f"--cartesian_vel={vec(jog['cartesian_vel'])} "
        f"--cartesian_acc={vec(jog['cartesian_acc'])} "
        f"--cartesian_dec={vec(jog['cartesian_dec'])} "
        f"--last_count={int(jog['last_count'])}||NotRunExecute}}"
    )
    return result


def temporal_ensemble_action(plans, global_step, cfg):
    """Blend all ACT chunks that predict the same global control step.

    Plans are (global_start_step, 50x11 chunk) pairs. Newer chunks receive
    higher exponential weight. Quaternion signs are aligned before averaging.
    """
    profile = cfg["inference"]["active_profile"]
    te = profile.get("temporal_ensemble", {})
    if not te.get("enabled", False):
        return None, 0
    candidates = []
    for start, chunk in reversed(plans):
        offset = int(global_step - start)
        if 0 <= offset < len(chunk):
            candidates.append(np.asarray(chunk[offset], dtype=np.float64))
    if not candidates:
        raise base.ClientError("ACT temporal ensemble has no action for current step")
    candidates = candidates[:int(te["max_plans"])]
    weights = np.exp(-float(te["decay_per_plan"]) * np.arange(len(candidates)))
    weights /= weights.sum()
    newest_q = candidates[0][3:7]
    aligned = []
    for action in candidates:
        action = action.copy()
        if np.dot(action[3:7], newest_q) < 0:
            action[3:7] *= -1
        aligned.append(action)
    result = np.sum(np.stack(aligned) * weights[:, None], axis=0)
    qnorm = float(np.linalg.norm(result[3:7]))
    if qnorm < 0.25:
        raise base.ClientError("ACT temporal ensemble produced invalid quaternion")
    result[3:7] /= qnorm
    return result.astype(np.float32), len(candidates)


def print_observation(mode, phase, state, status, cfg):
    state = np.asarray(state, dtype=np.float64)
    configured_di = cfg["robot"]["digital_inputs"]
    model_di = {name: float(state[7 + i]) for i, name in enumerate(configured_di)}
    all_io = {name: float(value) for name, value in sorted(status["io_values"].items())}
    topic_diag = status.get("topic_diagnostics", {})
    compensated_ft = status.get("compensated_ft", {})
    left_ft = compensated_ft.get("left_arm1", {}).get("values")
    right_ft = compensated_ft.get("right_arm0", {}).get("values")
    if not status.get("io_available", True):
        print(
            f"[{mode}][IO][{phase}][WARNING] Topic has no configured DI source; "
            "zeros are preview placeholders and real mode remains blocked",
            flush=True,
        )
    print(
        f"[{mode}][STATE][{phase}] running={status['system_running_state']} "
        f"right_joints_enabled={status['all_moving_joints_enabled']}\n"
        f"[{mode}][STATE][{phase}] topic_pose_frame={status['topic_pose_frame']} "
        f"topic_pose_xyzw={[float(v) for v in status['topic_pose_xyzw']]}\n"
        f"[{mode}][STATE][{phase}] dataset_pose_xyzw={state[:7].tolist()}\n"
        f"[{mode}][TOPIC][{phase}] {json.dumps(topic_diag, sort_keys=True)}\n"
        f"[{mode}][FT][{phase}] components=[fx,fy,fz,mx,my,mz] "
        f"ftvalue[0].left_arm1_compensated={left_ft} "
        f"ftvalue[2].right_arm0_compensated={right_ft}\n"
        f"[{mode}][IO][{phase}] model_inputs={json.dumps(model_di, sort_keys=True)}\n"
        f"[{mode}][IO][{phase}] all_io={json.dumps(all_io, sort_keys=True)}",
        flush=True,
    )


def prepare_action(action, cfg, dry_run):
    topic_frame = cfg["frames"][
        "topic_pose_frame_without_setwobj" if dry_run else "topic_pose_frame_after_setwobj"
    ]
    live_state, status = base.capture_state(
        cfg, return_metadata=True, topic_pose_frame=topic_frame,
        allow_missing_io=(
            dry_run and int(cfg["inference"]["active_profile"]["state_dim"]) == 7
        ),
    )
    if not dry_run and (
        not status["system_running_state"] or not status["all_moving_joints_enabled"]
    ):
        raise base.ClientError("robot is not running with all right-arm joints enabled")
    if dry_run and (
        not status["system_running_state"] or not status["all_moving_joints_enabled"]
    ):
        print(
            "[DRY_RUN][WARNING] robot is not fully enabled; preview continues because no RPC is sent",
            flush=True,
        )
    target_dataset, do_values, guards = guarded_action(action, live_state, cfg)
    command_pose = dataset_to_command_pose(target_dataset, cfg)
    commands = robot_commands(command_pose, do_values, cfg)
    return live_state, status, target_dataset, command_pose, do_values, guards, commands


def set_wobj_before_inference(cfg, dry_run, rpc):
    wobj = str(cfg["execution"]["inference_preflight_wobj"])
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    if not wobj or any(character not in allowed for character in wobj):
        raise base.ClientError("execution.inference_preflight_wobj is invalid")
    command = f"{{SetWobj --wobj={wobj}||SetWobj --wobj={wobj}||NotRunExecute}}"
    mode = "DRY_RUN" if dry_run else "REAL"
    verb = "WOULD_SEND" if dry_run else "SEND"
    print(f"[{mode}][RPC][PRE-INFER][{verb}] {command}", flush=True)
    if dry_run:
        return {"success": True, "dry_run": True}
    response = rpc.call(command, int(cfg["execution"]["direct_rpc"]["sync_timeout_ms"]))
    print(f"[{mode}][RPC][PRE-INFER][ACK] {json.dumps(response, sort_keys=True)}", flush=True)
    return response


def inference_request(cfg, dry_run, rpc, previous=None, executed=None, delay=None):
    started = time.monotonic()
    set_wobj_before_inference(cfg, dry_run, rpc)
    wobj_ready = time.monotonic()
    topic_frame = cfg["frames"][
        "topic_pose_frame_without_setwobj" if dry_run else "topic_pose_frame_after_setwobj"
    ]
    state, status = base.capture_state(
        cfg, return_metadata=True, topic_pose_frame=topic_frame,
        allow_missing_io=(
            dry_run and int(cfg["inference"]["active_profile"]["state_dim"]) == 7
        ),
    )
    state_ready = time.monotonic()
    mode = "DRY_RUN" if dry_run else "REAL"
    print_observation(mode, "PRE-INFER", state, status, cfg)
    if dry_run:
        xyz = np.asarray(state[:3], dtype=np.float64)
        safety = cfg["execution"]["safety"]
        xyz_min = np.asarray(safety["dataset_xyz_min"], dtype=np.float64)
        xyz_max = np.asarray(safety["dataset_xyz_max"], dtype=np.float64)
        if np.any(xyz < xyz_min) or np.any(xyz > xyz_max):
            print(
                "[DRY_RUN][WARNING] current dataset pose is outside the training "
                "workspace; predictions are diagnostic only and may be unreliable",
                flush=True,
            )
    actions, info = base.make_request(cfg, previous, executed, delay, state=state)
    info["_client_timing"].update({
        "set_wobj_s": wobj_ready - started,
        "topic_s": state_ready - wobj_ready,
        "e2e_s": time.monotonic() - started,
    })
    return actions, info


def execution_schedule(cfg, profile):
    """Use the selected model's command rate for its guided delay in steps."""
    hz = float(profile.get("command_hz", cfg["execution"]["command_hz"]))
    if not math.isfinite(hz) or not 0 < hz <= 10:
        raise base.ClientError("execution command_hz must be in (0,10]")
    rtc = cfg["rtc"]
    expected_s = float(profile.get("expected_guided_latency_s", rtc["expected_guided_latency_s"]))
    margin = int(rtc["schedule_margin_steps"])
    if not math.isfinite(expected_s) or expected_s <= 0 or margin < 0:
        raise base.ClientError("RTC expected latency and schedule margin must be positive")
    delay = math.ceil(expected_s * hz) + 1 + margin
    if delay < 1 or delay >= base.HORIZON - 1:
        raise base.ClientError(f"RTC schedule infeasible: delay={delay}, horizon={base.HORIZON}")
    return hz, delay


def run(cfg, dry_run, max_actions, actions_per_inference):
    mode = "DRY_RUN" if dry_run else "REAL"
    # Verify model identity, state/camera contract and RTC semantics before a
    # real command socket is even opened.
    health = base.check_server_health(cfg)
    profile = cfg["inference"]["active_profile"]
    print(
        f"[EXEC][PREFLIGHT] profile={profile['name']} model={health['model']} "
        f"rtc_mode={health['rtc_mode']} state_dim={health['state_dim']} "
        f"cameras={health['camera_order']}", flush=True,
    )
    rpc = None
    if not dry_run:
        direct = cfg["execution"]["direct_rpc"]
        rpc = DirectRobotRpc(
            direct["host"], int(direct["port"]), int(direct["connect_timeout_ms"]),
            int(direct["async_timeout_ms"]),
        )
    actions, initial = inference_request(cfg, dry_run, rpc)
    hz, delay = execution_schedule(cfg, profile)
    period = 1.0 / hz
    max_buffer_wait_s = float(cfg["execution"].get("max_buffer_wait_s", 0.5))
    if not math.isfinite(max_buffer_wait_s) or not 0 <= max_buffer_wait_s <= 2:
        raise base.ClientError("execution.max_buffer_wait_s must be in [0,2]")
    log_dir = Path(cfg["logging"]["directory"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dualarm_lt_rtc_{mode.lower()}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
    print(f"[EXEC] mode={mode} task={cfg['inference']['instruction']}")
    print(
        f"[EXEC] initial={actions.shape} model={initial['inference_elapsed_s']:.3f}s "
        f"hz={hz} max_actions={max_actions} min_actions_per_plan={actions_per_inference} "
        f"configured_delay={delay}"
    )
    index = cycle = sent = 0
    plans = [(0, actions.copy())]
    plan_used = 0
    pending = None
    request_start = 0.0
    request_sent = request_index = request_delay = 0
    event_path = log_path.with_name(log_path.stem + "_events.jsonl")
    try:
      with log_path.open("x") as log, event_path.open("x") as events, ThreadPoolExecutor(max_workers=1) as pool:
        base.log_line(
            events, "initial", cycle=0, sent=0,
            model_s=float(initial["inference_elapsed_s"]),
            client_timing=initial.get("_client_timing", {}),
        )
        while max_actions == 0 or sent < max_actions:
            tick = time.monotonic()
            if pending is not None and pending.done() and pending.exception() is not None:
                pending.result()  # Do not send another motion command after failed inference.
            remaining_total = math.inf if max_actions == 0 else max_actions - sent
            will_need_another_plan = remaining_total > (actions_per_inference - plan_used)
            # Start the next request as soon as the current plan has a valid
            # suffix. Never consume the remaining valid actions while waiting
            # for an inference future at a fixed actions-per-plan boundary.
            if pending is None and will_need_another_plan and index >= 1:
                guided = profile["rtc_mode"] == "guided"
                request_delay = min(delay, base.HORIZON - index) if guided else 0
                if guided and request_delay < 1:
                    raise base.ClientError("RTC has no remaining actions for a guided request")
                request_index = index
                request_sent = sent
                pending = pool.submit(
                    inference_request, cfg, dry_run, rpc,
                    actions[index:].copy() if guided else None,
                    index if guided else None,
                    request_delay if guided else None,
                )
                request_start = tick
                print(
                    f"[RTC] {profile['rtc_mode']} request cycle={cycle} index={index} "
                    f"used={plan_used}/{actions_per_inference} delay={request_delay} "
                    f"buffer={base.HORIZON - index}"
                )
                base.log_line(
                    events, "request", cycle=cycle, sent=sent, index=index,
                    delay_steps=request_delay, buffer_steps=base.HORIZON - index,
                    reduced_delay=guided and request_delay < delay,
                )
            action_for_step, te_count = temporal_ensemble_action(plans, sent, cfg)
            if action_for_step is None:
                action_for_step = actions[index]
            elif te_count > 1:
                print(f"[ACT][TE] global_step={sent} overlapping_plans={te_count}", flush=True)
            live_state, status, target_dataset, command_pose, dos, guards, commands = prepare_action(
                action_for_step, cfg, dry_run
            )
            next_sent = sent + 1
            print_observation(mode, f"PRE-RPC step={next_sent}", live_state, status, cfg)
            print(
                f"[{mode}] step={next_sent} chunk={index + 1}/50 DO8..11={dos} guards={guards}"
                f"\n[{mode}] dataset_target={target_dataset.tolist()}"
                f"\n[{mode}] command_target_frame="
                f"{cfg['frames']['command_pose_frame_with_setwobj']} "
                f"command_target_xyzw={command_pose.tolist()}",
                flush=True,
            )
            for command in commands:
                verb = "WOULD_SEND" if dry_run else "SEND"
                print(f"[{mode}][RPC][{verb}] {command}", flush=True)
            if dry_run:
                response = {"success": True, "dry_run": True, "transport": "none"}
            else:
                submitted = []
                for command in commands:
                    submitted.append(rpc.submit(command))
                response = submitted[-1]
                print(
                    f"[{mode}][RPC][SUBMITTED] {json.dumps(submitted, sort_keys=True)}",
                    flush=True,
                )
            sent = next_sent
            row = {
                "utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "sent_or_previewed": sent,
                "cycle": cycle, "chunk_index": index, "dataset_target": target_dataset.tolist(),
                "command_target": command_pose.tolist(),
                "command_target_frame": cfg["frames"]["command_pose_frame_with_setwobj"],
                "do8_do11": dos, "guards": guards,
                "current_dataset_pose": live_state[:7].tolist(),
                "current_topic_pose": [float(v) for v in status["topic_pose_xyzw"]],
                "current_topic_pose_frame": status["topic_pose_frame"],
                "topic_diagnostics": status.get("topic_diagnostics", {}),
                "compensated_ft": status.get("compensated_ft", {}),
                "current_model_di": {
                    name: float(live_state[7 + i])
                    for i, name in enumerate(cfg["robot"]["digital_inputs"])
                },
                "current_all_io": {
                    name: float(value) for name, value in sorted(status["io_values"].items())
                },
                "rpc_success": response.get("success"), "rpc_dry_run": response.get("dry_run", False),
                "equivalent_commands": commands,
                "rtc_pending": pending is not None,
                "rtc_request_age_s": time.monotonic() - request_start if pending is not None else None,
            }
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            index += 1
            plan_used += 1
            if max_actions != 0 and sent >= max_actions:
                break
            if index >= base.HORIZON and pending is not None and not pending.done():
                wait_started = time.monotonic()
                try:
                    pending.result(timeout=max_buffer_wait_s)
                except FutureTimeoutError:
                    waited = time.monotonic() - wait_started
                    base.log_line(
                        events, "buffer_exhausted", cycle=cycle, sent=sent,
                        request_index=request_index, request_wall_s=time.monotonic() - request_start,
                        waited_s=waited, max_buffer_wait_s=max_buffer_wait_s,
                    )
                    raise base.ClientError(
                        "RTC 50-step buffer exhausted and replacement wait expired; no more motion RPC"
                    )
                waited = time.monotonic() - wait_started
                base.log_line(
                    events, "buffer_wait", cycle=cycle, sent=sent,
                    request_index=request_index, waited_s=waited,
                )
                print(f"[RTC] buffer end: waited {waited:.3f}s for replacement", flush=True)
            steps_since_request = sent - request_sent if pending is not None else 0
            switch_ready = (
                pending is not None and pending.done()
                and steps_since_request >= request_delay
                and (plan_used >= actions_per_inference or index >= base.HORIZON)
            )
            if switch_ready:
                replacement, info = pending.result()
                if not 0 <= steps_since_request < base.HORIZON:
                    raise base.ClientError("RTC replacement has no valid remaining action")
                actions, index = replacement, steps_since_request
                plans.append((sent - index, actions.copy()))
                plans = plans[-int(profile.get("temporal_ensemble", {}).get("max_plans", 1)):]
                elapsed = time.monotonic() - request_start
                timing = info.get("_client_timing", {})
                print(
                    f"[RTC] replacement cycle={cycle + 1} model={info['inference_elapsed_s']:.3f}s "
                    f"e2e={elapsed:.3f}s next_index={index} used={plan_used} "
                    f"timing={json.dumps(timing, sort_keys=True)}"
                )
                base.log_line(
                    events, "replacement", cycle=cycle + 1, sent=sent,
                    request_index=request_index, next_index=index,
                    delay_steps=request_delay, request_wall_s=elapsed,
                    model_s=float(info["inference_elapsed_s"]), client_timing=timing,
                )
                pending = None
                cycle += 1
                plan_used = 0
            elif index >= base.HORIZON:
                elapsed = time.monotonic() - request_start if pending is not None else None
                base.log_line(
                    events, "buffer_exhausted", cycle=cycle, sent=sent,
                    request_index=request_index if pending is not None else None,
                    request_wall_s=elapsed,
                )
                raise base.ClientError(
                    "RTC 50-step buffer exhausted before a valid replacement; no more RPC"
                )
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    except Exception as exc:
        topic_state = cfg.get("_topic_state")
        with event_path.open("a") as events:
            base.log_line(
                events, "client_failure", sent=sent,
                error=f"{type(exc).__name__}: {exc}",
                topic_diagnostics=(
                    {} if topic_state is None else topic_state.diagnostics()
                ),
            )
        raise
    finally:
        if rpc is not None:
            rpc.close()
    print(f"DUALARM_LT_RTC_{mode}_DONE actions={sent} log={log_path} events={event_path}")


def run_nonrtc(cfg, dry_run, max_actions):
    """Execute complete fresh 50-step chunks without guided replacement."""
    mode = "DRY_RUN" if dry_run else "REAL"
    health = base.check_server_health(cfg)
    profile = cfg["inference"]["active_profile"]
    hz = float(profile.get("command_hz", cfg["execution"]["command_hz"]))
    if not math.isfinite(hz) or not 0 < hz <= 10:
        raise base.ClientError("execution command_hz must be in (0,10]")
    period = 1.0 / hz
    print(
        f"[NONRTC][PREFLIGHT] profile={profile['name']} model={health['model']} "
        f"state_dim={health['state_dim']} cameras={health['camera_order']} "
        f"chunk=50 hz={hz} max_actions={max_actions}", flush=True,
    )
    rpc = None
    if not dry_run:
        direct = cfg["execution"]["direct_rpc"]
        rpc = DirectRobotRpc(
            direct["host"], int(direct["port"]), int(direct["connect_timeout_ms"]),
            int(direct["async_timeout_ms"]),
        )
    log_dir = Path(cfg["logging"]["directory"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dualarm_lt_nonrtc_{mode.lower()}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.jsonl"
    event_path = log_path.with_name(log_path.stem + "_events.jsonl")
    sent = cycle = 0
    try:
      with log_path.open("x") as log, event_path.open("x") as events:
        while max_actions == 0 or sent < max_actions:
            # No RPC is sent during inference. The next chunk uses a new
            # observation; no RTC history, skipped index, or temporal blend.
            actions, info = inference_request(cfg, dry_run, rpc)
            if info.get("rtc_guided") is not False or info.get("rtc_request_had_history", False) is not False:
                raise base.ClientError("non-RTC request unexpectedly used guided history")
            base.log_line(
                events, "fresh_chunk", cycle=cycle, sent=sent,
                model_s=float(info["inference_elapsed_s"]),
                client_timing=info.get("_client_timing", {}),
            )
            print(
                f"[NONRTC] cycle={cycle + 1} fresh={actions.shape} "
                f"model={info['inference_elapsed_s']:.3f}s "
                f"client_e2e={info.get('_client_timing', {}).get('e2e_s')}s; "
                "execute up to 50 actions", flush=True,
            )
            for index in range(base.HORIZON):
                if max_actions != 0 and sent >= max_actions:
                    break
                tick = time.monotonic()
                live_state, status, target_dataset, command_pose, dos, guards, commands = prepare_action(
                    actions[index], cfg, dry_run
                )
                next_sent = sent + 1
                print_observation(mode, f"PRE-RPC step={next_sent}", live_state, status, cfg)
                print(
                    f"[NONRTC][{mode}] step={next_sent} cycle={cycle + 1} "
                    f"chunk={index + 1}/50 DO8..11={dos} guards={guards}"
                    f"\n[NONRTC][{mode}] dataset_target={target_dataset.tolist()}"
                    f"\n[NONRTC][{mode}] command_target_xyzw={command_pose.tolist()}",
                    flush=True,
                )
                for command in commands:
                    print(
                        f"[{mode}][RPC][{'WOULD_SEND' if dry_run else 'SEND'}] {command}",
                        flush=True,
                    )
                if dry_run:
                    response = {"success": True, "dry_run": True, "transport": "none"}
                else:
                    submitted = [rpc.submit(command) for command in commands]
                    response = submitted[-1]
                    print(
                        f"[{mode}][RPC][SUBMITTED] {json.dumps(submitted, sort_keys=True)}",
                        flush=True,
                    )
                sent = next_sent
                row = {
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "mode": mode, "execution_mode": "non_rtc", "sent_or_previewed": sent,
                    "cycle": cycle, "chunk_index": index,
                    "dataset_target": target_dataset.tolist(),
                    "command_target": command_pose.tolist(),
                    "command_target_frame": cfg["frames"]["command_pose_frame_with_setwobj"],
                    "do8_do11": dos, "guards": guards,
                    "current_dataset_pose": live_state[:7].tolist(),
                    "current_topic_pose": [float(v) for v in status["topic_pose_xyzw"]],
                    "current_topic_pose_frame": status["topic_pose_frame"],
                    "topic_diagnostics": status.get("topic_diagnostics", {}),
                    "compensated_ft": status.get("compensated_ft", {}),
                    "current_model_di": {
                        name: float(live_state[7 + i])
                        for i, name in enumerate(cfg["robot"]["digital_inputs"])
                    },
                    "current_all_io": {
                        name: float(value) for name, value in sorted(status["io_values"].items())
                    },
                    "rpc_success": response.get("success"),
                    "rpc_dry_run": response.get("dry_run", False),
                    "equivalent_commands": commands,
                }
                log.write(json.dumps(row, allow_nan=False) + "\n")
                log.flush()
                remaining = period - (time.monotonic() - tick)
                if remaining > 0 and index + 1 < base.HORIZON:
                    time.sleep(remaining)
            cycle += 1
    except Exception as exc:
        topic_state = cfg.get("_topic_state")
        with event_path.open("a") as events:
            base.log_line(
                events, "client_failure", sent=sent,
                error=f"{type(exc).__name__}: {exc}",
                topic_diagnostics=(
                    {} if topic_state is None else topic_state.diagnostics()
                ),
            )
        raise
    finally:
        if rpc is not None:
            rpc.close()
    print(f"DUALARM_LT_NONRTC_{mode}_DONE actions={sent} log={log_path} events={event_path}")


def self_test(cfg):
    live = np.asarray([0.23952933, -0.62075734, -0.25309893, 0.63933945, -0.41043657, -0.44729048, 0.47193021] + [0] * 17)
    action = np.r_[live[:7], [1, 0, 1, 0]]
    guarded, dos, _ = guarded_action(action, live, cfg)
    command_pose = dataset_to_command_pose(guarded, cfg)
    assert np.allclose(command_pose, guarded, atol=1e-9)
    assert dos == [1, 0, 1, 0]
    commands = robot_commands(command_pose, dos, cfg)
    assert "--do_num=8" in commands[0] and "--do_num=11" in commands[3]
    assert "--robottarget_value={" in commands[4] and "robottarget_pa_value" not in commands[4]
    assert cfg["execution"]["inference_preflight_wobj"] == "wobj0"
    if cfg["inference"]["active_profile"]["name"] == "act_v2":
        older = np.zeros((50, 11), dtype=np.float32)
        newer = np.zeros((50, 11), dtype=np.float32)
        older[:, 6] = newer[:, 6] = 1.0
        newer[:, 0] = 1.0
        blended, count = temporal_ensemble_action([(0, older), (0, newer)], 3, cfg)
        assert count == 2 and 0.5 < float(blended[0]) < 1.0
        assert np.allclose(np.linalg.norm(blended[3:7]), 1.0, atol=1e-6)
    print("DUALARM_LT_RTC_EXECUTOR_SELF_TEST_PASS; NO NETWORK/ROBOT")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("dualarm_lt_rtc_16.yaml")))
    parser.add_argument("--profile", choices=("qwenpi_v3_v2", "qwenpi_v3_v3", "act_v2"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--real", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    parser.add_argument("--confirm-real", help=argparse.SUPPRESS)
    parser.add_argument("--no-rtc", action="store_true", help="execute each fresh 50-step prediction sequentially")
    parser.add_argument("--max-actions", type=int)
    parser.add_argument("--actions-per-inference", type=int)
    parser.add_argument(
        "--topic-pose-frame-for-dry-run", choices=("auto", "controller", "dataset"),
        help="Explicit RT pose frame for read-only dry-run when workspace-based auto detection is ambiguous",
    )
    args = parser.parse_args()
    cfg = base.load_config(args.config, args.profile)
    if args.topic_pose_frame_for_dry_run is not None:
        if not args.dry_run:
            parser.error("--topic-pose-frame-for-dry-run requires --dry-run")
        cfg["frames"]["topic_pose_frame_without_setwobj"] = args.topic_pose_frame_for_dry_run
        print(
            "[DRY_RUN][FRAME] using explicit Topic frame "
            f"{args.topic_pose_frame_for_dry_run}; no robot RPC",
            flush=True,
        )
    if args.self_test:
        return self_test(cfg)
    max_actions = args.max_actions
    if max_actions is None:
        max_actions = int(cfg["execution"]["default_max_actions"])
    if max_actions < 0:
        raise base.ClientError("--max-actions must be >=0; 0 means until Ctrl+C")
    with base.PersistentTopicState(cfg) as topic_state:
        cfg["_topic_state"] = topic_state
        try:
            if args.no_rtc:
                if args.actions_per_inference is not None and args.actions_per_inference != base.HORIZON:
                    raise base.ClientError("--no-rtc uses all 50 actions per inference")
                return run_nonrtc(cfg, dry_run=args.dry_run, max_actions=max_actions)
            actions_per_inference = args.actions_per_inference
            if actions_per_inference is None:
                actions_per_inference = int(cfg["execution"]["default_actions_per_inference"])
            if not 2 <= actions_per_inference <= 50:
                raise base.ClientError("--actions-per-inference must be in [2, 50]")
            run(
                cfg, dry_run=args.dry_run, max_actions=max_actions,
                actions_per_inference=actions_per_inference,
            )
        finally:
            cfg.pop("_topic_state", None)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        prefix = "DUALARM_LT_NONRTC" if "--no-rtc" in sys.argv else "DUALARM_LT_RTC"
        print(f"{prefix}_OPERATOR_STOP")
    except (base.ClientError, OSError, ValueError) as exc:
        prefix = "DUALARM_LT_NONRTC" if "--no-rtc" in sys.argv else "DUALARM_LT_RTC_EXECUTOR"
        print(f"{prefix}_FAIL_CLOSED {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)

