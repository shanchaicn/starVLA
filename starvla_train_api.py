"""StarVLA training API for the qwenpi3.txt recipe."""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import signal
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal

import yaml
from fastapi import FastAPI, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

REPO = Path(__file__).resolve().parent
TRAIN_FILES = REPO / "examples/realRobots/dualarmLT/train_files"
LAUNCHER = TRAIN_FILES / "train_qwenpi_v3_v2_h50.sh"
TRAIN_RECIPE = TRAIN_FILES / "qwenpi_v3_v2_h50.yaml"
TRAIN_PYTHON = REPO / ".venv/bin/python"
TRAIN_ACCELERATE = REPO / ".venv/bin/accelerate"
CONFIG_ROOT = REPO / "starvla_api_configs"
JOB_ROOT = Path("/root/data/outputs/starVLA/api-jobs")
DATA_ROOT = Path("/root/data/datasets")
HOST = socket.gethostname()
BOOT_ID = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")]
ConfigPath = Annotated[str, Field(min_length=1, max_length=4096)]
CONFIG_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}\.ya?ml")
COMMON_CONFIG_FIELDS = {
    "config_name": "profile",
    "batch_size": "per_device_batch_size",
    "num_train_steps": "max_train_steps",
}
DeviceList = Annotated[str, Field(pattern=r"^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$", max_length=128)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]
Tail = Annotated[int, Query(ge=1, le=5000)]

LOSS_STEP = re.compile(r"\bStep\s+(\d+),\s*Loss:")
LOSS_SOURCE = re.compile(r"(?:^|\s+)[\w./-]+\.py:\d+\s*$")
LOSS_VALUE = re.compile(
    r"""^\{[^{}]*?['"]action_dit_loss['"]\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)(?=\s*[,}])"""
)
MAX_LOSS_RECORD = 8192
CHECKPOINT = re.compile(r"steps_(\d+)_(?:pytorch_model\.pt|model\.safetensors)$")
ACTIVE = {"starting", "running", "stopping"}
LOCK = threading.RLock()
PROCESSES: dict[str, subprocess.Popen] = {}

app = FastAPI(title="StarVLA Training API", version="1.0")


Rate = Annotated[float, Field(gt=0, allow_inf_nan=False)]


class LearningRates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base: Rate
    qwen_vl_interface: Rate
    action_model: Rate
    project_layers: Rate

    @field_validator("*", mode="before")
    @classmethod
    def reject_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("learning rates must be numbers")
        return value


def recipe_learning_rates() -> LearningRates:
    recipe = yaml.safe_load(TRAIN_RECIPE.read_text(encoding="utf-8"))
    return LearningRates.model_validate(recipe["trainer"]["learning_rate"])


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: Literal["qwenpi_v3_v2_h50"] = "qwenpi_v3_v2_h50"
    exp_name: Name = "dualarm_lt_suck"
    cuda_visible_devices: DeviceList = "0,1"
    num_processes: PositiveInt = 2
    per_device_batch_size: PositiveInt = 4
    max_train_steps: PositiveInt = 20000
    learning_rate: LearningRates = Field(default_factory=recipe_learning_rates)

    @model_validator(mode="before")
    @classmethod
    def normalize_common_fields(cls, data):
        if not isinstance(data, dict):
            return data
        values = dict(data)
        for common, native in COMMON_CONFIG_FIELDS.items():
            if common in values:
                value = values.pop(common)
                if native in values and values[native] != value:
                    raise ValueError(f"{common} and {native} must have the same value")
                values[native] = value
        return values

    @model_validator(mode="after")
    def check_devices(self) -> TrainConfig:
        devices = self.cuda_visible_devices.split(",")
        if len(devices) != len(set(devices)) or self.num_processes != len(devices):
            raise ValueError("num_processes must match the number of distinct visible GPU IDs")
        return self


class YamlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml_path: ConfigPath | None = None
    content: Annotated[str, Field(max_length=65536)]


class YamlPatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml_path: ConfigPath | None = None
    updates: dict[str, Any]


class StartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    yaml_path: ConfigPath | None = None
    job_id: Name | None = None
    dry_run: bool = False


def atomic_write(path: Path, content: str) -> None:
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def config_path(name: str | None) -> Path:
    root = CONFIG_ROOT.resolve()
    try:
        path = (root / (name or "default.yaml")).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, "Invalid YAML path") from exc
    if path.parent != root or CONFIG_FILENAME.fullmatch(path.name) is None:
        raise HTTPException(400, "YAML file must be under the StarVLA configuration root")
    return path


def public_config(config: TrainConfig) -> dict:
    data = config.model_dump()
    return {**data, **{common: data[native] for common, native in COMMON_CONFIG_FIELDS.items()}}


def effective_yaml(config: TrainConfig) -> str:
    data = config.model_dump()
    for common, native in COMMON_CONFIG_FIELDS.items():
        data[common] = data.pop(native)
    return yaml.safe_dump(data, sort_keys=False)


def parse_config(content: str) -> TrainConfig:
    if len(content) > 65536:
        raise HTTPException(422, "YAML exceeds 65536 characters")
    try:
        data = yaml.safe_load(content)
        if not isinstance(data, dict):
            raise ValueError("YAML must contain a configuration mapping")
        if isinstance(data.get("learning_rate"), dict):
            data["learning_rate"] = {**recipe_learning_rates().model_dump(), **data["learning_rate"]}
        return TrainConfig.model_validate(data)
    except (ValidationError, yaml.YAMLError, ValueError, RecursionError) as exc:
        raise HTTPException(422, str(exc)) from exc


def read_config(name: str | None) -> tuple[str, TrainConfig]:
    path = config_path(name)
    if not path.is_file():
        if path.name == "default.yaml":
            defaults = TrainConfig()
            return effective_yaml(defaults), defaults
        raise HTTPException(404, "YAML file not found")
    with path.open(encoding="utf-8") as handle:
        content = handle.read(65537)
    return content, parse_config(content)


def launch_command(config: TrainConfig, job_id: str, mode: str) -> list[str]:
    command = [
        str(LAUNCHER), mode,
        "--num-processes", str(config.num_processes),
        "--batch-size", str(config.per_device_batch_size),
        "--steps", str(config.max_train_steps),
        "--run-root-dir", str(JOB_ROOT),
        "--run-id", job_id,
    ]
    for group, rate in config.learning_rate.model_dump().items():
        command.extend(("--learning-rate", group, str(rate)))
    return command


def train_env(config: TrainConfig) -> dict[str, str]:
    return {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": config.cuda_visible_devices,
        "DATA_ROOT_DIR": str(DATA_ROOT),
        "PYTHON_BIN": str(TRAIN_PYTHON),
        "ACCELERATE_BIN": str(TRAIN_ACCELERATE),
        "WANDB_MODE": "disabled",
    }


def save_job(job: dict) -> None:
    atomic_write(JOB_ROOT / job["job_id"] / "job.json", json.dumps(job, indent=2) + "\n")


def load_job(job_id: str) -> dict:
    path = JOB_ROOT / job_id / "job.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(404, "Job not found") from exc


def process_token(pid: int) -> str | None:
    try:
        with Path(f"/proc/{pid}/stat").open() as handle:
            fields = handle.read().rsplit(")", 1)[1].split()
        if fields[0] != "Z" and int(fields[2]) == pid:
            return f"{BOOT_ID}:{fields[19]}"
    except (OSError, ValueError, IndexError):
        pass
    return None


def local_process_alive(job: dict) -> bool:
    token = job.get("process_token")
    return bool(
        token and job.get("host") == HOST and token == process_token(job["pid"])
    )


def read_status(job: dict) -> str:
    status = job["status"]
    if (status in ACTIVE and job.get("host") == HOST
            and job["job_id"] not in PROCESSES and not local_process_alive(job)):
        return "interrupted"
    return status


def finish_job(job: dict, process: subprocess.Popen) -> None:
    code = process.wait()
    with LOCK:
        job["returncode"] = code
        job["finished_at"] = time.time()
        try:
            last_status = load_job(job["job_id"])["status"]
            job["status"] = (
                "stopped" if last_status == "stopping" else ("succeeded" if code == 0 else "failed")
            )
            save_job(job)
        finally:
            PROCESSES.pop(job["job_id"], None)


def terminate_if_running(job: dict, process: subprocess.Popen | None) -> None:
    if process is not None:
        try:
            process.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            pass
    else:
        for _ in range(20):
            if not local_process_alive(job):
                break
            time.sleep(1)
    if local_process_alive(job):
        try:
            os.killpg(job["pid"], signal.SIGKILL)
        except ProcessLookupError:
            pass

    if process is None:
        for _ in range(5):
            if not local_process_alive(job):
                break
            time.sleep(1)
        with LOCK:
            if not local_process_alive(job):
                job.update(status="stopped", finished_at=time.time())
                save_job(job)


@contextmanager
def launch_lock():
    JOB_ROOT.mkdir(parents=True, exist_ok=True)
    with (JOB_ROOT / f".launch-{HOST}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def metrics(log_path: Path, step: int | None = None) -> dict | None:
    latest = None
    if not log_path.is_file():
        return latest
    current = None
    payload = ""
    skipping_line = False
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in iter(lambda: handle.readline(MAX_LOSS_RECORD + 1), ""):
            if skipping_line or len(line) > MAX_LOSS_RECORD:
                skipping_line = not line.endswith("\n")
                current = None
                continue
            header = LOSS_STEP.search(line)
            if header:
                current = int(header[1])
                payload = ""
                part = line[header.end():]
            elif current is not None and line[:1].isspace():
                part = line
            else:
                current = None
                continue
            if step is not None and current != step:
                current = None
                continue
            # Rich puts the source location on the first row and indents wrapped rows.
            part = LOSS_SOURCE.sub("", part).strip()
            if not part:
                continue
            if not payload and not part.startswith("{"):
                current = None
                continue
            if len(payload) + len(part) + 1 > MAX_LOSS_RECORD:
                current = None
                continue
            payload = (payload + " " + part) if payload else part
            match = LOSS_VALUE.match(payload)
            if match:
                loss = float(match[1])
                if math.isfinite(loss):
                    latest = {"step": current, "action_dit_loss": loss}
                current = None
            elif "}" in payload or "{" in payload[1:]:
                current = None
    return latest


def checkpoints(job_id: str, status: str) -> dict:
    output = JOB_ROOT / job_id
    completed_steps = set()
    summary = output / "summary.jsonl"
    if summary.is_file():
        with summary.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and type(value.get("steps")) is int:
                    completed_steps.add(value["steps"])
    artifacts = []
    checkpoint_dir = output / "checkpoints"
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.iterdir():
            match = CHECKPOINT.fullmatch(path.name)
            if match and int(match[1]) in completed_steps and path.is_file():
                artifacts.append({"step": int(match[1]), "path": str(path)})
    artifacts.sort(key=lambda item: item["step"])
    final = next(
        (str(path) for name in ("pytorch_model.pt", "model.safetensors")
         if (path := output / "final_model" / name).is_file()),
        None,
    ) if status == "succeeded" else None
    return {"checkpoints": artifacts, "final_model": final}


def job_identity(job: dict) -> dict:
    config = job["config"]
    return {
        "job_id": job["job_id"], "config_name": config["profile"], "exp_name": config["exp_name"],
        "status": read_status(job), "checkpoint_base_dir": str(JOB_ROOT),
        "checkpoint_dir": job["run_dir"], "log_path": str(Path(job["run_dir"]) / "train.log"),
    }


def status_payload(job: dict) -> dict:
    identity = job_identity(job)
    current = metrics(Path(identity["log_path"]))
    total = job["config"]["max_train_steps"]
    step = total if identity["status"] == "succeeded" else (current["step"] if current else 0)
    loss = current["action_dit_loss"] if current else None
    metric = {**current, "loss": loss} if current else None
    return {
        **job, **identity, "step": step, "total_steps": total,
        "current_step": step, "total_step": total, "last_loss_step": current["step"] if current else None,
        "progress": round(min(step / total, 1) * 100, 2),
        "loss": loss, "metric": metric, "latest_loss": current,
    }


@app.get("/starvla/health")
def health() -> dict:
    return {"status": "ok", "backend": "starvla"}


@app.get("/starvla/train/yaml/files")
def list_configs() -> dict:
    files = {"default.yaml"}
    if CONFIG_ROOT.is_dir():
        files.update(
            path.name for path in CONFIG_ROOT.iterdir()
            if path.is_file() and not path.is_symlink() and CONFIG_FILENAME.fullmatch(path.name)
        )
    return {
        "yaml_root": str(CONFIG_ROOT.resolve()),
        "count": len(files),
        "files": [{"filename": name, "yaml_path": str(config_path(name))} for name in sorted(files)],
    }


@app.get("/starvla/train/yaml")
def get_config(yaml_path: ConfigPath | None = None) -> dict:
    content, config = read_config(yaml_path)
    return {
        "yaml_path": str(config_path(yaml_path)), "content": content,
        "config": public_config(config),
    }


@app.post("/starvla/train/yaml/validate")
def validate_yaml(request: YamlRequest) -> dict:
    return {
        "valid": True, "yaml_path": str(config_path(request.yaml_path)),
        "config": public_config(parse_config(request.content)),
    }


def write_config(yaml_path: str | None, content: str, config: TrainConfig) -> dict:
    path = config_path(yaml_path)
    with LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, content)
    return {"yaml_path": str(path), "content": content, "config": public_config(config)}


@app.put("/starvla/train/yaml")
def put_config(request: YamlRequest) -> dict:
    return write_config(request.yaml_path, request.content, parse_config(request.content))


@app.patch("/starvla/train/yaml")
def patch_config(request: YamlPatchRequest) -> dict:
    with LOCK:
        _, original = read_config(request.yaml_path)
        fields = original.model_dump()
        updates = dict(request.updates)
        for common, native in COMMON_CONFIG_FIELDS.items():
            if common in updates:
                fields.pop(native)
        if isinstance(updates.get("learning_rate"), dict):
            updates["learning_rate"] = {**fields["learning_rate"], **updates["learning_rate"]}
        config = parse_config(yaml.safe_dump({**fields, **updates}))
        return write_config(request.yaml_path, effective_yaml(config), config)


@app.get("/starvla/train/yaml/download")
def download_config(yaml_path: ConfigPath | None = None) -> Response:
    path = config_path(yaml_path)
    content, _ = read_config(yaml_path)
    return Response(
        content, media_type="application/yaml",
        headers={"Content-Disposition": f'attachment; filename="{path.name}"', "Cache-Control": "no-store"},
    )


@app.post("/starvla/train/start", status_code=202)
def start_train(request: StartRequest | None = None) -> dict:
    request = request or StartRequest()
    _, config = read_config(request.yaml_path)
    job_id = request.job_id or uuid.uuid4().hex[:12]
    command = launch_command(config, job_id, "--run")
    env = train_env(config)
    run_dir = JOB_ROOT / job_id
    preview = {
        "job_id": job_id, "yaml_path": str(config_path(request.yaml_path)),
        "command": command, "cwd": str(REPO),
        "env": {"CUDA_VISIBLE_DEVICES": config.cuda_visible_devices},
        "config": public_config(config), "run_dir": str(run_dir),
    }
    if request.dry_run:
        try:
            checked = subprocess.run(
                launch_command(config, job_id, "--dry-run"),
                cwd=REPO, env=env, capture_output=True, text=True, timeout=90, check=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, "Training preflight timed out") from exc
        except subprocess.CalledProcessError as exc:
            raise HTTPException(422, (exc.stderr or exc.stdout or "Training preflight failed")[-4000:]) from exc
        except OSError as exc:
            raise HTTPException(503, "StarVLA launcher is unavailable") from exc
        return {**preview, "dry_run": True, "preflight": checked.stdout}
    if not any(Path("/dev").glob("nvidia[0-9]*")):
        raise HTTPException(503, "No NVIDIA GPU is visible in the training container")
    with LOCK, launch_lock():
        for manifest in JOB_ROOT.glob("*/job.json"):
            existing = load_job(manifest.parent.name)
            if existing.get("host") == HOST and read_status(existing) in ACTIVE:
                raise HTTPException(409, f"StarVLA training slot already in use by {existing['job_id']}")
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise HTTPException(409, "Job ID already exists") from exc
        config_dict = config.model_dump()
        job = {
            "job_id": job_id, "host": HOST, "status": "starting", "created_at": time.time(),
            "started_at": None, "finished_at": None, "pid": None, "returncode": None,
            "command": command, "config": config_dict, "yaml_path": preview["yaml_path"],
            "run_dir": str(run_dir),
        }
        atomic_write(run_dir / "train.yaml", yaml.safe_dump(config_dict, sort_keys=False))
        save_job(job)
        try:
            with (run_dir / "train.log").open("ab", buffering=0) as log:
                process = subprocess.Popen(
                    command, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
        except OSError as exc:
            job.update(status="failed", finished_at=time.time(), error=str(exc))
            save_job(job)
            raise HTTPException(503, "StarVLA launcher failed to start") from exc
        PROCESSES[job_id] = process
        job.update(
            status="running", started_at=time.time(), pid=process.pid,
            process_token=process_token(process.pid),
        )
        try:
            save_job(job)
        except OSError:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            raise
        finally:
            threading.Thread(target=finish_job, args=(job, process), daemon=True).start()
        return {**preview, "status": "running", "config_name": config.profile, "exp_name": config.exp_name}


def matching_jobs(config_name: str | None = None, exp_name: str | None = None) -> list[dict]:
    jobs = [load_job(path.parent.name) for path in JOB_ROOT.glob("*/job.json")]
    jobs = [
        job for job in jobs
        if (config_name is None or job["config"]["profile"] == config_name)
        and (exp_name is None or job["config"]["exp_name"] == exp_name)
    ]
    return sorted(jobs, key=lambda item: item["created_at"], reverse=True)


def resolve_job(config_name: str, exp_name: str, job_id: str | None) -> dict:
    if job_id is not None:
        job = load_job(job_id)
        identity = job_identity(job)
        if identity["config_name"] == config_name and identity["exp_name"] == exp_name:
            return job
    else:
        jobs = matching_jobs(config_name, exp_name)
        if jobs:
            return jobs[0]
    raise HTTPException(404, "Job not found")


@app.get("/starvla/jobs/search")
def search_jobs(yaml_path: ConfigPath) -> dict:
    _, config = read_config(yaml_path)
    jobs = matching_jobs(config.profile, config.exp_name)
    return {"count": len(jobs), "jobs": [
        {"job_id": job["job_id"], "config_name": config.profile, "exp_name": config.exp_name} for job in jobs
    ]}


@app.get("/starvla/jobs")
def list_jobs(config_name: Name | None = None, exp_name: Name | None = None) -> dict:
    jobs = matching_jobs(config_name, exp_name)
    return {
        "config_name": config_name, "exp_name": exp_name, "count": len(jobs),
        "job_ids": [job["job_id"] for job in jobs], "jobs": [{**job, **job_identity(job)} for job in jobs],
    }


@app.get("/starvla/jobs/{job_id}")
@app.get("/starvla/jobs/{job_id}/job.json")
def get_job(job_id: Name) -> dict:
    return status_payload(load_job(job_id))


@app.get("/starvla/jobs/{job_id}/logs")
def get_logs(job_id: Name, tail: Tail = 200) -> dict:
    identity = job_identity(load_job(job_id))
    path = Path(identity["log_path"])
    lines = deque(maxlen=tail)
    if path.is_file():
        with path.open(encoding="utf-8", errors="replace") as handle:
            lines.extend(handle)
    return {**identity, "logs": "".join(lines)}


@app.get("/starvla/jobs/{job_id}/loss")
def get_loss(job_id: Name, step: Annotated[int | None, Query(ge=0)] = None) -> dict:
    identity = job_identity(load_job(job_id))
    result = metrics(Path(identity["log_path"]), step)
    if result is None:
        raise HTTPException(404, "Loss record not found")
    return {**identity, **result, "loss": result["action_dit_loss"]}


@app.get("/starvla/jobs/{job_id}/checkpoints")
def get_checkpoints(job_id: Name) -> dict:
    identity = job_identity(load_job(job_id))
    artifacts = checkpoints(job_id, identity["status"])
    items = [
        {**item, "checkpoint_path": item["path"], "params_path": None,
         "weight_path": item["path"], "completion_verified": True}
        for item in artifacts["checkpoints"]
    ]
    latest = items[-1] if items else None
    latest_path = latest["path"] if latest else None
    return {
        **identity, **artifacts, "checkpoints": items, "count": len(items),
        "latest_step": latest["step"] if latest else None,
        "latest_checkpoint_path": latest_path, "latest_complete_checkpoint_path": latest_path,
        "latest_params_path": None, "latest_complete_params_path": None,
        "weight_path": artifacts["final_model"] or latest_path,
    }


@app.post("/starvla/jobs/{job_id}/stop")
def stop_job(job_id: Name) -> dict:
    with LOCK:
        job = load_job(job_id)
        status = read_status(job)
        if status != "running":
            return {"job_id": job_id, "status": status}
        if not local_process_alive(job):
            raise HTTPException(409, "Training process has already exited or belongs to another container")
        job["status"] = "stopping"
        save_job(job)
        try:
            os.killpg(job["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        threading.Thread(
            target=terminate_if_running,
            args=(job, PROCESSES.get(job_id)),
            daemon=True,
        ).start()
        return {"job_id": job_id, "status": "stopping"}


@app.get("/starvla/train/logs")
def training_logs(
    config_name: Name, exp_name: Annotated[list[Name], Query()],
    job_id: Name | None = None, tail: Tail = 200,
) -> dict:
    experiments = set(exp_name)
    if len(experiments) == 1:
        job = resolve_job(config_name, next(iter(experiments)), job_id)
    elif job_id is not None:
        job = load_job(job_id)
        identity = job_identity(job)
        if identity["config_name"] != config_name or identity["exp_name"] not in experiments:
            raise HTTPException(404, "Job not found")
    else:
        raise HTTPException(422, "job_id is required when filtering multiple experiments")
    return get_logs(job["job_id"], tail)


@app.get("/starvla/train/current")
@app.get("/starvla/train/status")
@app.get("/starvla/train/progress")
def training_status(config_name: Name, exp_name: Name, job_id: Name | None = None) -> dict:
    job = resolve_job(config_name, exp_name, job_id)
    payload = status_payload(job)
    weights = get_checkpoints(job["job_id"])
    return {**payload, "weights": weights, "weight_path": weights["weight_path"]}


@app.get("/starvla/train/loss")
def training_loss(config_name: Name, exp_name: Name, step: Annotated[int, Query(ge=0)], job_id: Name | None = None) -> dict:
    return get_loss(resolve_job(config_name, exp_name, job_id)["job_id"], step)


@app.get("/starvla/train/weights")
@app.get("/starvla/train/checkpoints")
def training_weights(config_name: Name, exp_name: Name, job_id: Name | None = None) -> dict:
    return get_checkpoints(resolve_job(config_name, exp_name, job_id)["job_id"])


@app.get("/starvla/train/folders")
def training_folders(config_name: Name) -> dict:
    return {"config_name": config_name, "folders": sorted({
        job["config"]["exp_name"] for job in matching_jobs(config_name)
    })}


@app.post("/starvla/train/stop")
def training_stop(config_name: Name, exp_name: Name, job_id: Name | None = None) -> dict:
    return stop_job(resolve_job(config_name, exp_name, job_id)["job_id"])


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description=app.title)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
