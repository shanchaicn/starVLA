#!/usr/bin/env python3
"""Freeze installed dependency files for an offline ARM64/Python 3.12 build.

Never copies .pth files or the host torch/CUDA runtime. The pinned NGC image
supplies torch; all other runtime dependencies are included in the image.
"""
from collections import deque
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import shutil

from packaging.requirements import Requirement

root = Path(__file__).resolve().parent.parent
sites = [root / "server/.venv/lib/python3.12/site-packages",
         Path("/home/test/Public/zpf/venvs/lerobot061/lib/python3.12/site-packages")]
vla = Path("/home/test/Public/cs/venv/vla/lib/python3.12/site-packages")


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def snapshot(group, roots, search):
    available = {}
    for site in search:
        for dist in metadata.distributions(path=[str(site)]):
            available.setdefault(canonical(dist.metadata["Name"]), dist)
    dest = root / "docker/context" / ("deps_" + group)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    todo = deque(roots)
    versions = {}
    while todo:
        name = canonical(todo.popleft())
        if name in versions or name in {"torch", "torchvision", "triton"} or name.startswith(("nvidia-", "cuda-")):
            continue
        dist = available[name]
        versions[name] = dist.version
        site = Path(dist.locate_file(".")).resolve()
        for entry in dist.files or []:
            source = Path(dist.locate_file(entry)).resolve()
            if not source.is_file() or not source.is_relative_to(site):
                continue
            rel = source.relative_to(site)
            if source.suffix in {".pth", ".pyc"} or "__pycache__" in rel.parts:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker and not req.marker.evaluate({"extra": ""}):
                continue
            todo.append(req.name)
    (root / "docker" / ("dependencies-" + group + ".json")).write_text(json.dumps(versions, indent=2) + "\n")
    print(group, versions, flush=True)


snapshot("server", ["transformers", "huggingface-hub", "tokenizers", "omegaconf", "antlr4-python3-runtime", "numpy", "pillow", "PyYAML", "einops", "accelerate", "psutil", "rich"], sites)
snapshot("client", ["numpy", "pillow", "PyYAML", "opencv-python-headless", "protobuf", "pyzmq", "tb6r5-get-status"], [vla] + sites)
