#!/usr/bin/env python3
"""Download two private ModelScope subtrees without storing a token in code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from modelscope.hub.api import HubApi
from modelscope.hub.snapshot_download import dataset_snapshot_download

REPO = "chaishan/huashu"
PREFIX = "newsmy_usb_h264_640x400_30fps_20261120_4cam"
FOLDERS = {"snap": "Snap-c", "insert": "Insert-c"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=tuple(FOLDERS), default=list(FOLDERS))
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/datasets"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    api = HubApi()
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        api.login(token)
    plans = {}
    for task in args.tasks:
        prefix = f"{PREFIX}/{FOLDERS[task]}"
        target = args.data_root / f"{task}_lt"
        marker = target / ".modelscope_download_complete.json"
        if target.exists():
            if marker.is_file():
                saved = json.loads(marker.read_text())
                if saved.get("repo") == REPO and saved.get("prefix") == prefix:
                    print(f"ALREADY_DOWNLOADED={target}", flush=True)
                    continue
            raise RuntimeError(f"Existing dataset is not this downloader's completed output: {target}")
        files = []
        page = 1
        while True:
            batch = api.get_dataset_files(REPO, root_path=prefix, recursive=True,
                                          page_size=1000, page_number=page, token=token)
            files.extend(f for f in batch if f.get("Type") == "blob")
            if len(batch) < 1000:
                break
            page += 1
        if not files:
            raise RuntimeError(f"Empty dataset subtree: {prefix}")
        plans[task] = (prefix, files)
    needed = sum(int(f["Size"]) for _, files in plans.values() for f in files)
    args.data_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.data_root).free
    # Include conservative allowance for SDK staging plus training checkpoints.
    if free < 2 * needed + 6 * 1024**3:
        raise RuntimeError(f"Insufficient free space: free={free}, download={needed}; retain >=6GiB for outputs")
    for task, (prefix, files) in plans.items():
        stage = args.data_root / ".huashu_snap_insert_download" / task
        stage.mkdir(parents=True, exist_ok=True)
        print(f"DOWNLOAD task={task} files={len(files)} bytes={sum(int(f['Size']) for f in files)}", flush=True)
        dataset_snapshot_download(dataset_id=REPO, revision="master", local_dir=str(stage),
                                  allow_patterns=[prefix + "/**"], max_workers=args.workers,
                                  token=token)
        source = stage / prefix
        for file in files:
            rel = Path(file["Path"]).relative_to(prefix)
            path = source / rel
            if not path.is_file() or path.stat().st_size != int(file["Size"]):
                raise RuntimeError(f"Incomplete download: {path}")
            if file.get("Sha256"):
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b""):
                        digest.update(block)
                if digest.hexdigest() != file["Sha256"]:
                    raise RuntimeError(f"SHA256 mismatch: {path}")
        if not (source / "meta/info.json").is_file():
            raise RuntimeError(f"No LeRobot meta/info.json in {source}")
        (source / ".modelscope_download_complete.json").write_text(json.dumps({
            "repo": REPO, "revision": "master", "prefix": prefix, "files": len(files),
            "bytes": sum(int(f["Size"]) for f in files), "sha256_checked": True,
        }, indent=2) + "\n")
        target = args.data_root / f"{task}_lt"
        if target.exists():
            raise RuntimeError(f"Refusing to replace existing dataset: {target}")
        source.rename(target)
        print(f"DATASET_READY={target}; files={len(files)}; SHA256 verified", flush=True)


if __name__ == "__main__":
    main()
