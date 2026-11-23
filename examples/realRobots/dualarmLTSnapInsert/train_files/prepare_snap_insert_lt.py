#!/usr/bin/env python3
"""Build a small training view; retain the original snap_lt / insert_lt data unchanged."""
import argparse
import json
from pathlib import Path


def link(source, destination):
    if destination.is_symlink() or destination.exists():
        if not destination.is_symlink() or destination.resolve() != source.resolve():
            raise RuntimeError(f"Existing path does not reference the requested source: {destination}")
    else:
        destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/datasets"))
    parser.add_argument("--task", choices=("snap", "insert"), required=True)
    args = parser.parse_args()
    source = args.data_root / f"{args.task}_lt"
    view = args.data_root / f"{args.task}_lt_v2"
    for relative in ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/episodes", "data", "videos"):
        if not (source / relative).exists():
            raise FileNotFoundError(source / relative)
    if view.is_symlink() or view.resolve() == source.resolve():
        raise RuntimeError("Training view must be a separate directory")
    (view / "meta").mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        if child.name != "meta":
            link(child, view / child.name)
    for child in (source / "meta").iterdir():
        if child.name != "modality.json":
            link(child, view / "meta" / child.name)
    expected = Path(__file__).with_name("modality.json").read_text()
    destination = view / "meta/modality.json"
    if destination.is_symlink():
        raise RuntimeError("Refusing to write a symlinked modality.json")
    if destination.exists() and json.loads(destination.read_text()) != json.loads(expected):
        raise RuntimeError(f"Existing mapping differs; not overwriting: {destination}")
    if not destination.exists():
        destination.write_text(expected)
    print(f"DATASET_VIEW_READY={view}; raw dataset unchanged at {source}")


if __name__ == "__main__":
    main()
