#!/usr/bin/env python3
"""Create a task-filtered Push or Pull LeRobot v3 view without copying media."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


TASKS = {
    "push": "Use the robot's left arm to slowly push in the charger box drawer.",
    "pull": "Use the robot's left arm to slowly pull out the charger box drawer.",
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def link_exact(source: Path, destination: Path):
    source = source.resolve()
    if destination.is_symlink():
        require(destination.resolve() == source, f"Wrong existing link: {destination}")
        return
    require(not destination.exists(), f"Refusing to replace existing path: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def write_json_exact(path: Path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        require(path.read_text() == text, f"Existing derived metadata differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def write_parquet_exact(path: Path, frame: pd.DataFrame):
    if path.exists():
        current = pd.read_parquet(path)
        require(current.equals(frame), f"Existing derived metadata differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("/root/data/datasets"))
    args = parser.parse_args()

    source = args.data_root / "push_pull_lt"
    view = args.data_root / f"{args.task}_lt_v2"
    prompt = TASKS[args.task]
    for relative in ("meta/info.json", "meta/tasks.parquet", "meta/episodes", "data", "videos"):
        require((source / relative).exists(), f"Missing source data: {source / relative}")
    require(not view.is_symlink(), f"Filtered view must be a real directory: {view}")

    episode_parts = []
    for path in sorted((source / "meta/episodes").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        selected = frame[frame["tasks"].map(lambda values: prompt in set(values))]
        if not selected.empty:
            episode_parts.append(selected)
    require(episode_parts, f"No {args.task} episodes found")
    episodes = pd.concat(episode_parts).sort_values("episode_index")
    require(episodes["episode_index"].is_unique, "Duplicate episode indices")
    require(episodes["tasks"].map(lambda values: list(values) == [prompt]).all(), "Ambiguous task labels")

    selected_ids = set(int(value) for value in episodes["episode_index"])
    linked_ids = set()
    for path in sorted((source / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "task_index"])
        episode_ids = set(int(value) for value in table["episode_index"].to_pylist())
        require(len(episode_ids) == 1, f"Expected one episode per file: {path}")
        episode_id = next(iter(episode_ids))
        if episode_id not in selected_ids:
            continue
        task_indices = set(int(value) for value in table["task_index"].to_pylist())
        require(len(task_indices) == 1, f"Mixed task indices: {path}")
        relative = path.relative_to(source)
        link_exact(path, view / relative)
        linked_ids.add(episode_id)
    require(linked_ids == selected_ids, f"Missing data files for episodes: {sorted(selected_ids - linked_ids)}")

    # Video lookup uses the original chunk/file indices stored in episode metadata.
    # Linking the media root keeps this view small; only selected trajectories are loaded.
    link_exact(source / "videos", view / "videos")

    tasks = pd.read_parquet(source / "meta/tasks.parquet")
    require(prompt in tasks.index, f"Task prompt missing from tasks.parquet: {prompt}")
    write_parquet_exact(view / "meta/tasks.parquet", tasks.loc[[prompt]])
    write_parquet_exact(view / "meta/episodes/chunk-000/file-000.parquet", episodes)

    info = json.loads((source / "meta/info.json").read_text())
    info["total_episodes"] = int(len(episodes))
    info["total_frames"] = int(episodes["length"].sum())
    info["total_tasks"] = 1
    info["splits"] = {"train": f"0:{len(episodes)}"}
    write_json_exact(view / "meta/info.json", info)

    modality = json.loads(Path(__file__).with_name("modality.json").read_text())
    write_json_exact(view / "meta/modality.json", modality)
    write_json_exact(view / "meta/filter_manifest.json", {
        "source": str(source.resolve()),
        "task": args.task,
        "prompt": prompt,
        "episodes": int(len(episodes)),
        "frames": int(episodes["length"].sum()),
        "episode_indices": sorted(selected_ids),
    })
    print(
        f"FILTERED_DATASET_READY task={args.task} path={view} "
        f"episodes={len(episodes)} frames={int(episodes['length'].sum())}"
    )


if __name__ == "__main__":
    main()
