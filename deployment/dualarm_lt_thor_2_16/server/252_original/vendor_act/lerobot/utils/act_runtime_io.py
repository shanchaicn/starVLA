"""Small inference-only helpers used to avoid importing dataset/pyarrow code."""
from __future__ import annotations

import json


def flatten_dict(value, parent_key="", sep="/"):
    result = {}
    for key, item in value.items():
        full = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(item, dict):
            result.update(flatten_dict(item, full, sep))
        else:
            result[full] = item
    return result


def unflatten_dict(value, sep="/"):
    result = {}
    for key, item in value.items():
        cursor = result
        parts = str(key).split(sep)
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = item
    return result


def write_json(value, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def deserialize_json_into_object(path, obj):
    """Training/checkpoint helper retained for import compatibility only."""
    with open(path, encoding="utf-8") as handle:
        values = json.load(handle)
    for key, value in values.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
    return obj
