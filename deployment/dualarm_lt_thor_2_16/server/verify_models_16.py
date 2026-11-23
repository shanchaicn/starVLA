#!/usr/bin/env python3
"""Verify copied files against SHA256 values captured on 252."""
import hashlib
import json
from pathlib import Path
import sys
import time

import yaml

here = Path(__file__).resolve().parent
cfg = yaml.safe_load((here / "server_16.yaml").read_text())
models = (here / cfg["paths"]["models_root"]).resolve()
manifest = json.loads((here / "models.sha256.json").read_text())
rows = []
for relative, expected in manifest.items():
    path = models / relative
    ok = path.is_file() and path.stat().st_size == expected["bytes"]
    sha = None
    if ok:
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        ok = sha == expected["sha256"]
    rows.append({"file": relative, "ok": ok, "sha256": sha})
    print(relative, "PASS" if ok else "FAIL", flush=True)
result = {"checked_at_unix": time.time(), "all_ok": all(row["ok"] for row in rows), "files": rows}
(here / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
sys.exit(0 if result["all_ok"] else 1)
