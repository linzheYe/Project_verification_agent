#!/usr/bin/env python3
from __future__ import annotations

"""
Batch launcher for model-response jsonl files.

Parameter ownership:
- This file only controls batch discovery/scheduling.
- Do NOT define clean/split/prefetch/retrieve thresholds here.
- Business params live in runs/run_simpleqa_entry.py.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Directory to scan for model-response JSONL files (relative to project root).
MODELS_DIR = Path("data/models")
# Python module entrypoint invoked for each pending model file.
ENTRY_MODULE = "runs.run_simpleqa_entry"


def _to_run_name(path: Path) -> str:
    stem = path.stem.strip()
    if "_" in stem:
        stem = stem.split("_", 1)[1]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", stem).strip("._-") or "model_run"


def main() -> int:
    models_dir = PROJECT_ROOT / MODELS_DIR
    if not models_dir.exists() or not models_dir.is_dir():
        print(f"[error] models dir not found: {models_dir}", file=sys.stderr)
        return 2

    jsonl_files = sorted(models_dir.glob("*.jsonl"))
    run_pairs: list[tuple[Path, str, Path]] = []
    for jsonl_path in jsonl_files:
        run_name = _to_run_name(jsonl_path)
        out_dir = models_dir / run_name
        if out_dir.exists():
            print(f"[skip] {jsonl_path.name} -> {out_dir.name}/ exists")
            continue
        run_pairs.append((jsonl_path, run_name, out_dir))

    print(f"[info] total_jsonl={len(jsonl_files)} pending={len(run_pairs)}")
    failed: list[tuple[str, int]] = []

    for idx, (jsonl_path, run_name, out_dir) in enumerate(run_pairs, start=1):
        print(f"[{idx}/{len(run_pairs)}] run={run_name} input={jsonl_path.name}")
        out_dir.mkdir(parents=True, exist_ok=False)

        env = os.environ.copy()
        env["SIMPLEQA_INPUT_MODEL_JSONL"] = str(jsonl_path)
        env["SIMPLEQA_RUN_NAME_OVERRIDE"] = run_name

        cmd = [sys.executable, "-m", ENTRY_MODULE]
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            failed.append((jsonl_path.name, proc.returncode))

    print(f"[summary] attempted={len(run_pairs)} failed={len(failed)}")
    for name, code in failed:
        print(f"  - {name}: exit_code={code}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
