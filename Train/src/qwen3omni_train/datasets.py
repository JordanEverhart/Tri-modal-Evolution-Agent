from __future__ import annotations

from pathlib import Path
from typing import Any


def enabled_paths(dataset_config: dict[str, Any], stage: str, split: str) -> list[str]:
    stage_cfg = dataset_config.get("datasets", {}).get(stage, {})
    rows = stage_cfg.get(split, [])
    if not isinstance(rows, list):
        raise ValueError(f"datasets.{stage}.{split} must be a list")
    paths: list[str] = []
    for row in rows:
        if not row or not row.get("enabled", True):
            continue
        path = row.get("path")
        if not path:
            raise ValueError(f"Missing path in datasets.{stage}.{split}: {row}")
        paths.append(str(Path(path).expanduser()))
    return paths


def conversion_job(dataset_config: dict[str, Any], job: str) -> dict[str, Any]:
    jobs = dataset_config.get("conversion_jobs", {})
    if job not in jobs:
        available = ", ".join(sorted(jobs))
        raise KeyError(f"Unknown conversion job {job!r}. Available: {available}")
    return jobs[job]
