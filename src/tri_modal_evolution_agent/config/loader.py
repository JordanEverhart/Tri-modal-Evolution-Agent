from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_path(value: Any, *, base_dir: Path) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_path(item, base_dir=base_dir) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_path(item, base_dir=base_dir) for item in value]
    if not isinstance(value, str):
        return value
    if value.startswith("${CONFIG_DIR}"):
        suffix = value.replace("${CONFIG_DIR}", "", 1).lstrip("/")
        return str((base_dir / suffix).resolve())
    if value.startswith("${REPO_ROOT}"):
        repo_root = base_dir.parents[1]
        suffix = value.replace("${REPO_ROOT}", "", 1).lstrip("/")
        return str((repo_root / suffix).resolve())
    return value


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {config_path} must be a mapping.")
    return _resolve_path(raw, base_dir=config_path.parent)
