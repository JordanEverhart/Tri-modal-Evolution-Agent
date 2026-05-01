from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from .paths import REPO_ROOT, TRAIN_ROOT, WORKSPACE_ROOT


_TEMPLATE_RE = re.compile(r"\$\{([^}]+)\}")


def _context(extra: dict[str, Any] | None = None) -> dict[str, str]:
    ctx = {
        "TRAIN_ROOT": str(TRAIN_ROOT),
        "REPO_ROOT": str(REPO_ROOT),
        "WORKSPACE_ROOT": str(WORKSPACE_ROOT),
    }
    ctx.update({key: str(value) for key, value in os.environ.items()})
    if extra:
        ctx.update({key: str(value) for key, value in extra.items() if value is not None})
    return ctx


def expand_value(value: Any, extra: dict[str, Any] | None = None) -> Any:
    if isinstance(value, dict):
        return {key: expand_value(item, extra=extra) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_value(item, extra=extra) for item in value]
    if not isinstance(value, str):
        return value
    ctx = _context(extra)

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return ctx.get(key, match.group(0))

    return _TEMPLATE_RE.sub(replace, value)


def load_yaml(path: str | Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {resolved}")
    return expand_value(payload, extra=extra)
