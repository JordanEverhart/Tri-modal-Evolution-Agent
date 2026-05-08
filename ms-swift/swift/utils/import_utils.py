"""External plugin importer used by ``--external_plugins``."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def import_external_file(path: str | Path) -> ModuleType:
    """Import a Python file for side effects.

    Reward plugins register themselves by mutating ``swift.rewards.orms`` during
    import. This is the only plugin behavior used by this project.
    """

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"external plugin file does not exist: {resolved}")

    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
    module_name = f"swift_external_plugin_{resolved.stem}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import external plugin: {resolved}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_external_plugins(paths: str | Path | list[str | Path] | tuple[str | Path, ...] | None) -> list[ModuleType]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    return [import_external_file(path) for path in paths]
