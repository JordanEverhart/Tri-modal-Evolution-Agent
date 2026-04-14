from __future__ import annotations

from collections.abc import Callable

from ..types import BenchmarkSample
from .daily_omni import load_daily_omni_samples
from .omnibench import load_omnibench_samples
from .worldsense import load_worldsense_samples


DatasetLoader = Callable[[dict, dict], list[BenchmarkSample]]


DATASET_LOADERS: dict[str, DatasetLoader] = {
    "worldsense": load_worldsense_samples,
    "daily_omni": load_daily_omni_samples,
    "omnibench": load_omnibench_samples,
}


def get_dataset_loader(loader_key: str) -> DatasetLoader:
    try:
        return DATASET_LOADERS[loader_key]
    except KeyError as exc:
        available = ", ".join(sorted(DATASET_LOADERS))
        raise KeyError(f"Unknown dataset loader {loader_key!r}. Available: {available}") from exc
