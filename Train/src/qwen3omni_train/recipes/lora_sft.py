from __future__ import annotations

from pathlib import Path

from ..paths import TRAIN_ROOT
from ..swift_launcher import LaunchSpec, build_launch_spec, launch


DEFAULT_RECIPE = TRAIN_ROOT / "configs" / "recipes" / "lora_sft.yaml"


def build(recipe: str | Path = DEFAULT_RECIPE) -> LaunchSpec:
    return build_launch_spec(recipe)


def run(recipe: str | Path = DEFAULT_RECIPE, *, execute: bool = False) -> LaunchSpec:
    return launch(recipe, execute=execute)
