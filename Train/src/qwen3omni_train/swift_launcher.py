from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import expand_value, load_yaml
from .datasets import enabled_paths


@dataclass(frozen=True)
class LaunchSpec:
    env: dict[str, str]
    command: list[str]
    output_dir: Path
    run_name: str


def _bool_str(value: bool) -> str:
    return "true" if value else "false"


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return _bool_str(value)
    return str(value)


def _append_args(command: list[str], args: dict[str, Any]) -> None:
    for key, value in args.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, list):
            if not value:
                continue
            command.append(flag)
            command.extend(_stringify(item) for item in value)
        else:
            command.extend([flag, _stringify(value)])


def _load_recipe(recipe_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_recipe = load_yaml(recipe_path)
    run_name = raw_recipe.get("run_name") or f"{raw_recipe.get('method', 'run')}_{time.strftime('%Y%m%d_%H%M%S')}"
    recipe = expand_value(raw_recipe, extra={"RUN_NAME": run_name})
    env_config = load_yaml(recipe["env_config"], extra={"RUN_NAME": run_name})
    model_config = load_yaml(recipe["model_config"], extra={"RUN_NAME": run_name})
    dataset_config = load_yaml(recipe["dataset_config"], extra={"RUN_NAME": run_name})
    return recipe, env_config, model_config, dataset_config


def build_launch_spec(recipe_path: str | Path) -> LaunchSpec:
    recipe, env_config, model_config, dataset_config = _load_recipe(recipe_path)
    run_name = str(recipe["run_name"])
    model = model_config["model"]
    thinker = model_config.get("thinker_only", {})
    lora = model_config.get("lora", {})
    stage = str(recipe["dataset_stage"])

    env = {key: _stringify(value) for key, value in env_config.get("env", {}).items()}
    runtime = env_config.get("runtime", {})
    env["CUDA_VISIBLE_DEVICES"] = _stringify(runtime.get("cuda_visible_devices", os.environ.get("CUDA_VISIBLE_DEVICES", "0")))
    env["NPROC_PER_NODE"] = _stringify(runtime.get("nproc_per_node", os.environ.get("NPROC_PER_NODE", "1")))
    env["ENABLE_AUDIO_OUTPUT"] = _stringify(thinker.get("enable_audio_output", False))
    env["USE_AUDIO_IN_VIDEO"] = _stringify(thinker.get("use_audio_in_video", False))

    swift_bin = str(runtime.get("swift_bin", "swift"))
    command = [swift_bin, str(recipe["entrypoint"])]
    if recipe.get("rlhf_type"):
        command.extend(["--rlhf_type", str(recipe["rlhf_type"])])

    command.extend(["--model", str(model["path"])])
    if model.get("model_type"):
        command.extend(["--model_type", str(model["model_type"])])
    if model.get("torch_dtype"):
        command.extend(["--torch_dtype", str(model["torch_dtype"])])

    train_datasets = enabled_paths(dataset_config, stage, "train")
    if not train_datasets:
        raise ValueError(f"No enabled train datasets for stage={stage}")
    command.append("--dataset")
    command.extend(train_datasets)

    val_datasets = enabled_paths(dataset_config, stage, "val")
    if val_datasets:
        command.append("--val_dataset")
        command.extend(val_datasets)

    adapters = recipe.get("adapters")
    if adapters:
        command.append("--adapters")
        if isinstance(adapters, list):
            command.extend(str(item) for item in adapters)
        else:
            command.append(str(adapters))

    lora_args = {
        "tuner_type": lora.get("tuner_type", "lora"),
        "target_modules": lora.get("target_modules", "all-linear"),
        "lora_rank": lora.get("lora_rank"),
        "lora_alpha": lora.get("lora_alpha"),
        "freeze_llm": thinker.get("freeze_llm", False),
        "freeze_vit": thinker.get("freeze_vit", True),
        "freeze_aligner": thinker.get("freeze_aligner", True),
    }
    _append_args(command, lora_args)

    reward_cfg = recipe.get("reward", {})
    _append_args(command, {
        "external_plugins": reward_cfg.get("external_plugins"),
        "reward_funcs": reward_cfg.get("reward_funcs"),
        "reward_weights": reward_cfg.get("reward_weights"),
    })

    output_dir = Path(str(recipe["output_dir"])).expanduser().resolve()
    _append_args(command, {**recipe.get("swift_args", {}), "output_dir": str(output_dir)})
    return LaunchSpec(env=env, command=command, output_dir=output_dir, run_name=run_name)


def shell_command(spec: LaunchSpec) -> str:
    env_items = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(spec.env.items()))
    cmd = " ".join(shlex.quote(item) for item in spec.command)
    return f"{env_items} {cmd}"


def launch(recipe_path: str | Path, *, execute: bool = False) -> LaunchSpec:
    spec = build_launch_spec(recipe_path)
    print(shell_command(spec), flush=True)
    if not execute:
        return spec
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(spec.env)
    subprocess.run(spec.command, env=env, check=True)
    return spec
