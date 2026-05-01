from __future__ import annotations

from pathlib import Path

from ..config.loader import load_yaml_config
from ..datasets.registry import get_dataset_loader
from ..engine.api_client import LocalOmniApiClient
from ..paths import OUTPUT_ROOT
from .runner import evaluate_samples


def _resolve_run_name(model_config: dict, dataset_config: dict, run_name: str | None) -> str:
    if run_name:
        return run_name
    return model_config["model"]["key"]


def run_dataset_evaluation(
    *,
    model_config_path: str | Path,
    dataset_config_path: str | Path,
    server_url: str | None = None,
    run_name: str | None = None,
    limit: int | None = None,
    resume: bool = False,
) -> dict[str, Path]:
    model_config = load_yaml_config(model_config_path)
    dataset_config = load_yaml_config(dataset_config_path)

    loader = get_dataset_loader(str(dataset_config["loader"]))
    samples = loader(dataset_config, model_config)
    if limit is not None:
        samples = samples[:limit]

    resolved_server_url = server_url or model_config["server"]["base_url"]
    client = LocalOmniApiClient(resolved_server_url)
    client.wait_until_ready()

    dataset_key = str(dataset_config["dataset_key"])
    resolved_run_name = _resolve_run_name(model_config, dataset_config, run_name)
    output_dir = OUTPUT_ROOT / dataset_key / resolved_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] dataset={dataset_key} selected_samples={len(samples)}", flush=True)
    return evaluate_samples(
        samples=samples,
        client=client,
        model_key=str(model_config["model"]["key"]),
        results_path=output_dir / "results.jsonl",
        summary_path=output_dir / "summary.json",
        errors_path=output_dir / "errors.jsonl",
        group_fields=list(dataset_config.get("group_fields", [])),
        resume=resume,
    )
