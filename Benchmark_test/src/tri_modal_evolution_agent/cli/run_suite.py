from __future__ import annotations

import argparse

from ..config.loader import load_yaml_config
from ..evaluation.pipeline import run_dataset_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a suite of benchmark datasets with one model config.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--suite-config", type=str, required=True)
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    suite_config = load_yaml_config(args.suite_config)

    jobs = suite_config.get("jobs", [])
    if not jobs:
        raise ValueError(f"No jobs defined in suite config: {args.suite_config}")

    for job in jobs:
        dataset_config = job["dataset_config"]
        job_run_name = job.get("run_name") or args.run_name
        print(f"[suite] running dataset config: {dataset_config}", flush=True)
        outputs = run_dataset_evaluation(
            model_config_path=args.model_config,
            dataset_config_path=dataset_config,
            server_url=args.server_url,
            run_name=job_run_name,
            limit=args.limit,
            resume=args.resume,
        )
        for key, path in outputs.items():
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
