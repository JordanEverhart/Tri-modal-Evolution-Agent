from __future__ import annotations

import argparse

from ..evaluation.pipeline import run_dataset_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run inference and evaluation for a single benchmark dataset.")
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--server-url", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    outputs = run_dataset_evaluation(
        model_config_path=args.model_config,
        dataset_config_path=args.dataset_config,
        server_url=args.server_url,
        run_name=args.run_name,
        limit=args.limit,
        resume=args.resume,
    )
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
