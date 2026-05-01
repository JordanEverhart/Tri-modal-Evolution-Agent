from __future__ import annotations

import argparse

from .config import load_yaml
from .converters import convert_records
from .datasets import conversion_job
from .swift_launcher import launch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-Omni ms-swift training utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch_parser = subparsers.add_parser("launch", help="Build or execute a swift training command.")
    launch_parser.add_argument("--recipe", required=True)
    mode = launch_parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print command only.")
    mode.add_argument("--execute", action="store_true", help="Run command. Use only on a GPU node.")

    convert_parser = subparsers.add_parser("convert", help="Convert manifest records to ms-swift JSONL.")
    convert_parser.add_argument("--datasets", required=True)
    convert_parser.add_argument("--job", required=True, choices=["sft", "grpo", "gspo"])
    convert_parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "launch":
        launch(args.recipe, execute=bool(args.execute))
        return

    if args.command == "convert":
        dataset_config = load_yaml(args.datasets)
        job = conversion_job(dataset_config, args.job)
        output_path = convert_records(
            converter=str(job["converter"]),
            input_path=str(job["input"]),
            output_path=str(job["output"]),
            default_system=job.get("default_system"),
            limit=args.limit,
        )
        print(f"converted: {output_path}")
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
