from __future__ import annotations

import argparse

from .eval import main as eval_main
from .run_suite import main as run_suite_main
from .serve import main as serve_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for Tri-modal-Evolution-Agent.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("serve", help="Start the local Qwen3-Omni HTTP server.")
    subparsers.add_parser("eval", help="Run one benchmark dataset.")
    subparsers.add_parser("run-suite", help="Run a configured dataset suite.")
    return parser


def main() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    if args.command == "serve":
        import sys

        sys.argv = [sys.argv[0], *remaining]
        serve_main()
        return

    if args.command == "eval":
        import sys

        sys.argv = [sys.argv[0], *remaining]
        eval_main()
        return

    if args.command == "run-suite":
        import sys

        sys.argv = [sys.argv[0], *remaining]
        run_suite_main()
        return

    raise ValueError(f"Unsupported command: {args.command}")
