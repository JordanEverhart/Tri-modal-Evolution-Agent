#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAIN_ROOT / "src"))

from qwen3omni_train.recipes.gspo import DEFAULT_RECIPE, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Qwen3-Omni GSPO via ms-swift.")
    parser.add_argument("--recipe", default=str(DEFAULT_RECIPE))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(args.recipe, execute=bool(args.execute))


if __name__ == "__main__":
    main()
