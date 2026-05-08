#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path(os.environ.get("OMNIBENCH_SFT_INPUT", TRAIN_ROOT / "data" / "sft" / "train.jsonl"))
DEFAULT_OUTPUT = Path(os.environ.get("OMNIBENCH_GRPO_OUTPUT", TRAIN_ROOT / "data" / "grpo" / "train.jsonl"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_grpo_record(row: dict[str, Any]) -> dict[str, Any]:
    messages = list(row["messages"])
    assistant = messages[-1]
    if assistant.get("role") != "assistant":
        raise ValueError(f"Expected assistant as last message: {assistant}")
    answer = str(assistant["content"]).strip()
    prompt_messages = messages[:-1]
    meta = dict(row.get("meta") or {})
    return {
        "messages": prompt_messages,
        "images": row.get("images", []),
        "audios": row.get("audios", []),
        "solution": answer,
        "answer": answer,
        "meta": meta,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare OmniBench GRPO JSONL from SFT-style answer records.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=0, help="Optional positive row limit for a tiny smoke run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = _load_jsonl(source)
    if args.limit > 0:
        rows = rows[: args.limit]
    converted = [_to_grpo_record(row) for row in rows]
    _write_jsonl(output, converted)
    summary = {
        "source": str(source),
        "output": str(output),
        "source_selected": len(rows),
        "written": len(converted),
        "limit": args.limit,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"source_selected: {len(rows)}")
    print(f"written: {len(converted)}")
    print(f"output: {output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
