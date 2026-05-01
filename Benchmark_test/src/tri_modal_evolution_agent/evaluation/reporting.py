from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def dump_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = ensure_parent(path)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_done_sample_ids(results_path: str | Path) -> set[str]:
    resolved = Path(results_path).expanduser().resolve()
    if not resolved.exists():
        return set()
    return {str(row["sample_id"]) for row in load_jsonl(resolved)}


def summarize_results(
    results: list[dict[str, Any]],
    *,
    dataset_key: str,
    model_key: str,
    group_fields: list[str],
) -> dict[str, Any]:
    total = len(results)
    correct = sum(1 for row in results if bool(row.get("is_correct")))
    invalid = sum(1 for row in results if row.get("parsed_response") == "N/A")
    wrong = total - correct
    summary: dict[str, Any] = {
        "dataset_key": dataset_key,
        "model_key": model_key,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "invalid_rate": round(invalid / total, 6) if total else 0.0,
    }

    by_field: dict[str, Any] = {}
    for field in group_fields:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            value = row.get(field, "N/A")
            if isinstance(value, list):
                value = "|".join(str(item) for item in value)
            grouped[str(value)].append(row)
        by_field[field] = {}
        for value, items in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
            item_total = len(items)
            item_correct = sum(1 for item in items if bool(item.get("is_correct")))
            item_invalid = sum(1 for item in items if item.get("parsed_response") == "N/A")
            by_field[field][value] = {
                "total": item_total,
                "correct": item_correct,
                "wrong": item_total - item_correct,
                "invalid": item_invalid,
                "accuracy": round(item_correct / item_total, 6) if item_total else 0.0,
            }
    summary["by_field"] = by_field
    return summary


def write_errors(errors_path: str | Path, results: list[dict[str, Any]]) -> Path:
    output_path = ensure_parent(errors_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in results:
            if bool(row.get("is_correct")):
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path
