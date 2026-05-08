#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any


TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OMNIBENCH_ROOT = Path(os.environ.get("OMNIBENCH_ROOT", TRAIN_ROOT / "data" / "raw" / "OmniBench"))
DEFAULT_DATASET_FILE = Path(
    os.environ.get(
        "OMNIBENCH_DATASET_FILE",
        DEFAULT_OMNIBENCH_ROOT / "dataset" / "batch-5_1142_20240817.jsonl",
    )
)
DEFAULT_MM_ROOT = Path(os.environ.get("OMNIBENCH_MM_ROOT", DEFAULT_OMNIBENCH_ROOT / "mm_data"))
DEFAULT_OUTPUT = Path(os.environ.get("OMNIBENCH_SFT_OUTPUT", TRAIN_ROOT / "data" / "sft" / "train.jsonl"))


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


def _option_letter(answer: str, options: list[str]) -> str:
    normalized = " ".join(str(answer).strip().split()).lower()
    for index, option in enumerate(options):
        candidate = " ".join(str(option).strip().split()).lower()
        if candidate == normalized:
            return chr(ord("A") + index)
    raise ValueError(f"Answer is not in options: {answer!r}")


def _prompt(question: str, options: list[str]) -> str:
    option_block = "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        "<image><audio>"
        "Answer this image-audio multiple-choice question using both modalities.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_block}\n\n"
        "Return exactly one uppercase option letter and no other text."
    )


def _to_sft_record(row: dict[str, Any], *, mm_root: Path, source_file: Path) -> dict[str, Any]:
    options = [str(option) for option in row["options"]]
    answer_letter = _option_letter(str(row["answer"]), options)
    image_path = (mm_root / "image" / str(row["image_path"])).resolve()
    audio_path = (mm_root / "audio" / str(row["audio_path"])).resolve()
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful audio-visual reasoning assistant.",
            },
            {
                "role": "user",
                "content": _prompt(str(row["question"]), options),
            },
            {
                "role": "assistant",
                "content": answer_letter,
                "loss": True,
            },
        ],
        "images": [str(image_path)],
        "audios": [str(audio_path)],
        "meta": {
            "source": "omnibench",
            "source_file": str(source_file),
            "index": int(row["index"]),
            "task_type": row.get("task type", ""),
            "audio_type": row.get("audio type", ""),
            "answer_text": row.get("answer", ""),
        },
    }


def _audio_decodable_like_swift(audio_path: Path, sampling_rate: int) -> tuple[bool, str]:
    """ms-swift loads local audio as BytesIO before calling librosa.load."""
    try:
        import librosa
    except Exception as exc:  # pragma: no cover - environment preflight
        return False, f"librosa_import_failed: {type(exc).__name__}: {exc}"

    try:
        audio_io = io.BytesIO(audio_path.read_bytes())
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            waveform, _ = librosa.load(audio_io, sr=sampling_rate)
        if getattr(waveform, "size", 0) == 0:
            return False, "decoded_empty"
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _validate_media(
    row: dict[str, Any],
    *,
    mm_root: Path,
    validate_audio: str,
    sampling_rate: int,
) -> list[str]:
    errors: list[str] = []
    image_path = mm_root / "image" / str(row["image_path"])
    audio_path = mm_root / "audio" / str(row["audio_path"])
    if not image_path.exists():
        errors.append(f"missing_image: {image_path}")
    if not audio_path.exists():
        errors.append(f"missing_audio: {audio_path}")
        return errors
    if audio_path.stat().st_size == 0:
        errors.append(f"empty_audio: {audio_path}")
        return errors
    if validate_audio == "swift-bytes":
        ok, reason = _audio_decodable_like_swift(audio_path, sampling_rate)
        if not ok:
            errors.append(f"undecodable_audio_swift_bytes: {audio_path}: {reason}")
    return errors


def build_split(
    rows: list[dict[str, Any]],
    *,
    group_fields: list[str],
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = " || ".join(str(row.get(field, "")) for field in group_fields)
        groups[key].append(row)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    summary: dict[str, dict[str, int]] = {}
    for key in sorted(groups):
        items = list(groups[key])
        items.sort(key=lambda item: int(item["index"]))
        rng.shuffle(items)
        keep = max(1, math.ceil(len(items) * fraction))
        chosen = sorted(items[:keep], key=lambda item: int(item["index"]))
        selected.extend(chosen)
        summary[key] = {"total": len(items), "selected": len(chosen)}

    selected.sort(key=lambda item: int(item["index"]))
    return selected, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a stratified OmniBench SFT JSONL for ms-swift.")
    parser.add_argument("--dataset-file", type=Path, default=DEFAULT_DATASET_FILE)
    parser.add_argument("--mm-root", type=Path, default=DEFAULT_MM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--group-fields", nargs="+", default=["task type", "audio type"])
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument(
        "--validate-audio",
        choices=["none", "swift-bytes"],
        default="swift-bytes",
        help="Validate audio files before sampling. 'swift-bytes' mirrors ms-swift local audio loading.",
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("--fraction must be in (0, 1].")

    dataset_file = args.dataset_file.expanduser().resolve()
    mm_root = args.mm_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = _load_jsonl(dataset_file)
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    for row in rows:
        errors = _validate_media(
            row,
            mm_root=mm_root,
            validate_audio=args.validate_audio,
            sampling_rate=args.sampling_rate,
        )
        if errors:
            invalid_rows.append({
                "index": int(row["index"]),
                "task_type": row.get("task type", ""),
                "audio_type": row.get("audio type", ""),
                "image_path": row.get("image_path", ""),
                "audio_path": row.get("audio_path", ""),
                "errors": errors,
            })
        else:
            valid_rows.append(row)
    selected, summary = build_split(
        valid_rows,
        group_fields=list(args.group_fields),
        fraction=args.fraction,
        seed=args.seed,
    )
    converted = [_to_sft_record(row, mm_root=mm_root, source_file=dataset_file) for row in selected]
    _write_jsonl(output, converted)

    summary_path = output.with_suffix(".summary.json")
    summary_payload = {
        "source_file": str(dataset_file),
        "mm_root": str(mm_root),
        "output": str(output),
        "group_fields": list(args.group_fields),
        "fraction": args.fraction,
        "seed": args.seed,
        "validate_audio": args.validate_audio,
        "sampling_rate": args.sampling_rate,
        "total": len(rows),
        "valid": len(valid_rows),
        "invalid": len(invalid_rows),
        "selected": len(converted),
        "invalid_rows": invalid_rows,
        "groups": summary,
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"source_total: {len(rows)}")
    print(f"valid_total: {len(valid_rows)}")
    print(f"invalid_total: {len(invalid_rows)}")
    print(f"selected_total: {len(converted)}")
    print(f"output: {output}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
