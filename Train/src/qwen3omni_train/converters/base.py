from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


MEDIA_KEYS = ("images", "videos", "audios")
SINGULAR_MEDIA = {"image": "images", "video": "videos", "audio": "audios"}


def _load_json_records(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Input manifest not found: {resolved}")
    if resolved.suffix.lower() == ".json":
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        raise ValueError(f"JSON input must be a list or contain a data list: {resolved}")

    records: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return resolved


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)]


def _media(row: dict[str, Any]) -> dict[str, list[str]]:
    media: dict[str, list[str]] = {}
    for key in MEDIA_KEYS:
        values = _as_list(row.get(key))
        if values:
            media[key] = values
    for src, dst in SINGULAR_MEDIA.items():
        values = _as_list(row.get(src))
        if values:
            media.setdefault(dst, []).extend(values)
    return media


def _media_prefix(media: dict[str, list[str]]) -> str:
    prefix = ""
    if media.get("images"):
        prefix += "<image>" * len(media["images"])
    if media.get("videos"):
        prefix += "<video>" * len(media["videos"])
    if media.get("audios"):
        prefix += "<audio>" * len(media["audios"])
    return prefix


def _question(row: dict[str, Any]) -> str:
    for key in ("query", "prompt", "question", "instruction", "input"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Could not find query/prompt/question in row: {row}")


def _answer(row: dict[str, Any]) -> str:
    for key in ("response", "answer", "output", "target", "solution", "completion"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Could not find response/answer/output in row: {row}")


def _solution(row: dict[str, Any]) -> str:
    for key in ("solution", "answer", "correct_answer", "target", "label"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Could not find solution/answer/correct_answer in row: {row}")


def _messages_from_row(row: dict[str, Any], *, default_system: str | None, include_answer: bool) -> list[dict[str, Any]]:
    if isinstance(row.get("messages"), list):
        messages = [dict(item) for item in row["messages"]]
    else:
        media = _media(row)
        query = _question(row)
        prefix = _media_prefix(media)
        if prefix and all(token not in query for token in ("<image>", "<video>", "<audio>")):
            query = f"{prefix}{query}"
        messages = []
        system = row.get("system") or row.get("system_prompt") or default_system
        if system:
            messages.append({"role": "system", "content": str(system)})
        messages.append({"role": "user", "content": query})
        if include_answer:
            messages.append({"role": "assistant", "content": _answer(row), "loss": True})

    if not include_answer and messages and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return messages


def _base_output(row: dict[str, Any], *, default_system: str | None, include_answer: bool) -> dict[str, Any]:
    output: dict[str, Any] = {
        "messages": _messages_from_row(row, default_system=default_system, include_answer=include_answer)
    }
    output.update(_media(row))
    meta = row.get("meta") or row.get("metadata")
    if isinstance(meta, dict):
        output["meta"] = meta
    elif row.get("sample_id") is not None:
        output["meta"] = {"sample_id": str(row["sample_id"])}
    return output


def to_sft(row: dict[str, Any], *, default_system: str | None = None) -> dict[str, Any]:
    return _base_output(row, default_system=default_system, include_answer=True)


def to_grpo(row: dict[str, Any], *, default_system: str | None = None) -> dict[str, Any]:
    output = _base_output(row, default_system=default_system, include_answer=False)
    solution = _solution(row)
    output["solution"] = solution
    output["answer"] = str(row.get("answer", solution))
    return output


def convert_records(
    *,
    converter: str,
    input_path: str | Path,
    output_path: str | Path,
    default_system: str | None = None,
    limit: int | None = None,
) -> Path:
    rows = _load_json_records(input_path)
    if limit is not None:
        rows = rows[:limit]

    if converter == "sft":
        converted = [to_sft(row, default_system=default_system) for row in rows]
    elif converter in {"grpo", "gspo"}:
        converted = [to_grpo(row, default_system=default_system) for row in rows]
    else:
        raise ValueError(f"Unsupported converter: {converter}")

    return _write_jsonl(output_path, converted)
