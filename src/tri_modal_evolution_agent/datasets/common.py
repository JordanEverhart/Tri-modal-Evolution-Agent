from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def choice_letters(n_choices: int) -> list[str]:
    if n_choices <= 0:
        raise ValueError("n_choices must be positive")
    if n_choices > 26:
        raise ValueError("At most 26 choices are supported")
    return [chr(ord("A") + idx) for idx in range(n_choices)]


def strip_option_prefix(text: str) -> str:
    stripped = (text or "").strip()
    return re.sub(r"^[A-Z][\.\):：]\s*", "", stripped)


def normalize_options(raw_options: list[str] | str) -> list[str]:
    if isinstance(raw_options, list):
        options = [strip_option_prefix(str(option)) for option in raw_options]
    else:
        pattern = r"([A-Z][\.\):：].*?)(?=(?:[A-Z][\.\):：])|$)"
        matches = re.findall(pattern, raw_options, flags=re.DOTALL)
        if not matches:
            raise ValueError(f"Could not split options from: {raw_options!r}")
        options = [strip_option_prefix(match) for match in matches]
    return [option.strip() for option in options if option.strip()]


def _normalize_text(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = (
        normalized.replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
        .replace("：", ":")
    )
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def build_index_to_answer(options: Iterable[str]) -> dict[str, str]:
    option_list = list(options)
    return dict(zip(choice_letters(len(option_list)), option_list))


def parse_choice_response(response: str, options: list[str], default_answer: str = "N/A") -> str:
    letters = choice_letters(len(options))
    raw_response = (response or "").strip()
    if not raw_response:
        return default_answer

    leading_match = re.match(r"^\s*\(?([A-Z])\)?[\.\):：]\s+", raw_response)
    if leading_match:
        answer = leading_match.group(1).upper()
        if answer in letters:
            return answer

    candidate_segments = [raw_response]
    lowered = raw_response.lower()
    for marker in ["assistant", "final answer", "answer", "option"]:
        position = lowered.rfind(marker)
        if position != -1:
            candidate_segments.append(raw_response[position:])

    patterns = [
        r"(?:final answer|answer|option|assistant)\s*[:：]?\s*\(?([A-Z])\)?[\.\):：]?\s*$",
        r"^\s*\(?([A-Z])\)?[\.\):：]?\s*$",
        r"\b([A-Z])\b[\.\):：]?\s*$",
    ]
    for segment in reversed(candidate_segments):
        tail = segment.strip()[-128:]
        for pattern in patterns:
            match = re.search(pattern, tail, flags=re.IGNORECASE)
            if not match:
                continue
            answer = match.group(1).upper()
            if answer in letters:
                return answer

    normalized_response = _normalize_text(raw_response)
    for letter, option in build_index_to_answer(options).items():
        normalized_option = _normalize_text(strip_option_prefix(option))
        if normalized_option and normalized_option in normalized_response:
            return letter

    return default_answer


def infer_gold_choice(answer_text: str, options: list[str], default_answer: str = "N/A") -> str:
    letters = choice_letters(len(options))
    normalized_answer = (answer_text or "").strip().upper()
    if normalized_answer in letters:
        return normalized_answer

    normalized_answer = _normalize_text(answer_text)
    for letter, option in build_index_to_answer(options).items():
        normalized_option = _normalize_text(strip_option_prefix(option))
        if not normalized_option:
            continue
        if normalized_answer == normalized_option:
            return letter
        if normalized_answer and (
            normalized_answer in normalized_option or normalized_option in normalized_answer
        ):
            return letter
    return parse_choice_response(answer_text, options, default_answer=default_answer)


def build_mcq_prompt(question: str, options: list[str], modality_hint: str, response_suffix: str) -> str:
    letters = choice_letters(len(options))
    option_block = "\n".join(f"{letter}. {option}" for letter, option in zip(letters, options))
    return (
        f"Answer this multiple-choice question using {modality_hint}.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{option_block}\n\n"
        f"{response_suffix}"
    )
