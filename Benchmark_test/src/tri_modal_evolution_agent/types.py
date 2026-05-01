from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    dataset_key: str
    question: str
    options: list[str]
    correct_answer: str
    messages: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)
    media_paths: dict[str, str] = field(default_factory=dict)
    generation: dict[str, Any] = field(default_factory=dict)
