from __future__ import annotations

import re
from typing import List

from swift.rewards import ORM, orms


def _normalize_answer(text: str) -> str:
    text = (text or "").strip()
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    match = re.search(r"\b([A-Z])\b", text.upper())
    if match:
        return match.group(1)
    return re.sub(r"\s+", " ", text).strip().lower()


class Qwen3OmniChoiceAccuracy(ORM):
    """Simple exact/letter reward for multiple-choice audio-visual GRPO."""

    def __call__(self, completions, solution=None, answer=None, **kwargs) -> List[float]:
        gold_values = solution if solution is not None else answer
        if gold_values is None:
            return [0.0 for _ in completions]
        rewards: list[float] = []
        for completion, gold in zip(completions, gold_values):
            pred = _normalize_answer(str(completion))
            target = _normalize_answer(str(gold))
            rewards.append(1.0 if pred and pred == target else 0.0)
        return rewards


orms["qwen3omni_choice_accuracy"] = Qwen3OmniChoiceAccuracy
