"""Minimal outcome reward model registry.

Reference-only implementation inspired by ModelScope ms-swift's public reward
plugin interface. Use the official ms-swift package for real training.

This mirrors the ms-swift reward plugin contract used by this project:

1. External plugin files import ``ORM`` and ``orms``.
2. A plugin defines a reward class derived from ``ORM``.
3. The plugin registers it with ``orms["reward_name"] = RewardClass``.
4. GRPO resolves names from ``--reward_funcs`` through this registry.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional


class ORM:
    """Base class for synchronous outcome reward functions."""

    def __init__(self, args: Optional[Any] = None, **kwargs: Any) -> None:
        self.args = args

    def __call__(self, completions: list[str], **kwargs: Any) -> List[float]:
        raise NotImplementedError


class AsyncORM:
    """Base class for asynchronous outcome reward functions."""

    def __init__(self, args: Optional[Any] = None, **kwargs: Any) -> None:
        self.args = args

    async def __call__(self, completions: list[str], **kwargs: Any) -> List[float]:
        raise NotImplementedError


class Format(ORM):
    """Reward the canonical ``<think>...</think><answer>...</answer>`` format."""

    def __call__(self, completions: list[str], **kwargs: Any) -> List[float]:
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])"
        matches = [re.match(pattern, text, re.DOTALL | re.MULTILINE) for text in completions]
        return [1.0 if match else 0.0 for match in matches]


orms: dict[str, type[ORM]] = {
    "format": Format,
}
