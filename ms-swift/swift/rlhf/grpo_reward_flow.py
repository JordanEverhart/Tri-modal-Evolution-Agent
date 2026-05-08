"""Compact GRPO reward flow matching the interface used by Train/.

The official trainer does considerably more work. This module keeps only the
reward resolution and reward-call contract that external plugins rely on.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable

from swift.rewards import AsyncORM, ORM, orms
from swift.utils import import_external_plugins


RewardCallable = Callable[..., list[float]]


@dataclass(frozen=True)
class GRPORewardResult:
    reward_func_names: list[str]
    rewards_per_func: list[list[float]]
    weighted_rewards: list[float]


def rows_to_batched(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Convert row-wise dataset records into column-wise reward kwargs."""

    keys: set[str] = set()
    for row in rows:
        keys.update(row)
    return {key: [row.get(key) for row in rows] for key in sorted(keys)}


def _completion_from_row(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        last = messages[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(row.get("completion", ""))


def resolve_reward_funcs(reward_funcs: list[str | RewardCallable], args: Any = None) -> list[RewardCallable]:
    """Resolve ``--reward_funcs`` names through the global ``orms`` registry."""

    resolved: list[RewardCallable] = []
    for reward_func in reward_funcs:
        if isinstance(reward_func, str):
            if reward_func not in orms:
                raise ValueError(f"reward_function {reward_func!r} is not implemented in swift.rewards")
            resolved.append(orms[reward_func](args=args))
        elif callable(reward_func):
            resolved.append(reward_func)
        else:
            raise TypeError(f"invalid reward function: {reward_func!r}")
    return resolved


async def _call_async_reward(reward_func: RewardCallable, completions: list[str], kwargs: dict[str, Any]) -> list[float]:
    output = reward_func(completions, **kwargs)
    if inspect.isawaitable(output):
        output = await output
    return list(output)


def _call_sync_reward(reward_func: RewardCallable, completions: list[str], kwargs: dict[str, Any]) -> list[float]:
    output = reward_func(completions, **kwargs)
    return list(output)


def _reward_name(reward_func: RewardCallable) -> str:
    if inspect.isfunction(reward_func):
        return reward_func.__name__
    return reward_func.__class__.__name__


def compute_grpo_rewards(
    rows: list[dict[str, Any]],
    *,
    reward_funcs: list[str | RewardCallable],
    external_plugins: list[str] | None = None,
    reward_weights: list[float] | None = None,
    args: Any = None,
    trainer_state: Any = None,
) -> GRPORewardResult:
    """Compute rewards for generated GRPO rows.

    Each row should already contain the assistant completion as the last message.
    Dataset columns such as ``solution``, ``answer``, ``messages``, and ``meta``
    are forwarded to reward functions as batched keyword arguments.
    """

    import_external_plugins(external_plugins)
    resolved = resolve_reward_funcs(reward_funcs, args=args)
    if not resolved:
        raise ValueError("reward_funcs is not set")

    weights = reward_weights if reward_weights is not None else [1.0] * len(resolved)
    if len(weights) != len(resolved):
        raise ValueError("reward_weights must match reward_funcs")

    completions = [_completion_from_row(row) for row in rows]
    reward_kwargs = rows_to_batched(rows)
    reward_kwargs["trainer_state"] = trainer_state

    columns: list[list[float]] = []
    for reward_func in resolved:
        if isinstance(reward_func, AsyncORM) or inspect.iscoroutinefunction(getattr(reward_func, "__call__", None)):
            rewards = asyncio.run(_call_async_reward(reward_func, completions, reward_kwargs))
        elif isinstance(reward_func, ORM) or callable(reward_func):
            rewards = _call_sync_reward(reward_func, completions, reward_kwargs)
        else:
            raise TypeError(f"invalid resolved reward function: {reward_func!r}")
        if len(rewards) != len(rows):
            raise ValueError(f"reward {_reward_name(reward_func)} returned {len(rewards)} values for {len(rows)} rows")
        columns.append([float(value) for value in rewards])

    rewards_per_func = [list(values) for values in zip(*columns)]
    weighted_rewards = [
        sum(value * float(weight) for value, weight in zip(row_rewards, weights))
        for row_rewards in rewards_per_func
    ]
    return GRPORewardResult(
        reward_func_names=[_reward_name(func) for func in resolved],
        rewards_per_func=rewards_per_func,
        weighted_rewards=weighted_rewards,
    )
