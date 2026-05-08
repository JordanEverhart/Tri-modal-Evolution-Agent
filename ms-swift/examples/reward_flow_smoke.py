#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

from swift.rlhf import compute_grpo_rewards


REPO_ROOT = Path(__file__).resolve().parents[2]
CHOICE_REWARD = REPO_ROOT / "Train" / "src" / "qwen3omni_train" / "rewards" / "choice_reward.py"


def main() -> None:
    rows = [
        {
            "messages": [
                {"role": "user", "content": "A or B?"},
                {"role": "assistant", "content": "A"},
            ],
            "solution": "A",
            "answer": "A",
            "meta": {"answer_text": "first option"},
        },
        {
            "messages": [
                {"role": "user", "content": "A or B?"},
                {"role": "assistant", "content": "B"},
            ],
            "solution": "A",
            "answer": "A",
            "meta": {"answer_text": "first option"},
        },
    ]
    result = compute_grpo_rewards(
        rows,
        external_plugins=[str(CHOICE_REWARD)],
        reward_funcs=["qwen3omni_choice_accuracy", "format"],
        reward_weights=[1.0, 0.2],
    )
    print(f"reward_funcs: {result.reward_func_names}")
    print(f"rewards_per_func: {result.rewards_per_func}")
    print(f"weighted_rewards: {result.weighted_rewards}")


if __name__ == "__main__":
    main()
