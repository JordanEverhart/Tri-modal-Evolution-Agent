# Minimal ms-swift Interface Reference

This folder is a compact, readable copy of the ms-swift interfaces that this
project touches. It is not a replacement for the official `modelscope/ms-swift`
training package.

Runtime training should still use the official ms-swift checkout through
`MS_SWIFT_ROOT` or an installed `swift` command. This folder exists so the
training code can be audited without opening the full upstream repository.

## Covered Surface

The `Train/` package uses only a small part of ms-swift directly:

- `swift sft`: command-line entrypoint used by LoRA SFT recipes.
- `swift rlhf --rlhf_type grpo`: command-line entrypoint used by GRPO/GSPO recipes.
- `megatron rlhf --rlhf_type grpo`: command-line entrypoint used by the 8-GPU Qwen3-Omni GRPO launch script.
- `swift.rewards.ORM`: base class for synchronous reward plugins.
- `swift.rewards.orms`: global reward registry updated by external plugin files.
- `--external_plugins`: imports a Python file for side-effect reward registration.
- `--reward_funcs`: resolves reward names from `orms` and calls each reward with model completions plus batched dataset columns.

The implemented files mirror that interface:

```text
ms-swift/
├── README.md
├── examples/reward_flow_smoke.py
└── swift/
    ├── __init__.py
    ├── cli/
    │   ├── contract.py
    │   ├── rlhf.py
    │   └── sft.py
    ├── megatron/rlhf_contract.py
    ├── rewards/
    │   ├── __init__.py
    │   └── orm.py
    ├── rlhf/grpo_reward_flow.py
    └── utils/import_utils.py
```

## Not Covered

The official package is still required for real training. This reference does
not implement:

- model loading
- Qwen3-Omni processors/templates
- LoRA module injection
- vLLM rollout workers
- Megatron tensor/expert/pipeline parallelism
- DeepSpeed/FSDP
- optimizer, scheduler, checkpoint, or tensorboard logic
- distributed dataset preprocessing

## How This Maps To Train

`Train/src/qwen3omni_train/swift_launcher.py` builds commands for `swift sft`
and `swift rlhf`. The simplified CLI parser here records those arguments as a
plain dictionary so the command contract is easy to inspect.

`Train/src/qwen3omni_train/rewards/*.py` imports:

```python
from swift.rewards import ORM, orms
```

The minimal `swift.rewards` implementation here is enough to import those
plugins and exercise the same reward-call contract locally.

The GRPO reward path is:

```text
external_plugins -> import Python file -> plugin writes orms[name] = RewardClass
reward_funcs -> look up names in orms -> instantiate RewardClass(args=args)
batch rows -> completions + batched columns -> reward(completions, **kwargs)
```

## Local Smoke Check

From the repository root:

```bash
PYTHONPATH=ms-swift:Train/src python ms-swift/examples/reward_flow_smoke.py
```

Expected output:

```text
reward_funcs: ['Qwen3OmniChoiceAccuracy', 'Format']
rewards_per_func: [[1.0, 0.0], [0.0, 0.0]]
weighted_rewards: [1.0, 0.0]
```

## License Note

The interface shape is derived from ModelScope ms-swift, which is distributed
under Apache-2.0. This compact reference is simplified for this project and
keeps attribution in file headers where the upstream behavior is mirrored.
