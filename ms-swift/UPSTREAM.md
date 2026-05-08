# Upstream Attribution

This directory is a compact reference implementation of the ms-swift interfaces
used by this repository. It is intentionally not a vendored copy of the official
training framework.

Official upstream:

- Repository: https://github.com/modelscope/ms-swift
- License: Apache License 2.0

Real training should use the official upstream package or a local checkout set
through `MS_SWIFT_ROOT`. The files in this folder are for interface auditing,
plugin smoke tests, and co-author readability.

Mirrored concepts:

- `swift.rewards.ORM`
- `swift.rewards.AsyncORM`
- `swift.rewards.orms`
- built-in `format` reward name
- `--external_plugins` side-effect import behavior
- `--reward_funcs` registry lookup behavior
- GRPO reward invocation with `completions` and batched dataset columns

Non-goals:

- no model training
- no model loading
- no official CLI replacement
- no Megatron/vLLM/DeepSpeed implementation
- no checkpoint compatibility layer
