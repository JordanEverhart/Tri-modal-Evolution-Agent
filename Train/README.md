# Qwen3-Omni ms-swift Training

This directory is the reusable training layer for Qwen3-Omni experiments with
`ms-swift`. It intentionally keeps datasets, model weights, checkpoints, logs,
API keys, and machine-specific paths out of Git.

Supported stages:

- LoRA SFT through `swift sft`
- GRPO through `swift rlhf --rlhf_type grpo`
- GSPO through `swift rlhf --rlhf_type grpo --importance_sampling_level sequence`
- Megatron GRPO smoke/full launches for Qwen3-Omni with a MIMO API judge reward

## Official Alignment

The package follows the `ms-swift` conventions used for Qwen3-Omni:

- Custom datasets are passed with `--dataset <jsonl_path>`.
- Multimodal examples use `messages` plus `images`, `videos`, and `audios` fields.
- RLHF/GRPO examples keep reward-side columns such as `solution`, `answer`, and `meta`; `ms-swift` forwards them to the ORM reward.
- `Qwen/Qwen3-Omni-30B-A3B-Instruct` uses `model_type: qwen3_omni_moe`.
- `ENABLE_AUDIO_OUTPUT=False` keeps training on the thinker side.
- `freeze_vit=true` and `freeze_aligner=true` avoid adapting the vision/audio towers and projector for LoRA runs.
- GSPO is represented as GRPO with sequence-level importance sampling.

Useful upstream references:

- `ms-swift`: https://github.com/modelscope/ms-swift
- Custom dataset docs: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md
- RLHF/GRPO docs: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/RLHF.md
- Supported models: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md
- GSPO docs: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/AdvancedResearch/GSPO.md

## Layout

```text
Train/
├── configs/
│   ├── datasets/datasets.yaml
│   ├── env/default.yaml
│   ├── models/qwen3_omni_30b_a3b_instruct.yaml
│   ├── recipes/
│   └── rewards/mimo_judge.example.yaml
├── examples/jsonl/
├── manifests/
├── schemas/
├── scripts/
└── src/qwen3omni_train/
```

## Setup

Install the lightweight training package inside the environment that already has
`ms-swift`, PyTorch, Megatron, and Qwen3-Omni dependencies:

```bash
cd /path/to/Tri-modal-Evolution-Agent/Train
pip install -e .
```

Set local paths in your shell, scheduler script, or an untracked `.env` file:

```bash
export CONDA_ROOT=/path/to/miniconda3
export CONDA_ENV=qwen3omni
export MS_SWIFT_ROOT=/path/to/ms-swift
export VLLM_ROOT=/path/to/vllm_qwen3_omni
export QWEN3OMNI_MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct
export CUDA_HOME=/path/to/cuda
```

For MIMO reward training, keep the real API key out of Git:

```bash
export XIAOMI_MIMO_API_KEY=...
export QWEN3OMNI_MIMO_CONFIG=configs/rewards/mimo_judge.example.yaml
```

If the compute node needs a proxy for the MIMO API:

```bash
export PROXY_URL=http://admin:10808
```

## Dataset Format

SFT JSONL:

```json
{"messages":[{"role":"system","content":"You are a helpful audio-visual reasoning assistant."},{"role":"user","content":"<image><audio>Question...\nA. ...\nB. ..."},{"role":"assistant","content":"A","loss":true}],"images":["/abs/path/image.jpg"],"audios":["/abs/path/audio.wav"],"meta":{"answer_text":"..."}}
```

GRPO/GSPO JSONL:

```json
{"messages":[{"role":"system","content":"You are a helpful audio-visual reasoning assistant."},{"role":"user","content":"<image><audio>Question...\nA. ...\nB. ..."}],"images":["/abs/path/image.jpg"],"audios":["/abs/path/audio.wav"],"solution":"A","answer":"A","meta":{"answer_text":"..."}}
```

The default config expects:

```text
data/sft/train.jsonl
data/grpo/train.jsonl
data/gspo/train.jsonl
```

These `data/` files are ignored by Git.

## OmniBench Preparation

Prepare a stratified SFT split from OmniBench:

```bash
export OMNIBENCH_ROOT=/path/to/OmniBench
python scripts/prepare_omnibench_sft.py \
  --fraction 0.5 \
  --output data/sft/train.jsonl
```

Convert answer-supervised records to GRPO prompt records:

```bash
python scripts/prepare_omnibench_grpo.py \
  --input data/sft/train.jsonl \
  --output data/grpo/train.jsonl
```

For a tiny GRPO smoke dataset:

```bash
python scripts/prepare_omnibench_grpo.py \
  --input data/sft/train.jsonl \
  --output data/grpo/train.jsonl \
  --limit 16
```

## CLI Dry Runs

The package CLI prints the exact `swift` command without launching training:

```bash
q3o-train launch --recipe configs/recipes/lora_sft.yaml --dry-run
q3o-train launch --recipe configs/recipes/grpo.yaml --dry-run
q3o-train launch --recipe configs/recipes/gspo.yaml --dry-run
```

Launch only from a GPU compute node:

```bash
q3o-train launch --recipe configs/recipes/lora_sft.yaml --execute
```

## Megatron GRPO With MIMO Reward

This is the path validated for Qwen3-Omni LoRA+GRPO with colocated vLLM and a
MIMO API judge reward. Run it on an 8-GPU node:

```bash
export CONDA_ROOT=/path/to/miniconda3
export CONDA_ENV=qwen3omni
export MS_SWIFT_ROOT=/path/to/ms-swift
export VLLM_ROOT=/path/to/vllm_qwen3_omni
export MODEL_DIR="$QWEN3OMNI_MODEL_PATH"
export DATASET=/path/to/grpo_train.jsonl
export XIAOMI_MIMO_API_KEY=...
export QWEN3OMNI_MIMO_CONFIG=configs/rewards/mimo_judge.example.yaml
export PROXY_URL=http://admin:10808

bash scripts/run_megatron_grpo_mimo_8gpu.sh
```

Or submit with Slurm:

```bash
sbatch scripts/sbatch_megatron_grpo_mimo_8gpu.sh
```

Important knobs are exposed as environment variables:

```bash
export TRAIN_ITERS=100
export GLOBAL_BATCH_SIZE=8
export NUM_GENERATIONS=2
export VLLM_GPU_MEMORY_UTILIZATION=0.55
export VLLM_MAX_MODEL_LEN=1024
export MAX_LENGTH=768
export MAX_COMPLETION_LENGTH=4
export OUTPUT_DIR=outputs/megatron_grpo/my_run
```

The script keeps the verified Qwen3-Omni runtime settings:

- `ENABLE_AUDIO_OUTPUT=0`
- `VLLM_USE_V1=0`
- `VLLM_QWEN3_FORCE_TORCH_ITERATIVE_MOE=1`
- vLLM compilation disabled through `vllm_engine_kwargs`
- tensor parallel size 2 and expert parallel size 4 on 8 GPUs
- `SWIFT_MEGATRON_FORCE_OS_EXIT_AFTER_MAIN=1`

## Reward Plugins

Available reward plugins:

- `qwen3omni_choice_accuracy`: exact multiple-choice letter reward.
- `qwen3omni_mimo_judge`: binary MIMO API judge reward with exact-match fallback.

The MIMO reward accepts config from `QWEN3OMNI_MIMO_CONFIG` or directly from
environment variables. It never requires committing a real key.

## Repository Hygiene

Ignored by design:

- `data/`
- `outputs/`
- `logs/`
- `runs/`
- checkpoints and safetensors
- `.env` and `.env.*`

Before pushing changes, run:

```bash
python -m compileall src scripts
PYTHONPATH=src python -m qwen3omni_train.cli launch --recipe configs/recipes/grpo.yaml --dry-run
rg -n --hidden -S "api_key:|BEGIN .*PRIVATE|password|secret|token" .
```

Do not start training, data preprocessing, or benchmarks on a login node.
