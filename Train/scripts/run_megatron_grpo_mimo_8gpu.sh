#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-qwen3omni}"
MS_SWIFT_ROOT="${MS_SWIFT_ROOT:-}"
VLLM_ROOT="${VLLM_ROOT:-}"
MODEL_DIR="${MODEL_DIR:-${QWEN3OMNI_MODEL_PATH:-}}"
DATASET="${DATASET:-${TRAIN_ROOT}/data/grpo/train.jsonl}"
REWARD_PLUGIN="${REWARD_PLUGIN:-${TRAIN_ROOT}/src/qwen3omni_train/rewards/mimo_judge_reward.py}"
MIMO_CONFIG="${MIMO_CONFIG:-${QWEN3OMNI_MIMO_CONFIG:-}}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ROOT}/outputs/megatron_grpo/qwen3omni_grpo_mimo_8gpu}"

TRAIN_ITERS="${TRAIN_ITERS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
STEPS_PER_GENERATION="${STEPS_PER_GENERATION:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.55}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1024}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
MAX_LENGTH="${MAX_LENGTH:-768}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4}"
SAVE_STEPS="${SAVE_STEPS:-1000}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
MASTER_PORT="${MASTER_PORT:-29545}"

require_path() {
    local name="$1"
    local value="$2"
    if [[ -z "${value}" ]]; then
        echo "[error] ${name} is required." >&2
        exit 2
    fi
}

require_file() {
    local name="$1"
    local value="$2"
    require_path "${name}" "${value}"
    if [[ ! -f "${value}" ]]; then
        echo "[error] ${name} does not exist: ${value}" >&2
        exit 2
    fi
}

require_dir() {
    local name="$1"
    local value="$2"
    require_path "${name}" "${value}"
    if [[ ! -d "${value}" ]]; then
        echo "[error] ${name} does not exist: ${value}" >&2
        exit 2
    fi
}

require_dir "MODEL_DIR or QWEN3OMNI_MODEL_PATH" "${MODEL_DIR}"
require_file "DATASET" "${DATASET}"
if [[ -n "${MIMO_CONFIG}" ]]; then
    require_file "MIMO_CONFIG" "${MIMO_CONFIG}"
fi
if [[ -z "${MIMO_CONFIG}" && -z "${XIAOMI_MIMO_API_KEY:-}" ]]; then
    echo "[error] Set XIAOMI_MIMO_API_KEY or MIMO_CONFIG/QWEN3OMNI_MIMO_CONFIG for the MIMO reward." >&2
    exit 2
fi

cd "${TRAIN_ROOT}"
mkdir -p logs "${OUTPUT_DIR}"

if [[ -n "${CONDA_ROOT:-}" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
elif [[ -n "${CONDA_PREFIX:-}" ]]; then
    echo "[env] using active conda env at ${CONDA_PREFIX}"
else
    echo "[warn] CONDA_ROOT is not set and no active conda env was detected; using current PATH."
fi

module load compiler/gcc/11.4.0 2>/dev/null || true
module load cuda/12.6.3 2>/dev/null || true

export PYTHONUNBUFFERED=1
if [[ -n "${PROXY_URL:-}" ]]; then
    export http_proxy="${http_proxy:-${PROXY_URL}}"
    export https_proxy="${https_proxy:-${PROXY_URL}}"
    export HTTP_PROXY="${HTTP_PROXY:-${http_proxy}}"
    export HTTPS_PROXY="${HTTPS_PROXY:-${https_proxy}}"
fi
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export no_proxy="${no_proxy:-127.0.0.1,localhost}"
export CUDA_HOME="${CUDA_HOME:-}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export CUDNN_HOME="${CUDNN_HOME:-${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn}"
    export NCCL_HOME="${NCCL_HOME:-${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/nccl}"
    export CUDA_RT_HOME="${CUDA_RT_HOME:-${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime}"
    export LD_LIBRARY_PATH="${CUDNN_HOME}/lib:${NCCL_HOME}/lib:${CUDA_RT_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi
if [[ -n "${CUDA_HOME}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

PYTHONPATH_PARTS=("${TRAIN_ROOT}/src")
if [[ -n "${MS_SWIFT_ROOT}" ]]; then
    PYTHONPATH_PARTS=("${MS_SWIFT_ROOT}" "${PYTHONPATH_PARTS[@]}")
fi
if [[ -n "${VLLM_ROOT}" ]]; then
    PYTHONPATH_PARTS=("${VLLM_ROOT}" "${PYTHONPATH_PARTS[@]}")
fi
export PYTHONPATH="$(IFS=:; echo "${PYTHONPATH_PARTS[*]}"):${PYTHONPATH:-}"

export ENABLE_AUDIO_OUTPUT=0
export USE_AUDIO_IN_VIDEO=False
export MAX_PIXELS="${MAX_PIXELS:-262144}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-32768}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-4}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-128}"
export VIDEO_MAX_TOKEN_NUM="${VIDEO_MAX_TOKEN_NUM:-16}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export TORCHINDUCTOR_MAX_AUTOTUNE="${TORCHINDUCTOR_MAX_AUTOTUNE:-0}"
export MAX_JOBS="${MAX_JOBS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VLLM_QWEN3_FORCE_TORCH_ITERATIVE_MOE="${VLLM_QWEN3_FORCE_TORCH_ITERATIVE_MOE:-1}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
export SWIFT_MEGATRON_FORCE_OS_EXIT_AFTER_MAIN="${SWIFT_MEGATRON_FORCE_OS_EXIT_AFTER_MAIN:-1}"
if [[ -n "${MIMO_CONFIG}" ]]; then
    export QWEN3OMNI_MIMO_CONFIG="${MIMO_CONFIG}"
fi
export QWEN3OMNI_MIMO_ALLOW_FALLBACK="${QWEN3OMNI_MIMO_ALLOW_FALLBACK:-1}"
export QWEN3OMNI_MIMO_DISABLE_AFTER_AUTH_FAILURE="${QWEN3OMNI_MIMO_DISABLE_AFTER_AUTH_FAILURE:-1}"
export QWEN3OMNI_MIMO_TIMEOUT_SEC="${QWEN3OMNI_MIMO_TIMEOUT_SEC:-20}"
export QWEN3OMNI_MIMO_MAX_RETRIES="${QWEN3OMNI_MIMO_MAX_RETRIES:-0}"
export QWEN3OMNI_MIMO_RETRY_SLEEP_SEC="${QWEN3OMNI_MIMO_RETRY_SLEEP_SEC:-1}"
export QWEN3OMNI_MIMO_MAX_TOKENS="${QWEN3OMNI_MIMO_MAX_TOKENS:-8}"
export QWEN3OMNI_MIMO_WARN_LIMIT="${QWEN3OMNI_MIMO_WARN_LIMIT:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export MASTER_PORT
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/torchinductor_${USER}_${SLURM_JOB_ID:-manual}_8g_mimo}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_${USER}_${SLURM_JOB_ID:-manual}_8g_mimo}"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"

echo "[job] started_at=$(date --iso-8601=seconds)"
echo "[job] hostname=$(hostname)"
echo "[job] pwd=$(pwd)"
echo "[job] slurm_job_id=${SLURM_JOB_ID:-}"
echo "[job] slurm_step_id=${SLURM_STEP_ID:-}"
echo "[job] slurm_nodelist=${SLURM_NODELIST:-}"
echo "[job] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[job] model=${MODEL_DIR}"
echo "[job] dataset=${DATASET}"
echo "[job] reward_plugin=${REWARD_PLUGIN}"
echo "[job] mimo_config=${QWEN3OMNI_MIMO_CONFIG:-<environment>}"
echo "[job] output_dir=${OUTPUT_DIR}"
echo "[job] train_iters=${TRAIN_ITERS}"
echo "[job] global_batch_size=${GLOBAL_BATCH_SIZE}"
echo "[job] num_generations=${NUM_GENERATIONS}"
wc -l "${DATASET}"

echo "[env] conda_prefix=${CONDA_PREFIX:-<unset>}"
echo "[env] python=$(command -v python)"
python --version
echo "[env] megatron=$(command -v megatron || true)"
echo "[env] cuda_home=${CUDA_HOME:-<unset>}"
echo "[env] nvcc=$(command -v nvcc || true)"
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 4 || true
fi
echo "[env] VLLM_QWEN3_FORCE_TORCH_ITERATIVE_MOE=${VLLM_QWEN3_FORCE_TORCH_ITERATIVE_MOE}"
echo "[env] VLLM_USE_V1=${VLLM_USE_V1}"
echo "[env] SWIFT_MEGATRON_FORCE_OS_EXIT_AFTER_MAIN=${SWIFT_MEGATRON_FORCE_OS_EXIT_AFTER_MAIN}"
echo "[env] proxy=${https_proxy:-<unset>}"
nvidia-smi

echo "[preflight] imports and reward registration"
python - <<'PY'
import importlib
import os
import sys

import torch

print("python_executable=", sys.executable)
print("torch=", torch.__version__, "cuda=", torch.version.cuda, "cuda_available=", torch.cuda.is_available(), "device_count=", torch.cuda.device_count())

for name in ["transformers", "swift", "vllm", "megatron.core", "transformer_engine", "transformer_engine_torch", "peft", "requests", "yaml"]:
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "<no __version__>")
    path = getattr(mod, "__file__", "<namespace>")
    print(f"{name}= {version} @ {path}")

from swift.rewards import orms
from qwen3omni_train.rewards.mimo_judge_reward import Qwen3OmniMimoJudge

reward = Qwen3OmniMimoJudge()
print("reward_registered=", "qwen3omni_mimo_judge" in orms)
print("reward_config_summary=", reward.config_summary())
print("PYTHONPATH=", os.environ.get("PYTHONPATH"))
PY

echo "[preflight] mimo api text judge"
python - <<'PY'
from qwen3omni_train.rewards.mimo_judge_reward import Qwen3OmniMimoJudge

reward = Qwen3OmniMimoJudge()
scores = reward(
    ["A"],
    solution=["A"],
    messages=[[{"role": "user", "content": "Options:\nA. yes\nB. no\nReturn exactly one uppercase option letter."}]],
    meta=[{"answer_text": "yes"}],
)
print("mimo_preflight_scores=", scores)
print("mimo_preflight_api_success_count=", reward.api_success_count)
print("mimo_preflight_api_failure_count=", reward.api_failure_count)
if reward.api_success_count < 1:
    print("mimo_preflight_warning=API judge did not succeed; training will continue with exact-match fallback.")
PY

echo "[train] megatron rlhf starting"
megatron rlhf \
    --rlhf_type grpo \
    --model "${MODEL_DIR}" \
    --save_safetensors true \
    --merge_lora false \
    --context_parallel_size 1 \
    --tensor_model_parallel_size 2 \
    --expert_model_parallel_size 4 \
    --pipeline_model_parallel_size 1 \
    --dataset "${DATASET}" \
    --train_iters "${TRAIN_ITERS}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --micro_batch_size "${MICRO_BATCH_SIZE}" \
    --steps_per_generation "${STEPS_PER_GENERATION}" \
    --num_generations "${NUM_GENERATIONS}" \
    --external_plugins "${REWARD_PLUGIN}" \
    --reward_funcs qwen3omni_mimo_judge \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
    --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
    --vllm_enforce_eager true \
    --vllm_enable_prefix_caching false \
    --vllm_limit_mm_per_prompt '{"image": 1, "audio": 1}' \
    --vllm_engine_kwargs '{"compilation_config":{"level":0,"use_inductor":false,"use_cudagraph":false}}' \
    --max_length "${MAX_LENGTH}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --tuner_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --freeze_aligner true \
    --lr 1e-5 \
    --bf16 true \
    --beta 0.0 \
    --importance_sampling_level token \
    --epsilon 0.2 \
    --loss_type grpo \
    --dynamic_sample false \
    --overlong_filter true \
    --sleep_level 2 \
    --offload_model true \
    --offload_optimizer true \
    --offload_bridge false \
    --logging_steps "${LOGGING_STEPS}" \
    --recompute_granularity selective \
    --finetune true \
    --dataloader_num_workers 1 \
    --dataset_num_proc 1 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend unfused \
    --gradient_accumulation_fusion false \
    --sequence_parallel true \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --padding_free false \
    --log_completions true \
    --report_to tensorboard \
    --eval_iters 0 \
    --eval_steps 1000 \
    --save_steps "${SAVE_STEPS}" \
    --output_dir "${OUTPUT_DIR}"

echo "[job] finished_at=$(date --iso-8601=seconds)"
