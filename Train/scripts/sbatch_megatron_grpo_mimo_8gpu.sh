#!/usr/bin/env bash
#SBATCH --job-name=q3o_grpo_mimo
#SBATCH --partition=gpuA800
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --output=logs/q3o_grpo_mimo_%j.out
#SBATCH --error=logs/q3o_grpo_mimo_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_ROOT="${TRAIN_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${TRAIN_ROOT}"
mkdir -p logs

bash scripts/run_megatron_grpo_mimo_8gpu.sh
