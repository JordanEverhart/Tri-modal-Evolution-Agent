#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=/public/home/202492301216/Workplace/Tri-modal-Evolution-Agent/Benchmark_test
WORKSPACE_ROOT=/public/home/202492301216/Workplace
LOG_ROOT="${WORKSPACE_ROOT}/logs"
RUN_NAME=qwen3_omni_30b_instruct_20260421_seq
MODEL_CONFIG="${REPO_ROOT}/configs/models/qwen3_omni_30b_a3b_instruct.yaml"
AGG_SUMMARY="${REPO_ROOT}/outputs/${RUN_NAME}_aggregate_summary.json"
SERVICE_LOG=/tmp/tri_agent_serve.log
SERVICE_PID=/tmp/tri_agent_serve.pid
DATASETS=("worldsense" "omnibench")

mkdir -p "${LOG_ROOT}"

source "${WORKSPACE_ROOT}/miniconda3/etc/profile.d/conda.sh"
conda activate qwen3omni

export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"

cleanup() {
  if [[ -f "${SERVICE_PID}" ]]; then
    local pid
    pid="$(cat "${SERVICE_PID}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
    fi
  fi
}
trap cleanup EXIT

start_service() {
  cleanup
  rm -f "${SERVICE_PID}"
  : > "${SERVICE_LOG}"
  nohup tri-agent-serve --model-config "${MODEL_CONFIG}" > "${SERVICE_LOG}" 2>&1 < /dev/null &
  echo "$!" > "${SERVICE_PID}"
}

wait_for_service() {
  python - <<'PY'
import sys
import time
import requests

s = requests.Session()
s.trust_env = False
last_error = None
for _ in range(180):
    try:
        r = s.get("http://127.0.0.1:18080/health", timeout=10)
        if r.ok:
            print(r.text, flush=True)
            sys.exit(0)
        last_error = f"{r.status_code} {r.text}"
    except Exception as exc:  # noqa: BLE001
        last_error = repr(exc)
    time.sleep(5)
raise SystemExit(f"service did not become ready: {last_error}")
PY
}

prepare_dataset_output() {
  local dataset="$1"
  local output_dir="${REPO_ROOT}/outputs/${dataset}/${RUN_NAME}"
  if [[ -d "${output_dir}" ]]; then
    local backup_dir="${output_dir}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "${output_dir}" "${backup_dir}"
    echo "[batch] backed up ${output_dir} -> ${backup_dir} $(date --iso-8601=seconds)"
  fi
}

run_dataset() {
  local dataset="$1"
  local attempts=0
  local resume_flag=""
  while true; do
    attempts=$((attempts + 1))
    start_service
    wait_for_service
    echo "[batch] starting ${dataset} attempt=${attempts} $(date --iso-8601=seconds)"
    if tri-agent eval \
      --model-config "${MODEL_CONFIG}" \
      --dataset-config "configs/datasets/${dataset}.yaml" \
      --run-name "${RUN_NAME}" \
      ${resume_flag}; then
      echo "[batch] finished ${dataset} status=0 $(date --iso-8601=seconds)"
      cleanup
      return 0
    fi

    if [[ "${attempts}" -ge 5 ]]; then
      echo "[batch] finished ${dataset} status=1 attempts=${attempts} $(date --iso-8601=seconds)"
      return 1
    fi

    echo "[batch] retrying ${dataset} after failure $(date --iso-8601=seconds)"
    resume_flag="--resume"
    cleanup
    sleep 5
  done
}

for dataset in "${DATASETS[@]}"; do
  prepare_dataset_output "${dataset}"
  run_dataset "${dataset}"
done

python - <<'PY'
import json
from pathlib import Path

workspace_root = Path("/public/home/202492301216/Workplace/Tri-modal-Evolution-Agent/Benchmark_test")
run_name = "qwen3_omni_30b_instruct_20260421_seq"
dataset_keys = ["daily_omni", "worldsense", "omnibench"]
output_path = workspace_root / "outputs" / f"{run_name}_aggregate_summary.json"

per_dataset = {}
total = correct = wrong = invalid = 0
for dataset_key in dataset_keys:
    summary_path = workspace_root / "outputs" / dataset_key / run_name / "summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    per_dataset[dataset_key] = data
    total += int(data.get("total", 0))
    correct += int(data.get("correct", 0))
    wrong += int(data.get("wrong", 0))
    invalid += int(data.get("invalid", 0))

payload = {
    "run_name": run_name,
    "datasets": per_dataset,
    "overall": {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "invalid": invalid,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "invalid_rate": round(invalid / total, 6) if total else 0.0,
    },
}
output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[batch] aggregate_summary {output_path}", flush=True)
PY

echo "[batch] all done $(date --iso-8601=seconds)"
