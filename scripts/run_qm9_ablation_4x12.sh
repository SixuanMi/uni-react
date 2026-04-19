#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ------------------------------------------------------------
# Usage:
#   bash scripts/run_qm9_ablation_4x12.sh
#
# Optional env overrides:
#   GPU_LIST="0,1,2,3"      # physical GPU ids
#   MAX_CONCURRENT=4         # max concurrent ablation jobs (<= number of GPUs)
#   LOG_DIR="runs/qm9_ablation_logs"
#   BATCH_SIZE=16 NUM_WORKERS=16 EPOCHS=100 ...
#
# This launcher reuses scripts/train_qm9_all_targets.sh for each ablation.
# Each ablation job runs 12 single-target finetunes sequentially.
# ------------------------------------------------------------

GPU_LIST_STR="${GPU_LIST:-0,1,2,3}"
IFS=',' read -r -a GPUS <<< "${GPU_LIST_STR}"
if [[ "${#GPUS[@]}" -eq 0 ]]; then
  echo "ERROR: GPU_LIST is empty." >&2
  exit 2
fi

MAX_CONCURRENT="${MAX_CONCURRENT:-${#GPUS[@]}}"
if (( MAX_CONCURRENT < 1 )); then
  echo "ERROR: MAX_CONCURRENT must be >= 1, got ${MAX_CONCURRENT}" >&2
  exit 2
fi
if (( MAX_CONCURRENT > ${#GPUS[@]} )); then
  MAX_CONCURRENT="${#GPUS[@]}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-runs/qm9_ablation_logs_${TS}}"
mkdir -p "${LOG_DIR}"

ABLATION_NAMES=(
  "baseline"
  "lidi-node"
  "lidi-edge"
  "lidi-full"
)
ABLATION_CKPTS=(
  "runs/ablation_stage1_baseline/best.pt"
  "runs/ablation_stage1_lidi_node/best.pt"
  "runs/ablation_stage1_lidi_edge/best.pt"
  "runs/ablation_stage1_lidi_full/best.pt"
)

if [[ "${#ABLATION_NAMES[@]}" -ne "${#ABLATION_CKPTS[@]}" ]]; then
  echo "ERROR: internal ablation list mismatch." >&2
  exit 2
fi

for ckpt in "${ABLATION_CKPTS[@]}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "ERROR: missing checkpoint: ${ckpt}" >&2
    exit 2
  fi
done

echo "============================================================"
echo "QM9 ablation launcher"
echo "repo: ${REPO_ROOT}"
echo "gpus: ${GPU_LIST_STR}"
echo "max_concurrent: ${MAX_CONCURRENT}"
echo "log_dir: ${LOG_DIR}"
echo "worker script: scripts/train_qm9_all_targets.sh"
echo "============================================================"

PIDS=()
PID_NAMES=()
PID_LOGS=()
FAILURES=()

reap_finished_one() {
  local i pid status name log
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ! kill -0 "${pid}" 2>/dev/null; then
      set +e
      wait "${pid}"
      status=$?
      set -e
      name="${PID_NAMES[$i]}"
      log="${PID_LOGS[$i]}"
      if (( status == 0 )); then
        echo "[done] ${name} (pid=${pid})"
      else
        echo "[fail] ${name} (pid=${pid}, code=${status})" >&2
        FAILURES+=("${name}:${log}")
      fi
      unset 'PIDS[i]' 'PID_NAMES[i]' 'PID_LOGS[i]'
      PIDS=("${PIDS[@]}")
      PID_NAMES=("${PID_NAMES[@]}")
      PID_LOGS=("${PID_LOGS[@]}")
      return 0
    fi
  done
  return 1
}

launch_job() {
  local idx="$1"
  local gpu="$2"
  local name="${ABLATION_NAMES[$idx]}"
  local ckpt="${ABLATION_CKPTS[$idx]}"
  local out_prefix="qm9_pretrain_ablation_stage1_${name}"
  local log="${LOG_DIR}/${name}.log"

  echo "[start] ${name} on GPU ${gpu} -> ${log}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export NPROC_PER_NODE=1
    export PRETRAINED_CKPT="${ckpt}"
    export PRETRAINED_STRICT=0
    export ENCODER_TYPE="${ENCODER_TYPE:-single_mol}"
    export OUT_PREFIX="${out_prefix}"
    bash "${SCRIPT_DIR}/train_qm9_all_targets.sh"
  ) >"${log}" 2>&1 &

  local pid=$!
  PIDS+=("${pid}")
  PID_NAMES+=("${name}")
  PID_LOGS+=("${log}")
}

next_idx=0
total_jobs="${#ABLATION_NAMES[@]}"

while (( next_idx < total_jobs )) || (( ${#PIDS[@]} > 0 )); do
  while (( next_idx < total_jobs )) && (( ${#PIDS[@]} < MAX_CONCURRENT )); do
    gpu="${GPUS[$(( next_idx % ${#GPUS[@]} ))]}"
    launch_job "${next_idx}" "${gpu}"
    next_idx=$((next_idx + 1))
  done

  if (( ${#PIDS[@]} > 0 )); then
    if ! reap_finished_one; then
      sleep 5
    fi
  fi
done

if (( ${#FAILURES[@]} > 0 )); then
  echo "============================================================" >&2
  echo "Some ablation jobs failed:" >&2
  for item in "${FAILURES[@]}"; do
    echo "  - ${item}" >&2
  done
  echo "============================================================" >&2
  exit 1
fi

echo "============================================================"
echo "All ablation jobs finished successfully."
echo "Logs: ${LOG_DIR}"
echo "============================================================"
