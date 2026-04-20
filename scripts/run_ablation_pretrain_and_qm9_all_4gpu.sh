#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ------------------------------------------------------------
# End-to-end launcher:
#   Stage-1 geometric pretrain (4 ablations) in parallel on 4 GPUs
#   -> then QM9 all-target multi-task finetune (4 ablations) in parallel
#
# Usage:
#   bash scripts/run_ablation_pretrain_and_qm9_all_4gpu.sh
#
# Optional env overrides:
#   GPU_LIST="0,1,2,3"
#   MAX_CONCURRENT=4
#   RUN_PRETRAIN=1
#   RUN_FINETUNE=1
#   FINETUNE_CONFIG="configs/finetune_qm9_all.yaml"
#   PRETRAIN_STRICT="false"
#   LOG_ROOT="runs/ablation_e2e_logs"
#   PYTHON_BIN="python"
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

RUN_PRETRAIN="${RUN_PRETRAIN:-1}"
RUN_FINETUNE="${RUN_FINETUNE:-1}"
if [[ "${RUN_PRETRAIN}" != "1" && "${RUN_FINETUNE}" != "1" ]]; then
  echo "ERROR: RUN_PRETRAIN and RUN_FINETUNE cannot both be 0." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FINETUNE_CONFIG="${FINETUNE_CONFIG:-configs/finetune_qm9_all.yaml}"
PRETRAIN_STRICT="${PRETRAIN_STRICT:-false}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-runs/ablation_e2e_logs_${TS}}"
PRETRAIN_LOG_DIR="${LOG_ROOT}/pretrain"
FINETUNE_LOG_DIR="${LOG_ROOT}/finetune"
mkdir -p "${PRETRAIN_LOG_DIR}" "${FINETUNE_LOG_DIR}"

ABLATION_NAMES=(
  "baseline"
  "lidi-node"
  "lidi-edge"
  "lidi-full"
)

PRETRAIN_CONFIGS=(
  "configs/single_mol/geometric_ablation_baseline.yaml"
  "configs/single_mol/geometric_ablation_lidi_node.yaml"
  "configs/single_mol/geometric_ablation_lidi_edge.yaml"
  "configs/single_mol/geometric_ablation_lidi_full.yaml"
)

PRETRAIN_OUT_DIRS=(
  "runs/ablation_stage1_baseline"
  "runs/ablation_stage1_lidi_node"
  "runs/ablation_stage1_lidi_edge"
  "runs/ablation_stage1_lidi_full"
)

if [[ "${#ABLATION_NAMES[@]}" -ne "${#PRETRAIN_CONFIGS[@]}" ]] || [[ "${#ABLATION_NAMES[@]}" -ne "${#PRETRAIN_OUT_DIRS[@]}" ]]; then
  echo "ERROR: internal ablation arrays mismatch." >&2
  exit 2
fi

if [[ ! -f "${FINETUNE_CONFIG}" ]]; then
  echo "ERROR: missing finetune config: ${FINETUNE_CONFIG}" >&2
  exit 2
fi

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

run_stage() {
  local stage="$1"
  local total_jobs="${#ABLATION_NAMES[@]}"
  local next_idx=0

  PIDS=()
  PID_NAMES=()
  PID_LOGS=()
  FAILURES=()

  launch_job() {
    local idx="$1"
    local gpu="$2"
    local name="${ABLATION_NAMES[$idx]}"
    local log
    local cmd=()

    if [[ "${stage}" == "pretrain" ]]; then
      local cfg="${PRETRAIN_CONFIGS[$idx]}"
      if [[ ! -f "${cfg}" ]]; then
        echo "ERROR: missing pretrain config: ${cfg}" >&2
        exit 2
      fi
      log="${PRETRAIN_LOG_DIR}/${name}.log"
      cmd=("${PYTHON_BIN}" -m uni_react.train_pretrain_geometric --config "${cfg}")
    elif [[ "${stage}" == "finetune" ]]; then
      local ckpt="${PRETRAIN_OUT_DIRS[$idx]}/best.pt"
      if [[ ! -f "${ckpt}" ]]; then
        echo "ERROR: missing checkpoint for ${name}: ${ckpt}" >&2
        exit 2
      fi
      local out_dir="runs/qm9_pretrain_ablation_stage1_${name}_single_mol_egnn_multi"
      log="${FINETUNE_LOG_DIR}/${name}.log"
      cmd=(
        "${PYTHON_BIN}" -m uni_react.train_finetune_qm9
        --config "${FINETUNE_CONFIG}"
        --pretrained_ckpt "${ckpt}"
        --pretrained_strict "${PRETRAIN_STRICT}"
        --out_dir "${out_dir}"
      )
    else
      echo "ERROR: unknown stage: ${stage}" >&2
      exit 2
    fi

    echo "[start:${stage}] ${name} on GPU ${gpu} -> ${log}"
    (
      set -euo pipefail
      export CUDA_VISIBLE_DEVICES="${gpu}"
      export NPROC_PER_NODE=1
      "${cmd[@]}"
    ) >"${log}" 2>&1 &

    local pid=$!
    PIDS+=("${pid}")
    PID_NAMES+=("${name}")
    PID_LOGS+=("${log}")
  }

  while (( next_idx < total_jobs )) || (( ${#PIDS[@]} > 0 )); do
    while (( next_idx < total_jobs )) && (( ${#PIDS[@]} < MAX_CONCURRENT )); do
      local gpu="${GPUS[$(( next_idx % ${#GPUS[@]} ))]}"
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
    echo "Stage '${stage}' has failures:" >&2
    for item in "${FAILURES[@]}"; do
      echo "  - ${item}" >&2
    done
    echo "============================================================" >&2
    return 1
  fi

  echo "[ok] Stage '${stage}' finished."
  return 0
}

echo "============================================================"
echo "Ablation E2E launcher"
echo "repo: ${REPO_ROOT}"
echo "gpus: ${GPU_LIST_STR}"
echo "max_concurrent: ${MAX_CONCURRENT}"
echo "run_pretrain: ${RUN_PRETRAIN}"
echo "run_finetune: ${RUN_FINETUNE}"
echo "finetune_config: ${FINETUNE_CONFIG}"
echo "logs: ${LOG_ROOT}"
echo "============================================================"

if [[ "${RUN_PRETRAIN}" == "1" ]]; then
  run_stage "pretrain"
fi

if [[ "${RUN_FINETUNE}" == "1" ]]; then
  run_stage "finetune"
fi

echo "============================================================"
echo "All requested stages finished successfully."
echo "Logs: ${LOG_ROOT}"
echo "============================================================"
