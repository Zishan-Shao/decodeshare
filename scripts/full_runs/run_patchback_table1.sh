#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}/experiments/03_patchback"

MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_TAG="${MODEL_TAG:-$(echo "${MODEL}" | tr '/:' '__')}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/03_patchback/table1}"
mkdir -p "${OUT_DIR}"

RUN_TAG="${RUN_TAG:-patchback_${MODEL_TAG}_layer${LAYER:-10}_task${TASK:-aqua}_seed${SEED:-123}}"
QS_OUT="${QS_OUT:-${OUT_DIR}/Q_shared_layer${LAYER:-10}_seed${SEED:-123}.npy}"
QS_ARGS=(--compute_Qs "${COMPUTE_QS:-1}" --Qs_out "${QS_OUT}")
if [[ "${COMPUTE_QS:-1}" != "1" ]]; then
  QS_ARGS+=(--Qs_path "${QS_PATH:?Set QS_PATH when COMPUTE_QS=0}")
fi

run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --dtype "${MODEL_DTYPE:-fp32}" \
  --layer "${LAYER:-10}" \
  --seed "${SEED:-123}" \
  "${QS_ARGS[@]}" \
  --basis_tasks "${BASIS_TASKS:-gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa}" \
  --basis_n_subspace "${N_SUBSPACE:-128}" \
  --task "${TASK:-aqua}" \
  --candidate_labels "${CANDIDATE_LABELS:-ABCDE}" \
  --n_eval "${N_EVAL:-254}" \
  --max_flips "${MAX_FLIPS:-128}" \
  --out_json "${OUT_DIR}/${RUN_TAG}.json"

if [[ "${SUMMARIZE:-1}" == "1" ]]; then
  run_python summarize_patching_jsons.py \
    --dir "${OUT_DIR}" \
    --pattern "*.json" \
    --out_csv "${OUT_DIR}/summary.csv" \
    --out_md "${OUT_DIR}/summary.md" \
    --out_paper_md "${OUT_DIR}/paper_table.md" \
    --out_alpha_csv "${OUT_DIR}/alpha_sweep.csv" \
    --out_alpha_md "${OUT_DIR}/alpha_sweep.md"
fi
