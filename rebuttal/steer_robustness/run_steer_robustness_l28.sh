#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp16}"
LAYER="${LAYER:-28}"

TASKS="${TASKS:-commonsenseqa,arc_challenge,openbookqa,qasc,logiqa}"
TASKS_SUBSPACE="${TASKS_SUBSPACE:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa}"

N_EVAL="${N_EVAL:-128}"
N_SUBSPACE="${N_SUBSPACE:-128}"
MAX_VECTORS="${MAX_VECTORS:-16}"

REPAIR_TEMPLATE_SEEDS="${REPAIR_TEMPLATE_SEEDS:-1234,2345,3456,4567,5678,6789,7890,8901}"
RANK_TEMPLATE_SEEDS="${RANK_TEMPLATE_SEEDS:-1234,2345,3456}"
REAL_TEMPLATE_SEEDS="${REAL_TEMPLATE_SEEDS:-4567,5678,6789,7890,8901}"

MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
REASONING_TOKENS="${REASONING_TOKENS:-64}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-12345}"

POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-0.0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"
BOOTSTRAP_SEED="${BOOTSTRAP_SEED:-123}"

CAA_MANIFEST="${CAA_MANIFEST:-results/rebuttal_mechanism/cross_method_l28_med/manifests/caa.jsonl}"
SAE_MANIFEST="${SAE_MANIFEST:-results/rebuttal_mechanism/cross_method_l28_med/manifests/sae.jsonl}"

CAA_RANKFLIP_JSON="${CAA_RANKFLIP_JSON:-results/rebuttal_mechanism/cross_method_l28_med/rankflip/rankflip_caa.json}"
SAE_RANKFLIP_JSON="${SAE_RANKFLIP_JSON:-results/rebuttal_mechanism/cross_method_l28_med/rankflip/rankflip_sae.json}"
USE_EXISTING_RANKFLIP="${USE_EXISTING_RANKFLIP:-1}"

BASIS_NPZ="${BASIS_NPZ:-results/rebuttal_mechanism/logit_lens_l28/basis_layer28_tseed1234.npz}"
BASIS_Q_NPY="${BASIS_Q_NPY:-results/rebuttal_mechanism/logit_lens_l28/Q_shared_layer28_tseed1234.npy}"

OUT_ROOT="${OUT_ROOT:-results/rebuttal_mechanism/steer_robustness_l28}"
PAIR_DIR="${OUT_ROOT}/paired_repair"
SELECT_DIR="${OUT_ROOT}/selected_deployment"
LOG_DIR="${OUT_ROOT}/logs"

PARALLEL="${PARALLEL:-0}"
RUN_PAIRED_REPAIR="${RUN_PAIRED_REPAIR:-1}"
RUN_SELECTED_DEPLOYMENT="${RUN_SELECTED_DEPLOYMENT:-1}"
GPU_REPAIR_CAA="${GPU_REPAIR_CAA:-}"
GPU_REPAIR_SAE="${GPU_REPAIR_SAE:-}"
GPU_SELECT_CAA="${GPU_SELECT_CAA:-}"
GPU_SELECT_SAE="${GPU_SELECT_SAE:-}"

mkdir -p "${PAIR_DIR}" "${SELECT_DIR}" "${LOG_DIR}"

require_file() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "[Error] Missing file: ${path}" >&2
    exit 1
  fi
}

rankflip_json_is_raw() {
  local path="$1"
  [[ -f "${path}" ]] || return 1
  "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import json
path = ${path@Q}
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
vectors = data.get("vectors", {})
ok = False
if isinstance(vectors, dict) and vectors:
    sample = next(iter(vectors.values()))
    ok = isinstance(sample.get("score_real_summary"), dict) and isinstance(sample["score_real_summary"].get("by_seed"), dict)
    ok = ok and isinstance(sample.get("delta_real_decode_by_seed"), dict)
raise SystemExit(0 if ok else 1)
PY
}

prepare_basis_q() {
  require_file "${BASIS_NPZ}"
  if [[ ! -f "${BASIS_Q_NPY}" ]]; then
    echo "[Prep] Extracting Q from ${BASIS_NPZ} -> ${BASIS_Q_NPY}"
    "${PYTHON_BIN}" - <<PY
import os
import numpy as np
src = os.path.expanduser("${BASIS_NPZ}")
dst = os.path.expanduser("${BASIS_Q_NPY}")
os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
z = np.load(src)
if "Q" not in z:
    raise KeyError(f"{src} does not contain key 'Q'")
np.save(dst, np.asarray(z["Q"], dtype=np.float32))
print(f"[Saved] {dst}")
PY
  fi
  require_file "${BASIS_Q_NPY}"
}

run_cmd() {
  local gpu="$1"
  local log="$2"
  shift 2
  if [[ -n "${gpu}" ]]; then
    echo "[Run] CUDA_VISIBLE_DEVICES=${gpu} $*"
    CUDA_VISIBLE_DEVICES="${gpu}" "$@" 2>&1 | tee "${log}"
  else
    echo "[Run] $*"
    "$@" 2>&1 | tee "${log}"
  fi
}

launch_or_run() {
  local gpu="$1"
  local log="$2"
  shift 2
  if [[ "${PARALLEL}" == "1" && -n "${gpu}" ]]; then
    run_cmd "${gpu}" "${log}" "$@" &
    PIDS+=($!)
  else
    run_cmd "${gpu}" "${log}" "$@"
  fi
}

run_paired_repair() {
  local tag="$1"
  local manifest="$2"
  local gpu="$3"
  require_file "${manifest}"

  local out_json="${PAIR_DIR}/paired_repair_${tag}.json"
  local out_md="${PAIR_DIR}/paired_repair_${tag}.md"
  local log="${LOG_DIR}/paired_repair_${tag}.log"

  launch_or_run "${gpu}" "${log}" \
    "${PYTHON_BIN}" rebuttal/steer_robustness/exp_steer_robustness_paired_repair.py \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --model_dtype "${DTYPE}" \
      --vectors_manifest "${manifest}" \
      --max_vectors "${MAX_VECTORS}" \
      --tasks_eval "${TASKS}" \
      --tasks_subspace "${TASKS_SUBSPACE}" \
      --n_eval "${N_EVAL}" \
      --n_subspace "${N_SUBSPACE}" \
      --template_seeds "${REPAIR_TEMPLATE_SEEDS}" \
      --decoding greedy \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --reasoning_tokens "${REASONING_TOKENS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_prompt_len "${MAX_PROMPT_LEN}" \
      --sample_seed "${SAMPLE_SEED}" \
      --staged 1 \
      --alpha_proj 1.0 \
      --norm_match 1 \
      --basis_layers "${LAYER}" \
      --shared_basis_npy_pattern "${BASIS_Q_NPY}" \
      --include_pca_prefill 1 \
      --pca_var 0.95 \
      --pca_max_rows 200000 \
      --pca_max_dim 4096 \
      --per_task_max_states 20000 \
      --tau 0.001 \
      --m_shared all \
      --positive_threshold "${POSITIVE_THRESHOLD}" \
      --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
      --bootstrap_seed "${BOOTSTRAP_SEED}" \
      --seed "${SEED}" \
      --resume_json "${out_json}" \
      --save_every_vectors 1 \
      --out_json "${out_json}" \
      --out_md "${out_md}"
}

run_selected_deployment() {
  local tag="$1"
  local manifest="$2"
  local rankflip_json="$3"
  local gpu="$4"
  local out_json="${SELECT_DIR}/selected_deployment_${tag}.json"
  local out_md="${SELECT_DIR}/selected_deployment_${tag}.md"
  local log="${LOG_DIR}/selected_deployment_${tag}.log"

  if [[ "${USE_EXISTING_RANKFLIP}" == "1" && -n "${rankflip_json}" ]] && rankflip_json_is_raw "${rankflip_json}"; then
    launch_or_run "${gpu}" "${log}" \
      "${PYTHON_BIN}" rebuttal/steer_robustness/exp_steer_robustness_selected_deployment.py \
        --rankflip_json "${rankflip_json}" \
        --positive_threshold "${POSITIVE_THRESHOLD}" \
        --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
        --bootstrap_seed "${BOOTSTRAP_SEED}" \
        --out_json "${out_json}" \
        --out_md "${out_md}"
    return
  fi
  if [[ "${USE_EXISTING_RANKFLIP}" == "1" && -n "${rankflip_json}" && -f "${rankflip_json}" ]]; then
    echo "[Info] ${rankflip_json} is not a raw rankflip JSON; falling back to direct rerun."
  fi

  require_file "${manifest}"
  launch_or_run "${gpu}" "${log}" \
    "${PYTHON_BIN}" rebuttal/steer_robustness/exp_steer_robustness_selected_deployment.py \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --model_dtype "${DTYPE}" \
      --vectors_manifest "${manifest}" \
      --max_vectors "${MAX_VECTORS}" \
      --tasks "${TASKS}" \
      --n_eval "${N_EVAL}" \
      --template_seeds_rank "${RANK_TEMPLATE_SEEDS}" \
      --template_seeds_real "${REAL_TEMPLATE_SEEDS}" \
      --decoding greedy \
      --max_new_tokens "${MAX_NEW_TOKENS}" \
      --reasoning_tokens "${REASONING_TOKENS}" \
      --batch_size "${BATCH_SIZE}" \
      --max_prompt_len "${MAX_PROMPT_LEN}" \
      --sample_seed "${SAMPLE_SEED}" \
      --trad_mode prefill \
      --decode_mode decode \
      --staged 1 \
      --agg mean \
      --positive_threshold "${POSITIVE_THRESHOLD}" \
      --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
      --bootstrap_seed "${BOOTSTRAP_SEED}" \
      --seed "${SEED}" \
      --out_json "${out_json}" \
      --out_md "${out_md}"
}

PIDS=()

prepare_basis_q

if [[ "${RUN_PAIRED_REPAIR}" == "1" ]]; then
  run_paired_repair "caa" "${CAA_MANIFEST}" "${GPU_REPAIR_CAA}"
  run_paired_repair "sae" "${SAE_MANIFEST}" "${GPU_REPAIR_SAE}"
fi

if [[ "${RUN_SELECTED_DEPLOYMENT}" == "1" ]]; then
  run_selected_deployment "caa" "${CAA_MANIFEST}" "${CAA_RANKFLIP_JSON}" "${GPU_SELECT_CAA}"
  run_selected_deployment "sae" "${SAE_MANIFEST}" "${SAE_RANKFLIP_JSON}" "${GPU_SELECT_SAE}"
fi

if [[ "${#PIDS[@]}" -gt 0 ]]; then
  echo "[Wait] Waiting for ${#PIDS[@]} background jobs ..."
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
  done
fi

echo "[Done] Outputs under ${OUT_ROOT}"
