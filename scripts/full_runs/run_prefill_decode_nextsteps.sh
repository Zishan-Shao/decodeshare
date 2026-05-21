#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

cd "${REPO_ROOT}"

SCRIPT="${REPO_ROOT}/experiments/04_prefill_decode/run_prefill_decode_reasoning_sweeps.py"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/04_prefill_decode/prefill_decode_sweeps}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --model "${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
  --device "${DEVICE}"
  --dtype "${MODEL_DTYPE:-fp32}"
  --layer "${LAYER:-10}"
  --tasks "${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}"
  --eval_tasks "${EVAL_TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}"
  --n_prompts "${N_PROMPTS:-256}"
  --eval_n "${N_EVAL:-2048}"
  --max_prompt_len "${MAX_PROMPT_LEN:-512}"
  --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW_TOKENS:-128}"
  --per_task_max_states "${PER_TASK_MAX_STATES:-20000}"
  --do_generation "${DO_GENERATION:-0}"
  --fc_warmup_tokens "${FC_WARMUP_TOKENS:-0}"
  --fc_prefix_mode "${FC_PREFIX_MODE:-auto}"
  --fc_answer_prefix "${FC_ANSWER_PREFIX:-${FINAL_ANSWER_PREFIX_DEFAULT}}"
  --bootstrap_iters "${BOOTSTRAP_ITERS:-5000}"
  --perm_iters "${PERM_ITERS:-10000}"
  --seed "${SEED:-42}"
  --fc_save_scores "${FC_SAVE_SCORES:-0}"
)

for ALPHA in ${ALPHAS:-0.25 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.90 1.00}; do
  TAG="alpha_${ALPHA}"
  run_python_gpu "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --alpha_remove "${ALPHA}" \
    --out_json "${OUT_DIR}/${TAG}.json" \
    --out_txt "${OUT_DIR}/${TAG}.txt" \
    --out_md "${OUT_DIR}/${TAG}.md" \
    --out_tex "${OUT_DIR}/${TAG}.tex"
done

for K in ${KS:-16 32 64 128}; do
  TAG="k_${K}"
  run_python_gpu "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --alpha_remove 1.0 \
    --k_eval "${K}" \
    --out_json "${OUT_DIR}/${TAG}.json" \
    --out_txt "${OUT_DIR}/${TAG}.txt" \
    --out_md "${OUT_DIR}/${TAG}.md" \
    --out_tex "${OUT_DIR}/${TAG}.tex"
done
