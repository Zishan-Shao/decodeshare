#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Next steps sweeps for prefill-vs-decode (forced-choice)
#
# 1) alpha sweep: alpha_remove ∈ {0.25, 0.5, 0.75, 1.0}
# 2) k sweep:     k_eval ∈ {16, 32, 64, 126}
# 3) qasc eval_n=2048 (optionally eval only qasc)
#
# REQUIREMENTS:
#   - Use the patched script:
#       prefill_vs_decode_alignment_experiment_reasoning_fixed_sweeps_metrics.py
#     which supports:
#       --eval_tasks, --k_eval, --fc_save_scores
# ------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/src/prefill_vs_decode_alignment_experiment_reasoning_fixed_sweeps_metrics.py"

# Model/config
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp32}"
LAYER="${LAYER:-10}"

# Basis tasks (used for BOTH basis estimation and default evaluation pool)
TASKS="${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}"

# For the sweeps, it's usually enough to evaluate only the "not significant / floor-ish" tasks
# (still estimating bases on the full TASKS list above).
# EVAL_TASKS_SWEEP="${EVAL_TASKS_SWEEP:-openbookqa,qasc,logiqa}"
EVAL_TASKS_SWEEP="${EVAL_TASKS_SWEEP:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}"

# Data sizes
N_PROMPTS="${N_PROMPTS:-256}"   # basis estimation prompts per task
EVAL_N="${EVAL_N:-2048}"         # eval examples per task (unless you override in step 3)
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"

# FC settings (match your current run)
FC_WARMUP="${FC_WARMUP:-0}"
FC_PREFIX_MODE="${FC_PREFIX_MODE:-auto}"   # auto|always|never
FC_PREFIX="${FC_PREFIX:-$'\nFinal answer:'}"

# Shared-subspace calibration collection
CALIB_DECODE_MAX_NEW="${CALIB_DECODE_MAX_NEW:-128}"
PER_TASK_MAX_STATES="${PER_TASK_MAX_STATES:-20000}"

# Stats (can reduce if you're just scouting)
BOOT_ITERS="${BOOT_ITERS:-5000}"
PERM_ITERS="${PERM_ITERS:-10000}"
SEED="${SEED:-42}"

OUT_DIR="${OUT_DIR:-results/prefill_decode_nextsteps}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --model "${MODEL}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --layer "${LAYER}"
  --tasks "${TASKS}"
  --n_prompts "${N_PROMPTS}"
  --eval_n "${EVAL_N}"
  --max_prompt_len "${MAX_PROMPT_LEN}"
  --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW}"
  --per_task_max_states "${PER_TASK_MAX_STATES}"
  --do_generation 0
  --fc_warmup_tokens "${FC_WARMUP}"
  --fc_prefix_mode "${FC_PREFIX_MODE}"
  --fc_answer_prefix "${FC_PREFIX}"
  --bootstrap_iters "${BOOT_ITERS}"
  --perm_iters "${PERM_ITERS}"
  --seed "${SEED}"
  # Save per-example candidate scores if you want deeper analysis.
  # Metrics (gold_logprob/margin) are saved regardless.
  --fc_save_scores 0
)

echo "[Run] Using script: ${SCRIPT}"
echo "[Run] OUT_DIR=${OUT_DIR}"
echo

# # -------------------------
# # (1) alpha sweep
# # -------------------------
# ALPHAS=(0.25 0.50 0.75 1.00)
# for A in "${ALPHAS[@]}"; do
#   TAG="alpha_${A}"
#   CUDA_VISIBLE_DEVICES=1 python "${SCRIPT}" \
#     "${COMMON_ARGS[@]}" \
#     --eval_tasks "${EVAL_TASKS_SWEEP}" \
#     --alpha_remove "${A}" \
#     --out_json "${OUT_DIR}/${TAG}.json" \
#     --out_txt  "${OUT_DIR}/${TAG}.txt" \
#     --out_md   "${OUT_DIR}/${TAG}.md" \
#     --out_tex  "${OUT_DIR}/${TAG}.tex"
# done


# 只改这段即可：更密的 alpha grid（重点扫 0.5~0.8）
ALPHAS=(0.25 0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.90 1.00)

for A in "${ALPHAS[@]}"; do
  TAG="alpha_${A}"
  CUDA_VISIBLE_DEVICES=1 python "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --eval_tasks "${EVAL_TASKS_SWEEP}" \
    --alpha_remove "${A}" \
    --out_json "${OUT_DIR}/${TAG}.json" \
    --out_txt  "${OUT_DIR}/${TAG}.txt" \
    --out_md   "${OUT_DIR}/${TAG}.md" \
    --out_tex  "${OUT_DIR}/${TAG}.tex"
done


# -------------------------
# (2) k sweep (dimension-matched k)
# -------------------------
# NOTE: k_eval must be <= the run's k_match_raw, otherwise it will clamp with a warning.
KS=(16 32 64 128)
for K in "${KS[@]}"; do
  TAG="k_${K}"
  CUDA_VISIBLE_DEVICES=1 python "${SCRIPT}" \
    "${COMMON_ARGS[@]}" \
    --eval_tasks "${EVAL_TASKS_SWEEP}" \
    --alpha_remove 1.0 \
    --k_eval "${K}" \
    --out_json "${OUT_DIR}/${TAG}.json" \
    --out_txt  "${OUT_DIR}/${TAG}.txt" \
    --out_md   "${OUT_DIR}/${TAG}.md" \
    --out_tex  "${OUT_DIR}/${TAG}.tex"
done

# # -------------------------
# # (3) qasc larger eval_n
# # -------------------------
# # "Strictly comparable" option: keep TASKS as the full set, just evaluate qasc, and bump eval_n.
# # This will still load eval_n for all tasks (because the loader is per-task), but only *evaluates* qasc.
# TAG="qasc_evaln_2048"
# CUDA_VISIBLE_DEVICES=1 python "${SCRIPT}" \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" --layer "${LAYER}" \
#   --tasks "${TASKS}" \
#   --eval_tasks "qasc" \
#   --n_prompts "${N_PROMPTS}" \
#   --eval_n 2048 \
#   --max_prompt_len "${MAX_PROMPT_LEN}" \
#   --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW}" \
#   --per_task_max_states "${PER_TASK_MAX_STATES}" \
#   --do_generation 0 \
#   --fc_warmup_tokens "${FC_WARMUP}" \
#   --fc_prefix_mode "${FC_PREFIX_MODE}" \
#   --fc_answer_prefix "${FC_PREFIX}" \
#   --alpha_remove 1.0 \
#   --bootstrap_iters "${BOOT_ITERS}" \
#   --perm_iters "${PERM_ITERS}" \
#   --seed "${SEED}" \
#   --fc_save_scores 0 \
#   --out_json "${OUT_DIR}/${TAG}.json" \
#   --out_txt  "${OUT_DIR}/${TAG}.txt" \
#   --out_md   "${OUT_DIR}/${TAG}.md" \
#   --out_tex  "${OUT_DIR}/${TAG}.tex"

echo
echo "[Done] Results written under: ${OUT_DIR}"
