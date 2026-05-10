#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# User-editable config
###############################################################################
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp32}"              # bf16 recommended for both Qwen/Falcon if supported
LAYER="${LAYER:-4}"
SEED="${SEED:-123}"

# Your base script (shared hooks / Qs compute helper)
BASE_SCRIPT_PATH="${BASE_SCRIPT_PATH:-subspace_patching_transfer.py}"

# Shared subspace basis
QS_PATH="${QS_PATH:-Q_shared_layer10.npy}"

# Eval sizing
N_EVAL="${N_EVAL:-256}"
MAX_FLIPS="${MAX_FLIPS:-64}"
PATCH_N_STEPS="${PATCH_N_STEPS:-4}"

# Token limits
GOLD_MAX_TOKENS_HE="${GOLD_MAX_TOKENS_HE:-128}"
MAX_NEW_TOKENS_MATH="${MAX_NEW_TOKENS_MATH:-96}"
MAX_NEW_TOKENS_CODE="${MAX_NEW_TOKENS_CODE:-256}"

# Datasets
HUMANEVAL_HF_ID="${HUMANEVAL_HF_ID:-openai_humaneval}"
HUMANEVAL_HF_SPLIT="${HUMANEVAL_HF_SPLIT:-test}"

# Output root
OUT_ROOT="${OUT_ROOT:-results/openanswer_full}"

###############################################################################
# Model IDs (edit if you want other sizes/variants)
###############################################################################
# Qwen
QWEN_MATH_MODEL="${QWEN_MATH_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
QWEN_CODE_MODEL="${QWEN_CODE_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}"

# Falcon
FALCON_MODEL="${FALCON_MODEL:-tiiuae/falcon-7b-instruct}"

###############################################################################
# Scripts (must exist)
###############################################################################
QWEN_SCRIPT="${QWEN_SCRIPT:-openanswer_subspace_patching_qwen.py}"
FALCON_SCRIPT="${FALCON_SCRIPT:-openanswer_subspace_patching_falcon.py}"

###############################################################################
# Prompting (tuned to make generation “obedient”)
###############################################################################
# Qwen: keep simple system; you can override via env if needed
QWEN_SYS_PROMPT="${QWEN_SYS_PROMPT:-You are a helpful assistant.}"

# Falcon: two variants for math/code
FALCON_SYS_MATH_STRICT="${FALCON_SYS_MATH_STRICT:-You are a helpful assistant. For math problems: output ONLY the final numeric answer. No words, no units, no punctuation, no explanation.}"
FALCON_SYS_MATH_COT="${FALCON_SYS_MATH_COT:-You are a helpful assistant. Solve the problem step by step. At the end, output a single line exactly: Final answer (number only): <number>}"
FALCON_SYS_CODE="${FALCON_SYS_CODE:-You are a senior CUDA_VISIBLE_DEVICES=3 python engineer. Return ONLY valid CUDA_VISIBLE_DEVICES=3 python code. Do NOT include Markdown fences, explanations, or commentary.}"

###############################################################################
# Helpers
###############################################################################
die() { echo "[Error] $*" >&2; exit 1; }

need_file() {
  [[ -f "$1" ]] || die "Missing file: $1"
}

run_cmd() {
  echo ""
  echo "=================================================================="
  echo "[Run] $*"
  echo "=================================================================="
  "$@"
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

###############################################################################
# Checks
###############################################################################
need_file "$BASE_SCRIPT_PATH"
need_file "$QS_PATH"
need_file "$QWEN_SCRIPT"
need_file "$FALCON_SCRIPT"

mkdir -p "$OUT_ROOT"

###############################################################################
# Common args (shared across runs)
###############################################################################
COMMON_ARGS=(
  --base_script_path "$BASE_SCRIPT_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --layer "$LAYER"
  --seed "$SEED"
  --Qs_path "$QS_PATH"
  --patch_n_steps "$PATCH_N_STEPS"
  --n_eval "$N_EVAL"
  --max_flips "$MAX_FLIPS"
  --prompt_format chat
)

###############################################################################
# Qwen runs
###############################################################################
run_qwen() {
  local tag="qwen"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"

  local log_dir="$out_dir/logs"
  mkdir -p "$log_dir"
  local ts
  ts="$(timestamp)"

  # 1) GSM8K pair_logprob
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$QWEN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$QWEN_MATH_MODEL" \
    --system_prompt "$QWEN_SYS_PROMPT" \
    --task gsm8k \
    --eval_mode pair_logprob \
    --out_json "$out_dir/gsm8k_pairlogprob.json" \
    2>&1 | tee "$log_dir/${ts}_gsm8k_pairlogprob.log"

  # 2) GSM8K gen_math
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$QWEN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$QWEN_MATH_MODEL" \
    --system_prompt "$QWEN_SYS_PROMPT" \
    --task gsm8k \
    --eval_mode gen_math \
    --max_new_tokens "$MAX_NEW_TOKENS_MATH" \
    --out_json "$out_dir/gsm8k_genmath.json" \
    2>&1 | tee "$log_dir/${ts}_gsm8k_genmath.log"

  # 3) HumanEval pair_logprob (HF loader)
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$QWEN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$QWEN_CODE_MODEL" \
    --system_prompt "$QWEN_SYS_PROMPT" \
    --task humaneval \
    --use_benchmark_loader 0 --hf_id "$HUMANEVAL_HF_ID" --hf_split "$HUMANEVAL_HF_SPLIT" \
    --eval_mode pair_logprob \
    --gold_max_tokens "$GOLD_MAX_TOKENS_HE" \
    --out_json "$out_dir/humaneval_pairlogprob.json" \
    2>&1 | tee "$log_dir/${ts}_humaneval_pairlogprob.log"

  # 4) HumanEval gen_code_compile
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$QWEN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$QWEN_CODE_MODEL" \
    --system_prompt "$QWEN_SYS_PROMPT" \
    --task humaneval \
    --use_benchmark_loader 0 --hf_id "$HUMANEVAL_HF_ID" --hf_split "$HUMANEVAL_HF_SPLIT" \
    --eval_mode gen_code_compile \
    --max_new_tokens "$MAX_NEW_TOKENS_CODE" \
    --out_json "$out_dir/humaneval_gencode_compile.json" \
    2>&1 | tee "$log_dir/${ts}_humaneval_gencode_compile.log"
}

###############################################################################
# Falcon runs
###############################################################################
run_falcon() {
  local tag="falcon"
  local out_dir="$OUT_ROOT/$tag"
  mkdir -p "$out_dir"

  local log_dir="$out_dir/logs"
  mkdir -p "$log_dir"
  local ts
  ts="$(timestamp)"

  # 1) GSM8K pair_logprob (strict: push first tokens toward numbers)
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$FALCON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$FALCON_MODEL" \
    --system_prompt "$FALCON_SYS_MATH_STRICT" \
    --answer_prefix $'\nFinal answer (number only):' \
    --task gsm8k \
    --eval_mode pair_logprob \
    --out_json "$out_dir/gsm8k_pairlogprob.json" \
    2>&1 | tee "$log_dir/${ts}_gsm8k_pairlogprob.log"

  # 2) GSM8K gen_math (CoT but still enforce final line format)
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$FALCON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$FALCON_MODEL" \
    --system_prompt "$FALCON_SYS_MATH_COT" \
    --answer_prefix $'\nLet'\''s think step by step.\nFinal answer (number only):' \
    --task gsm8k \
    --eval_mode gen_math \
    --max_new_tokens "$MAX_NEW_TOKENS_MATH" \
    --out_json "$out_dir/gsm8k_genmath.json" \
    2>&1 | tee "$log_dir/${ts}_gsm8k_genmath.log"

  # 3) HumanEval pair_logprob
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$FALCON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$FALCON_MODEL" \
    --system_prompt "$FALCON_SYS_CODE" \
    --task humaneval \
    --use_benchmark_loader 0 --hf_id "$HUMANEVAL_HF_ID" --hf_split "$HUMANEVAL_HF_SPLIT" \
    --eval_mode pair_logprob \
    --gold_max_tokens "$GOLD_MAX_TOKENS_HE" \
    --out_json "$out_dir/humaneval_pairlogprob.json" \
    2>&1 | tee "$log_dir/${ts}_humaneval_pairlogprob.log"

  # 4) HumanEval gen_code_compile
  run_cmd CUDA_VISIBLE_DEVICES=3 python "$FALCON_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --model "$FALCON_MODEL" \
    --system_prompt "$FALCON_SYS_CODE" \
    --task humaneval \
    --use_benchmark_loader 0 --hf_id "$HUMANEVAL_HF_ID" --hf_split "$HUMANEVAL_HF_SPLIT" \
    --eval_mode gen_code_compile \
    --max_new_tokens "$MAX_NEW_TOKENS_CODE" \
    --out_json "$out_dir/humaneval_gencode_compile.json" \
    2>&1 | tee "$log_dir/${ts}_humaneval_gencode_compile.log"
}

###############################################################################
# Main
###############################################################################
echo "[Info] DEVICE=$DEVICE DTYPE=$DTYPE LAYER=$LAYER SEED=$SEED"
echo "[Info] QS_PATH=$QS_PATH"
echo "[Info] N_EVAL=$N_EVAL MAX_FLIPS=$MAX_FLIPS PATCH_N_STEPS=$PATCH_N_STEPS"
echo "[Info] OUT_ROOT=$OUT_ROOT"
echo "[Info] QWEN_SCRIPT=$QWEN_SCRIPT FALCON_SCRIPT=$FALCON_SCRIPT"

# Run both (you can comment one out if needed)
run_falcon
run_qwen

echo ""
echo "[Done] All runs completed."
echo "  Falcon outputs: $OUT_ROOT/falcon/"
echo "  Qwen outputs:   $OUT_ROOT/qwen/"
