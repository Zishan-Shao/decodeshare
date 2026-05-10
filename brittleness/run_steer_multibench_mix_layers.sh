#!/usr/bin/env bash
set -euo pipefail

# Run the sharedness-protecting steering benchmarks across layers.
# Order: for each layer (4,10,24), run 3 models.

LAYERS=(10 24 4)

# BASIS_SOURCE=multitask \
# BASIS_TASKS="csqa_pair,arc_challenge_pair,boolq,sst2,strategyqa,piqa" \
# BASIS_TASKS_PER_CLASS=64 \
# BASIS_HOLDOUT_CURRENT_TASK=1 \
# bash run_steer_multibench_mix_layers.sh

# Three-model suite (edit as you like)
MODELS=(
  "Qwen/Qwen2.5-7B-Instruct"
  "tiiuae/falcon-7b-instruct"
  "meta-llama/Llama-2-7b-chat-hf"
)

SCRIPTS=(
  "steering_vector_reliability_multibench_patch_qwen.py"
  "steering_vector_reliability_multibench_patch_falcon.py"
  "steering_vector_reliability_multibench_patch_v3.py"
)

DTYPES=(
  "fp16"
  "fp16"
  "bf16"
  # "fp32"
  # "fp32"
  # "fp32"
)

# Default task mix (override via env var TASKS="..."):
# - boolq: verification binary (similar flavor to strategyqa)
# - sst2: sentiment (aux)
# - csqa_pair / arc_challenge_pair: paired 2-way MCQ
TASKS="${TASKS:-boolq,sst2,csqa_pair,arc_challenge_pair}"

# Common knobs (match your usual settings)
CALIB_PER_CLASS="${CALIB_PER_CLASS:-256}"
EVAL_PER_CLASS="${EVAL_PER_CLASS:-128}"
BASIS_K="${BASIS_K:-512}"
BASIS_MAX_STATES="${BASIS_MAX_STATES:-1024}"
# Default: neutral basis. You can override to multitask via env var.
# Example:
#   BASIS_SOURCE=multitask BASIS_TASKS="csqa_pair,arc_challenge_pair,boolq,sst2,strategyqa,piqa" bash run_steer_multibench_mix_layers.sh
BASIS_SOURCE="${BASIS_SOURCE:-neutral}"
# Only include tasks that this repo's scripts currently implement.
BASIS_TASKS="${BASIS_TASKS:-csqa_pair,arc_challenge_pair,boolq,sst2,strategyqa,piqa}"
BASIS_TASKS_PER_CLASS="${BASIS_TASKS_PER_CLASS:-64}"
# Leave-one-out multitask basis: exclude the current task when estimating B (optional).
BASIS_HOLDOUT_CURRENT_TASK="${BASIS_HOLDOUT_CURRENT_TASK:-0}"
BETAS="${BETAS:-0,0.25,0.5,0.75,1.0}"
LAMBDAS="${LAMBDAS:-0,0.5,1.0}"
SEED="${SEED:-0}"
MAX_PROMPT_TOKENS="${MAX_PROMPT_TOKENS:-512}"

OUT_ROOT="${OUT_ROOT:-brittleness/results/steer_multibench_full}"

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//-/_}"
  s="${s//./_}"
  echo "$s"
}

run_one() {
  local script="$1"
  local model="$2"
  local dtype="$3"
  local layer="$4"

  local model_slug
  model_slug="$(slugify "$model")"

  local out_dir="${OUT_ROOT}/${model_slug}/layer${layer}"
  mkdir -p "$out_dir"

  if [[ "$script" == *"_qwen.py" || "$script" == *"_falcon.py" ]]; then
    python "$script" \
      --model "$model" \
      --device cuda --dtype "$dtype" \
      --layer "$layer" \
      --tasks "$TASKS" \
      --seed "$SEED" \
      --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
      --calib_per_class "$CALIB_PER_CLASS" --eval_per_class "$EVAL_PER_CLASS" \
      --basis_source "$BASIS_SOURCE" --basis_k "$BASIS_K" --basis_max_states "$BASIS_MAX_STATES" \
      --basis_tasks "$BASIS_TASKS" --basis_tasks_per_class "$BASIS_TASKS_PER_CLASS" \
      --basis_holdout_current_task "$BASIS_HOLDOUT_CURRENT_TASK" \
      --betas "$BETAS" --lambdas "$LAMBDAS" \
      --use_chat_template 1 \
      --v_est_templates all --sign_calib_templates all \
      --out_dir "$out_dir" \
      --show_per_template 1
  else
    python "$script" \
      --model "$model" \
      --device cuda --dtype "$dtype" \
      --layer "$layer" \
      --tasks "$TASKS" \
      --seed "$SEED" \
      --max_prompt_tokens "$MAX_PROMPT_TOKENS" \
      --calib_per_class "$CALIB_PER_CLASS" --eval_per_class "$EVAL_PER_CLASS" \
      --basis_source "$BASIS_SOURCE" --basis_k "$BASIS_K" --basis_max_states "$BASIS_MAX_STATES" \
      --basis_tasks "$BASIS_TASKS" --basis_tasks_per_class "$BASIS_TASKS_PER_CLASS" \
      --basis_holdout_current_task "$BASIS_HOLDOUT_CURRENT_TASK" \
      --betas "$BETAS" --lambdas "$LAMBDAS" \
      --v_est_templates all \
      --out_dir "$out_dir" \
      --show_per_template 1
  fi
}

for layer in "${LAYERS[@]}"; do
  echo
  echo "=== Layer ${layer} ==="
  for i in "${!MODELS[@]}"; do
    echo
    echo "--- Model: ${MODELS[$i]} | Script: ${SCRIPTS[$i]} ---"
    run_one "${SCRIPTS[$i]}" "${MODELS[$i]}" "${DTYPES[$i]}" "$layer"
  done
done

echo
echo "Done. Results in: ${OUT_ROOT}"
