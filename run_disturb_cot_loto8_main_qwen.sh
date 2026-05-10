#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Env / paths
# -----------------------------
WORKDIR="src"
cd "${WORKDIR}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flashsvd

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# -----------------------------
# Main-text reproducible suite:
# 3 models × 3 layers, and for each:
#   (1) generation (greedy)
#   (2) LOTO (greedy; heldout)
#   (3) forced-choice (MC/YesNo forced-choice; gsm8k stays generation)
#   (4) LOTO + forced-choice
#
# Outputs go to: src/results/disturb_cot_main/
# -----------------------------

# GPU / core toggles
GPU_ID="${GPU_ID:-1}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
DEVICE="${DEVICE:-cuda}"

RESULTS_DIR="${RESULTS_DIR:-${WORKDIR}/results/disturb_cot_main_qwen}"
mkdir -p "${RESULTS_DIR}"

# We use the reasoning version because it supports BOTH:
# - generation protocol (--use_forced_choice 0)
# - forced-choice protocol for MC/YesNo tasks (--use_forced_choice 1)
SCRIPT="${SCRIPT:-${WORKDIR}/disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8_reasoning.py}"

# 8-task set used elsewhere in the repo (keep consistent)
TASKS="${TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,openbookqa,arc_challenge,qasc,boolq,piqa}"
LAYERS=(4 10 24)

# 3 representative models for main text
MODEL_NAMES=(
  #"meta-llama/Llama-2-7b-chat-hf"
  "Qwen/Qwen2.5-7B-Instruct"
  "tiiuae/falcon-7b-instruct"
)

# fixed knobs (keep identical across runs)
N_SUBSPACE="${N_SUBSPACE:-128}"
N_EVAL="${N_EVAL:-2048}"
CALIB_NEW="${CALIB_NEW:-128}"
PER_TASK_MAX_STATES="${PER_TASK_MAX_STATES:-20000}"
REASONING_TOKENS="${REASONING_TOKENS:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"

TEMPLATE_RANDOMIZATION="${TEMPLATE_RANDOMIZATION:-1}"
TEMPLATE_SEED="${TEMPLATE_SEED:-1234}"
SHUFFLE_CHOICES="${SHUFFLE_CHOICES:-1}"

ADD_ANSWER_PREFIX="${ADD_ANSWER_PREFIX:-1}"
ANSWER_PREFIX="${ANSWER_PREFIX:-$'\nFinal answer:'}"

SEED="${SEED:-42}"
SAMPLE_SEED="${SAMPLE_SEED:-12345}"

run_one () {
  local model="$1"
  local layer="$2"
  local mode="$3"             # all / loto
  local loto_eval_mode="$4"   # heldout / all (only used if mode=loto)
  local protocol="$5"         # gen / fc

  local model_tag
  model_tag="$(echo "${model}" | tr '/:' '__')"

  local use_forced_choice=0
  if [ "${protocol}" = "fc" ]; then
    use_forced_choice=1
  fi

  local out_base="${RESULTS_DIR}/${protocol}_${mode}_${model_tag}_layer${layer}"
  local out_json="${out_base}.json"
  local out_md="${out_base}.md"
  local out_txt="${out_base}.txt"

  echo ""
  echo "================================================================================"
  echo "[Run] model=${model}"
  echo "      layer=${layer} mode=${mode} loto_eval_mode=${loto_eval_mode} protocol=${protocol}"
  echo "      gpu=${GPU_ID} dtype=${MODEL_DTYPE}"
  echo "      out=${out_base}"
  echo "================================================================================"

  CUDA_VISIBLE_DEVICES="${GPU_ID}" python "${SCRIPT}" \
    --model "${model}" --device "${DEVICE}" --model_dtype "${MODEL_DTYPE}" \
    --tasks "${TASKS}" \
    --mode "${mode}" --loto_eval_mode "${loto_eval_mode}" \
    --n_subspace "${N_SUBSPACE}" --n_eval "${N_EVAL}" --layer "${layer}" \
    --calib_decode_max_new_tokens "${CALIB_NEW}" --per_task_max_states "${PER_TASK_MAX_STATES}" \
    --reasoning_tokens "${REASONING_TOKENS}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --batch_size "${BATCH_SIZE}" --max_prompt_len "${MAX_PROMPT_LEN}" \
    --template_randomization "${TEMPLATE_RANDOMIZATION}" --template_seed "${TEMPLATE_SEED}" \
    --shuffle_choices "${SHUFFLE_CHOICES}" \
    --add_answer_prefix "${ADD_ANSWER_PREFIX}" --answer_prefix "${ANSWER_PREFIX}" \
    --do_sample 0 \
    --use_forced_choice "${use_forced_choice}" \
    --fc_warmup_tokens 0 --fc_warmup_decoding greedy --fc_prefix_mode auto --fc_debug_print 0 \
    --seed "${SEED}" --sample_seed "${SAMPLE_SEED}" \
    --out_json "${out_json}" --out_md "${out_md}" \
    2>&1 | tee "${out_txt}"
}

for model in "${MODEL_NAMES[@]}"; do
  for layer in "${LAYERS[@]}"; do
    # (1) generation (greedy)
    run_one "${model}" "${layer}" "all"  "heldout" "gen"
    # (2) LOTO generation (greedy, heldout)
    run_one "${model}" "${layer}" "loto" "heldout" "gen"
    # (3) forced choice (MC/YesNo forced-choice; gsm8k stays generation)
    run_one "${model}" "${layer}" "all"  "heldout" "fc"
    # (4) LOTO + forced choice
    run_one "${model}" "${layer}" "loto" "heldout" "fc"
  done
done
