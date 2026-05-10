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

# 允许外部指定 GPU：GPU=1 bash xxx.sh
GPU="${GPU:-0}"
export CUDA_VISIBLE_DEVICES=0

# （可选）如果你需要 HF token，取消注释并确保已设置环境变量
# export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"

# -----------------------------
# Output dirs
# -----------------------------
OUT_ALPHA_DIR="results/alpha_sweep_decode_aligned"
OUT_KMATCH_DIR="results/kmatch"
LOG_DIR="results/logs_disturb_energy_kmatch"

mkdir -p "${OUT_ALPHA_DIR}" "${OUT_KMATCH_DIR}" "${LOG_DIR}"

# -----------------------------
# Config
# -----------------------------
MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_TAG="${MODEL_NAME//\//_}"
LAYER=10
TASKS="commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq"
# gsm8k 先别放进去（见下面解释）


BATCH_SIZE=4
MAX_PROMPT_LEN=512
N_PROMPTS=128
CALIB_MAX_NEW_TOKENS=128
PER_TASK_MAX_STATES=20000
EVAL_N=2048

PCA_VAR=0.95
TAU=0.001
M_SHARED="all"

BOOTSTRAP_ITERS=5000
PERM_ITERS=10000
CI_ALPHA=0.05

SEED=42
USE_CHAT_TEMPLATE=1

# -----------------------------
# Helper: run one job
# -----------------------------
run_one () {
  local alpha="$1"
  local out_json="$2"
  local out_txt="$3"
  local log_file="$4"

  echo "[Run] alpha_shared_base=${alpha}"
  echo "[Run] out_json=${out_json}"
  echo "[Run] out_txt=${out_txt}"
  echo "[Run] log=${log_file}"
  echo

  CUDA_VISIBLE_DEVICES=0 python disturb_energy_matched_sharedness_kmatch.py \
    --model "${MODEL_NAME}" --device cuda --dtype fp32 \
    --layer "${LAYER}" \
    --n_prompts "${N_PROMPTS}" \
    --calib_max_new_tokens "${CALIB_MAX_NEW_TOKENS}" \
    --per_task_max_states "${PER_TASK_MAX_STATES}" \
    --pca_var "${PCA_VAR}" --tau "${TAU}" --m_shared "${M_SHARED}" \
    --eval_n "${EVAL_N}" --max_prompt_len "${MAX_PROMPT_LEN}" \
    --control_basis joint_nonshared_topk \
    --batch_size "${BATCH_SIZE}" \
    --bootstrap_iters "${BOOTSTRAP_ITERS}" --perm_iters "${PERM_ITERS}" --ci_alpha "${CI_ALPHA}" \
    --seed "${SEED}" \
    --use_chat_template "${USE_CHAT_TEMPLATE}" \
    --alpha_shared_base "${alpha}" \
    --eval_tasks "${TASKS}" \
    --out_json "${out_json}" \
    --out_txt "${out_txt}" 2>&1 | tee "${log_file}"
}

# -----------------------------
# (1) Alpha sweep
#   注意：你的 python 脚本里每次都会同时做：
#   - shared_alpha / ctrl_alpha（alpha-match）
#   - shared_full / ctrl_struct / ctrl_energy / rand...
#   所以这里 sweep alpha_shared_base 就行
# -----------------------------
ALPHAS=(0 0.25 0.5 0.75 1.0 1.25)

for a in "${ALPHAS[@]}"; do
  tag="${a/./p}"

  run_one \
    "${a}" \
    "${OUT_ALPHA_DIR}/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.json" \
    "${OUT_ALPHA_DIR}/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.txt" \
    "${LOG_DIR}/${MODEL_TAG}_layer${LAYER}_alphaShared_${tag}.log"
done

# -----------------------------
# (2) K-match single run (alpha=1.0)
#   你脚本内部的 ctrl_energy 是 K-match(alpha=1) 得出的 headline，
#   这里单独跑一次输出到 results/kmatch 方便你写主结果
# -----------------------------
run_one \
  "1.0" \
  "${OUT_KMATCH_DIR}/${MODEL_TAG}_layer${LAYER}_kmatch.json" \
  "${OUT_KMATCH_DIR}/${MODEL_TAG}_layer${LAYER}_kmatch.txt" \
  "${LOG_DIR}/${MODEL_TAG}_layer${LAYER}_kmatch.log"

echo
echo "[Done] All runs finished."
echo "[Done] Alpha outputs: ${OUT_ALPHA_DIR}"
echo "[Done] Kmatch outputs: ${OUT_KMATCH_DIR}"
echo "[Done] Logs: ${LOG_DIR}"
