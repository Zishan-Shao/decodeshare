#!/usr/bin/env bash
set -euo pipefail

SCRIPT="prove_sharedness_decode_fair.py"

MODEL="meta-llama/Llama-2-7b-chat-hf"
DEVICE="cuda"
DTYPE="fp32"

LAYER=10
N_PROMPTS=128
BATCH=4
MAX_PROMPT_LEN=512
MAX_NEW_TOKENS=128
DECODING="greedy"

# 固定 joint PCA 维度 k（对齐不同子集 run）
K=512

TAU=0.001

# addition curve 不一定需要很重的 null；你也可以先关掉 scramble
NULL_PERM_TRIALS=200
NULL_SCRAMBLE_TRIALS=0

BASE_SEED=42
REPEATS=20

#TASKS=(gsm8k commonsenseqa strategyqa aqua arc_challenge openbookqa qasc boolq piqa)
TASKS=("gsm8k" "commonsenseqa" "strategyqa" "aqua" "openbookqa" "qasc" "boolq" "piqa")
TASKS_STR="$(IFS=,; echo "${TASKS[*]}")"
T_TOTAL=${#TASKS[@]}

OUTDIR="results/task_add_curve/layer${LAYER}_k${K}"
mkdir -p "${OUTDIR}"

echo "[Info] All tasks: ${TASKS_STR}"
echo "[Info] Output dir: ${OUTDIR}"

# 1) Reference run: all tasks -> save Q_all
REF_BASIS="${OUTDIR}/Q_all.npy"
python "${SCRIPT}" \
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --model_dtype "${DTYPE}" \
  --layer "${LAYER}" \
  --n_prompts "${N_PROMPTS}" \
  --batch_size "${BATCH}" \
  --max_prompt_len "${MAX_PROMPT_LEN}" \
  --calib_max_new_tokens "${MAX_NEW_TOKENS}" \
  --calib_decoding "${DECODING}" \
  --min_dim "${K}" --max_dim "${K}" \
  --tau "${TAU}" --m_shared all \
  --null_perm_trials "${NULL_PERM_TRIALS}" \
  --null_scramble_trials "${NULL_SCRAMBLE_TRIALS}" \
  --seed "${BASE_SEED}" \
  --tasks "${TASKS_STR}" \
  --save_joint_basis "${REF_BASIS}" \
  --out_json "${OUTDIR}/full.json" \
  --out_txt  "${OUTDIR}/full.txt"

# 2) Addition curve: random subsets of size t
export TASKS_STR
for t in $(seq 2 "${T_TOTAL}"); do
  RUN_DIR="${OUTDIR}/t${t}"
  mkdir -p "${RUN_DIR}"

  for r in $(seq 1 "${REPEATS}"); do
    SEED=$((BASE_SEED + 10000*t + r))
    export SEED
    export K_SUBSET="${t}"

    SUBSET=$(
      python - <<'PY'
import os, random
tasks = os.environ["TASKS_STR"].split(",")
seed  = int(os.environ["SEED"])
k     = int(os.environ["K_SUBSET"])
random.seed(seed)
subset = random.sample(tasks, k)
print(",".join(subset))
PY
    )

    echo "[Run] t=${t} rep=${r} seed=${SEED} tasks=${SUBSET}"

    python "${SCRIPT}" \
      --model "${MODEL}" \
      --device "${DEVICE}" \
      --model_dtype "${DTYPE}" \
      --layer "${LAYER}" \
      --n_prompts "${N_PROMPTS}" \
      --batch_size "${BATCH}" \
      --max_prompt_len "${MAX_PROMPT_LEN}" \
      --calib_max_new_tokens "${MAX_NEW_TOKENS}" \
      --calib_decoding "${DECODING}" \
      --min_dim "${K}" --max_dim "${K}" \
      --tau "${TAU}" --m_shared all \
      --null_perm_trials "${NULL_PERM_TRIALS}" \
      --null_scramble_trials "${NULL_SCRAMBLE_TRIALS}" \
      --seed "${SEED}" \
      --tasks "${SUBSET}" \
      --ref_joint_basis "${REF_BASIS}" \
      --out_json "${RUN_DIR}/rep${r}.json" \
      --out_txt  "${RUN_DIR}/rep${r}.txt"
  done
done

echo "[Done] task addition curve runs saved to ${OUTDIR}"
