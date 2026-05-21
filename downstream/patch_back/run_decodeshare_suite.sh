#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "${SCRIPT_DIR}"
conda activate "${CONDA_ENV:-decodeshare}"

# ====== 基本配置 ======
MODEL="Qwen/Qwen2.5-7B-Instruct" #"meta-llama/Llama-2-7b-chat-hf"
DEVICE="cuda"
DTYPE="fp32"
LAYER="10"
SEED="123"

# 你现在用的 basis tasks（保持不变即可）
BASIS_TASKS="gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa"
BASIS_N_SUBSPACE="128"

# 输出目录（避免覆盖）
OUTDIR="results/subspace_patching_transfer/runs_layer${LAYER}_seed${SEED}_${MODEL//\//_}"
mkdir -p "${OUTDIR}"

QS_PATH="${OUTDIR}/Q_shared_layer${LAYER}.npy"

echo "[Run] Output dir: ${OUTDIR}"
echo "[Run] Q_shared will be saved to: ${QS_PATH}"

# ====== 实验组 0：计算 Q_shared（会顺带跑一次 task=aqua；也可接受）======
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 1 --Qs_out "${QS_PATH}" \
  --basis_tasks "${BASIS_TASKS}" --basis_n_subspace "${BASIS_N_SUBSPACE}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 254 --max_flips 128 \
  --out_json "${OUTDIR}/aqua_computeQ.json"

# ====== 实验组 1：Held-out transfer（强烈建议保留）======

# 1) AQuA (5-choice)
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 254 --max_flips 128 \
  --out_json "${OUTDIR}/aqua.json"

# 2) ARC-Challenge (通常 4-choice) —— 如果 benchmark_dataloaders 不支持就注释掉
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task arc_challenge --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/arc_challenge.json"

# 3) LogiQA (通常 4-choice) —— 如果不支持就注释掉
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task logiqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/logiqa.json"

# ====== 实验组 2：In-basis（可选但很推荐，增强读者信心）======

# CommonsenseQA (5-choice)
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task commonsenseqa --candidate_labels ABCDE \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/commonsenseqa.json"

# OpenBookQA (4-choice)
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task openbookqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/openbookqa.json"

# PIQA (2-choice)
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task piqa --candidate_labels AB \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/piqa.json"

# QASC（如果你的 loader 是 8-choice，用 ABCDEFGH；若不是就改回 ABCD/ABCDE）
python subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task qasc --candidate_labels ABCDEFGH \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/qasc.json"

echo "[Done] All commands finished. Results in ${OUTDIR}/"

# # ====== 实验组 0：计算 Q_shared（会顺带跑一次 task=aqua；也可接受）======
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 1 --Qs_out "${QS_PATH}" \
#   --basis_tasks "${BASIS_TASKS}" --basis_n_subspace "${BASIS_N_SUBSPACE}" \
#   --task aqua --candidate_labels ABCDE \
#   --n_eval 254 --max_flips 128 \
#   --out_json "${OUTDIR}/aqua_computeQ.json"

# # ====== 实验组 1：Held-out transfer（强烈建议保留）======

# # 1) AQuA (5-choice)
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task aqua --candidate_labels ABCDE \
#   --n_eval 254 --max_flips 128 \
#   --out_json "${OUTDIR}/aqua.json"

# # 2) ARC-Challenge (通常 4-choice) —— 如果 benchmark_dataloaders 不支持就注释掉
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task arc_challenge --candidate_labels ABCD \
#   --n_eval 256 --max_flips 128 \
#   --out_json "${OUTDIR}/arc_challenge.json"

# # 3) LogiQA (通常 4-choice) —— 如果不支持就注释掉
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task logiqa --candidate_labels ABCD \
#   --n_eval 256 --max_flips 128 \
#   --out_json "${OUTDIR}/logiqa.json"

# # ====== 实验组 2：In-basis（可选但很推荐，增强读者信心）======

# # CommonsenseQA (5-choice)
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task commonsenseqa --candidate_labels ABCDE \
#   --n_eval 256 --max_flips 128 \
#   --out_json "${OUTDIR}/commonsenseqa.json"

# # OpenBookQA (4-choice)
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task openbookqa --candidate_labels ABCD \
#   --n_eval 256 --max_flips 128 \
#   --out_json "${OUTDIR}/openbookqa.json"

# # PIQA (2-choice)
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task piqa --candidate_labels AB \
#   --n_eval 256 --max_flips 128 \
#   --out_json "${OUTDIR}/piqa.json"

# # QASC（如果你的 loader 是 8-choice，用 ABCDEFGH；若不是就改回 ABCD/ABCDE）
# python subspace_patching_transfer.py \
#   --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
#   --layer "${LAYER}" --seed "${SEED}" \
#   --compute_Qs 0 --Qs_path "${QS_PATH}" \
#   --task qasc --candidate_labels ABCDEFGH \
#   --n_eval 256 --max_flips 128 \
#   --out_json "${OUTDIR}/qasc.json"

# echo "[Done] All commands finished. Results in ${OUTDIR}/"
