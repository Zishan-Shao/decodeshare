#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"

PATCHBACK_DIR="${REPO_ROOT}/downstream/patch_back"
cd "${PATCHBACK_DIR}"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-${MODEL_DTYPE:-fp32}}"
LAYER="${LAYER:-10}"
SEED="${SEED:-123}"

BASIS_TASKS="${BASIS_TASKS:-gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa}"
BASIS_N_SUBSPACE="${BASIS_N_SUBSPACE:-128}"

OUTDIR="${OUT_DIR:-${REPO_ROOT}/outputs/03_patchback/decodeshare_suite/runs_layer${LAYER}_seed${SEED}_${MODEL//\//_}}"
mkdir -p "${OUTDIR}"

QS_PATH="${OUTDIR}/Q_shared_layer${LAYER}.npy"

echo "[Run] Output dir: ${OUTDIR}"
echo "[Run] Q_shared will be saved to: ${QS_PATH}"

# ====== 实验组 0：计算 Q_shared（会顺带跑一次 task=aqua；也可接受）======
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 1 --Qs_out "${QS_PATH}" \
  --basis_tasks "${BASIS_TASKS}" --basis_n_subspace "${BASIS_N_SUBSPACE}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 254 --max_flips 128 \
  --out_json "${OUTDIR}/aqua_computeQ.json"

# ====== 实验组 1：Held-out transfer（强烈建议保留）======

# 1) AQuA (5-choice)
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 254 --max_flips 128 \
  --out_json "${OUTDIR}/aqua.json"

# 2) ARC-Challenge (通常 4-choice) —— 如果 benchmark_dataloaders 不支持就注释掉
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task arc_challenge --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/arc_challenge.json"

# 3) LogiQA (通常 4-choice) —— 如果不支持就注释掉
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task logiqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/logiqa.json"

# ====== 实验组 2：In-basis（可选但很推荐，增强读者信心）======

# CommonsenseQA (5-choice)
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task commonsenseqa --candidate_labels ABCDE \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/commonsenseqa.json"

# OpenBookQA (4-choice)
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task openbookqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/openbookqa.json"

# PIQA (2-choice)
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task piqa --candidate_labels AB \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/piqa.json"

# QASC（如果你的 loader 是 8-choice，用 ABCDEFGH；若不是就改回 ABCD/ABCDE）
run_python_gpu subspace_patching_transfer.py \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --compute_Qs 0 --Qs_path "${QS_PATH}" \
  --task qasc --candidate_labels ABCDEFGH \
  --n_eval 256 --max_flips 128 \
  --out_json "${OUTDIR}/qasc.json"

echo "[Done] All commands finished. Results in ${OUTDIR}/"
