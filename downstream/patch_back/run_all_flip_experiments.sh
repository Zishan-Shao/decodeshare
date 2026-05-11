#!/usr/bin/env bash
set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd /home/zs89/decodeshare/patch_back
conda activate flashsvd

MODEL="Qwen/Qwen2.5-7B-Instruct" #"meta-llama/Llama-2-7b-chat-hf"
DEVICE="cuda"
DTYPE="fp32"
LAYER="10"

BASE_SCRIPT="subspace_patching_transfer.py"
FLIP_SCRIPT="flipset_alpha_sweep_and_transfer.py"

# Absolute output directory as requested
OUTDIR="/home/zs89/decodeshare/patch_back/results/runs_flip_supplement/layer${LAYER}_${MODEL}"
mkdir -p "${OUTDIR}"

echo "[Info] OUTDIR=${OUTDIR}"
echo "[Info] MODEL=${MODEL} LAYER=${LAYER} DTYPE=${DTYPE}"

echo "[Info] BASE_SCRIPT=${BASE_SCRIPT}"
echo "[Info] FLIP_SCRIPT=${FLIP_SCRIPT}"

if [ ! -f "${BASE_SCRIPT}" ]; then
  echo "[Error] ${BASE_SCRIPT} not found in $(pwd)"
  exit 1
fi
if [ ! -f "${FLIP_SCRIPT}" ]; then
  echo "[Error] ${FLIP_SCRIPT} not found in $(pwd)"
  exit 1
fi

# ============================================================
# SEED=123 suite
# ============================================================
SEED="123"
QS_SEED123="Q_shared_layer${LAYER}_seed${SEED}.npy"

echo ""
echo "============================================================"
echo "[Suite] SEED=${SEED}"
echo "============================================================"

# (0) Ensure Q_shared exists (seed-specific)
if [ ! -f "${QS_SEED123}" ]; then
  echo "[Run] Computing Q_shared for seed=${SEED} -> ${QS_SEED123}"
  CUDA_VISIBLE_DEVICES=3 python "${BASE_SCRIPT}" \
    --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --layer "${LAYER}" --seed "${SEED}" \
    --compute_Qs 1 --Qs_out "${QS_SEED123}" \
    --basis_tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
    --basis_n_subspace 128 \
    --task aqua --candidate_labels ABCDE \
    --n_eval 256 --max_flips 64 \
    --out_json "${OUTDIR}/compute_Qs_seed${SEED}.json"
else
  echo "[Info] Found existing ${QS_SEED123}, skip compute."
fi

# (1) Alpha sweep on AQuA flip-set
echo "[Run] Alpha sweep on AQuA flip-set (seed=${SEED})"
CUDA_VISIBLE_DEVICES=3 python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_SEED123}" \
  --run_alpha_sweep 1 --alpha_list 0,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --run_transfer_patching 0 \
  --out_json "${OUTDIR}/aqua_flipset_alpha_sweep_seed${SEED}.json"

# (2) Transfer patch: same-task donors (random pick), with self reference
echo "[Run] Transfer patch SAME-TASK donors on AQuA flip-set (seed=${SEED})"
CUDA_VISIBLE_DEVICES=3 python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_SEED123}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source same_task_eval --donor_n_eval 512 --donor_pick random \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${OUTDIR}/aqua_flipset_transfer_same_task_seed${SEED}.json"

# (3) Transfer patch: cross-task "generation-ish" donors (no baseline-correct filtering)
echo "[Run] Transfer patch CROSS-TASK donors (gsm8k,strategyqa) on AQuA flip-set (seed=${SEED})"
CUDA_VISIBLE_DEVICES=3 python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_SEED123}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source cross_task_eval --donor_tasks gsm8k,strategyqa \
  --donor_n_eval 128 --donor_pick random \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${OUTDIR}/aqua_flipset_transfer_cross_generative_seed${SEED}.json"

# (4) Transfer patch: cross-task MC donors, filtered to gold in candidates + baseline-correct
echo "[Run] Transfer patch CROSS-TASK MC donors (commonsenseqa,openbookqa) baseline-correct on AQuA flip-set (seed=${SEED})"
CUDA_VISIBLE_DEVICES=3 python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_SEED123}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source cross_task_eval --donor_tasks commonsenseqa,openbookqa \
  --donor_n_eval 256 --donor_pick random \
  --donor_require_gold_in_candidates 1 --donor_require_baseline_correct 1 \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${OUTDIR}/aqua_flipset_transfer_cross_mc_baselinecorrect_seed${SEED}.json"

# ============================================================
# SEED=456 robustness suite (IMPORTANT: recompute Q_shared for this seed)
# ============================================================
SEED2="456"
QS_SEED456="Q_shared_layer${LAYER}_seed${SEED2}.npy"

echo ""
echo "============================================================"
echo "[Suite] Robustness SEED=${SEED2}"
echo "============================================================"

# (5) Ensure Q_shared exists for seed=456
if [ ! -f "${QS_SEED456}" ]; then
  echo "[Run] Computing Q_shared for seed=${SEED2} -> ${QS_SEED456}"
  CUDA_VISIBLE_DEVICES=3 python "${BASE_SCRIPT}" \
    --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --layer "${LAYER}" --seed "${SEED2}" \
    --compute_Qs 1 --Qs_out "${QS_SEED456}" \
    --basis_tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
    --basis_n_subspace 128 \
    --task aqua --candidate_labels ABCDE \
    --n_eval 256 --max_flips 64 \
    --out_json "${OUTDIR}/compute_Qs_seed${SEED2}.json"
else
  echo "[Info] Found existing ${QS_SEED456}, skip compute."
fi

# (6) Robustness run: alpha sweep (optional but nice)
echo "[Run] Alpha sweep on AQuA flip-set (seed=${SEED2})"
CUDA_VISIBLE_DEVICES=3 python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED2}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_SEED456}" \
  --run_alpha_sweep 1 --alpha_list 0,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --run_transfer_patching 0 \
  --out_json "${OUTDIR}/aqua_flipset_alpha_sweep_seed${SEED2}.json"

# (7) Robustness run: cross-task MC donors baseline-correct (the key setting)
echo "[Run] Robustness transfer CROSS-TASK MC donors baseline-correct (seed=${SEED2})"
CUDA_VISIBLE_DEVICES=3 python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED2}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_SEED456}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source cross_task_eval --donor_tasks commonsenseqa,openbookqa \
  --donor_n_eval 256 --donor_pick random \
  --donor_require_gold_in_candidates 1 --donor_require_baseline_correct 1 \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${OUTDIR}/aqua_flipset_transfer_cross_mc_baselinecorrect_seed${SEED2}.json"

echo ""
echo "[Done] All supplement flip-set experiments finished."
echo "[Done] Outputs in: ${OUTDIR}"
