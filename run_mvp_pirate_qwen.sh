#!/usr/bin/env bash
set -euo pipefail

# Run Qwen pirate MVP (v5): probe + story_clean
# Outputs:
#  - probe logs/results:  brittleness/results/mvp_pirate_story_clean_qwen_probe/
#  - main  logs/results:  brittleness/results/mvp_pirate_story_clean_qwen/
#
# Usage:
#   GPU=0 ./run_mvp_pirate_qwen.sh
#   CUDA_VISIBLE_DEVICES=1 ./run_mvp_pirate_qwen.sh
#
# Notes:
# - This script does NOT activate conda automatically. If needed:
#     source "$HOME/miniconda3/etc/profile.d/conda.sh" && conda activate <env>

export TOKENIZERS_PARALLELISM=false

GPU="${GPU:-0}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B-Instruct}"
DTYPE="${DTYPE:-fp32}"
LAYER="${LAYER:-28}"

OUT_MAIN="brittleness/results/mvp_pirate_story_clean_qwen"
OUT_PROBE="brittleness/results/mvp_pirate_story_clean_qwen_probe"

mkdir -p "${OUT_MAIN}" "${OUT_PROBE}"

echo "[Info] MODEL_ID=${MODEL_ID}"
echo "[Info] DTYPE=${DTYPE} LAYER=${LAYER} GPU=${GPU}"
echo "[Info] OUT_PROBE=${OUT_PROBE}"
echo "[Info] OUT_MAIN=${OUT_MAIN}"

if [ ! -f "brittleness/mvp_projection_patch_pirate_v5.py" ]; then
  echo "[Error] Missing script: brittleness/mvp_projection_patch_pirate_v5.py"
  exit 1
fi

# Respect user-provided CUDA_VISIBLE_DEVICES if already set; otherwise use GPU var.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU}"
fi

echo ""
echo "=============================="
echo "[1/2] Probe (v + next-token delta sanity)"
echo "=============================="
python brittleness/mvp_projection_patch_pirate_v5.py \
  --model "${MODEL_ID}" \
  --device cuda --dtype "${DTYPE}" \
  --layer "${LAYER}" \
  --v_mode decode --v_decode_steps 16 --v_n 16 \
  --probe_alpha 12 --fail_fast 1 --debug_only 1 \
  --out_dir "${OUT_PROBE}" \
  2>&1 | tee "${OUT_PROBE}/run.log"

echo ""
echo "=============================="
echo "[2/2] Main (smoke test auto-alpha + small eval)"
echo "=============================="
python brittleness/mvp_projection_patch_pirate_v5.py \
  --model "${MODEL_ID}" \
  --device cuda --dtype "${DTYPE}" \
  --layer "${LAYER}" \
  --v_mode decode --v_decode_steps 16 --v_n 16 \
  --pirate_threshold 2 \
  --temperature 0.9 --top_p 0.9 \
  --smoke_test 1 --smoke_decoding sample --smoke_alphas 10,20,40,60,80,120 \
  --auto_use_best_alpha 1 --stop_on_smoke_success 1 \
  --do_greedy 0 --do_sample 1 --sample_seeds 1 \
  --inject_first_n 24 \
  --eval_n_base 4 --eval_n_templates 2 --max_eval_prompts 20 \
  --early_abort_after 40 --early_abort_if_all_zero 1 \
  --out_dir "${OUT_MAIN}" \
  2>&1 | tee "${OUT_MAIN}/run.log"

echo ""
echo "[Done]"
echo " - Probe outputs: ${OUT_PROBE}"
echo " - Main outputs : ${OUT_MAIN}"
