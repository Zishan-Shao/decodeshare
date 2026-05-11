#!/usr/bin/env bash
set -euo pipefail

# Run Part A (mechanism) scripts end-to-end and save outputs under
# rebuttal/mechanism/PartA/results/.
#
# Usage (defaults):
#   bash rebuttal/mechanism/PartA/run_partA_all.sh
#
# Override via env vars, e.g.:
#   MODEL=meta-llama/Llama-2-13b-chat-hf DEVICE=cuda DTYPE=fp16 LAYER=10 \
#   RUN_ID=my_run_001 \
#   bash rebuttal/mechanism/PartA/run_partA_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT="$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

# -------------------------
# Shared config
# -------------------------
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
DEVICE="${DEVICE:-cuda}"                    # cuda|cpu
DTYPE="${DTYPE:-fp16}"                      # fp32|fp16|bf16
# Backward-compat:
# - If you set LAYERS, we run all layers in that CSV.
# - Else if you set LAYER, we run just that layer.
# - Else default sweep is 10,24,28.
if [[ -z "${LAYERS:-}" ]]; then
  if [[ -n "${LAYER:-}" ]]; then
    LAYERS="${LAYER}"
  else
    LAYERS="10,24,28"
  fi
fi
LAYER="${LAYER:-10}"                        # kept for legacy callers
SEED="${SEED:-42}"
TEMPLATE_SEED="${TEMPLATE_SEED:-1234}"

ANSWER_PREFIX="${ANSWER_PREFIX:-$'\nFinal answer:'}"
FC_PREFIX_MODE="${FC_PREFIX_MODE:-auto}"    # auto|always|never

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_BASE="${OUT_BASE:-${SCRIPT_DIR}/results}"
OUT_DIR="${OUT_DIR:-${OUT_BASE}/${RUN_ID}}"
mkdir -p "${OUT_DIR}"

COMMANDS_SH="${OUT_DIR}/commands.sh"
ENV_TXT="${OUT_DIR}/env.txt"

echo "#!/usr/bin/env bash" > "${COMMANDS_SH}"
echo "set -euo pipefail" >> "${COMMANDS_SH}"
echo >> "${COMMANDS_SH}"

{
  echo "[Run] RUN_ID=${RUN_ID}"
  echo "[Run] OUT_DIR=${OUT_DIR}"
  echo "[Run] MODEL=${MODEL}"
  echo "[Run] DEVICE=${DEVICE} DTYPE=${DTYPE}"
  echo "[Run] LAYERS=${LAYERS}"
  echo "[Run] SEED=${SEED} TEMPLATE_SEED=${TEMPLATE_SEED}"
  echo
  echo "[Env] python=$(${PYTHON_BIN} -V 2>&1 || true)"
  #echo "[Env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "[Env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}"
  echo "[Env] torch:"
  ${PYTHON_BIN} - <<'PY' || true
import torch
print("  torch.__version__ =", torch.__version__)
print("  torch.version.cuda =", getattr(torch.version, "cuda", None))
print("  torch.cuda.is_available() =", torch.cuda.is_available())
PY
  echo
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[Env] nvidia-smi:"
    nvidia-smi || true
  fi
} | tee "${ENV_TXT}"

# Parse layers
IFS=',' read -r -a LAYERS_ARR <<< "${LAYERS}"
LAYERS_ARR_CLEAN=()
for _l in "${LAYERS_ARR[@]}"; do
  _l="${_l//[[:space:]]/}"
  [[ -z "${_l}" ]] && continue
  if [[ "${_l}" =~ ^[0-9]+$ ]]; then
    LAYERS_ARR_CLEAN+=("${_l}")
  else
    echo "[Error] Bad layer in LAYERS: ${_l}" >&2
    exit 2
  fi
done
if [[ "${#LAYERS_ARR_CLEAN[@]}" -eq 0 ]]; then
  echo "[Error] No valid layers parsed from LAYERS=${LAYERS}" >&2
  exit 2
fi

# -------------------------
# (A1) Computational path (KV cache)
# -------------------------
DEFAULT_A1_SCRIPT="${SCRIPT_DIR}/exp_A1_computational_path_kv_cache.py"
FALLBACK_A1_SCRIPT="${REPO_ROOT}/rebuttal/mechanism/exp_A1_computational_path_kv_cache.py"
A1_SCRIPT="${A1_SCRIPT:-${DEFAULT_A1_SCRIPT}}"
if [[ ! -f "${A1_SCRIPT}" && -f "${FALLBACK_A1_SCRIPT}" ]]; then
  A1_SCRIPT="${FALLBACK_A1_SCRIPT}"
fi
A1_TASKS="${A1_TASKS:-commonsenseqa,arc_challenge,openbookqa,qasc,logiqa}"
A1_N_PROMPTS="${A1_N_PROMPTS:-32}"
A1_BATCH_SIZE="${A1_BATCH_SIZE:-4}"
A1_MAX_PROMPT_LEN="${A1_MAX_PROMPT_LEN:-512}"
A1_ALPHA="${A1_ALPHA:-1.0}"

A1_OUT="${OUT_DIR}/A1_comp_path"
mkdir -p "${A1_OUT}"

for L in "${LAYERS_ARR_CLEAN[@]}"; do
  A1_LOG="${OUT_DIR}/A1_comp_path_layer${L}.log"

  # If present, A1 uses a "real" decode-shared basis saved by exp_1; otherwise it falls back to a random basis.
  DEFAULT_BASIS_NPZ="${REPO_ROOT}/results/rebuttal_mechanism/logit_lens_l${L}/basis_layer${L}_tseed${TEMPLATE_SEED}.npz"
  BASIS_NPZ_LAYER="${BASIS_NPZ:-${DEFAULT_BASIS_NPZ}}"

  A1_CMD=(
    "${PYTHON_BIN}" "${A1_SCRIPT}"
    --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}"
    --layer "${L}" --alpha "${A1_ALPHA}"
    --tasks "${A1_TASKS}"
    --n_prompts "${A1_N_PROMPTS}"
    --seed "${SEED}"
    --template_seed "${TEMPLATE_SEED}"
    --template_randomization 1
    --shuffle_choices 1
    --add_answer_prefix 1
    --answer_prefix "${ANSWER_PREFIX}"
    --batch_size "${A1_BATCH_SIZE}"
    --max_prompt_len "${A1_MAX_PROMPT_LEN}"
    --out_dir "${A1_OUT}"
  )
  if [[ -f "${BASIS_NPZ_LAYER}" ]]; then
    A1_CMD+=(--basis_npz "${BASIS_NPZ_LAYER}")
  else
    echo "[Warn] BASIS_NPZ not found: ${BASIS_NPZ_LAYER} (A1 layer=${L} will use a random basis)"
  fi

  echo >> "${COMMANDS_SH}"
  printf "%q " "${A1_CMD[@]}" >> "${COMMANDS_SH}"
  echo >> "${COMMANDS_SH}"

  echo
  echo "[Run] A1(layer=${L}) -> ${A1_OUT}"
  ("${A1_CMD[@]}" |& tee "${A1_LOG}")
done

# -------------------------
# (A2) Geometry (prefill vs decode subspace mismatch)
# -------------------------
DEFAULT_A2_SCRIPT="${SCRIPT_DIR}/exp_A2_geometric_subspace_misalignment.py"
FALLBACK_A2_SCRIPT="${REPO_ROOT}/rebuttal/mechanism/exp_A2_geometric_subspace_misalignment.py"
A2_SCRIPT="${A2_SCRIPT:-${DEFAULT_A2_SCRIPT}}"
if [[ ! -f "${A2_SCRIPT}" && -f "${FALLBACK_A2_SCRIPT}" ]]; then
  A2_SCRIPT="${FALLBACK_A2_SCRIPT}"
fi
A2_LAYERS="${A2_LAYERS:-${LAYERS}}"
A2_TASKS="${A2_TASKS:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}"
A2_N_PROMPTS="${A2_N_PROMPTS:-128}"
A2_CALIB_DECODE_MAX_NEW="${A2_CALIB_DECODE_MAX_NEW:-128}"
A2_PER_TASK_MAX_STATES="${A2_PER_TASK_MAX_STATES:-20000}"
A2_KS="${A2_KS:-32,64,128}"
A2_PCA_MAX_ROWS="${A2_PCA_MAX_ROWS:-80000}"
A2_PCA_DEVICE="${A2_PCA_DEVICE:-cpu}"   # cpu is safer; set to cuda if you want speed and have GPU.

A2_OUT="${OUT_DIR}/A2_geometry"
mkdir -p "${A2_OUT}"
A2_LOG="${OUT_DIR}/A2_geometry.log"

A2_CMD=(
  "${PYTHON_BIN}" "${A2_SCRIPT}"
  --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}"
  --layers "${A2_LAYERS}"
  --tasks "${A2_TASKS}"
  --n_prompts "${A2_N_PROMPTS}"
  --seed "${SEED}"
  --template_seed "${TEMPLATE_SEED}"
  --template_randomization 1
  --shuffle_choices 1
  --add_answer_prefix 1
  --answer_prefix "${ANSWER_PREFIX}"
  --batch_size 4
  --max_prompt_len 512
  --calib_decode_max_new_tokens "${A2_CALIB_DECODE_MAX_NEW}"
  --per_task_max_states "${A2_PER_TASK_MAX_STATES}"
  --ks "${A2_KS}"
  --pca_max_rows "${A2_PCA_MAX_ROWS}"
  --pca_device "${A2_PCA_DEVICE}"
  --out_dir "${A2_OUT}"
)

echo >> "${COMMANDS_SH}"
printf "%q " "${A2_CMD[@]}" >> "${COMMANDS_SH}"
echo >> "${COMMANDS_SH}"

echo
echo "[Run] A2 -> ${A2_OUT}"
("${A2_CMD[@]}" |& tee "${A2_LOG}")

# -------------------------
# (A3) Causal decode-only test + matched controls
# -------------------------
DEFAULT_A3_SCRIPT="${SCRIPT_DIR}/exp_A3_causal_decode_only_controls.py"
FALLBACK_A3_SCRIPT="${REPO_ROOT}/rebuttal/mechanism/exp_A3_causal_decode_only_controls.py"
A3_SCRIPT="${A3_SCRIPT:-${DEFAULT_A3_SCRIPT}}"
if [[ ! -f "${A3_SCRIPT}" && -f "${FALLBACK_A3_SCRIPT}" ]]; then
  A3_SCRIPT="${FALLBACK_A3_SCRIPT}"
fi
A3_TASKS_SUBSPACE="${A3_TASKS_SUBSPACE:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq}"
A3_TASKS_EVAL="${A3_TASKS_EVAL:-commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq}"

A3_N_PROMPTS="${A3_N_PROMPTS:-128}"
A3_EVAL_N="${A3_EVAL_N:-256}"
A3_CALIB_DECODE_MAX_NEW="${A3_CALIB_DECODE_MAX_NEW:-128}"
A3_PER_TASK_MAX_STATES="${A3_PER_TASK_MAX_STATES:-20000}"

PCA_VAR="${PCA_VAR:-0.95}"
MIN_DIM="${MIN_DIM:-8}"
MAX_DIM="${MAX_DIM:-256}"
TAU="${TAU:-0.001}"
M_SHARED="${M_SHARED:-all}"

K_EVAL="${K_EVAL:-128}"
ALPHA_REMOVE="${ALPHA_REMOVE:-1.0}"

BOOT_ITERS="${BOOT_ITERS:-5000}"
PERM_ITERS="${PERM_ITERS:-10000}"
CI_ALPHA="${CI_ALPHA:-0.05}"

A3_OUT="${OUT_DIR}/A3_causal"
mkdir -p "${A3_OUT}"

for L in "${LAYERS_ARR_CLEAN[@]}"; do
  A3_LOG="${OUT_DIR}/A3_causal_layer${L}.log"

  A3_CMD=(
    "${PYTHON_BIN}" "${A3_SCRIPT}"
    --model "${MODEL}" --device "${DEVICE}" --dtype "${DTYPE}"
    --layer "${L}"
    --tasks_subspace "${A3_TASKS_SUBSPACE}"
    --tasks_eval "${A3_TASKS_EVAL}"
    --n_prompts "${A3_N_PROMPTS}"
    --eval_n "${A3_EVAL_N}"
    --seed "${SEED}"
    --template_seed "${TEMPLATE_SEED}"
    --template_randomization 1
    --shuffle_choices 1
    --add_answer_prefix 1
    --answer_prefix "${ANSWER_PREFIX}"
    --batch_size 4
    --max_prompt_len 512
    --calib_decode_max_new_tokens "${A3_CALIB_DECODE_MAX_NEW}"
    --per_task_max_states "${A3_PER_TASK_MAX_STATES}"
    --pca_var "${PCA_VAR}"
    --min_dim "${MIN_DIM}"
    --max_dim "${MAX_DIM}"
    --tau "${TAU}"
    --m_shared "${M_SHARED}"
    --k_eval "${K_EVAL}"
    --alpha_remove "${ALPHA_REMOVE}"
    --fc_prefix_mode "${FC_PREFIX_MODE}"
    --fc_answer_prefix "${ANSWER_PREFIX}"
    --bootstrap_iters "${BOOT_ITERS}"
    --perm_iters "${PERM_ITERS}"
    --alpha "${CI_ALPHA}"
    --out_dir "${A3_OUT}"
  )

  echo >> "${COMMANDS_SH}"
  printf "%q " "${A3_CMD[@]}" >> "${COMMANDS_SH}"
  echo >> "${COMMANDS_SH}"

  echo
  echo "[Run] A3(layer=${L}) -> ${A3_OUT}"
  ("${A3_CMD[@]}" |& tee "${A3_LOG}")
done

# Convenience: update "latest" pointer
mkdir -p "${OUT_BASE}"
ln -sfn "${OUT_DIR}" "${OUT_BASE}/latest" || true

echo
echo "[Done] All PartA runs finished."
echo "[Done] OUT_DIR=${OUT_DIR}"
echo "[Done] Latest -> ${OUT_BASE}/latest"
