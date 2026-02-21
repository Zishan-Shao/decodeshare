#!/usr/bin/env bash
set -euo pipefail

# Run the two rebuttal experiments and save results under a single run directory.
#
# Usage (defaults):
#   bash rebuttal/run_ranking_flip_and_repair_controls.sh
#
# Override any parameter via env vars, e.g.:
#   MODEL=meta-llama/Llama-2-13b-chat-hf \
#   VECTORS_MANIFEST_RANK=/path/to/your_100_vectors.jsonl \
#   VECTORS_MANIFEST_REPAIR=/path/to/your_vectors_for_repair.jsonl \
#   OUT_DIR=results/rebuttal_my_run \
#   bash rebuttal/run_ranking_flip_and_repair_controls.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"

# -------------------------
# Shared config
# -------------------------
MODEL="${MODEL:-meta-llama/Llama-2-7b-chat-hf}"
DEVICE="${DEVICE:-cuda}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"   # fp32|fp16
SEED="${SEED:-42}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/results/rebuttal_${RUN_ID}}"
mkdir -p "${OUT_DIR}"

# -------------------------
# Experiment 1: Ranking flip
# -------------------------
RANKING_SCRIPT="${RANKING_SCRIPT:-${REPO_ROOT}/rebuttal/exp_ranking_flip_steering.py}"
# Default to a later layer example (steering tends to be more meaningful in later layers).
VECTORS_MANIFEST_RANK="${VECTORS_MANIFEST_RANK:-${REPO_ROOT}/rebuttal/steering_vectors_layer28.jsonl}"
RANK_MAX_VECTORS="${RANK_MAX_VECTORS:-0}"  # 0=no limit
RANK_FILTER_REGEX="${RANK_FILTER_REGEX:-}"

RANK_TASKS="${RANK_TASKS:-commonsenseqa,arc_challenge,openbookqa,qasc,logiqa}"
RANK_N_EVAL="${RANK_N_EVAL:-128}"
TEMPLATE_SEEDS_RANK="${TEMPLATE_SEEDS_RANK:-1234,2345,3456}"
TEMPLATE_SEEDS_REAL="${TEMPLATE_SEEDS_REAL:-4567,5678,6789}"

DECODING="${DECODING:-greedy}"            # greedy|sample
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
REASONING_TOKENS="${REASONING_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-12345}"

TRAD_MODE="${TRAD_MODE:-prefill}"         # prefill|both
DECODE_MODE="${DECODE_MODE:-decode}"      # decode|both
STAGED="${STAGED:-1}"                     # 0|1
AGG="${AGG:-mean}"                        # mean|min|median

RANK_OUT_JSON="${RANK_OUT_JSON:-${OUT_DIR}/ranking_flip.json}"
RANK_LOG="${OUT_DIR}/ranking_flip.log"

RANK_CMD=(
  "${PYTHON_BIN}" "${RANKING_SCRIPT}"
  --model "${MODEL}"
  --device "${DEVICE}"
  --model_dtype "${MODEL_DTYPE}"
  --vectors_manifest "${VECTORS_MANIFEST_RANK}"
  --max_vectors "${RANK_MAX_VECTORS}"
  --tasks "${RANK_TASKS}"
  --n_eval "${RANK_N_EVAL}"
  --template_seeds_rank "${TEMPLATE_SEEDS_RANK}"
  --template_seeds_real "${TEMPLATE_SEEDS_REAL}"
  --decoding "${DECODING}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --reasoning_tokens "${REASONING_TOKENS}"
  --batch_size "${BATCH_SIZE}"
  --max_prompt_len "${MAX_PROMPT_LEN}"
  --sample_seed "${SAMPLE_SEED}"
  --trad_mode "${TRAD_MODE}"
  --decode_mode "${DECODE_MODE}"
  --staged "${STAGED}"
  --agg "${AGG}"
  --seed "${SEED}"
  --out_json "${RANK_OUT_JSON}"
)
if [[ -n "${RANK_FILTER_REGEX}" ]]; then
  RANK_CMD+=(--filter_regex "${RANK_FILTER_REGEX}")
fi

# -------------------------
# Experiment 2: Repair vs controls
# -------------------------
REPAIR_SCRIPT="${REPAIR_SCRIPT:-${REPO_ROOT}/rebuttal/exp_repair_controls_steering.py}"
# Default to the same later-layer manifest; override with VECTORS_MANIFEST_REPAIR if needed.
VECTORS_MANIFEST_REPAIR="${VECTORS_MANIFEST_REPAIR:-${REPO_ROOT}/rebuttal/steering_vectors_layer28.jsonl}"
REPAIR_MAX_VECTORS="${REPAIR_MAX_VECTORS:-0}"  # 0=no limit
REPAIR_FILTER_REGEX="${REPAIR_FILTER_REGEX:-}"

TASKS_SUBSPACE="${TASKS_SUBSPACE:-gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa}"
N_SUBSPACE="${N_SUBSPACE:-128}"
TASKS_EVAL="${TASKS_EVAL:-commonsenseqa,arc_challenge,openbookqa,qasc,logiqa}"
N_EVAL="${N_EVAL:-128}"
TEMPLATE_SEEDS="${TEMPLATE_SEEDS:-1234,2345,3456,4567,5678}"

ALPHA_PROJ="${ALPHA_PROJ:-1.0}"
NORM_MATCH="${NORM_MATCH:-1}"                 # 0|1
TAU="${TAU:-0.001}"
M_SHARED="${M_SHARED:-all}"                   # all|<int>|<comma list>
SHARED_DIM="${SHARED_DIM:-0}"                 # 0=all (may be clamped if INCLUDE_PCA_PREFILL=1)

PCA_VAR="${PCA_VAR:-0.95}"
PCA_MAX_ROWS="${PCA_MAX_ROWS:-200000}"        # 0=no limit
PCA_MAX_DIM="${PCA_MAX_DIM:-4096}"
PER_TASK_MAX_STATES="${PER_TASK_MAX_STATES:-20000}"
CALIB_DECODE_MAX_NEW_TOKENS="${CALIB_DECODE_MAX_NEW_TOKENS:--1}"

# If 1, adds "pca_prefill" as an extra strong control (prefill-distribution PCA).
INCLUDE_PCA_PREFILL="${INCLUDE_PCA_PREFILL:-1}"  # 0|1

# Prefill PCA uses **one row per prompt** (prefill last-token state), so max PCA rank is (n_rows-1).
# When INCLUDE_PCA_PREFILL=1, we must ensure k (=shared_dim / #shared comps) does not exceed that.
if [[ "${INCLUDE_PCA_PREFILL}" == "1" ]]; then
  IFS=',' read -r -a _tasks_subspace_arr <<< "${TASKS_SUBSPACE}"
  _n_tasks_subspace=0
  for _t in "${_tasks_subspace_arr[@]}"; do
    _t="${_t#"${_t%%[![:space:]]*}"}"   # ltrim
    _t="${_t%"${_t##*[![:space:]]}"}"   # rtrim
    if [[ -n "${_t}" ]]; then
      _n_tasks_subspace=$((_n_tasks_subspace + 1))
    fi
  done
  _prefill_n_rows=$((_n_tasks_subspace * N_SUBSPACE))
  _max_k_prefill=$((_prefill_n_rows - 1))
  if [[ "${PCA_MAX_DIM}" -gt 0 && "${PCA_MAX_DIM}" -lt "${_max_k_prefill}" ]]; then
    _max_k_prefill="${PCA_MAX_DIM}"
  fi
  if [[ "${_max_k_prefill}" -lt 1 ]]; then
    echo "[Error] INCLUDE_PCA_PREFILL=1 but prefill PCA has insufficient rows: n_rows≈${_prefill_n_rows} (need >=2)." >&2
    exit 1
  fi
  if [[ "${SHARED_DIM}" -le 0 ]]; then
    SHARED_DIM="${_max_k_prefill}"
    echo "[Info] INCLUDE_PCA_PREFILL=1 and SHARED_DIM=0 → set SHARED_DIM=${SHARED_DIM} (prefill n_rows≈${_prefill_n_rows}, max_k_prefill=${_max_k_prefill})"
  elif [[ "${SHARED_DIM}" -gt "${_max_k_prefill}" ]]; then
    echo "[Warn] SHARED_DIM=${SHARED_DIM} > max_k_prefill=${_max_k_prefill} (prefill n_rows≈${_prefill_n_rows}); clamping to ${_max_k_prefill} to avoid crash." >&2
    SHARED_DIM="${_max_k_prefill}"
  fi
fi

REPAIR_OUT_JSON="${REPAIR_OUT_JSON:-${OUT_DIR}/repair_controls.json}"
REPAIR_LOG="${OUT_DIR}/repair_controls.log"

REPAIR_CMD=(
  "${PYTHON_BIN}" "${REPAIR_SCRIPT}"
  --model "${MODEL}"
  --device "${DEVICE}"
  --model_dtype "${MODEL_DTYPE}"
  --vectors_manifest "${VECTORS_MANIFEST_REPAIR}"
  --max_vectors "${REPAIR_MAX_VECTORS}"
  --tasks_subspace "${TASKS_SUBSPACE}"
  --n_subspace "${N_SUBSPACE}"
  --tasks_eval "${TASKS_EVAL}"
  --n_eval "${N_EVAL}"
  --template_seeds "${TEMPLATE_SEEDS}"
  --decoding "${DECODING}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --reasoning_tokens "${REASONING_TOKENS}"
  --batch_size "${BATCH_SIZE}"
  --max_prompt_len "${MAX_PROMPT_LEN}"
  --sample_seed "${SAMPLE_SEED}"
  --staged "${STAGED}"
  --alpha_proj "${ALPHA_PROJ}"
  --norm_match "${NORM_MATCH}"
  --tau "${TAU}"
  --m_shared "${M_SHARED}"
  --shared_dim "${SHARED_DIM}"
  --pca_var "${PCA_VAR}"
  --pca_max_rows "${PCA_MAX_ROWS}"
  --pca_max_dim "${PCA_MAX_DIM}"
  --per_task_max_states "${PER_TASK_MAX_STATES}"
  --calib_decode_max_new_tokens "${CALIB_DECODE_MAX_NEW_TOKENS}"
  --include_pca_prefill "${INCLUDE_PCA_PREFILL}"
  --seed "${SEED}"
  --out_json "${REPAIR_OUT_JSON}"
)
if [[ -n "${REPAIR_FILTER_REGEX}" ]]; then
  REPAIR_CMD+=(--filter_regex "${REPAIR_FILTER_REGEX}")
fi

count_jsonl() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo 0
    return
  fi
  # Count non-empty, non-comment JSON lines (starts with "{" after optional whitespace).
  if command -v rg >/dev/null 2>&1; then
    rg --no-filename -c '^[[:space:]]*\\{' "${f}" 2>/dev/null | tr -d ' '
  else
    grep -c '^[[:space:]]*{' "${f}" 2>/dev/null | tr -d ' '
  fi
}

manifest_layers() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo ""
    return
  fi
  "${PYTHON_BIN}" - "${f}" <<'PY' 2>/dev/null || true
import json, sys
path = sys.argv[1]
layers = set()
with open(path, "r", encoding="utf-8") as fh:
    for line in fh:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if "layer" in obj:
            try:
                layers.add(int(obj["layer"]))
            except Exception:
                pass
print(",".join(str(x) for x in sorted(layers)))
PY
}

{
  echo "# Auto-generated: $(date -Iseconds)"
  echo
  echo "# OUT_DIR"
  printf '%q\n' "${OUT_DIR}"
  echo
  echo "# Ranking flip"
  printf '%q ' "${RANK_CMD[@]}"; echo
  echo
  echo "# Repair controls"
  printf '%q ' "${REPAIR_CMD[@]}"; echo
  echo
} > "${OUT_DIR}/commands.sh"

echo "[Run] OUT_DIR=${OUT_DIR}"
echo "[Run] MODEL=${MODEL}  DEVICE=${DEVICE}  MODEL_DTYPE=${MODEL_DTYPE}"
echo

if [[ ! -f "${VECTORS_MANIFEST_RANK}" ]]; then
  echo "[Error] Missing VECTORS_MANIFEST_RANK: ${VECTORS_MANIFEST_RANK}" >&2
  exit 1
fi
if [[ ! -f "${VECTORS_MANIFEST_REPAIR}" ]]; then
  echo "[Error] Missing VECTORS_MANIFEST_REPAIR: ${VECTORS_MANIFEST_REPAIR}" >&2
  exit 1
fi

N_RANK_VECS="$(count_jsonl "${VECTORS_MANIFEST_RANK}")"
RANK_LAYERS="$(manifest_layers "${VECTORS_MANIFEST_RANK}")"
echo "[Info] Ranking-flip manifest vectors: ${N_RANK_VECS}"
echo "[Info] Ranking-flip manifest layers: ${RANK_LAYERS:-unknown}"
if [[ "${N_RANK_VECS}" -lt 50 ]]; then
  echo "[Warn] Ranking-flip typically uses >=50 vectors; got ${N_RANK_VECS} (set VECTORS_MANIFEST_RANK=...)." >&2
fi
echo

N_REPAIR_VECS="$(count_jsonl "${VECTORS_MANIFEST_REPAIR}")"
REPAIR_LAYERS="$(manifest_layers "${VECTORS_MANIFEST_REPAIR}")"
echo "[Info] Repair-controls manifest vectors: ${N_REPAIR_VECS}"
echo "[Info] Repair-controls manifest layers: ${REPAIR_LAYERS:-unknown}"
echo

echo "[Run] (1/2) Ranking flip → ${RANK_OUT_JSON}"
printf '%q ' "${RANK_CMD[@]}"; echo
"${RANK_CMD[@]}" 2>&1 | tee "${RANK_LOG}"
echo

echo "[Run] (2/2) Repair controls → ${REPAIR_OUT_JSON}"
printf '%q ' "${REPAIR_CMD[@]}"; echo
"${REPAIR_CMD[@]}" 2>&1 | tee "${REPAIR_LOG}"
echo

echo "[Done] Wrote:"
echo "  - ${RANK_OUT_JSON}"
echo "  - ${REPAIR_OUT_JSON}"
echo "  - ${OUT_DIR}/commands.sh"
echo "  - ${RANK_LOG}"
echo "  - ${REPAIR_LOG}"
