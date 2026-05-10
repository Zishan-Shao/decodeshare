#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Run openanswer experiments for (Falcon Instruct, Qwen Instruct) on layers 4/10/24
# - No local Q_shared_layerX.npy required
# - Compute Q_shared on-the-fly via subspace_patching_transfer.py (maybe_compute_Qs)
# - Cache Q_shared under results/... so reruns reuse it
###############################################################################

# Always run relative to this script directory (so relative paths work)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

###############################################################################
# User config (override via env vars if needed)
###############################################################################
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp32}"          # bf16 recommended if supported; can set fp16/fp32
SEED="${SEED:-123}"

LAYERS=(${LAYERS:-4 10 24})

# Paths (assume files live in patch_back/)
BASE_SCRIPT_PATH="${BASE_SCRIPT_PATH:-subspace_patching_transfer.py}"

# Your two openanswer scripts (already exist)
FALCON_SCRIPT="${FALCON_SCRIPT:-openanswer_subspace_patching_falcon.py}"
QWEN_SCRIPT="${QWEN_SCRIPT:-openanswer_subspace_patching_qwen.py}"

# Aux module paths (same defaults as your python)
LOTO8_PATH="${LOTO8_PATH:-disturb_CoT_shared_acc_lasttoken_fp32_sanity_energy_balance_loto8.py}"
DATALOADERS_PATH="${DATALOADERS_PATH:-benchmark_dataloaders.py}"

# Output roots
OUT_ROOT="${OUT_ROOT:-results/openanswer_instruct}"
QS_CACHE_ROOT="${QS_CACHE_ROOT:-results/qshared_cache_openanswer}"

# Eval sizes
N_EVAL="${N_EVAL:-256}"
MAX_FLIPS="${MAX_FLIPS:-64}"
PATCH_N_STEPS="${PATCH_N_STEPS:-4}"

# Token limits
GOLD_MAX_TOKENS_HE="${GOLD_MAX_TOKENS_HE:-128}"
MAX_NEW_TOKENS_MATH="${MAX_NEW_TOKENS_MATH:-96}"
MAX_NEW_TOKENS_CODE="${MAX_NEW_TOKENS_CODE:-256}"

# HumanEval dataset (HF)
HUMANEVAL_HF_ID="${HUMANEVAL_HF_ID:-openai_humaneval}"
HUMANEVAL_HF_SPLIT="${HUMANEVAL_HF_SPLIT:-test}"

# Q_shared compute config (match your openanswer defaults; override if desired)
BASIS_TASKS="${BASIS_TASKS:-gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa}"
BASIS_N_SUBSPACE="${BASIS_N_SUBSPACE:-128}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-8}"
CALIB_MAX_NEW_TOKENS="${CALIB_MAX_NEW_TOKENS:-128}"
PER_TASK_MAX_STATES="${PER_TASK_MAX_STATES:-20000}"
MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-1024}"
VARIANCE_THRESHOLD="${VARIANCE_THRESHOLD:-0.95}"
MIN_DIM="${MIN_DIM:-8}"
MAX_DIM="${MAX_DIM:-1024}"
TAU="${TAU:-0.001}"
M_SHARED="${M_SHARED:-all}"

# Recompute Qs even if cached?
FORCE_RECOMPUTE_QS="${FORCE_RECOMPUTE_QS:-0}"

###############################################################################
# Models (INSTRUCT ONLY)
###############################################################################
FALCON_MODEL="${FALCON_MODEL:-tiiuae/falcon-7b-instruct}"
QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen2.5-7B-Instruct}"

###############################################################################
# Prompting (strong guidance)
###############################################################################
# Math strict (good for pair_logprob: encourage immediate numeric)
SYS_MATH_STRICT="${SYS_MATH_STRICT:-You are a helpful assistant. For math problems: output ONLY the final numeric answer. No words, no units, no punctuation, no explanation.}"
# Math CoT but enforce final line
SYS_MATH_COT="${SYS_MATH_COT:-You are a helpful assistant. Solve the problem step by step. At the end, output a single line exactly: Final answer (number only): <number>}"
# Code strict
SYS_CODE="${SYS_CODE:-You are a senior Python engineer. Return ONLY valid Python code. Do NOT include Markdown fences, explanations, or commentary.}"

###############################################################################
# Helpers
###############################################################################
die() { echo "[Error] $*" >&2; exit 1; }
need_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
ts() { date +"%Y%m%d_%H%M%S"; }

###############################################################################
# Checks
###############################################################################
need_file "$BASE_SCRIPT_PATH"
need_file "$FALCON_SCRIPT"
need_file "$QWEN_SCRIPT"
need_file "$LOTO8_PATH"
need_file "$DATALOADERS_PATH"

mkdir -p "$OUT_ROOT" "$QS_CACHE_ROOT"

###############################################################################
# Compute Q_shared via subspace_patching_transfer.py (maybe_compute_Qs)
# Cache path: ${QS_CACHE_ROOT}/{tag}/layer{L}/Q_shared.npy
###############################################################################
compute_qs_if_needed() {
  local model_id="$1"     # HF model id
  local tag="$2"          # "falcon" or "qwen"
  local layer="$3"        # e.g. 4
  local out_dir="$QS_CACHE_ROOT/$tag/layer${layer}"
  local out_path="$out_dir/Q_shared.npy"

  mkdir -p "$out_dir"

  if [[ "$FORCE_RECOMPUTE_QS" == "1" ]]; then
    echo "[Info] FORCE_RECOMPUTE_QS=1 -> recompute Q_shared for $tag layer $layer"
  else
    if [[ -f "$out_path" ]]; then
      echo "[Info] Reuse cached Q_shared: $out_path"
      echo "$out_path"
      return 0
    fi
  fi

  echo "[Info] Computing Q_shared on-the-fly for model=$model_id tag=$tag layer=$layer"
  echo "[Info]  -> will write: $out_path"

  # Run a tiny python program that imports subspace_patching_transfer.py and calls maybe_compute_Qs
  python - <<PY
import os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import importlib.util

BASE_SCRIPT_PATH = os.path.abspath("${BASE_SCRIPT_PATH}")
LOTO8_PATH = os.path.abspath("${LOTO8_PATH}")
DATALOADERS_PATH = os.path.abspath("${DATALOADERS_PATH}")

MODEL_ID = "${model_id}"
DEVICE = "${DEVICE}"
DTYPE = "${DTYPE}"
LAYER = int("${layer}")
SEED = int("${SEED}")

OUT_PATH = os.path.abspath("${out_path}")

BASIS_TASKS = "${BASIS_TASKS}"
BASIS_N_SUBSPACE = int("${BASIS_N_SUBSPACE}")
CALIB_BATCH_SIZE = int("${CALIB_BATCH_SIZE}")
CALIB_MAX_NEW_TOKENS = int("${CALIB_MAX_NEW_TOKENS}")
PER_TASK_MAX_STATES = int("${PER_TASK_MAX_STATES}")
MAX_PROMPT_LEN = int("${MAX_PROMPT_LEN}")
VARIANCE_THRESHOLD = float("${VARIANCE_THRESHOLD}")
MIN_DIM = int("${MIN_DIM}")
MAX_DIM = int("${MAX_DIM}")
TAU = float("${TAU}")
M_SHARED = "${M_SHARED}"

ANSWER_PREFIX = "\\nFinal answer:"  # used inside basis tasks prompting, OK for Qs basis

def import_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod

# Seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

base_mod = import_module_from_path("subspace_patching_transfer_runtime", BASE_SCRIPT_PATH)

# Load aux modules (loto8 hooks + benchmark dataloaders)
loto8, dl = base_mod.load_aux_modules(LOTO8_PATH, DATALOADERS_PATH)

# dtype
dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
torch_dtype = dtype_map.get(DTYPE, torch.bfloat16)

# Load model/tokenizer (trust_remote_code for falcon is often needed; safe to try for both)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=None,
        trust_remote_code=True,
    ).to(DEVICE)
except TypeError:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch_dtype,
        device_map=None,
    ).to(DEVICE)

model.eval()

try:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
except TypeError:
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

if tok.pad_token_id is None and tok.eos_token_id is not None:
    tok.pad_token = tok.eos_token

tasks = [x.strip() for x in BASIS_TASKS.split(",") if x.strip()]

# Compute Q_shared
Qs = base_mod.maybe_compute_Qs(
    loto8=loto8,
    dl=dl,
    model=model,
    tokenizer=tok,
    layer_idx=LAYER,
    seed=SEED,
    tasks=tasks,
    n_subspace=BASIS_N_SUBSPACE,
    template_randomization=True,
    shuffle_choices=True,
    answer_prefix=ANSWER_PREFIX,
    calib_batch_size=CALIB_BATCH_SIZE,
    calib_max_new_tokens=CALIB_MAX_NEW_TOKENS,
    per_task_max_states=PER_TASK_MAX_STATES,
    max_prompt_len=MAX_PROMPT_LEN,
    variance_threshold=VARIANCE_THRESHOLD,
    min_dim=MIN_DIM,
    max_dim=MAX_DIM,
    tau=TAU,
    m_shared=M_SHARED,
    out_path=OUT_PATH,
)

print(f"[OK] wrote Q_shared: {OUT_PATH} shape={tuple(Qs.shape)}")
PY

  [[ -f "$out_path" ]] || die "Q_shared compute finished but file not found: $out_path"
  echo "$out_path"
}

###############################################################################
# Run openanswer experiments for a given (model, script, tag, layer, Qs_path)
###############################################################################
run_openanswer_for_layer() {
  local script="$1"
  local model_id="$2"
  local tag="$3"
  local layer="$4"
  local qs_path="$5"

  local out_dir="$OUT_ROOT/$tag/layer${layer}"
  local log_dir="$out_dir/logs"
  mkdir -p "$out_dir" "$log_dir"
  local tstamp
  tstamp="$(ts)"

  # Common args
  local common=(
    --base_script_path "$BASE_SCRIPT_PATH"
    --model "$model_id"
    --device "$DEVICE"
    --dtype "$DTYPE"
    --layer "$layer"
    --seed "$SEED"
    --Qs_path "$qs_path"
    --patch_n_steps "$PATCH_N_STEPS"
    --n_eval "$N_EVAL"
    --max_flips "$MAX_FLIPS"
    --prompt_format chat
  )

  echo "------------------------------------------------------------------"
  echo "[Run] $tag instruct @ layer=$layer"
  echo "[Run] script=$script"
  echo "[Run] model=$model_id"
  echo "[Run] Qs_path=$qs_path"
  echo "------------------------------------------------------------------"

  # 1) GSM8K pair_logprob
  python "$script" \
    "${common[@]}" \
    --system_prompt "$SYS_MATH_STRICT" \
    --answer_prefix $'\nFinal answer (number only):' \
    --task gsm8k \
    --eval_mode pair_logprob \
    --out_json "$out_dir/gsm8k_pairlogprob.json" \
    2>&1 | tee "$log_dir/${tstamp}_gsm8k_pairlogprob.log"

  # 2) GSM8K gen_math
  python "$script" \
    "${common[@]}" \
    --system_prompt "$SYS_MATH_COT" \
    --answer_prefix $'\nLet'\''s think step by step.\nFinal answer (number only):' \
    --task gsm8k \
    --eval_mode gen_math \
    --max_new_tokens "$MAX_NEW_TOKENS_MATH" \
    --out_json "$out_dir/gsm8k_genmath.json" \
    2>&1 | tee "$log_dir/${tstamp}_gsm8k_genmath.log"

  # 3) HumanEval pair_logprob (HF loader)
  python "$script" \
    "${common[@]}" \
    --system_prompt "$SYS_CODE" \
    --task humaneval \
    --use_benchmark_loader 0 --hf_id "$HUMANEVAL_HF_ID" --hf_split "$HUMANEVAL_HF_SPLIT" \
    --eval_mode pair_logprob \
    --gold_max_tokens "$GOLD_MAX_TOKENS_HE" \
    --out_json "$out_dir/humaneval_pairlogprob.json" \
    2>&1 | tee "$log_dir/${tstamp}_humaneval_pairlogprob.log"

  # 4) HumanEval gen_code_compile
  python "$script" \
    "${common[@]}" \
    --system_prompt "$SYS_CODE" \
    --task humaneval \
    --use_benchmark_loader 0 --hf_id "$HUMANEVAL_HF_ID" --hf_split "$HUMANEVAL_HF_SPLIT" \
    --eval_mode gen_code_compile \
    --max_new_tokens "$MAX_NEW_TOKENS_CODE" \
    --out_json "$out_dir/humaneval_gencode_compile.json" \
    2>&1 | tee "$log_dir/${tstamp}_humaneval_gencode_compile.log"
}

###############################################################################
# Main
###############################################################################
echo "[Info] DEVICE=$DEVICE DTYPE=$DTYPE SEED=$SEED"
echo "[Info] LAYERS=${LAYERS[*]}"
echo "[Info] OUT_ROOT=$OUT_ROOT"
echo "[Info] QS_CACHE_ROOT=$QS_CACHE_ROOT"
echo "[Info] FALCON_MODEL=$FALCON_MODEL"
echo "[Info] QWEN_MODEL=$QWEN_MODEL"
echo "[Info] FORCE_RECOMPUTE_QS=$FORCE_RECOMPUTE_QS"
echo ""

for layer in "${LAYERS[@]}"; do
  echo "=================================================================="
  echo "[Layer $layer] Prepare Q_shared (on-the-fly) and run experiments"
  echo "=================================================================="

  # Falcon Qs + run
  falcon_qs_path="$(compute_qs_if_needed "$FALCON_MODEL" "falcon" "$layer")"
  run_openanswer_for_layer "$FALCON_SCRIPT" "$FALCON_MODEL" "falcon" "$layer" "$falcon_qs_path"

  # Qwen Qs + run
  qwen_qs_path="$(compute_qs_if_needed "$QWEN_MODEL" "qwen" "$layer")"
  run_openanswer_for_layer "$QWEN_SCRIPT" "$QWEN_MODEL" "qwen" "$layer" "$qwen_qs_path"
done

echo ""
echo "[Done] All runs completed."
echo "  Results: $OUT_ROOT/{falcon,qwen}/layer{4,10,24}/"
echo "  Qs cache: $QS_CACHE_ROOT/{falcon,qwen}/layer{4,10,24}/Q_shared.npy"
