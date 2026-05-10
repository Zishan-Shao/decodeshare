#!/usr/bin/env bash
set -euo pipefail

# ========= Environment =========
source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd patch_back
conda activate flashsvd

# ========= Config =========
MODEL_FALCON="${MODEL_FALCON:-tiiuae/falcon-7b-instruct}"
TEMPLATE_FALCON="${TEMPLATE_FALCON:-falcon_instruct}"   # template key (only used if a script supports a template flag)
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp32}"                                  # fp32 for comparability
LAYER="${LAYER:-10}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

SEED_MAIN="${SEED_MAIN:-123}"
SEED_ROBUST="${SEED_ROBUST:-456}"

# If scripts expose these flags, we can force them deterministically
TEMPLATE_RANDOMIZATION="${TEMPLATE_RANDOMIZATION:-0}"    # int (0/1) if supported
SHUFFLE_CHOICES="${SHUFFLE_CHOICES:-0}"                  # int (0/1) if supported
USE_HF_CHAT_TEMPLATE="${USE_HF_CHAT_TEMPLATE:-1}"        # int (0/1) if supported

BASE_SCRIPT="subspace_patching_transfer.py"
FLIP_SCRIPT="flipset_alpha_sweep_and_transfer.py"
OPEN_SCRIPT="openanswer_subspace_patching.py"
SUM_SCRIPT="summarize_patching_jsons.py"

MODEL_TAG="$(echo "${MODEL_FALCON}" | sed 's|/|__|g')"
OUTROOT="patch_back/results/${MODEL_TAG}/layer${LAYER}"
mkdir -p "${OUTROOT}"

echo "[Info] MODEL_FALCON=${MODEL_FALCON}"
echo "[Info] TEMPLATE_FALCON=${TEMPLATE_FALCON}"
echo "[Info] OUTROOT=${OUTROOT}"
echo "[Info] LAYER=${LAYER} DTYPE=${DTYPE} DEVICE=${DEVICE}"
echo "[Info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# ========= Ensure scripts exist =========
for f in "${BASE_SCRIPT}" "${FLIP_SCRIPT}" "${OPEN_SCRIPT}" "${SUM_SCRIPT}"; do
  if [ ! -f "${f}" ]; then
    echo "[Error] Missing script: ${f} in $(pwd)"
    exit 1
  fi
done

# ========= Template arg auto-detect (EXACT TOKEN MATCH) =========
escape_re() {
  # escape for grep -E
  printf '%s' "$1" | sed -e 's/[][(){}.^$+*?|\\/]/\\&/g'
}

has_arg() {
  # True only if the script help contains the exact option token (not substring)
  # e.g. "--template" will NOT match "--template_randomization"
  local script="$1"
  local needle="$2"
  local nre
  nre="$(escape_re "$needle")"
  python "${script}" -h 2>&1 | grep -Eq "^[[:space:]]*${nre}([[:space:],=\[]|$)"
}

template_flags_for() {
  local script="$1"
  local flags=()

  # template name flags (string)
  if has_arg "${script}" "--template"; then
    flags+=(--template "${TEMPLATE_FALCON}")
  elif has_arg "${script}" "--prompt_template"; then
    flags+=(--prompt_template "${TEMPLATE_FALCON}")
  elif has_arg "${script}" "--chat_template"; then
    flags+=(--chat_template "${TEMPLATE_FALCON}")
  elif has_arg "${script}" "--format"; then
    flags+=(--format "${TEMPLATE_FALCON}")
  fi

  # deterministic toggles (ints) if supported
  if has_arg "${script}" "--template_randomization"; then
    flags+=(--template_randomization "${TEMPLATE_RANDOMIZATION}")
  fi
  if has_arg "${script}" "--shuffle_choices"; then
    flags+=(--shuffle_choices "${SHUFFLE_CHOICES}")
  fi
  if has_arg "${script}" "--use_hf_chat_template"; then
    flags+=(--use_hf_chat_template "${USE_HF_CHAT_TEMPLATE}")
  fi

  echo "${flags[@]}"
}

BASE_TFLAGS=( $(template_flags_for "${BASE_SCRIPT}") )
FLIP_TFLAGS=( $(template_flags_for "${FLIP_SCRIPT}") )
OPEN_TFLAGS=( $(template_flags_for "${OPEN_SCRIPT}") )

echo "[Info] BASE template flags: ${BASE_TFLAGS[*]:-(none)}"
echo "[Info] FLIP template flags: ${FLIP_TFLAGS[*]:-(none)}"
echo "[Info] OPEN template flags: ${OPEN_TFLAGS[*]:-(none)}"

# ============================================================
# (0) Compute / load Q_shared for each seed (Falcon-specific)
# ============================================================
QS_MAIN="${OUTROOT}/Q_shared_layer${LAYER}_seed${SEED_MAIN}.npy"
QS_ROBUST="${OUTROOT}/Q_shared_layer${LAYER}_seed${SEED_ROBUST}.npy"

if [ ! -f "${QS_MAIN}" ]; then
  echo "[Run] Compute Q_shared (seed=${SEED_MAIN}) -> ${QS_MAIN}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
    --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --layer "${LAYER}" --seed "${SEED_MAIN}" \
    "${BASE_TFLAGS[@]}" \
    --compute_Qs 1 --Qs_out "${QS_MAIN}" \
    --basis_tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
    --basis_n_subspace 128 \
    --task aqua --candidate_labels ABCDE \
    --n_eval 256 --max_flips 64 \
    --out_json "${OUTROOT}/compute_Qs_seed${SEED_MAIN}.json"
else
  echo "[Info] Found ${QS_MAIN}"
fi

if [ ! -f "${QS_ROBUST}" ]; then
  echo "[Run] Compute Q_shared (seed=${SEED_ROBUST}) -> ${QS_ROBUST}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
    --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --layer "${LAYER}" --seed "${SEED_ROBUST}" \
    "${BASE_TFLAGS[@]}" \
    --compute_Qs 1 --Qs_out "${QS_ROBUST}" \
    --basis_tasks gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa \
    --basis_n_subspace 128 \
    --task aqua --candidate_labels ABCDE \
    --n_eval 256 --max_flips 64 \
    --out_json "${OUTROOT}/compute_Qs_seed${SEED_ROBUST}.json"
else
  echo "[Info] Found ${QS_ROBUST}"
fi

# ============================================================
# (1) Multiple-choice patchback suite (seed=SEED_MAIN)
# ============================================================
MC_DIR="${OUTROOT}/subspace_mc_seed${SEED_MAIN}"
mkdir -p "${MC_DIR}"

echo "[Run] MC patching: AQuA"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 254 --max_flips 128 \
  --out_json "${MC_DIR}/aqua.json"

echo "[Run] MC patching: ARC-Challenge"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task arc_challenge --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/arc_challenge.json"

echo "[Run] MC patching: CommonsenseQA"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task commonsenseqa --candidate_labels ABCDE \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/commonsenseqa.json"

echo "[Run] MC patching: LogiQA"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task logiqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/logiqa.json"

echo "[Run] MC patching: OpenBookQA"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task openbookqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/openbookqa.json"

echo "[Run] MC patching: PIQA"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task piqa --candidate_labels AB \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/piqa.json"

echo "[Run] MC patching: QASC (8-choice; if your loader differs, change candidate_labels)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${BASE_TFLAGS[@]}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task qasc --candidate_labels ABCDEFGH \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/qasc.json"

# ============================================================
# (2) Flip-set alpha sweep + transfer donor patching (AQuA)
# ============================================================
FLIP_DIR="${OUTROOT}/flipset"
mkdir -p "${FLIP_DIR}"

echo "[Run] Flipset alpha sweep (seed=${SEED_MAIN})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${FLIP_TFLAGS[@]}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_MAIN}" \
  --run_alpha_sweep 1 --alpha_list 0,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --run_transfer_patching 0 \
  --out_json "${FLIP_DIR}/aqua_alpha_sweep_seed${SEED_MAIN}.json"

echo "[Run] Flipset alpha sweep (seed=${SEED_ROBUST})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_ROBUST}" \
  "${FLIP_TFLAGS[@]}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_ROBUST}" \
  --run_alpha_sweep 1 --alpha_list 0,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --run_transfer_patching 0 \
  --out_json "${FLIP_DIR}/aqua_alpha_sweep_seed${SEED_ROBUST}.json"

echo "[Run] Flipset transfer donors (same task, seed=${SEED_MAIN})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${FLIP_TFLAGS[@]}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_MAIN}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source same_task_eval --donor_n_eval 512 --donor_pick random \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${FLIP_DIR}/aqua_transfer_same_task_seed${SEED_MAIN}.json"

echo "[Run] Flipset transfer donors (cross-task MC baseline-correct, seed=${SEED_MAIN})"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${FLIP_TFLAGS[@]}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_MAIN}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source cross_task_eval --donor_tasks commonsenseqa,openbookqa \
  --donor_n_eval 256 --donor_pick random \
  --donor_require_gold_in_candidates 1 --donor_require_baseline_correct 1 \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${FLIP_DIR}/aqua_transfer_cross_mc_baselinecorrect_seed${SEED_MAIN}.json"

# ============================================================
# (3) Open-answer suite (seed=SEED_MAIN) — patch_n_steps=4 for stability
# ============================================================
OA_DIR="${OUTROOT}/openanswer_seed${SEED_MAIN}"
mkdir -p "${OA_DIR}"

echo "[Run] OpenAnswer GSM8K pair_logprob"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${OPEN_TFLAGS[@]}" \
  --task gsm8k --n_eval 256 --max_flips 64 \
  --eval_mode pair_logprob \
  --Qs_path "${QS_MAIN}" \
  --patch_n_steps 4 \
  --out_json "${OA_DIR}/gsm8k_pairlogprob.json"

echo "[Run] OpenAnswer GSM8K gen_math (max_new_tokens=64)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${OPEN_TFLAGS[@]}" \
  --task gsm8k --n_eval 256 --max_flips 64 \
  --eval_mode gen_math \
  --Qs_path "${QS_MAIN}" \
  --patch_n_steps 4 --max_new_tokens 64 \
  --out_json "${OA_DIR}/gsm8k_genmath.json"

echo "[Run] OpenAnswer HumanEval pair_logprob"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${OPEN_TFLAGS[@]}" \
  --task humaneval --use_benchmark_loader 0 \
  --hf_id openai_humaneval --hf_split test \
  --n_eval 164 --max_flips 64 \
  --eval_mode pair_logprob \
  --Qs_path "${QS_MAIN}" \
  --gold_max_tokens 128 \
  --patch_n_steps 4 \
  --out_json "${OA_DIR}/humaneval_pairlogprob.json"

echo "[Run] OpenAnswer HumanEval gen_code_compile (safe proxy; max_new_tokens=256)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_FALCON}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  "${OPEN_TFLAGS[@]}" \
  --task humaneval --use_benchmark_loader 0 \
  --hf_id openai_humaneval --hf_split test \
  --n_eval 164 --max_flips 64 \
  --eval_mode gen_code_compile \
  --Qs_path "${QS_MAIN}" \
  --patch_n_steps 4 --max_new_tokens 256 \
  --out_json "${OA_DIR}/humaneval_gencode_compile.json"

# ============================================================
# (4) Summarize all Falcon JSONs
# ============================================================
SUM_DIR="${OUTROOT}/_summary"
mkdir -p "${SUM_DIR}"

echo "[Run] Summarize all JSONs under ${OUTROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" python "${SUM_SCRIPT}" \
  --dir "${OUTROOT}" \
  --pattern "**/*.json" \
  --recursive \
  --no_dedupe \
  --out_csv "${SUM_DIR}/summary.csv" \
  --out_md "${SUM_DIR}/summary.md" \
  --out_paper_md "${SUM_DIR}/paper_table.md" \
  --out_alpha_csv "${SUM_DIR}/alpha_sweep.csv" \
  --out_alpha_md "${SUM_DIR}/alpha_sweep.md"

# ============================================================
# (5) Detailed analysis report + LaTeX tables + PDF plots
# ============================================================
ANALYSIS_PY="${SUM_DIR}/analyze_falcon_results.py"
REPORT_MD="${SUM_DIR}/falcon_report.md"
TABLES_TEX="${SUM_DIR}/falcon_tables.tex"
PLOT_DIR="${SUM_DIR}/plots"
mkdir -p "${PLOT_DIR}"

cat > "${ANALYSIS_PY}" <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_CSV = os.environ.get("SUMMARY_CSV", "summary.csv")
ALPHA_CSV = os.environ.get("ALPHA_CSV", "alpha_sweep.csv")
OUT_MD = os.environ.get("OUT_MD", "falcon_report.md")
OUT_TEX = os.environ.get("OUT_TEX", "falcon_tables.tex")
PLOT_DIR = os.environ.get("PLOT_DIR", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(SUMMARY_CSV)
alpha_df = pd.read_csv(ALPHA_CSV) if os.path.exists(ALPHA_CSV) else pd.DataFrame()

mc = df[df.get("kind","") == "subspace_mc"].copy()
oa = df[df.get("kind","") == "openanswer"].copy()
fs = df[df.get("kind","") == "flipset"].copy()

def select_cols(kind_df, cols):
    keep = [c for c in cols if c in kind_df.columns]
    return kind_df[keep].copy()

mc_cols = [
    "task","eval_mode","seed","base_acc_scan","ablt_acc_scan","flips_scan",
    "patched_0_rescued_pct","patched_full_rescued_pct",
    "control_time_shuffled_rescued_pct","control_shared_randvec_rescued_pct",
    "control_rand_subspace_rescued_pct","control_patch_nonshared_rescued_pct",
]
oa_cols = [
    "task","eval_mode","seed","base_acc_scan","ablt_acc_scan","flips_scan",
    "patched_self_rescued_pct","control_time_shuffled_rescued_pct",
    "control_shared_randvec_rescued_pct","control_rand_subspace_rescued_pct",
    "control_patch_nonshared_rescued_pct",
]
fs_cols = [
    "file","seed","task","base_acc_scan","ablt_acc_scan","flips_scan",
    "patched_self_rescued_pct","patched_transfer_rescued_pct",
]

mc_tbl = select_cols(mc, mc_cols).sort_values(["task","eval_mode","seed"]) if len(mc) else pd.DataFrame()
oa_tbl = select_cols(oa, oa_cols).sort_values(["task","eval_mode","seed"]) if len(oa) else pd.DataFrame()
fs_tbl = select_cols(fs, fs_cols).sort_values(["seed","file"]) if len(fs) else pd.DataFrame()

# Plot: MC patched_0 rescue by task
if len(mc_tbl) and "patched_0_rescued_pct" in mc_tbl.columns:
    mc_plot = mc_tbl.groupby("task", as_index=False)["patched_0_rescued_pct"].mean()
    plt.figure()
    plt.bar(mc_plot["task"], mc_plot["patched_0_rescued_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rescue% on flips (patched_0)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "mc_patched0_rescue.pdf"), dpi=300)
    plt.close()

# Plot: alpha sweep flip rate
if len(alpha_df) and {"alpha","flip_rate","seed"}.issubset(alpha_df.columns):
    a = alpha_df.copy()
    a["alpha"] = pd.to_numeric(a["alpha"], errors="coerce")
    a = a.dropna(subset=["alpha"])
    plt.figure()
    for seed in sorted(a["seed"].dropna().unique()):
        sub = a[a["seed"] == seed].sort_values("alpha")
        plt.plot(sub["alpha"], sub["flip_rate"]*100.0, marker="o", label=f"seed={int(seed)}")
    plt.xlabel("alpha")
    plt.ylabel("Flip rate on flip-set (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "alpha_sweep_fliprate.pdf"), dpi=300)
    plt.close()

def df_to_md_table(dfx: pd.DataFrame, max_rows: int = 40) -> str:
    if dfx is None or len(dfx) == 0:
        return "_(none)_"
    d = dfx.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_markdown(index=False)

lines = []
lines.append("# Falcon subspace patching + flipset report\n")
lines.append(f"Generated from `{os.path.basename(SUMMARY_CSV)}` and `{os.path.basename(ALPHA_CSV)}`.\n")
lines.append("## Overview\n")
lines.append(f"- Runs: {len(df)} total JSON summaries\n")
lines.append(f"- MC runs: {len(mc)}; Open-answer runs: {len(oa)}; Flipset runs: {len(fs)}\n")
lines.append("## Key plots (PDF)\n")
for fn in ["mc_patched0_rescue.pdf", "alpha_sweep_fliprate.pdf"]:
    p = os.path.join(PLOT_DIR, fn)
    if os.path.exists(p):
        lines.append(f"- `{fn}`")
lines.append("\n## Multiple-choice patchback (subspace_mc)\n")
lines.append(df_to_md_table(mc_tbl))
lines.append("\n## Open-answer patchback (openanswer)\n")
lines.append(df_to_md_table(oa_tbl))
lines.append("\n## Flipset transfer patching (flipset)\n")
lines.append(df_to_md_table(fs_tbl))
lines.append("\n")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

tex_lines = []
tex_lines.append("% Auto-generated LaTeX tables for Falcon results\n")
tex_lines.append("% Requires \\usepackage{booktabs}\n\n")

if len(mc_tbl):
    tex_lines.append(mc_tbl.to_latex(index=False, escape=True,
                                     caption="Falcon: multiple-choice summary.",
                                     label="tab:falcon_mc_summary"))
    tex_lines.append("\n")
if len(oa_tbl):
    tex_lines.append(oa_tbl.to_latex(index=False, escape=True,
                                     caption="Falcon: open-answer summary.",
                                     label="tab:falcon_openanswer_summary"))
    tex_lines.append("\n")

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

print(f"[OK] Wrote report: {OUT_MD}")
print(f"[OK] Wrote LaTeX tables: {OUT_TEX}")
print(f"[OK] Plots in: {PLOT_DIR}")
PY

chmod +x "${ANALYSIS_PY}"

echo "[Run] Analyze Falcon results -> report/md/tex/plots"
SUMMARY_CSV="${SUM_DIR}/summary.csv" \
ALPHA_CSV="${SUM_DIR}/alpha_sweep.csv" \
OUT_MD="${REPORT_MD}" \
OUT_TEX="${TABLES_TEX}" \
PLOT_DIR="${PLOT_DIR}" \
python "${ANALYSIS_PY}"

echo ""
echo "[Done] Falcon suite complete."
echo " - Summary CSV: ${SUM_DIR}/summary.csv"
echo " - Summary MD:  ${SUM_DIR}/summary.md"
echo " - Paper table: ${SUM_DIR}/paper_table.md"
echo " - Report:      ${REPORT_MD}"
echo " - LaTeX tables:${TABLES_TEX}"
echo " - Plots dir:   ${PLOT_DIR}"
