#!/usr/bin/env bash
set -euo pipefail

# ========= Environment =========
source "$HOME/miniconda3/etc/profile.d/conda.sh"
cd patch_back
conda activate flashsvd

# ========= Config =========
MODEL_ID="${MODEL_ID:-meta-llama/Llama-2-7b-chat-hf}"   # default: llama-2-7b-chat-hf
MODEL_TAG="${MODEL_TAG:-${MODEL_ID//\//__}}"            # filesystem-safe tag
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-fp32}"                                 # fp32 for comparability
LAYER="${LAYER:-24}"
GPU="${GPU:-0}"

SEED_MAIN="123"
SEED_ROBUST="456"

BASE_SCRIPT="subspace_patching_transfer.py"
FLIP_SCRIPT="flipset_alpha_sweep_and_transfer.py"
OPEN_SCRIPT="openanswer_subspace_patching.py"
SUM_SCRIPT="summarize_patching_jsons.py"                # your unified summarizer

OUTROOT="patch_back/results/${MODEL_TAG}/layer${LAYER}"
mkdir -p "${OUTROOT}"

echo "[Info] MODEL_ID=${MODEL_ID}"
echo "[Info] MODEL_TAG=${MODEL_TAG}"
echo "[Info] OUTROOT=${OUTROOT}"
echo "[Info] LAYER=${LAYER} DTYPE=${DTYPE} DEVICE=${DEVICE} GPU=${GPU}"

# ========= Ensure scripts exist =========
for f in "${BASE_SCRIPT}" "${FLIP_SCRIPT}" "${OPEN_SCRIPT}" "${SUM_SCRIPT}"; do
  if [ ! -f "${f}" ]; then
    echo "[Error] Missing script: ${f} in $(pwd)"
    exit 1
  fi
done

# ============================================================
# (0) Compute / load Q_shared for each seed (model-specific!)
# ============================================================
QS_MAIN="${OUTROOT}/Q_shared_layer${LAYER}_seed${SEED_MAIN}.npy"
QS_ROBUST="${OUTROOT}/Q_shared_layer${LAYER}_seed${SEED_ROBUST}.npy"

if [ ! -f "${QS_MAIN}" ]; then
  echo "[Run] Compute Q_shared (seed=${SEED_MAIN}) -> ${QS_MAIN}"
  CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
    --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --layer "${LAYER}" --seed "${SEED_MAIN}" \
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
  CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
    --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
    --layer "${LAYER}" --seed "${SEED_ROBUST}" \
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
# (1) Multiple-choice patchback suite (seed=123)
# ============================================================
MC_DIR="${OUTROOT}/subspace_mc_seed${SEED_MAIN}"
mkdir -p "${MC_DIR}"

echo "[Run] MC patching: AQuA"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 254 --max_flips 128 \
  --out_json "${MC_DIR}/aqua.json"

echo "[Run] MC patching: ARC-Challenge"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task arc_challenge --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/arc_challenge.json"

echo "[Run] MC patching: CommonsenseQA"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task commonsenseqa --candidate_labels ABCDE \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/commonsenseqa.json"

echo "[Run] MC patching: LogiQA"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task logiqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/logiqa.json"

echo "[Run] MC patching: OpenBookQA"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task openbookqa --candidate_labels ABCD \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/openbookqa.json"

echo "[Run] MC patching: PIQA"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --compute_Qs 0 --Qs_path "${QS_MAIN}" \
  --task piqa --candidate_labels AB \
  --n_eval 256 --max_flips 128 \
  --out_json "${MC_DIR}/piqa.json"

echo "[Run] MC patching: QASC (assuming 8-choice; if your loader differs, change candidate_labels)"
CUDA_VISIBLE_DEVICES="${GPU}" python "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
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
CUDA_VISIBLE_DEVICES="${GPU}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_MAIN}" \
  --run_alpha_sweep 1 --alpha_list 0,0.02,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --run_transfer_patching 0 \
  --out_json "${FLIP_DIR}/aqua_alpha_sweep_seed${SEED_MAIN}.json"

echo "[Run] Flipset alpha sweep (seed=${SEED_ROBUST})"
CUDA_VISIBLE_DEVICES="${GPU}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_ROBUST}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_ROBUST}" \
  --run_alpha_sweep 1 --alpha_list 0,0.05,0.1,0.2,0.3,0.5,0.75,1.0 \
  --run_transfer_patching 0 \
  --out_json "${FLIP_DIR}/aqua_alpha_sweep_seed${SEED_ROBUST}.json"

echo "[Run] Flipset transfer donors (same task, seed=${SEED_MAIN})"
CUDA_VISIBLE_DEVICES="${GPU}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --task aqua --candidate_labels ABCDE \
  --n_eval 1024 --flipset_max 128 \
  --Qs_path "${QS_MAIN}" \
  --run_alpha_sweep 0 --run_transfer_patching 1 \
  --donor_source same_task_eval --donor_n_eval 512 --donor_pick random \
  --patch_window steps_0 --run_self_patch_ref 1 \
  --out_json "${FLIP_DIR}/aqua_transfer_same_task_seed${SEED_MAIN}.json"

echo "[Run] Flipset transfer donors (cross-task MC baseline-correct, seed=${SEED_MAIN})"
CUDA_VISIBLE_DEVICES="${GPU}" python "${FLIP_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
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
# (3) Open-answer suite (seed=123) — patch_n_steps=4 for stability
# ============================================================
OA_DIR="${OUTROOT}/openanswer_seed${SEED_MAIN}"
mkdir -p "${OA_DIR}"

echo "[Run] OpenAnswer GSM8K pair_logprob"
CUDA_VISIBLE_DEVICES="${GPU}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --task gsm8k --n_eval 256 --max_flips 64 \
  --eval_mode pair_logprob \
  --Qs_path "${QS_MAIN}" \
  --patch_n_steps 4 \
  --out_json "${OA_DIR}/gsm8k_pairlogprob.json"

echo "[Run] OpenAnswer GSM8K gen_math (max_new_tokens=64; appendix-style)"
CUDA_VISIBLE_DEVICES="${GPU}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --task gsm8k --n_eval 256 --max_flips 64 \
  --eval_mode gen_math \
  --Qs_path "${QS_MAIN}" \
  --patch_n_steps 4 --max_new_tokens 64 \
  --out_json "${OA_DIR}/gsm8k_genmath.json"

echo "[Run] OpenAnswer HumanEval pair_logprob"
CUDA_VISIBLE_DEVICES="${GPU}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --task humaneval --use_benchmark_loader 0 \
  --hf_id openai_humaneval --hf_split test \
  --n_eval 164 --max_flips 64 \
  --eval_mode pair_logprob \
  --Qs_path "${QS_MAIN}" \
  --gold_max_tokens 128 \
  --patch_n_steps 4 \
  --out_json "${OA_DIR}/humaneval_pairlogprob.json"

echo "[Run] OpenAnswer HumanEval gen_code_compile (safe proxy; max_new_tokens=256)"
CUDA_VISIBLE_DEVICES="${GPU}" python "${OPEN_SCRIPT}" \
  --base_script_path "${BASE_SCRIPT}" \
  --model "${MODEL_ID}" --device "${DEVICE}" --dtype "${DTYPE}" \
  --layer "${LAYER}" --seed "${SEED_MAIN}" \
  --task humaneval --use_benchmark_loader 0 \
  --hf_id openai_humaneval --hf_split test \
  --n_eval 164 --max_flips 64 \
  --eval_mode gen_code_compile \
  --Qs_path "${QS_MAIN}" \
  --patch_n_steps 4 --max_new_tokens 256 \
  --out_json "${OA_DIR}/humaneval_gencode_compile.json"

# ============================================================
# (4) Summarize all JSONs (MD/LaTeX + alpha sweep)
# ============================================================
SUM_DIR="${OUTROOT}/_summary"
mkdir -p "${SUM_DIR}"

echo "[Run] Summarize all JSONs under ${OUTROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" python "${SUM_SCRIPT}" \
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
# (5) Detailed analysis report + LaTeX tables + PDF plots (dpi=300)
# ============================================================
ANALYSIS_PY="${SUM_DIR}/analyze_${MODEL_TAG}_results.py"
REPORT_MD="${SUM_DIR}/${MODEL_TAG}_report.md"
TABLES_TEX="${SUM_DIR}/${MODEL_TAG}_tables.tex"
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
ALPHA_CSV   = os.environ.get("ALPHA_CSV", "alpha_sweep.csv")
OUT_MD      = os.environ.get("OUT_MD", "model_report.md")
OUT_TEX     = os.environ.get("OUT_TEX", "model_tables.tex")
PLOT_DIR    = os.environ.get("PLOT_DIR", "plots")
MODEL_LABEL = os.environ.get("MODEL_LABEL", "Model")

os.makedirs(PLOT_DIR, exist_ok=True)

df = pd.read_csv(SUMMARY_CSV)
alpha_df = pd.read_csv(ALPHA_CSV) if os.path.exists(ALPHA_CSV) else pd.DataFrame()

# Split by kind
mc = df[df["kind"] == "subspace_mc"].copy()
oa = df[df["kind"] == "openanswer"].copy()
fs = df[df["kind"] == "flipset"].copy()

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

mc_tbl = select_cols(mc, mc_cols).sort_values(["task","eval_mode","seed"])
oa_tbl = select_cols(oa, oa_cols).sort_values(["task","eval_mode","seed"])
fs_tbl = select_cols(fs, fs_cols).sort_values(["seed","file"])

# ---- Plot 1: MC patched_0 rescue by task ----
if len(mc_tbl) > 0 and "patched_0_rescued_pct" in mc_tbl.columns:
    mc_plot = mc_tbl.groupby("task", as_index=False)["patched_0_rescued_pct"].mean()
    plt.figure()
    plt.bar(mc_plot["task"], mc_plot["patched_0_rescued_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rescue% on flips (patched_0)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "mc_patched0_rescue.pdf"), dpi=300)
    plt.close()

# ---- Plot 2: MC controls gap ----
if len(mc_tbl) > 0 and "patched_0_rescued_pct" in mc_tbl.columns:
    tmp = mc_tbl.groupby("task", as_index=False).agg({
        "patched_0_rescued_pct":"mean",
        "control_shared_randvec_rescued_pct":"mean",
        "control_patch_nonshared_rescued_pct":"mean",
        "control_rand_subspace_rescued_pct":"mean",
    })
    plt.figure()
    x = np.arange(len(tmp))
    w = 0.2
    plt.bar(x - 1.5*w, tmp["patched_0_rescued_pct"], width=w, label="patched_0")
    plt.bar(x - 0.5*w, tmp["control_shared_randvec_rescued_pct"], width=w, label="rand vec in shared")
    plt.bar(x + 0.5*w, tmp["control_rand_subspace_rescued_pct"], width=w, label="rand subspace")
    plt.bar(x + 1.5*w, tmp["control_patch_nonshared_rescued_pct"], width=w, label="nonshared patch")
    plt.xticks(x, tmp["task"], rotation=45, ha="right")
    plt.ylabel("Rescue% on flips")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "mc_controls_gap.pdf"), dpi=300)
    plt.close()

# ---- Plot 3: Alpha sweep flip_rate curves ----
if len(alpha_df) > 0 and {"alpha","flip_rate","seed"}.issubset(alpha_df.columns):
    alpha_df2 = alpha_df.copy()
    alpha_df2["alpha"] = pd.to_numeric(alpha_df2["alpha"], errors="coerce")
    alpha_df2 = alpha_df2.dropna(subset=["alpha"])
    plt.figure()
    for seed in sorted(alpha_df2["seed"].dropna().unique()):
        sub = alpha_df2[alpha_df2["seed"] == seed].sort_values("alpha")
        plt.plot(sub["alpha"], sub["flip_rate"]*100.0, marker="o", label=f"seed={int(seed)}")
    plt.xlabel("alpha")
    plt.ylabel("Flip rate on flip-set (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "alpha_sweep_fliprate.pdf"), dpi=300)
    plt.close()

# ---- Plot 4: Alpha sweep mean delta margin curves ----
if len(alpha_df) > 0 and {"alpha","mean_delta_margin_vs_baseline","seed"}.issubset(alpha_df.columns):
    alpha_df2 = alpha_df.copy()
    alpha_df2["alpha"] = pd.to_numeric(alpha_df2["alpha"], errors="coerce")
    alpha_df2 = alpha_df2.dropna(subset=["alpha"])
    plt.figure()
    for seed in sorted(alpha_df2["seed"].dropna().unique()):
        sub = alpha_df2[alpha_df2["seed"] == seed].sort_values("alpha")
        plt.plot(sub["alpha"], sub["mean_delta_margin_vs_baseline"], marker="o", label=f"seed={int(seed)}")
    plt.xlabel("alpha")
    plt.ylabel("Mean Δmargin vs baseline (on flip-set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "alpha_sweep_deltam.pdf"), dpi=300)
    plt.close()

# ---- Plot 5: Open-answer patched_self rescue ----
if len(oa_tbl) > 0 and "patched_self_rescued_pct" in oa_tbl.columns:
    oa_plot = oa_tbl.copy()
    oa_plot["label"] = oa_plot["task"].astype(str) + ":" + oa_plot["eval_mode"].astype(str)
    plt.figure()
    plt.bar(oa_plot["label"], oa_plot["patched_self_rescued_pct"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Rescue% on flips (patched_self)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "openanswer_patchedself_rescue.pdf"), dpi=300)
    plt.close()

def df_to_md_table(dfx: pd.DataFrame, max_rows: int = 30) -> str:
    if dfx is None or len(dfx) == 0:
        return "_(none)_"
    d = dfx.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_markdown(index=False)

# ---- Markdown report ----
lines = []
lines.append(f"# {MODEL_LABEL} subspace patching + flipset report\n")
lines.append(f"Generated from `{os.path.basename(SUMMARY_CSV)}` and `{os.path.basename(ALPHA_CSV)}`.\n")
lines.append("## Overview\n")
lines.append(f"- Runs: {len(df)} total JSON summaries\n")
lines.append(f"- MC runs: {len(mc)}; Open-answer runs: {len(oa)}; Flipset runs: {len(fs)}\n")
lines.append("## Key plots (PDF, dpi=300)\n")
for fn in [
    "mc_patched0_rescue.pdf",
    "mc_controls_gap.pdf",
    "alpha_sweep_fliprate.pdf",
    "alpha_sweep_deltam.pdf",
    "openanswer_patchedself_rescue.pdf",
]:
    p = os.path.join(PLOT_DIR, fn)
    if os.path.exists(p):
        lines.append(f"- `{fn}`")
lines.append("\n")

lines.append("## Multiple-choice patchback (subspace_mc)\n")
lines.append(df_to_md_table(mc_tbl))
lines.append("\n")

lines.append("## Open-answer patchback (openanswer)\n")
lines.append(df_to_md_table(oa_tbl))
lines.append("\n")

lines.append("## Flipset transfer patching (flipset)\n")
lines.append(df_to_md_table(fs_tbl))
lines.append("\n")

if len(alpha_df) > 0:
    lines.append("## Alpha sweep (flip-set)\n")
    a = alpha_df.copy()
    a["alpha"] = pd.to_numeric(a["alpha"], errors="coerce")
    a = a.dropna(subset=["alpha"])
    keep = a[a["alpha"].isin([0.0, 0.5, 0.75, 1.0])].copy()
    if len(keep) == 0:
        keep = a
    keep = keep.sort_values(["seed","alpha"])
    cols = [c for c in ["file","seed","alpha","n","flip_rate","ablated_acc","mean_delta_margin_vs_baseline"] if c in keep.columns]
    lines.append(df_to_md_table(keep[cols], max_rows=60))
    lines.append("\n")

with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

# ---- LaTeX tables ----
def to_latex_table(dfx: pd.DataFrame, caption: str, label: str) -> str:
    if dfx is None or len(dfx) == 0:
        return f"% {caption}\n% (empty)\n"
    return dfx.to_latex(index=False, escape=True, caption=caption, label=label)

tex_lines = []
tex_lines.append(f"% Auto-generated LaTeX tables for {MODEL_LABEL} results\n")
tex_lines.append("% Requires \\usepackage{booktabs}\n\n")

if len(mc_tbl) > 0:
    tex_lines.append(to_latex_table(
        mc_tbl,
        caption=f"{MODEL_LABEL}: multiple-choice (subspace patching) summary.",
        label="tab:model_mc_summary"
    ))
    tex_lines.append("\n")

if len(oa_tbl) > 0:
    tex_lines.append(to_latex_table(
        oa_tbl,
        caption=f"{MODEL_LABEL}: open-answer (openanswer subspace patching) summary.",
        label="tab:model_openanswer_summary"
    ))
    tex_lines.append("\n")

if len(alpha_df) > 0:
    tex_lines.append(to_latex_table(
        alpha_df.sort_values(["seed","alpha"]).head(40),
        caption=f"{MODEL_LABEL}: alpha sweep (first 40 rows shown).",
        label="tab:model_alpha_sweep_head"
    ))
    tex_lines.append("\n")

with open(OUT_TEX, "w", encoding="utf-8") as f:
    f.write("\n".join(tex_lines))

print(f"[OK] Wrote report: {OUT_MD}")
print(f"[OK] Wrote LaTeX tables: {OUT_TEX}")
print(f"[OK] Plots in: {PLOT_DIR}")
PY

chmod +x "${ANALYSIS_PY}"

echo "[Run] Analyze results -> report/md/tex/plots"
SUMMARY_CSV="${SUM_DIR}/summary.csv" \
ALPHA_CSV="${SUM_DIR}/alpha_sweep.csv" \
OUT_MD="${REPORT_MD}" \
OUT_TEX="${TABLES_TEX}" \
PLOT_DIR="${PLOT_DIR}" \
MODEL_LABEL="${MODEL_ID}" \
python "${ANALYSIS_PY}"

echo ""
echo "[Done] Llama-2 suite complete."
echo " - Summary CSV: ${SUM_DIR}/summary.csv"
echo " - Summary MD:  ${SUM_DIR}/summary.md"
echo " - Paper table: ${SUM_DIR}/paper_table.md"
echo " - Report:      ${REPORT_MD}"
echo " - LaTeX tables:${TABLES_TEX}"
echo " - Plots dir:   ${PLOT_DIR}"
