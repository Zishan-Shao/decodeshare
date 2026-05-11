# DecodeShare — Added Experiments (Protocol-Focused)

This note summarizes **two experiments** that strengthen a protocol-focused story without turning the paper into a “new steering method” paper.

---

## Part A) Mechanism (Protocol-Level)

These scripts support the “mechanism” story **at the protocol level**: computational path (KV cache), geometry (subspace mismatch), and causal decode-only tests with matched controls.

### A1 — Computational path (KV cache / decode boundary)

- Script: `rebuttal/mechanism/exp_A1_computational_path_kv_cache.py`
- What it shows: a decode-only (seq_len==1) intervention is **invisible** under a naive prefill forward, but **visible** under decode-aligned boundary caching.

### A2 — Geometry (prefill vs decode subspace misalignment)

- Script: `rebuttal/mechanism/exp_A2_geometric_subspace_misalignment.py`
- What it shows: principal angles + cross-distribution explained-variance (e.g., EV(decode|prefill PCs) vs EV(decode|decode PCs)).

### A3 — Causal test (decode-only removal + matched controls)

- Script: `rebuttal/mechanism/exp_A3_causal_decode_only_controls.py`
- What it shows: decode-only shared-subspace removal vs **dimension/energy matched** controls (nonshared + random).

---

## 1) Ranking Flip (TRAD vs DECODE vs REAL)

### Purpose
Show that **the ranking of steering vectors** can **flip** depending on *how you evaluate*:
- **TRAD**: “traditional” evaluation = **prefill-only** intervention (steering applied only during the prefill forward).
- **DECODE**: your **decode-aligned protocol** = **decode-only** intervention (steering applied only during KV-cached decode steps).
- **REAL**: the “ground truth” you care about — KV-cached decode-time performance on held-out templates/seeds.

This directly supports the claim: **prefill-like / non-KV eval can mis-rank interventions**, and the decode protocol correlates better with real serving-time decoding.

### Expected outcome
- **Higher correlation** between **DECODE ranking** and **REAL ranking** than between **TRAD ranking** and **REAL ranking** (e.g., Spearman ↑).
- Identify concrete **rank flips** (vectors that look good under TRAD but fail under REAL; DECODE predicts that failure).
- Flips are most visible under **template variation** and **decode-only** settings.

### Command to run
```bash
# Recommended (multi-template): reduces template-seed noise and makes correlations meaningful.
python exp_ranking_flip_steering.py   --model meta-llama/Llama-2-7b-chat-hf   --device cuda   --vectors_manifest steering_vectors_example.jsonl   --tasks commonsenseqa,arc_challenge,openbookqa,qasc,logiqa   --n_eval 128   --template_seeds_rank 1234,2345,3456   --template_seeds_real 4567,5678,6789   --trad_mode prefill   --decode_mode decode   --staged 1   --reasoning_tokens 128   --decoding greedy   --out_json ranking_flip_results.json
```

**Output:** `ranking_flip_results.json`  
Contains per-vector scores and correlations such as:
- `spearman(trad, real)` vs `spearman(decode, real)`
- top-k vectors under each ranking
- per-task / per-template deltas (if enabled)

---

## 2) Repair vs Strong Controls (Shared / Random / PCA / Shrink)

### Purpose
Evaluate whether “repairing” steering vectors by removing the **decode-time shared subspace** is genuinely *targeted*, by comparing against **three strong controls**:

- **Shared repair**: project vector away from the estimated decode-time shared basis  
  \(v' = (I - \alpha QQ^T) v\)
- **Random subspace** control: same dimensionality, random orthonormal basis
- **PCA** control: top-PCA directions (same dim or same explained variance)
- **Shrinkage** control: simple scaling \(\gamma v\), with **norm-matching** to the repaired vector (energy fairness)

This supports the causal claim: the shared subspace is **special**, not just “any low-rank removal / energy reduction”.

### Expected outcome
- **Worst-case template performance improves** more for **Shared repair** than for Random/PCA/Shrink.
- **Template variance decreases** (std/range across templates) under Shared repair.
- Controls should **not** match the shared repair improvements unless the effect is trivial (e.g., only due to smaller norm).

### Command to run
```bash
# Recommended: provide your own `vectors_manifest` (JSONL). For a quick sanity check, you can use the included example.
python exp_repair_controls_steering.py   --model meta-llama/Llama-2-7b-chat-hf   --device cuda   --vectors_manifest steering_vectors_repair_example.jsonl   --basis_layers auto   --tasks_subspace gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa   --n_subspace 128   --tasks_eval commonsenseqa,arc_challenge,openbookqa,qasc,logiqa   --n_eval 128   --template_seeds 1234,2345,3456,4567,5678   --staged 1   --reasoning_tokens 128   --alpha_proj 1.0   --norm_match 1   --tau 0.001   --m_shared all   --pca_var 0.95   --calib_decode_max_new_tokens -1   --out_json repair_controls_results.json
```

**Output:** `repair_controls_results.json`  
Includes, for each vector and each condition (orig/shared/rand/pca/shrink):
- mean delta vs baseline
- **worst-case delta** (min over templates)
- **template variance** (std/range over templates)
- per-task, per-template breakdown for plots

---

## Input: `vectors_manifest` (JSONL)
Each line is one steering vector:
```json
{"name":"truthful_l10_seed0","concept":"truthful","layer":10,"alpha":1.0,"path":"vectors/truthful_l10_seed0.npy"}
{"name":"refusal_l15_seed123","concept":"refusal","layer":15,"alpha":0.8,"path":"vectors/refusal_l15_seed123.pt"}
```

Supported formats:
- `.npy` (shape `[D]`)
- `.pt/.pth` (tensor `[D]` or dict with key in `vector/v/direction/dir`)

---

## What to emphasize in the paper
- Ranking Flip: “**evaluation protocol changes conclusions** (rankings), decode protocol predicts real KV-cached decode.”
- Repair Controls: “shared-subspace repair improves **worst-case** and reduces **template variance**, beating strong controls → effect is targeted, not trivial.”
