# DecodeShare Important Results Summary

This note consolidates the most important experimental results currently available in the workspace, with an emphasis on:

- core paper evidence (`H1/H2/H3`)
- deployment-facing utility (`rank-flip`)
- direct steering robustness (`repair`)
- after-review additions (scale, probing, `tau`)

## Executive Takeaways

- The strongest deployment result is `rank-flip`: decode-aligned ranking tracks real KV-cached decode utility far better than TRAD / prefill-style ranking.
- The strongest causal / mechanistic result is `patchback`: patching shared decode directions rescues almost all flips, while nonshared patch controls stay near zero.
- `H3` explains why protocol matters: decode and prefill shared bases are far apart, and only decode-estimated decode-time intervention consistently damages decode-time behavior.
- `H1` supports a compact shared decode workspace across several model families, although the strictest `full` criterion is not passed by every model.
- Direct steering repair supports a qualified claim: shared repair reliably reduces template dispersion, but mean and worst-case utility remain task- and vector-dependent.
- After-review additions strengthen the package further: there is now direct `13B` H1 evidence, an unembedding probe of the shared basis, and an empirical robustness justification for `tau=1e-3`.

## Table 1. Most Important Results At A Glance

| Family | Representative result | Headline numbers | Main take-away | Main caveat |
|---|---|---|---|---|
| `H1` existence | multi-model full benchmark | `Qwen-7B full PASS`, `Falcon-7B full PASS`, `Llama-3.1-8B full PASS`, `Llama-2-13B lite full PASS@L10` | shared decode workspace is real and not model-specific | some models only pass in pooled / loosened settings |
| `H2` disturbance | LOTO(8) decode-time shared ablation | large held-out drops on `arc_challenge`, `openbookqa`, `qasc`, `logiqa` vs random | removing shared decode directions causally hurts reasoning / MC behavior | effect size varies by task |
| `H2` patchback | aggregated flip-set rescue | `Patched@full = 100%` on many model/layer settings, `NonsharedPatch ~= 0%` | shared decode directions are high-leverage causal pathways | random-subspace controls can be nontrivial on Qwen |
| `H3` protocol mismatch | prefill vs decode basis mismatch | mean principal angle `78.6°`; decode-on-decode energy `0.315` vs prefill-on-decode `0.050` | prefill is not a faithful proxy for decode-time behavior | mismatch magnitude varies across depth |
| `rank-flip` | random-100 layer-28 stress test | `rho(TRAD,REAL)=0.065`, `rho(DECODE,REAL)=0.700` | evaluation protocol changes which vector looks best | strongest evidence is currently on layer 28 |
| `repair` | shared repair vs controls | lower template std, but mixed mean / worst-case gains | best claim is “variance / robustness knob” | not a universal utility gain |
| after-review | scale + probing + `tau` | direct `13B` H1, unembedding probe, `tau` sensitivity sweep | package is materially stronger than the original rebuttal draft | probing is still qualitative; no linear-classifier probe yet |

## 1. `H1`: Decode-Time Shared Workspace Exists

Source:

- `Hype1/results/full_benchmark/H1_full_benchmark_summary.md`
- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h1_l10_lite.json`
- `rebuttal/after_review/exp_4_scale_extension/llama3_8b_h1_lite.json`

Criterion:

- `supports_H1 = (p_null1_perm < 0.05) AND (p_null2_scramble < 0.05) AND (shared_count > 0)`

### Table 2. Selected `H1` Results

| Model | Variant | Layer | tau | m_shared | cross_dim | \|S\| | \|S\|/cross_dim | p_perm | p_scramble | H1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `meta-llama/Llama-3.1-8B-Instruct` | full | 10 | 0.001 | all | 2812 | 110 | 0.039 | 0.0005 | 0.0099 | PASS |
| `meta-llama/Llama-3.1-8B-Instruct` | pooled | 10 | 0.001 | 8 | 2812 | 143 | 0.051 | 0.0005 | 0.0099 | PASS |
| `Qwen/Qwen2.5-7B-Instruct` | full | 10 | 0.001 | 13 | 2549 | 56 | 0.022 | 0.0005 | 0.0099 | PASS |
| `Qwen/Qwen2.5-7B-Instruct` | pooled | 10 | 0.001 | 8 | 2549 | 85 | 0.033 | 0.0005 | 0.0099 | PASS |
| `tiiuae/falcon-7b-instruct` | full | 10 | 0.001 | all | 2479 | 122 | 0.049 | 0.0005 | 0.0099 | PASS |
| `tiiuae/falcon-7b-instruct` | pooled | 10 | 0.001 | 8 | 2479 | 154 | 0.062 | 0.0005 | 0.0099 | PASS |
| `mistralai/Mistral-7B-Instruct-v0.3` | pooled | 10 | 0.001 | 8 | 2808 | 134 | 0.048 | 0.0005 | 0.0099 | PASS |
| `meta-llama/Llama-2-7b-chat-hf` | pooled | 10 | 0.001 | 8 | 2423 | 35 | 0.014 | 0.0005 | 0.0099 | PASS |
| `google/gemma-3-12b-it` | loosened | 10 | 0.0003 | 8 | 2581 | 1498 | 0.580 | 0.0005 | 0.0099 | PASS |
| `meta-llama/Llama-2-13b-chat-hf` | full-lite | 10 | 0.001 | all | 2772 | 77 | 0.028 | 0.0154 | 0.0476 | PASS |
| `meta-llama/Llama-3.1-8B-Instruct` | full-lite | 10 | 0.001 | all | 2515 | 103 | 0.041 | 0.0154 | 0.0476 | PASS |

Analysis:

- The clean positive cases under the main threshold tend to have a compact shared set: roughly `2%` to `6%` of the pooled PCA subspace.
- The strict `full` criterion is genuinely demanding; some models only pass under `pooled`, which is useful to mention as a feature of the test rather than as a weakness.
- There is now direct `13B` evidence in the workspace: `Llama-2-13B-chat-hf` passes a lite `full` H1 check at `layer=10`.
- Across the benchmark summary, `12/21` model-variant rows support `H1`, including `4` strict `full` passes and `5` `pooled` passes.

## 2. `H2`: Decode-Time Shared Directions Are Causally Important

Sources:

- `results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc.md`
- `patch_back/paper/patchback_tables_all_models_all_layers.tex`

### Table 3A. LOTO(8) Held-Out Performance Under Shared Decode Ablation

| Held-out | n | Baseline | Shared(full) | Rand(full) | Δ(shared-baseline) | p(shared-baseline) |
|---|---:|---|---|---|---|---:|
| `gsm8k` | 256 | `4.7 [2.3, 7.4]` | `3.5 [1.6, 5.9]` | `5.1 [2.7, 7.8]` | `-1.2 [-4.3, +2.0]` | 0.6310 |
| `commonsenseqa` | 256 | `51.6 [45.7, 57.8]` | `47.7 [41.4, 53.9]` | `50.8 [44.5, 57.0]` | `-3.9 [-9.8, +1.6]` | 0.2190 |
| `strategyqa` | 256 | `57.0 [51.2, 62.9]` | `53.5 [47.7, 59.4]` | `57.8 [52.0, 63.7]` | `-3.5 [-7.4, +0.0]` | 0.0947 |
| `aqua` | 254 | `24.0 [18.9, 29.5]` | `17.3 [12.6, 22.0]` | `22.0 [16.9, 27.2]` | `-6.7 [-13.4, -0.4]` | 0.0547 |
| `arc_challenge` | 256 | `51.6 [45.3, 57.4]` | `42.6 [36.7, 48.4]` | `51.6 [45.7, 57.8]` | `-9.0 [-14.8, -2.7]` | 0.0099 |
| `openbookqa` | 256 | `52.7 [46.5, 59.0]` | `41.4 [35.2, 47.3]` | `54.7 [48.4, 60.5]` | `-11.3 [-18.4, -4.3]` | 0.0027 |
| `qasc` | 256 | `50.4 [44.5, 56.6]` | `41.8 [35.5, 47.7]` | `51.2 [44.9, 57.4]` | `-8.6 [-14.5, -2.7]` | 0.0060 |
| `logiqa` | 256 | `35.5 [29.7, 41.8]` | `24.2 [19.1, 29.7]` | `37.1 [31.2, 43.0]` | `-11.3 [-19.1, -3.5]` | 0.0062 |

### Table 3B. Patchback On Flip Sets (Multiple-Choice Aggregate)

| Model | Layer | `|F|` | Patched@full | RandVec(shared) | RandSubspace | NonsharedPatch |
|---|---:|---:|---|---|---|---|
| `Llama-2-7b-chat` | 4 | 202 | `100.0% (202/202)` | `19.3%` | `16.3%` | `0.0%` |
| `Llama-2-7b-chat` | 10 | 449 | `100.0% (449/449)` | `7.6%` | `5.3%` | `0.0%` |
| `Llama-2-7b-chat` | 24 | 297 | `100.0% (297/297)` | `8.8%` | `6.1%` | `0.0%` |
| `Falcon-7b-instruct` | 4 | 91 | `100.0% (91/91)` | `9.9%` | `0.0%` | `0.0%` |
| `Falcon-7b-instruct` | 10 | 97 | `100.0% (97/97)` | `0.0%` | `1.0%` | `0.0%` |
| `Falcon-7b-instruct` | 24 | 192 | `100.0% (192/192)` | `19.3%` | `19.8%` | `0.0%` |
| `Qwen2.5-7B-Instruct` | 4 | 325 | `100.0% (325/325)` | `32.6%` | `32.9%` | `0.0%` |
| `Qwen2.5-7B-Instruct` | 10 | 548 | `100.0% (548/548)` | `36.1%` | `49.8%` | `0.0%` |
| `Qwen2.5-7B-Instruct` | 24 | 453 | `100.0% (453/453)` | `37.5%` | `39.3%` | `0.0%` |

### Table 3C. Selected Transfer-Patching Results

| Model | Layer | Flip set | Patched(self) | Patched(transfer) |
|---|---:|---:|---|---|
| `Llama-2-7b-chat` | 10 | 169 | `75.1%` | `78.7%` |
| `Falcon-7b-instruct` | 24 | 88 | `100.0%` | `92.0%` |
| `Qwen2.5-7B-Instruct` | 10 | 128 | `100.0%` | `96.1%` |
| `Qwen2.5-7B-Instruct` | 24 | 122 | `100.0%` | `24.6%` |

Analysis:

- LOTO held-out ablation shows that removing the shared decode basis hurts downstream performance substantially on several held-out MC tasks, while the random full-basis control is much weaker.
- Patchback is the cleanest causal result in the entire project: patching shared decode directions rescues essentially all flip-set failures, and patching nonshared directions stays at or near zero.
- Transfer patching is strong but depth-sensitive: it stays high for Falcon and Qwen layer 10, but can collapse at deeper layers such as `Qwen layer 24`.

## 3. `H3`: Prefill and Decode Are Mechanistically Mismatched

Sources:

- `results/h3_grid/h3_grid_reasoning.json`
- `rebuttal/pca_prefill_decode_mismatch_layer28.json`
- `rebuttal/after_review/exp_2_tau_layer_robustness/exp_2_tau_layer_robustness_summary.md`

### Table 4A. Basis-Level Mismatch (`Llama-2-7b-chat-hf`, layer 10)

| Quantity | Value |
|---|---:|
| `k_decode` | 132 |
| `k_prefill` | 66 |
| matched-basis mean cosine | 0.195 |
| mean principal angle | `78.56°` |
| decode-on-decode energy | 0.315 |
| prefill-on-decode energy | 0.050 |
| random-on-decode energy | 0.016 |

### Table 4B. Task-Level Forced-Choice Counterfactuals (`decode-est@decode` vs `prefill-est@decode`)

| Task | n | Baseline | Decode-est@Decode | Prefill-est@Decode | Random@Decode | Δ(decode-prefill) |
|---|---:|---:|---:|---:|---:|---:|
| `commonsenseqa` | 1221 | 52.4 | 26.5 | 53.2 | 53.2 | -26.7 |
| `strategyqa` | 687 | 54.3 | 50.8 | 52.5 | 53.9 | -1.7 |
| `piqa` | 1838 | 66.1 | 52.5 | 65.9 | 66.7 | -13.4 |
| `arc_challenge` | 1172 | 49.6 | 27.1 | 49.7 | 49.5 | -22.6 |
| `openbookqa` | 500 | 50.6 | 27.8 | 49.0 | 51.2 | -21.2 |
| `qasc` | 926 | 48.9 | 15.1 | 49.7 | 49.2 | -34.6 |
| `logiqa` | 651 | 33.3 | 23.8 | 31.8 | 33.6 | -8.0 |
| `boolq` | 2048 | 59.8 | 52.8 | 60.8 | 60.4 | -8.1 |

### Table 4C. Layer-28 PCA Mismatch Summary

| k | mean principal angle | decode var explained by decode PCs | decode var explained by prefill PCs |
|---:|---:|---:|---:|
| 32 | `80.1°` | 22.4% | 4.0% |
| 64 | `79.1°` | 28.8% | 5.7% |
| 128 | `77.2°` | 36.3% | 8.0% |
| 256 | `74.1°` | 44.8% | 12.0% |

Analysis:

- At the basis level, decode and prefill shared subspaces are far apart; the matched-basis mean cosine is only `0.195`, and the mean principal angle is `78.6°`.
- At the behavioral level, decode-estimated decode intervention is damaging on every forced-choice task, with an average unweighted drop of about `17.0` points relative to prefill-estimated decode intervention.
- This is the central reason `prefill` is not a faithful serving-time proxy.

## 4. Rank-Flip: The Strongest Deployment-Side Result

Sources:

- `results/rebuttal_rankflip_layer28_rand100_full/ranking_flip_layer28_rand100.json`
- `results/rebuttal_rankflip_layer28_rand100_full/ranking_flip_trad_family_tradB.json`
- `rebuttal/orthogonal_steer/results/A1_cross_method/summary.md`
- `rebuttal/orthogonal_steer/results/A1_caa_v2_label_text_l28_n32/rankflip/rankflip_caa.json`

### Table 5. Rank-Flip Results

| Setting | #Vec | Spearman(TRAD,REAL) | Spearman(DECODE,REAL) | Spearman(TRAD,DECODE) | regret@1 TRAD | regret@1 DECODE | top10 overlap TRAD∩REAL | top10 overlap DECODE∩REAL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random-100 stress test | 100 | 0.065 | 0.700 | 0.036 | 0.0141 | 0.0042 | - | - |
| TRAD-B stronger baseline | 100 | 0.454 (`TRAD-both`) | 0.700 | 0.480 (`TRAD-both` vs DECODE) | 0.0047 | 0.0042 | - | - |
| CAA pool | 8 | 0.455 | 0.563 | 0.163 | 0.0047 | 0.0000 | 8 | 8 |
| INSTR pool | 64 | 0.172 | 0.767 | 0.072 | 0.0156 | 0.0031 | 2 | 2 |
| SAE pool | 64 | -0.064 | 0.594 | -0.134 | 0.0125 | 0.0031 | 1 | 6 |
| CAA-v2 stronger construction | 32 | -0.370 | 0.700 | -0.239 | 0.0203 | 0.0000 | 1 | 7 |

Analysis:

- This is the strongest practical result in the repo: the ranking of vectors changes materially depending on whether evaluation is TRAD / prefill-style or decode-aligned.
- On the hardest random-100 stress test, `DECODE` tracks `REAL` much better than `TRAD` (`0.700` vs `0.065`), which is a large effect.
- The effect generalizes beyond random vectors to `CAA`, `INSTR`, and `SAE` pools.
- Even a stronger always-on TRAD baseline (`TRAD-both`) still underperforms decode-aligned evaluation.

## 5. Direct Repair / Steering Robustness

Sources:

- `results/rebuttal_20260208_212945/repair_controls.json`
- `results/steer_robust_partial_latest/partial_summary.md`

### Table 6A. Original Layer-28 Repair Controls (`n=5` vectors)

| Method | mean(mean_delta) | mean(worst_delta) | mean(std_delta) |
|---|---:|---:|---:|
| `orig` | -0.00525 | -0.01719 | 0.00992 |
| `shared` | -0.00394 | -0.01250 | 0.00632 |
| `rand` | -0.00500 | -0.01250 | 0.00677 |
| `pca` | -0.00450 | -0.01281 | 0.00622 |
| `pca_prefill` | -0.00825 | -0.01719 | 0.00854 |
| `shrink` | -0.00413 | -0.01344 | 0.00776 |

Wins:

- `shared_vs_rand_worst_delta_winrate = 0.6`
- `shared_vs_pca_worst_delta_winrate = 0.2`
- `shared_vs_pca_prefill_worst_delta_winrate = 1.0`
- `shared_vs_shrink_worst_delta_winrate = 0.8`

### Table 6B. Partial Live Robustness Summary (`n=18` vectors)

| Group | n | orig mean(mean) | shared mean(mean) | orig mean(worst) | shared mean(worst) | orig mean(std) | shared mean(std) | shared wins mean / worst / std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | 18 | +0.0100 | +0.0039 | -0.0034 | -0.0062 | +0.0097 | +0.0070 | `1/18, 3/18, 15/18` |
| `caa_v2` | 6 | +0.0108 | +0.0035 | -0.0057 | -0.0070 | +0.0117 | +0.0075 | `0/6, 2/6, 6/6` |
| `instr64` | 6 | +0.0125 | +0.0053 | -0.0010 | -0.0045 | +0.0093 | +0.0058 | `0/6, 0/6, 6/6` |
| `sae64` | 6 | +0.0067 | +0.0030 | -0.0037 | -0.0072 | +0.0082 | +0.0077 | `1/6, 1/6, 3/6` |

Analysis:

- The cleanest direct-repair claim is about dispersion, not raw utility: shared repair usually lowers template variance / range.
- In the larger partial live run, `shared` wins on template std for `15/18` vectors overall, but only wins on mean utility for `1/18` and on worst-case utility for `3/18`.
- This is why the safest phrasing is “robustness knob with task-dependent trade-offs”, not “better steering method”.

## 6. After-Review Additions

Sources:

- `rebuttal/after_review/exp_1_unembedding_probe/exp_1_comparison.md`
- `rebuttal/after_review/exp_2_tau_layer_robustness/exp_2_tau_layer_robustness_summary.md`
- `rebuttal/after_review/exp_4_scale_extension/llama2_13b_h1_l10_lite.json`
- `rebuttal/after_review/exp_4_scale_extension/llama3_8b_h1_lite.json`

### Table 7. After-Review Additions

| Addition | Key numbers | Why it matters | Caveat |
|---|---|---|---|
| direct `13B` H1 check | `Llama-2-13B-chat-hf`, `layer=10`, `cross_dim=2772`, `shared_count=77`, `p_perm=0.0154`, `p_scramble=0.0476` | direct evidence that the phenomenon extends to `13B` | current run is a lite check, not the full benchmark setting |
| extra `8B` confirmation | `Llama-3.1-8B-Instruct`, lite `full` run: `cross_dim=2515`, `shared_count=103`, `p_perm=0.0154`, `p_scramble=0.0476` | strengthens `>7B` scale-extension story | lite run uses smaller prompt / null budgets than the full benchmark |
| unembedding probe (`layer=28`) | `option_letters: 0.260 vs 0.244 / 0.208`, `answer_scaffold: 0.283 vs 0.213 / 0.208`, `newline: 0.374 vs 0.204 / 0.190`, `digits: 0.314 vs 0.261 / 0.159` | qualitative evidence that decode-shared directions align with answer / decision scaffold tokens | no linear-classifier probe yet |
| `tau` robustness | at `pca_var=0.95`, shared ratio `0.0903 -> 0.0365 -> 0.0164` as `tau=5e-4 -> 1e-3 -> 2e-3` | shows `tau=1e-3` is in a stable compact regime | not a principled automatic selector |
| convergence | overlap mean rises `0.921 -> 1.000` (`Llama-2-7B`) and `0.914 -> 1.000` (`Qwen-2.5-7B`) as tasks increase `2 -> 8` | basis estimate stabilizes with more tasks | still empirical rather than theoretical |

## Overall Interpretation

- If the goal is to defend the mechanistic claim, lead with `H2 patchback` and `H3 mismatch`.
- If the goal is to defend practical deployment relevance, lead with `rank-flip`.
- If the goal is to defend the abstract steering sentence, use the `repair` results conservatively: they support “reduced template sensitivity with trade-offs”, not universal gains.
- If the goal is to address reviewer requests after the fact, the package is now materially stronger because the workspace contains direct `13B` evidence, unembedding-based characterization, and a clean `tau` robustness story.
