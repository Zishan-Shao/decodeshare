# B2 / Rank-Flip Protocol Table

This note restores a clean rebuttal-ready table for the protocol comparison:

- `TRAD-A`: prefill-only ranking proxy
- `TRAD-B`: stronger always-on ranking proxy (prefill + decode on ranking templates)
- `DECODE`: decode-aligned ranking proxy
- `REAL`: held-out decode-only deployment target

Important scope note:

- The leftmost correlation column below comes from the separate `random100` rank-flip stress test.
- The remaining deployment columns come from the cached `B2` selection-to-deployment benchmark.
- `B2` is not just `CAA / INSTR / SAE`; it aggregates over `4` candidate pools:
  - `random100`
  - `instr64`
  - `sae64`
  - `caa_v2`

Primary sources:

- `results/rebuttal_rankflip_layer28_rand100_full/ranking_flip_layer28_rand100.json`
- `results/rebuttal_rankflip_layer28_rand100_full/ranking_flip_trad_family_tradB.json`
- `rebuttal/after_review/exp_6_downstream_utility/b2_cached/paper_table_b2.csv`
- `rebuttal/after_review/exp_6_downstream_utility/b2_cached/paper_table_b2_pairwise.csv`

## Main rebuttal table

| Protocol | Random-100 Spearman with `REAL` | B2 `REAL` mean utility | B2 `REAL` worst-case | B2 `REAL` flip rate | B2 Regret@1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `TRAD-A` | `0.065` | `-0.002 [-0.008, 0.003]` | `-0.003 [-0.011, 0.003]` | `0.750 [0.250, 1.000]` | `0.016 [0.013, 0.019]` |
| `TRAD-B` | `0.454` | `-0.002 [-0.008, 0.003]` | `-0.003 [-0.011, 0.003]` | `0.750 [0.250, 1.000]` | `0.016 [0.013, 0.019]` |
| `DECODE (ours)` | `0.700` | `+0.011 [0.002, 0.021]` | `+0.010 [-0.000, 0.021]` | `0.083 [0.000, 0.250]` | `0.003 [0.001, 0.004]` |

Suggested table note:

> `REAL` is omitted because it is the held-out deployment target rather than a ranking protocol. In the cached `B2` benchmark, `TRAD-A` and `TRAD-B` happened to select the same top-1 candidate in each pool, so their downstream `REAL` metrics coincide. This does not mean the two proxies are equivalent: in the larger `random100` stress test, their correlation with `REAL` differs substantially (`0.065` vs. `0.454`).

## Optional per-pool B2 breakdown

| Pool | Protocol | `REAL` mean utility | `REAL` worst-case | `REAL` flip rate | Regret@1 |
| --- | --- | ---: | ---: | ---: | ---: |
| `caa_v2` | `DECODE` | `0.027 [0.027, 0.027]` | `0.027 [0.027, 0.027]` | `0.000 [0.000, 0.000]` | `0.000 [0.000, 0.000]` |
| `caa_v2` | `TRAD-A/B` | `0.006 [0.006, 0.006]` | `0.006 [0.006, 0.006]` | `0.000 [0.000, 0.000]` | `0.020 [0.020, 0.020]` |
| `instr64` | `DECODE` | `0.010 [0.010, 0.010]` | `0.010 [0.010, 0.010]` | `0.000 [0.000, 0.000]` | `0.003 [0.003, 0.003]` |
| `instr64` | `TRAD-A/B` | `-0.002 [-0.002, -0.002]` | `-0.002 [-0.002, -0.002]` | `1.000 [1.000, 1.000]` | `0.016 [0.016, 0.016]` |
| `random100` | `DECODE` | `0.002 [0.002, 0.002]` | `-0.002 [-0.002, -0.002]` | `0.333 [0.333, 0.333]` | `0.004 [0.004, 0.004]` |
| `random100` | `TRAD-A/B` | `-0.008 [-0.008, -0.008]` | `-0.013 [-0.013, -0.013]` | `1.000 [1.000, 1.000]` | `0.014 [0.014, 0.014]` |
| `sae64` | `DECODE` | `0.004 [0.004, 0.004]` | `0.004 [0.004, 0.004]` | `0.000 [0.000, 0.000]` | `0.003 [0.003, 0.003]` |
| `sae64` | `TRAD-A/B` | `-0.005 [-0.005, -0.005]` | `-0.005 [-0.005, -0.005]` | `1.000 [1.000, 1.000]` | `0.012 [0.012, 0.012]` |

## Pairwise summary

| Comparison | Delta `REAL` mean utility | Delta Regret@1 | Delta `REAL` flip rate | Win rate |
| --- | ---: | ---: | ---: | ---: |
| `DECODE vs TRAD-A` | `+0.013 [0.010, 0.018]` | `-0.013 [-0.018, -0.010]` | `-0.667 [-1.000, -0.250]` | `100%` |
| `DECODE vs TRAD-B` | `+0.013 [0.010, 0.018]` | `-0.013 [-0.018, -0.010]` | `-0.667 [-1.000, -0.250]` | `100%` |

## Ready-to-paste caption

> **Table X.** Comparison of ranking proxies against the held-out deployment target `REAL`. `TRAD-A` is the original prefill-only ranking proxy, `TRAD-B` is a stronger prefill+decode proxy, and `DECODE` is our decode-aligned ranking proxy. The leftmost column reports ranking quality on the separate `random100` stress test, while the remaining columns report held-out deployment performance on the cached `B2` selection benchmark aggregated over `random100`, `instr64`, `sae64`, and `caa_v2`. `REAL` itself is omitted because it is the held-out target rather than a ranking protocol.

## Ready-to-paste LaTeX

```tex
\begin{table}[t]
\centering
\small
\begin{tabular}{lccccc}
\toprule
Protocol & Rand100 $\rho(\cdot,\mathrm{REAL})$ & B2 REAL Mean $\uparrow$ & B2 REAL Worst $\uparrow$ & B2 REAL Flip $\downarrow$ & B2 Regret@1 $\downarrow$ \\
\midrule
TRAD-A & 0.065 & -0.002 [-0.008, 0.003] & -0.003 [-0.011, 0.003] & 0.750 [0.250, 1.000] & 0.016 [0.013, 0.019] \\
TRAD-B & 0.454 & -0.002 [-0.008, 0.003] & -0.003 [-0.011, 0.003] & 0.750 [0.250, 1.000] & 0.016 [0.013, 0.019] \\
DECODE (ours) & 0.700 & +0.011 [0.002, 0.021] & +0.010 [-0.000, 0.021] & 0.083 [0.000, 0.250] & 0.003 [0.001, 0.004] \\
\bottomrule
\end{tabular}
\caption{Comparison of ranking proxies against the held-out deployment target $\mathrm{REAL}$. TRAD-A is the original prefill-only ranking proxy, TRAD-B is a stronger prefill+decode proxy, and DECODE is our decode-aligned ranking proxy. The leftmost column reports ranking quality on the separate random100 stress test, while the remaining columns report held-out deployment performance on the cached B2 selection benchmark aggregated over random100, instr64, sae64, and caa\_v2. REAL itself is omitted because it is the held-out target rather than a ranking protocol.}
\label{tab:rankflip-b2-protocol}
\end{table}
```
