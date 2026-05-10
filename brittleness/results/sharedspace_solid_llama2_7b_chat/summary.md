# Shared-space experiment summary

- model: meta-llama/Llama-2-7b-chat-hf
- layer: 28
- v_decode_steps: 16
- alpha used: 45.0
- inject_start_step: 1
- inject_first_n: 24
- pirate_threshold: 2
- decodings: ['greedy', 'sample'] (sample_seeds=[1, 2, 3])

## Sharedness sweep

| basis_k | sharedness(v) | sharedness(v_fixed) |
| --- | --- | --- |
| 16 | 0.4651 | 0.0000 |
| 32 | 0.4864 | 0.0000 |
| 64 | 0.5166 | 0.0000 |
| 128 | 0.5455 | 0.0000 |

## Aggregated metrics (per-template mean±std, worst)

| decoding | method | kind | basis_k | mean_success ± std | worst_success | mean_hits ± std | worst_hits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| greedy | no_steer | baseline | - | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 |
| greedy | v_orig_a45 | v_orig | - | 0.040 ± 0.058 | 0.000 | 0.090 ± 0.136 | 0.000 |
| greedy | v_fixed_k16_a45 | v_fixed | 16 | 0.120 ± 0.060 | 0.000 | 0.400 ± 0.130 | 0.150 |
| greedy | v_fixed_k32_a45 | v_fixed | 32 | 0.050 ± 0.032 | 0.000 | 0.210 ± 0.116 | 0.050 |
| greedy | v_fixed_k64_a45 | v_fixed | 64 | 0.090 ± 0.066 | 0.000 | 0.250 ± 0.145 | 0.050 |
| greedy | v_fixed_k128_a45 | v_fixed | 128 | 0.000 ± 0.000 | 0.000 | 0.040 ± 0.037 | 0.000 |
| greedy | rand0_a45 | rand | - | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 |
| sample | no_steer | baseline | - | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 |
| sample | v_orig_a45 | v_orig | - | 0.033 ± 0.024 | 0.000 | 0.127 ± 0.063 | 0.050 |
| sample | v_fixed_k16_a45 | v_fixed | 16 | 0.100 ± 0.015 | 0.083 | 0.360 ± 0.034 | 0.317 |
| sample | v_fixed_k32_a45 | v_fixed | 32 | 0.097 ± 0.029 | 0.067 | 0.340 ± 0.064 | 0.267 |
| sample | v_fixed_k64_a45 | v_fixed | 64 | 0.127 ± 0.017 | 0.100 | 0.413 ± 0.062 | 0.317 |
| sample | v_fixed_k128_a45 | v_fixed | 128 | 0.063 ± 0.027 | 0.033 | 0.213 ± 0.097 | 0.083 |
| sample | rand0_a45 | rand | - | 0.000 ± 0.000 | 0.000 | 0.000 ± 0.000 | 0.000 |

## LaTeX examples

- examples: `examples.tex`
