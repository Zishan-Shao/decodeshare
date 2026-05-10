# H1 summary: decode-time shared workspace exists (full_benchmark)
- results_dir: `Hype1/results/full_benchmark`
- alpha: 0.05

H1 support criterion used here: `supports_H1 = (p_null1_perm < alpha) AND (p_null2_scramble < alpha) AND (shared_count > 0)`.

| Model | Variant | Layer | tau | m_shared | cross_dim | |S| | |S|/cross_dim | p1 (perm) | p2 (scramble) | H1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| meta-llama/Llama-3.1-8B-Instruct | full | 10 | 0.001 | all | 2812 | 110 | 0.039 | 0.000500 | 0.009901 | PASS |
| meta-llama/Llama-3.1-8B-Instruct | pooled | 10 | 0.001 | 8 | 2812 | 143 | 0.051 | 0.000500 | 0.009901 | PASS |
| meta-llama/Llama-3.1-8B-Instruct | loosened | 10 | 0.0003 | 8 | 2812 | 670 | 0.238 | 0.000500 | 1.000000 | FAIL |
| mistralai/Mistral-7B-Instruct-v0.3 | full | 10 | None | None | None | None | N/A | N/A | N/A | FAIL |
| mistralai/Mistral-7B-Instruct-v0.3 | pooled | 10 | 0.001 | 8 | 2808 | 134 | 0.048 | 0.000500 | 0.009901 | PASS |
| mistralai/Mistral-7B-Instruct-v0.3 | loosened | 10 | 0.0003 | 8 | 2808 | 650 | 0.231 | 0.000500 | 1.000000 | FAIL |
| Qwen/Qwen2.5-1.5B-Instruct | full | 10 | 0.001 | all | 1131 | 144 | 0.127 | 0.000500 | 0.009901 | PASS |
| Qwen/Qwen2.5-1.5B-Instruct | pooled | 10 | 0.001 | 8 | 1131 | 193 | 0.171 | 0.000500 | 1.000000 | FAIL |
| Qwen/Qwen2.5-1.5B-Instruct | loosened | 10 | 0.0003 | 8 | 1131 | 1131 | 1.000 | 1.000000 | 1.000000 | FAIL |
| Qwen/Qwen2.5-7B-Instruct | full | 10 | 0.001 | 13 | 2549 | 56 | 0.022 | 0.000500 | 0.009901 | PASS |
| Qwen/Qwen2.5-7B-Instruct | pooled | 10 | 0.001 | 8 | 2549 | 85 | 0.033 | 0.000500 | 0.009901 | PASS |
| Qwen/Qwen2.5-7B-Instruct | loosened | 10 | 0.0003 | 8 | 2549 | 1163 | 0.456 | 0.000500 | 0.009901 | PASS |
| tiiuae/falcon-7b-instruct | full | 10 | 0.001 | all | 2479 | 122 | 0.049 | 0.000500 | 0.009901 | PASS |
| tiiuae/falcon-7b-instruct | pooled | 10 | 0.001 | 8 | 2479 | 154 | 0.062 | 0.000500 | 0.009901 | PASS |
| tiiuae/falcon-7b-instruct | loosened | 10 | 0.0003 | 8 | 2479 | 530 | 0.214 | 0.000500 | 1.000000 | FAIL |
| google/gemma-3-12b-it | full | 10 | 0.001 | 13 | 2581 | 0 | 0.000 | 1.000000 | 1.000000 | FAIL |
| google/gemma-3-12b-it | pooled | 10 | 0.001 | 8 | 2581 | 0 | 0.000 | 1.000000 | 1.000000 | FAIL |
| google/gemma-3-12b-it | loosened | 10 | 0.0003 | 8 | 2581 | 1498 | 0.580 | 0.000500 | 0.009901 | PASS |
| meta-llama/Llama-2-7b-chat-hf | full | 10 | 0.001 | 13 | 2423 | 0 | 0.000 | 1.000000 | 1.000000 | FAIL |
| meta-llama/Llama-2-7b-chat-hf | pooled | 10 | 0.001 | 8 | 2423 | 35 | 0.014 | 0.000500 | 0.009901 | PASS |
| meta-llama/Llama-2-7b-chat-hf | loosened | 10 | 0.0003 | 8 | 2423 | 1389 | 0.573 | 0.000500 | 0.009901 | PASS |

Notes:
- p2_min = 1/(N_scramble+1) is the minimum attainable p-value for Null-2 at finite trials; when p2==p2_min, it indicates zero exceedances in the scramble null.
