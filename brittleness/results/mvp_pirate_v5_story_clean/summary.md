# MVP v5: Pirate steering debug + smoke test

- model: meta-llama/Llama-2-7b-chat-hf
- layer: 28
- v_mode: decode
- v_decode_steps: 16
- pirate_threshold: 2
- inject_first_n: 24
- temperature/top_p: 0.9/0.9
- basis_k: 128
- sharedness(v): 0.546672
- sharedness(v_fixed): 0.000000

| Method | Greedy mean ± std | Greedy worst | Sample mean ± std | Sample worst |
| --- | --- | --- | --- | --- |
| no_steer |  |  | 0.000 ± 0.000 | 0.000 |
| v_orig_a50 |  |  | 0.125 ± 0.125 | 0.000 |
| v_fixed_a50 |  |  | 0.125 ± 0.125 | 0.000 |
| rand0_a50 |  |  | 0.000 ± 0.000 | 0.000 |
