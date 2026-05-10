# Prefill-Decode Mismatch Per-Task Breakdown (`H3`, `layer=27`)

Source status:

- Existing artifact: [llama32_3b_h3_l27_run.log](rebuttal/after_review/exp_4_scale_extension/llama32_3b_h3_l27_run.log)
- This run is **not complete**. It only contains finished summaries for `commonsenseqa` and `strategyqa`; later tasks reached the header/progress stage but do not have final summary lines in the saved log.
- Therefore, the table below is a **partial recovery from the existing run**, not a complete 8-task result table.

Run-level summary from the same log:

- model: `meta-llama/Llama-3.2-3B-Instruct`
- layer: `27`
- `k_match=57`
- mean basis angle: `72.09°`
- setup fragment visible in log: `calib_decode_max_new_tokens=256`, `per_task_max_states=12000`, `warmup_tokens=0`

## Partial Table: recovered rows from the existing `layer27` log

| Task | Baseline | `Δ(decode-est@decode)` | `Δ(prefill-est@decode)` | `Δ(rand@decode)` | `p` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `commonsenseqa` | `69.5` | `-33.4 pts` | `-3.7 pts` | `-0.4 pts` | `—` |
| `strategyqa` | `50.6` | `+5.8 pts` | `-1.4 pts` | `+0.4 pts` | `—` |

Notes:

- The `p` column is left as `—` on purpose. The saved `layer27` artifact is only a log, not a structured JSON with per-example correctness arrays, so there is no honest way to reconstruct paired significance from the existing file alone.
- The stronger `commonsenseqa` hit is real in the current log. `strategyqa` currently goes in the opposite direction, which is another reason not to overstate `layer27` before rerunning the full table.

## Ready-to-fill full table template for `layer27`

| Task | `Δ(decode-est@decode)` | `Δ(prefill-est@decode)` | `Δ(rand@decode)` | `p` |
| --- | ---: | ---: | ---: | ---: |
| `commonsenseqa` | `-33.4 pts` | `-3.7 pts` | `-0.4 pts` | `—` |
| `strategyqa` | `+5.8 pts` | `-1.4 pts` | `+0.4 pts` | `—` |
| `piqa` | `TBD` | `TBD` | `TBD` | `TBD` |
| `arc_challenge` | `TBD` | `TBD` | `TBD` | `TBD` |
| `openbookqa` | `TBD` | `TBD` | `TBD` | `TBD` |
| `qasc` | `TBD` | `TBD` | `TBD` | `TBD` |
| `logiqa` | `TBD` | `TBD` | `TBD` | `TBD` |
| `boolq` | `TBD` | `TBD` | `TBD` | `TBD` |

## Exact rerun command to produce a complete structured `layer27` artifact

Run from [reasoning](reasoning):

```bash
cd reasoning

CUDA_VISIBLE_DEVICES=0 python h3_killer_counterfactual_grid_reasoning_v2.py \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --device cuda \
  --model_dtype fp16 \
  --tasks gsm8k,commonsenseqa,strategyqa,piqa,arc_challenge,openbookqa,qasc,logiqa,boolq \
  --layer 27 \
  --n_subspace 128 \
  --n_eval 512 \
  --calib_decode_max_new_tokens 256 \
  --per_task_max_states 12000 \
  --max_prompt_len 1024 \
  --batch_size 8 \
  --tau 0.001 \
  --m_shared all \
  --answer_prefix $'\nFinal answer:' \
  --warmup_tokens 0 \
  --template_randomization 1 \
  --shuffle_choices 1 \
  --seed 0
```

Expected output file after a complete rerun:

```text
reasoning/h3_grid_v3_meta-llama_Llama-3.2-3B-Instruct_layer27_k<matched_k>_W0_seed0.json
```

Once that JSON exists, the remaining 6 rows can be filled exactly, and paired `p` values can also be computed offline if you decide which comparison the rebuttal should report.
