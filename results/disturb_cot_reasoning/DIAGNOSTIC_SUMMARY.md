# Disturb CoT Diagnostic Summary

Generated from **2** JSON file(s). Decoding summarized: **greedy**

# Model: Llama-2-7b-chat-hf

## Run: `energy_balance_loto8_reasoning_fc.json`

- Source file: `energy_balance_loto8_reasoning_fc.json`
- Tasks: gsm8k, commonsenseqa, strategyqa, aqua, arc_challenge, openbookqa, qasc, logiqa

## Model: Llama-2-7b-chat-hf
- **Model (raw)**: `meta-llama/Llama-2-7b-chat-hf`
- **Signature**: mode=loto | loto_eval=heldout | layer=10 | tau=0.001 | m=all | tr=1 | sc=1 | rand=joint_nonshared_varmatch | dtype=fp32
- **Decoding**: `greedy`

### Overview (means across tasks)
| Condition     | Avg Acc(%) | Avg Extr(%) | Avg EOS(%) | Avg Len |
|---------------|------------|-------------|------------|---------|
| baseline      | 4.7        | 100.0       | 100.0      | 5.2     |
| shared_full   | 3.5        | 98.8        | 93.4       | 24.4    |
| shared_staged | 3.5        | 98.8        | 94.1       | 23.4    |
| rand_full     | 5.1        | 100.0       | 100.0      | 5.2     |
| rand_staged   | 5.1        | 100.0       | 100.0      | 5.2     |

### Diagnostic flags (heuristic counts)
- Heuristics: baseline_extr≥0.80 & extr_drop≥0.20 → `PARSE↓`; extr≤0.50 → `PARSE low`; |Δeos|≥0.20 → `EOS shift`; len_ratio≤0.50 → `LEN↓`

| Condition     | Total flagged tasks | PARSE↓ | PARSE low | EOS shift | LEN↓ |
|---------------|---------------------|--------|-----------|-----------|------|
| shared_full   | 0                   | 0      | 0         | 0         | 0    |
| shared_staged | 0                   | 0      | 0         | 0         | 0    |
| rand_full     | 0                   | 0      | 0         | 0         | 0    |
| rand_staged   | 0                   | 0      | 0         | 0         | 0    |

### Extraction rate per task (%, with CI if available; Δ vs baseline)
| Task          | n   | Baseline | Shared(full)  | Shared(staged) | Rand(full)     | Rand(staged)   | Flags               |
|---------------|-----|----------|---------------|----------------|----------------|----------------|---------------------|
| aqua          | 254 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| arc_challenge | 256 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| commonsenseqa | 256 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| gsm8k         | 256 | 100.0    | 98.8 (-1.2pp) | 98.8 (-1.2pp)  | 100.0 (+0.0pp) | 100.0 (+0.0pp) | SF:- SS:- RF:- RS:- |
| logiqa        | 256 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| openbookqa    | 256 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| qasc          | 256 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| strategyqa    | 256 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |

### EOS rate per task (%, with CI if available; Δ vs baseline)
| Task          | n   | Baseline | Shared(full)  | Shared(staged) | Rand(full)     | Rand(staged)   |
|---------------|-----|----------|---------------|----------------|----------------|----------------|
| aqua          | 254 | N/A      | N/A           | N/A            | N/A            | N/A            |
| arc_challenge | 256 | N/A      | N/A           | N/A            | N/A            | N/A            |
| commonsenseqa | 256 | N/A      | N/A           | N/A            | N/A            | N/A            |
| gsm8k         | 256 | 100.0    | 93.4 (-6.6pp) | 94.1 (-5.9pp)  | 100.0 (+0.0pp) | 100.0 (+0.0pp) |
| logiqa        | 256 | N/A      | N/A           | N/A            | N/A            | N/A            |
| openbookqa    | 256 | N/A      | N/A           | N/A            | N/A            | N/A            |
| qasc          | 256 | N/A      | N/A           | N/A            | N/A            | N/A            |
| strategyqa    | 256 | N/A      | N/A           | N/A            | N/A            | N/A            |

### Avg new tokens per task (with CI if available; Δ and ratio vs baseline)
| Task          | n   | Baseline | Shared(full)         | Shared(staged)       | Rand(full)         | Rand(staged)       |
|---------------|-----|----------|----------------------|----------------------|--------------------|--------------------|
| aqua          | 254 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| arc_challenge | 256 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| commonsenseqa | 256 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| gsm8k         | 256 | 5.2      | 24.4 (+19.14, 4.67x) | 23.4 (+18.22, 4.49x) | 5.2 (+0.03, 1.01x) | 5.2 (+0.03, 1.01x) |
| logiqa        | 256 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| openbookqa    | 256 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| qasc          | 256 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| strategyqa    | 256 | N/A      | N/A                  | N/A                  | N/A                | N/A                |


---

## Run: `energy_balance_loto8_reasoning_fc_eval2048.json`

- Source file: `energy_balance_loto8_reasoning_fc_eval2048.json`
- Tasks: gsm8k, commonsenseqa, strategyqa, aqua, arc_challenge, openbookqa, qasc, logiqa

## Model: Llama-2-7b-chat-hf
- **Model (raw)**: `meta-llama/Llama-2-7b-chat-hf`
- **Signature**: mode=loto | loto_eval=heldout | layer=10 | tau=0.001 | m=all | tr=1 | sc=1 | rand=joint_nonshared_varmatch | dtype=fp32
- **Decoding**: `greedy`

### Overview (means across tasks)
| Condition     | Avg Acc(%) | Avg Extr(%) | Avg EOS(%) | Avg Len |
|---------------|------------|-------------|------------|---------|
| baseline      | 4.9        | 100.0       | 99.9       | 6.2     |
| shared_full   | 2.3        | 99.2        | 92.3       | 27.0    |
| shared_staged | 2.3        | 99.2        | 93.5       | 25.9    |
| rand_full     | 4.7        | 100.0       | 99.7       | 8.3     |
| rand_staged   | 4.7        | 100.0       | 99.8       | 8.2     |

### Diagnostic flags (heuristic counts)
- Heuristics: baseline_extr≥0.80 & extr_drop≥0.20 → `PARSE↓`; extr≤0.50 → `PARSE low`; |Δeos|≥0.20 → `EOS shift`; len_ratio≤0.50 → `LEN↓`

| Condition     | Total flagged tasks | PARSE↓ | PARSE low | EOS shift | LEN↓ |
|---------------|---------------------|--------|-----------|-----------|------|
| shared_full   | 0                   | 0      | 0         | 0         | 0    |
| shared_staged | 0                   | 0      | 0         | 0         | 0    |
| rand_full     | 0                   | 0      | 0         | 0         | 0    |
| rand_staged   | 0                   | 0      | 0         | 0         | 0    |

### Extraction rate per task (%, with CI if available; Δ vs baseline)
| Task          | n    | Baseline | Shared(full)  | Shared(staged) | Rand(full)     | Rand(staged)   | Flags               |
|---------------|------|----------|---------------|----------------|----------------|----------------|---------------------|
| aqua          | 254  | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| arc_challenge | 1172 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| commonsenseqa | 1221 | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| gsm8k         | 1319 | 100.0    | 99.2 (-0.8pp) | 99.2 (-0.8pp)  | 100.0 (+0.0pp) | 100.0 (+0.0pp) | SF:- SS:- RF:- RS:- |
| logiqa        | 651  | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| openbookqa    | 500  | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| qasc          | 926  | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |
| strategyqa    | 687  | N/A      | N/A           | N/A            | N/A            | N/A            | SF:- SS:- RF:- RS:- |

### EOS rate per task (%, with CI if available; Δ vs baseline)
| Task          | n    | Baseline | Shared(full)  | Shared(staged) | Rand(full)    | Rand(staged)  |
|---------------|------|----------|---------------|----------------|---------------|---------------|
| aqua          | 254  | N/A      | N/A           | N/A            | N/A           | N/A           |
| arc_challenge | 1172 | N/A      | N/A           | N/A            | N/A           | N/A           |
| commonsenseqa | 1221 | N/A      | N/A           | N/A            | N/A           | N/A           |
| gsm8k         | 1319 | 99.9     | 92.3 (-7.6pp) | 93.5 (-6.4pp)  | 99.7 (-0.2pp) | 99.8 (-0.2pp) |
| logiqa        | 651  | N/A      | N/A           | N/A            | N/A           | N/A           |
| openbookqa    | 500  | N/A      | N/A           | N/A            | N/A           | N/A           |
| qasc          | 926  | N/A      | N/A           | N/A            | N/A           | N/A           |
| strategyqa    | 687  | N/A      | N/A           | N/A            | N/A           | N/A           |

### Avg new tokens per task (with CI if available; Δ and ratio vs baseline)
| Task          | n    | Baseline | Shared(full)         | Shared(staged)       | Rand(full)         | Rand(staged)       |
|---------------|------|----------|----------------------|----------------------|--------------------|--------------------|
| aqua          | 254  | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| arc_challenge | 1172 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| commonsenseqa | 1221 | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| gsm8k         | 1319 | 6.2      | 27.0 (+20.75, 4.32x) | 25.9 (+19.62, 4.14x) | 8.3 (+2.04, 1.33x) | 8.2 (+1.98, 1.32x) |
| logiqa        | 651  | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| openbookqa    | 500  | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| qasc          | 926  | N/A      | N/A                  | N/A                  | N/A                | N/A                |
| strategyqa    | 687  | N/A      | N/A                  | N/A                  | N/A                | N/A                |


---
