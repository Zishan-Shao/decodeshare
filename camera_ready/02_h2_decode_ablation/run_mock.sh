#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIG="/home/zs89/decodeshare"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
read -r -a PYTHON_ARR <<< "${PYTHON_CMD:-conda run -n flashsvd python}"

(cd "${ROOT}/experiments/02_decode_ablation" && CUDA_VISIBLE_DEVICES="" "${PYTHON_ARR[@]}" run_loto_reasoning.py --help >/dev/null)
(cd "${ROOT}/experiments/02_decode_ablation" && CUDA_VISIBLE_DEVICES="" "${PYTHON_ARR[@]}" run_energy_kmatch_reasoning.py --help >/dev/null)

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT
cat > "${TMPDIR}/mixed_protocol_loto.json" <<'JSON'
{
  "config": {
    "model": "meta-llama/Llama-2-7b-chat-hf",
    "mode": "loto",
    "loto_eval_mode": "heldout",
    "layer_indices": [10],
    "tau": 0.001,
    "m_shared": "all",
    "template_randomization": 1,
    "shuffle_choices": 1,
    "rand_type": "joint_nonshared_varmatch",
    "model_dtype": "fp32",
    "tasks": ["gsm8k", "commonsenseqa"]
  },
  "folds": {
    "gsm8k": {
      "basis": {"cross_dim": 10, "shared_k": 2, "sanity": {"energy_ratio_shared": {"mean": 0.4}, "energy_ratio_rand": {"mean": 0.1}}},
      "by_dataset": {
        "gsm8k": {
          "n": 1,
          "runs": {
            "greedy/baseline": {"accuracy": 1.0, "ci_low": 1.0, "ci_high": 1.0},
            "greedy/shared_full": {"accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0},
            "greedy/rand_full": {"accuracy": 1.0, "ci_low": 1.0, "ci_high": 1.0}
          },
          "paired_tests": {"greedy": {"shared_full_vs_baseline": {"mean_diff": -1.0, "ci_low": -1.0, "ci_high": -1.0, "p_value": 0.0001}}}
        }
      }
    },
    "commonsenseqa": {
      "basis": {"cross_dim": 10, "shared_k": 2, "sanity": {"energy_ratio_shared": {"mean": 0.4}, "energy_ratio_rand": {"mean": 0.1}}},
      "by_dataset": {
        "commonsenseqa": {
          "n": 1,
          "runs": {
            "forced_choice/baseline": {"accuracy": 1.0, "ci_low": 1.0, "ci_high": 1.0},
            "forced_choice/shared_full": {"accuracy": 0.0, "ci_low": 0.0, "ci_high": 0.0},
            "forced_choice/rand_full": {"accuracy": 1.0, "ci_low": 1.0, "ci_high": 1.0}
          },
          "paired_tests": {"forced_choice": {"shared_full_vs_baseline": {"mean_diff": -1.0, "ci_low": -1.0, "ci_high": -1.0, "p_value": 0.0001}}}
        }
      }
    }
  }
}
JSON
(cd "${ROOT}" && "${PYTHON_ARR[@]}" experiments/02_decode_ablation/summarize_disturb_cot_results.py --results_dir "${TMPDIR}" --pattern mixed_protocol_loto.json --no_recursive --output "${TMPDIR}/summary.md" >/dev/null)
grep -E '\| gsm8k +\| 1 +\| greedy +' "${TMPDIR}/summary.md" >/dev/null
grep -E '\| commonsenseqa +\| 1 +\| forced_choice +' "${TMPDIR}/summary.md" >/dev/null

test -s "${ORIG}/results/disturb_cot_reasoning/energy_balance_loto8_reasoning_fc_eval2048.md"
test -s "${ORIG}/results/energy_kmatch_alpha_sweep/meta-llama_Llama-2-7b-chat-hf_L10_seed42_ts20260110_080440.tex"
echo "h2_ablation_mock_ok"
