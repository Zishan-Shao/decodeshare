# 01 Sharedness

Paper role: H1, shared decode-time structure.

This folder contains the canonical code needed to reproduce the H1 result tables
and diagnostics. New runs should use the top-level scripts below.

## Structure

- `run_full_benchmark.py`: canonical full H1 benchmark runner for Table 6 and appendix Tables 7-13.
- `sharedness_base.py`: thin compatibility CLI for the original fair-task existence runner.
- `summarize_full_benchmark.py`: aggregates full-benchmark JSON/TXT records into CSV, Markdown, and LaTeX summaries.
- `collect_activations.py`: collects balanced decode-phase hidden states for diagnostic figures.
- `analysis/`: H1 diagnostic analysis scripts.
- `configs/`: paper parameter records for full runs and diagnostics.

Shared method code lives in `decodeshare.sharedness` and
`decodeshare.activations`; files in this folder are experiment entry points.

## Results

Generated paper-facing artifacts default to:

- `outputs/01_sharedness/full_benchmark/`
- `outputs/01_sharedness/exp1/`
- `outputs/01_sharedness/exp2/`
- `outputs/01_sharedness/exp2.75/`
- `outputs/01_sharedness/exp3/`

The full-benchmark directory includes compact raw result records (`*_exist*.json` and `*_exist*.txt`) plus generated summaries:

- `H1_full_benchmark_summary.csv`
- `H1_full_benchmark_summary.md`
- `H1_full_benchmark_summary.tex`
- `H1_evidence_chain.tex`

## Reproduction Notes

- Use only `Node0` and `Node1` for camera-ready reruns unless the cluster availability changes.
- Keep `PYTHONPATH` pointed at the repo root when running direct Python commands.
- Full model reruns are expensive; the smoke tests only check CLI/import validity and summary regeneration.

```bash
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
bash scripts/01_h1_sharedness/run_mock.sh
```

See `PAPER_RESULTS.md` and `scripts/01_h1_sharedness/COMMANDS.md` for paper-output mapping and exact commands.
