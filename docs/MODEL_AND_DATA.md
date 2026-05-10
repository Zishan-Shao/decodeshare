# Models and Data

DecodeShare uses public LLM checkpoints and standard QA / reasoning benchmarks.
Exact model availability and licenses are controlled by the upstream model
providers.

## Models used in scripts and outputs

- `meta-llama/Llama-2-7b-chat-hf`
- `meta-llama/Llama-2-13b-chat-hf`
- `meta-llama/Llama-2-70b-chat-hf`
- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `Qwen/Qwen2.5-1.5B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`
- `tiiuae/falcon-7b-instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `google/gemma-3-12b-it`

Some of these repositories are gated. Run `huggingface-cli login` or
`hf auth login`, accept the upstream model terms, and set the model path in the
scripts before launching experiments.

## Benchmarks

The loaders include common multiple-choice and reasoning datasets such as:

- GSM8K
- CommonsenseQA
- StrategyQA
- AQuA
- ARC-Challenge
- OpenBookQA
- PIQA
- QASC
- LogiQA
- BoolQ
- SST-2
- RTE

Dataset loading is implemented in the `benchmark_dataloaders.py` files under
`src/`, `reasoning/`, `patch_back/`, and `brittleness/`.

## Generated data

The branch tracks compact summaries and small JSON/CSV outputs. It does not
track raw activations, cached bases, steering vectors, model checkpoints, or
downstream `.pt` outputs. See `docs/artifact_manifest.tsv` for those files.
