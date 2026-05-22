# Steering Projection Demo

This demo shows the core DecodeShare steering idea on a Llama-style model:

1. Collect short KV-cached decode rollouts from mixed tasks.
2. Estimate a small decode-time shared basis from those hidden states.
3. Build a simple activation steering vector from contrastive prompts.
4. Split the steering vector into its shared-channel component and residual.
5. Compare generations from the original vector and the shared-removed vector.

The demo is intentionally small. It is not a replacement for the paper's full
steering-ranking experiments, but it gives a concrete view of how DecodeShare
changes a steering vector.

## Run

```bash
conda activate decodeshare

python demo/run_steering_projection_demo.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --layer 28 \
  --out_dir outputs/demo_steering_projection
```

The default model matches the paper's Llama-2 setup and may require Hugging Face
access approval. For a faster architecture-compatible smoke run, you can use a
smaller Llama-family checkpoint:

```bash
python demo/run_steering_projection_demo.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cuda \
  --layer 16 \
  --out_dir outputs/demo_steering_projection_tinyllama
```

To inspect the command without importing or loading model dependencies:

```bash
python demo/run_steering_projection_demo.py --dry_run
```

## Outputs

The script writes:

```text
outputs/demo_steering_projection/
  steering_projection_report.html
  projection_summary.json
```

Open the HTML report to see the projection split, overlap metrics, top logit
changes, and side-by-side generations.
