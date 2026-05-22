# Steering Projection Demo

This demo shows the core DecodeShare steering idea on a Llama-style model:

1. Collect short KV-cached decode rollouts from mixed tasks.
2. Estimate a small decode-time shared basis from those hidden states.
3. Build a simple activation steering vector from contrastive prompts.
4. Split the steering vector into its shared-channel component and residual.
5. Show the paper-level rank-flip snapshot that motivates decode-time steering
   validation.
6. Compare generations from the original vector and the shared-removed vector.

The demo is intentionally small. It is not a replacement for the paper's full
steering-ranking experiments, but it gives a concrete view of how DecodeShare
changes a steering vector.

The script prints a concise rank-flip table and an example steering-vector
before/after directly to the terminal. It also writes the same content, plus a
visual vector split, to an HTML report.

By default, the script uses a controlled demo vector: it starts from the
CAA-style contrastive vector and amplifies the part that overlaps the
decode-shared basis, so the before/after effect is visible in a short run. To
use the untouched CAA-style vector, add `--demo_vector_mode caa`.

## Run

```bash
conda activate decodeshare

python demo/run_steering_projection_demo.py \
  --model meta-llama/Llama-2-7b-chat-hf \
  --device cuda \
  --layer 28 \
  --demo_vector_mode caa_plus_shared \
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
changes, the rank-flip snapshot, and side-by-side generations.

## Gradio App

The optional Gradio UI is a focused **Interactive Steering Chat**. It displays
three responses for each prompt:

- baseline,
- prefill-estimated steering vector,
- decode-estimated steering vector.

Both steering vectors are deployed during KV-cached decoding; only the
estimation source differs. This is a qualitative inspection tool and does not
claim that every preset is repaired or improved by projection.

```bash
pip install -r demo/requirements-demo.txt

python demo/app.py --server_port 7860
```

The app still has to load the selected model. To avoid repeated calibration
work, it can load a small cache containing the demo decode-shared basis and
preset steering vectors. The cache does not contain model weights.

```bash
python demo/build_interactive_cache.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cuda \
  --layer 16 \
  --cache demo/assets/interactive_tinyllama_chat_cache.pt
```

After that file exists, `demo/app.py` will reuse it by default and skip
basis/vector estimation during initialization.
