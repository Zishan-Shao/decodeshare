Optional Gradio cache files live here.

`demo.py` looks for `interactive_tinyllama_chat_cache.pt` by default. The
cache contains the demo decode-shared basis, preset steering vectors, and
metadata only; it does not contain model weights.
That default cache filename is allowed by `.gitignore`, so it can be committed
intentionally after being generated on a CUDA machine.

Build it with:

```bash
python demo/build_interactive_cache.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --device cuda \
  --layer 16 \
  --cache demo/assets/interactive_tinyllama_chat_cache.pt
```
