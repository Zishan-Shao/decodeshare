---
title: DecodeShare Steering Demo
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.10"
suggested_hardware: cpu-basic
short_description: Interactive prefill-vs-decode steering demo for DecodeShare.
models:
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0
---

# DecodeShare Steering Demo

This Space runs the public DecodeShare interactive steering demo. It compares
three generations for the same prompt:

- baseline,
- prefill-estimated steering vector,
- decode-estimated steering vector.

The bundled cache contains the demo decode-shared basis and preset steering
vectors for TinyLlama. It does not contain model weights. The Space still loads
`TinyLlama/TinyLlama-1.1B-Chat-v1.0` at startup. Free CPU hardware works, but
startup can take about 2-3 minutes if the Space is sleeping, and generation is
slow; keep `Max new tokens` small for CPU runs.

Recommended prompt:

```text
Explain the concept of 'Singular Value Decomposition' to a 5-year-old using a pirate metaphor.
```

This is a qualitative inspection demo. It is meant to make the
prefill-vs-decode deployment mismatch visible, not to claim that every preset is
improved by projection.
