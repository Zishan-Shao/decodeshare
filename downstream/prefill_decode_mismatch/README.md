# Prefill/Decode Mismatch

Paper role: diagnostics for the distribution shift between prompt prefill
hidden states and KV-cached decode hidden states.

Main entry point:

- `exp_pca_mismatch.py`: estimates PCA bases on prefill and decode states, then
  reports principal angles and cross-distribution variance capture.

The main H3 ablation wrappers live under `experiments/04_prefill_decode/`; this
folder contains the focused PCA diagnostic used as a downstream appendix check.
