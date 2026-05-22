# Steering Rank Flip

Paper role: evaluate whether decode-aligned validation ranks steering vectors
better than prefill-based proxies under held-out KV-cached deployment.

Main entry points:

- `exp_cross_method_rank_flip.py`: builds and evaluates CAA, instruction, and
  SAE candidate pools.
- `exp_diagnostic_rank_flip.py`: generates diagnostic random directions and
  runs the rank-flip protocol.
- `exp_trad_family_rank_flip.py`: compares prefill-only, always-on, and
  no-cache traditional steering proxies.
- `exp_rank_flip.py`: core rank-flip evaluator used by the wrappers above.

Recommended wrapper:

```bash
bash scripts/reproduce_steering_flip_tables.sh
```

Use `DRY_RUN=1` to print the commands without running model inference.
