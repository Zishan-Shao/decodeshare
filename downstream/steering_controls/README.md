# Steering Controls

Paper role: projection and robustness controls for steering vectors after
estimating decode-time shared structure.

Main entry point:

- `exp_projection_controls.py`: compares the original vector against
  shared-subspace projection, random-subspace projection, PCA controls,
  prefill-PCA controls, and norm-matched shrinkage.

This folder is separate from `steering_rank_flip/` because these controls test
what happens after choosing vectors, while rank flip tests how vectors should be
selected for held-out decode deployment.
