# Installation

Create the recorded conda environment:

```bash
conda env create -f environment.yml
conda activate flashsvd
```

Install the repository in editable mode:

```bash
pip install -e .
```

Run the lightweight checks:

```bash
bash scripts/run_all_smoke_tests.sh
```

The environment file reflects the local `flashsvd` environment used for the
camera-ready staging pass. Before public release, trim it if we want a smaller
cross-machine environment specification.
