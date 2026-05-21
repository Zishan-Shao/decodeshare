# Installation

Create the clean conda environment:

```bash
conda env create -f environment.yml
conda activate decodeshare
```

The environment installs the repository in editable mode through
`pyproject.toml`. If you are using an existing environment instead, install the
package and development tools manually:

```bash
pip install -e ".[dev]"
```

Run the lightweight checks:

```bash
bash scripts/run_all_smoke_tests.sh
```

The conda file keeps only the Python version, PyTorch/CUDA stack, core
scientific packages, and the editable local install. Project-level Python
dependencies live in `pyproject.toml` so pip-only and conda users share the
same package metadata.

For CPU-only machines, remove `pytorch-cuda=12.1` and the `nvidia` channel from
`environment.yml`, or install PyTorch separately before `pip install -e ".[dev]"`.
