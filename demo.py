#!/usr/bin/env python3

"""Launcher for the DecodeShare Gradio demo."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.gradio_app import *  # noqa: F401,F403,E402
from demo.gradio_app import main  # noqa: E402


if __name__ == "__main__":
    main()
