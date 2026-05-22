# -*- coding: utf-8 -*-
"""Compatibility CLI for the H1 sharedness existence runner."""

from pathlib import Path

from decodeshare.sharedness import *  # noqa: F401,F403
from decodeshare.sharedness import _should_write_txt  # noqa: F401
from decodeshare.sharedness import main as _sharedness_main


def main() -> None:
    _sharedness_main(default_output_dir=str(Path(__file__).resolve().parent))


if __name__ == "__main__":
    main()
