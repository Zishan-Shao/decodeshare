#!/usr/bin/env python3

"""Precompute the Gradio demo basis/vector cache."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.gradio_app import DEFAULT_CHAT_CACHE, DEFAULT_CHAT_MODEL, prepare_chat_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--basis_k", type=int, default=24)
    parser.add_argument("--calib_max_new_tokens", type=int, default=6)
    parser.add_argument("--steer_max_new_tokens", type=int, default=6)
    parser.add_argument("--max_prompt_tokens", type=int, default=384)
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cache", default=str(DEFAULT_CHAT_CACHE))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        _, status = prepare_chat_state(
            model=args.model,
            device=args.device,
            dtype=args.dtype,
            layer=args.layer,
            basis_k=args.basis_k,
            calib_max_new_tokens=args.calib_max_new_tokens,
            steer_max_new_tokens=args.steer_max_new_tokens,
            max_prompt_tokens=args.max_prompt_tokens,
            system=args.system,
            seed=args.seed,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            cache_path=str(Path(args.cache)),
            use_cache=False,
            save_cache=True,
        )
    except Exception as exc:
        raise SystemExit(f"Failed to build interactive cache: {exc}") from None
    print(status)


if __name__ == "__main__":
    main()
