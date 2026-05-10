#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_calib_mix_jsonl.py

Generate a local JSONL calibration mixture (with a single 'text' field per line)
that can be fed into SVD-LLM's get_calib_train_data().

Why JSONL?
- Simple.
- Compatible with HF `load_dataset("json")`.
- Lets you mix many tasks (language + reasoning + code) into one calibration pool.

This script intentionally generates *prompts* (not model outputs). Whitening only needs
realistic input distributions to estimate activation covariances.

Example:
  python make_calib_mix_jsonl_v2.py \
    --output ./calib_mix.jsonl \
    --tasks wikitext,gsm8k,commonsenseqa,boolq,arc_challenge,openbookqa,piqa,humaneval,mbpp,competition_math \
    --n_per_task 512 \
    --seed 0 \
    --template_randomization \
    --shuffle_choices \
    --add_answer_prefix \
    --answer_prefix "\nFinal answer:"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict

from benchmark_dataloaders_ext import load_selected_tasks, stable_int_seed


def _write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=str, required=True, help="Output JSONL path.")
    p.add_argument(
        "--tasks",
        type=str,
        required=True,
        help="Comma-separated task list. Examples: wikitext,gsm8k,commonsenseqa,humaneval",
    )
    p.add_argument("--n_per_task", type=int, default=256, help="How many prompts per task.")
    p.add_argument("--seed", type=int, default=0, help="Base seed (each task gets a derived seed).")

    # Prompt shaping controls (keep aligned with benchmark_dataloaders_ext)
    p.add_argument("--template_randomization", action="store_true", help="Randomize MC templates per example.")
    p.add_argument("--shuffle_choices", action="store_true", help="Shuffle MC choice order deterministically.")
    p.add_argument("--add_answer_prefix", action="store_true", help="Append an answer prefix to prompts.")
    p.add_argument("--answer_prefix", type=str, default="\nAnswer:", help="Answer prefix string.")

    # Robustness knob
    p.add_argument(
        "--skip_failed_tasks",
        action="store_true",
        help="If a dataset fails to load, skip it (and print a warning) instead of crashing.",
    )

    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(tasks) == 0:
        raise ValueError("No tasks provided.")

    # Try to load everything in one shot (fast path).
    try:
        sub_by, _, meta_by = load_selected_tasks(
            tasks=tasks,
            n_subspace=args.n_per_task,
            n_eval=0,
            seed=args.seed,
            template_randomization=bool(args.template_randomization),
            template_seed=args.seed,
            shuffle_choices=bool(args.shuffle_choices),
            add_answer_prefix=bool(args.add_answer_prefix),
            answer_prefix=args.answer_prefix,
        )
    except Exception as e:
        if not args.skip_failed_tasks:
            raise
        print(f"[WARN] Bulk loading failed: {type(e).__name__}: {e}")
        print("[WARN] Falling back to per-task loading (skipping failures).")

        sub_by, meta_by = {}, {}
        for t in tasks:
            t_seed = stable_int_seed(args.seed, t)
            try:
                sb, _, mb = load_selected_tasks(
                    tasks=[t],
                    n_subspace=args.n_per_task,
                    n_eval=0,
                    seed=t_seed,
                    template_randomization=bool(args.template_randomization),
                    template_seed=t_seed,
                    shuffle_choices=bool(args.shuffle_choices),
                    add_answer_prefix=bool(args.add_answer_prefix),
                    answer_prefix=args.answer_prefix,
                )
                sub_by.update(sb)
                meta_by.update(mb)
                print(f"[OK] Loaded {t}: {len(sb[t])} prompts")
            except Exception as ee:
                print(f"[SKIP] {t}: {type(ee).__name__}: {ee}")

    # Flatten to JSONL rows
    rows: List[Dict[str, str]] = []
    for t in tasks:
        exs = sub_by.get(t, [])
        for ex in exs:
            rows.append({"text": ex.prompt, "task": t})

    out = Path(args.output)
    _write_jsonl(out, rows)

    print("\n=== Calibration mix written ===")
    print(f"Output: {out.resolve()}")
    print(f"Total rows: {len(rows)}")
    print("Rows per task:")
    for t in tasks:
        print(f"  - {t}: {len(sub_by.get(t, []))}")

    # Light meta summary (helpful for reproducibility)
    # NOTE: meta_by may be empty if everything failed/skipped.
    if meta_by:
        # Keep JSON readable by pruning large fields.
        meta_out = out.with_suffix(".meta.json")
        meta_clean = {}
        for k, v in meta_by.items():
            vv = dict(v)
            # Avoid huge schema dumps; keep id + revision and a couple of basics
            vv.pop("schema", None)
            meta_clean[k] = vv
        meta_out.write_text(json.dumps(meta_clean, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Meta: {meta_out.resolve()}")


if __name__ == "__main__":
    main()
