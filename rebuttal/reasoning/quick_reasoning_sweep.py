from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_SCRIPT = ROOT / "reasoning" / "disturb_CoT_shared_loto_reasoning.py"
DEFAULT_RAW_DIR = Path(__file__).resolve().parent / "raw"

TASK_LABELS = {
    "gsm8k": "Open-ended numeric reasoning",
    "commonsenseqa": "Commonsense multiple choice",
    "strategyqa": "Yes/No reasoning",
    "aqua": "Math multiple choice",
    "arc_challenge": "Science reasoning multiple choice",
    "openbookqa": "Open-book science reasoning",
    "qasc": "Multi-hop science reasoning",
    "logiqa": "Logical reasoning multiple choice",
    "boolq": "Reading comprehension yes/no",
    "piqa": "Physical reasoning multiple choice",
}

CANDIDATE_COUNTS = {
    "commonsenseqa": 5,
    "strategyqa": 2,
    "aqua": 5,
    "arc_challenge": 4,
    "openbookqa": 4,
    "qasc": 8,
    "logiqa": 4,
    "boolq": 2,
    "piqa": 2,
}


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_command(args: argparse.Namespace, holdout: str, out_json: Path, out_md: Path) -> List[str]:
    cmd = [
        args.python,
        str(args.base_script),
        "--model",
        args.model,
        "--device",
        args.device,
        "--model_dtype",
        args.model_dtype,
        "--layer",
        str(args.layer),
        "--tasks",
        args.tasks,
        "--mode",
        "loto",
        "--loto_eval_mode",
        "heldout",
        "--loto_only",
        holdout,
        "--n_subspace",
        str(args.n_subspace),
        "--n_eval",
        str(args.n_eval),
        "--pca_var",
        str(args.pca_var),
        "--tau",
        str(args.tau),
        "--m_shared",
        args.m_shared,
        "--calib_decode_max_new_tokens",
        str(args.calib_decode_max_new_tokens),
        "--per_task_max_states",
        str(args.per_task_max_states),
        "--alpha_remove",
        str(args.alpha_remove),
        "--reasoning_tokens",
        str(args.reasoning_tokens),
        "--max_new_tokens",
        str(args.max_new_tokens),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
        "--do_sample",
        str(int(args.do_sample)),
        "--template_randomization",
        str(int(args.template_randomization)),
        "--template_seed",
        str(args.template_seed),
        "--shuffle_choices",
        str(int(args.shuffle_choices)),
        "--add_answer_prefix",
        str(int(args.add_answer_prefix)),
        "--answer_prefix",
        args.answer_prefix,
        "--use_forced_choice",
        str(int(args.use_forced_choice)),
        "--fc_warmup_tokens",
        str(args.fc_warmup_tokens),
        "--fc_prefix_mode",
        args.fc_prefix_mode,
        "--fc_answer_prefix",
        args.fc_answer_prefix,
        "--batch_size",
        str(args.batch_size),
        "--max_prompt_len",
        str(args.max_prompt_len),
        "--bootstrap_iters",
        str(args.bootstrap_iters),
        "--perm_iters",
        str(args.perm_iters),
        "--ci_alpha",
        str(args.ci_alpha),
        "--seed",
        str(args.seed),
        "--sample_seed",
        str(args.sample_seed),
        "--out_json",
        str(out_json),
        "--out_md",
        str(out_md),
    ]
    return cmd


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_metrics(payload: Dict[str, Any], holdout: str) -> Dict[str, Any]:
    fold = payload["folds"][holdout]
    result = fold["eval"][holdout]
    paired = result.get("paired", {})
    decode_minus_prefill = paired.get("decode_minus_prefill", {})
    decode_minus_baseline = paired.get("decode_minus_baseline", {})
    prefill_minus_baseline = paired.get("prefill_minus_baseline", {})
    random_minus_baseline = paired.get("random_minus_baseline", {})
    baseline_acc = float(result["baseline"]["acc"])
    decode_acc = float(result["decode_shared"]["acc"])
    prefill_acc = float(result["prefill_shared"]["acc"])
    random_acc = float(result["random"]["acc"])

    chance_acc: Optional[float] = None
    if holdout in CANDIDATE_COUNTS:
        chance_acc = 1.0 / float(CANDIDATE_COUNTS[holdout])

    floor_flag = baseline_acc <= (1.0 / max(int(result.get("n", 1)), 1))
    if chance_acc is not None:
        floor_flag = floor_flag or baseline_acc <= chance_acc

    return {
        "task": holdout,
        "task_label": TASK_LABELS.get(holdout, holdout),
        "n_eval": int(result.get("n", 0)),
        "protocol": result.get("protocol", ""),
        "k_eval": int(fold.get("bases", {}).get("k_eval", 0)),
        "chance_acc": chance_acc,
        "baseline_near_floor": bool(floor_flag),
        "baseline_acc": baseline_acc,
        "decode_shared_acc": decode_acc,
        "prefill_shared_acc": prefill_acc,
        "random_acc": random_acc,
        "decode_minus_prefill": float(decode_minus_prefill.get("mean_diff", 0.0)),
        "decode_minus_prefill_ci_low": float(decode_minus_prefill.get("ci_low", 0.0)),
        "decode_minus_prefill_ci_high": float(decode_minus_prefill.get("ci_high", 0.0)),
        "decode_minus_prefill_p": float(decode_minus_prefill.get("p_value", 1.0)),
        "decode_minus_baseline": float(decode_minus_baseline.get("mean_diff", decode_acc - baseline_acc)),
        "prefill_minus_baseline": float(prefill_minus_baseline.get("mean_diff", prefill_acc - baseline_acc)),
        "random_minus_baseline": float(random_minus_baseline.get("mean_diff", random_acc - baseline_acc)),
    }


def fmt_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def fmt_delta(value: float) -> str:
    return f"{100.0 * value:+.1f}"


def build_markdown(summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    cfg = summary["config"]
    aggregate = summary["aggregate"]
    lines.append("# Quick Reasoning Rebuttal Check")
    lines.append("")
    lines.append("This is a quick-turn held-out-task check for reasoning-heavy tasks.")
    lines.append("")
    lines.append(f"- Model: `{cfg['model']}` dtype={cfg['model_dtype']} device={cfg['device']}")
    lines.append(f"- Tasks used for basis/eval: `{cfg['tasks']}`")
    lines.append(f"- Held-out tasks run: `{cfg['heldout_tasks']}`")
    lines.append(f"- Per-task eval size: n_eval={cfg['n_eval']}, n_subspace={cfg['n_subspace']}, layer={cfg['layer']}")
    lines.append(f"- Protocol: LOTO heldout, forced_choice={cfg['use_forced_choice']}, do_sample={cfg['do_sample']}")
    lines.append("")
    lines.append("## Per-task results")
    lines.append("")
    header = [
        "Held-out",
        "Type",
        "n",
        "Baseline",
        "Decode-shared",
        "Prefill-shared",
        "Random",
        "D-P delta",
        "p",
    ]
    rows: List[List[str]] = []
    for item in summary["per_task"]:
        rows.append(
            [
                item["task"],
                item["task_label"],
                str(item["n_eval"]),
                fmt_pct(item["baseline_acc"]) if item["chance_acc"] is None else f"{fmt_pct(item['baseline_acc'])} (chance {fmt_pct(item['chance_acc'])})",
                fmt_pct(item["decode_shared_acc"]),
                fmt_pct(item["prefill_shared_acc"]),
                fmt_pct(item["random_acc"]),
                f"{fmt_delta(item['decode_minus_prefill'])} [{fmt_delta(item['decode_minus_prefill_ci_low'])}, {fmt_delta(item['decode_minus_prefill_ci_high'])}]",
                f"{item['decode_minus_prefill_p']:.3g}",
            ]
        )

    cols = list(zip(*([header] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(str(x).ljust(w) for x, w in zip(row, widths)) + " |"

    lines.append(fmt_row(header))
    lines.append("|-" + "-|-".join("-" * width for width in widths) + "-|")
    for row in rows:
        lines.append(fmt_row(row))
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(
        f"- Mean accuracy: baseline={fmt_pct(aggregate['baseline_acc_mean'])}, "
        f"decode_shared={fmt_pct(aggregate['decode_shared_acc_mean'])}, "
        f"prefill_shared={fmt_pct(aggregate['prefill_shared_acc_mean'])}, "
        f"random={fmt_pct(aggregate['random_acc_mean'])}"
    )
    lines.append(
        f"- Mean deltas vs baseline: decode={fmt_delta(aggregate['decode_minus_baseline_mean'])}, "
        f"prefill={fmt_delta(aggregate['prefill_minus_baseline_mean'])}, "
        f"random={fmt_delta(aggregate['random_minus_baseline_mean'])}"
    )
    lines.append(
        f"- Mean decode-minus-prefill delta: {fmt_delta(aggregate['decode_minus_prefill_mean'])}"
    )
    informative = summary["aggregate"].get("informative_tasks", [])
    inconclusive = summary["aggregate"].get("inconclusive_tasks", [])
    if informative:
        lines.append(f"- Informative held-out tasks: `{','.join(informative)}`")
    if inconclusive:
        lines.append(f"- Inconclusive due to baseline floor/chance: `{','.join(inconclusive)}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    for item in summary["per_task"]:
        if item["baseline_near_floor"]:
            lines.append(
                f"- `{item['task']}` is currently inconclusive: baseline is at or near floor/chance, so this fold does not say much about decode-vs-prefill selectivity."
            )
        else:
            lines.append(
                f"- `{item['task']}` is informative: decode-shared changes accuracy by {fmt_delta(item['decode_minus_baseline'])} vs baseline and {fmt_delta(item['decode_minus_prefill'])} vs prefill-shared."
            )
    lines.append(
        "- Use informative folds as rebuttal evidence that the decode-shared phenomenon is not confined to short classification tasks."
    )
    return "\n".join(lines) + "\n"


def aggregate_metrics(per_task: List[Dict[str, Any]]) -> Dict[str, float]:
    informative = [item for item in per_task if not item["baseline_near_floor"]]
    return {
        "baseline_acc_mean": mean(item["baseline_acc"] for item in per_task),
        "decode_shared_acc_mean": mean(item["decode_shared_acc"] for item in per_task),
        "prefill_shared_acc_mean": mean(item["prefill_shared_acc"] for item in per_task),
        "random_acc_mean": mean(item["random_acc"] for item in per_task),
        "decode_minus_baseline_mean": mean(item["decode_minus_baseline"] for item in per_task),
        "prefill_minus_baseline_mean": mean(item["prefill_minus_baseline"] for item in per_task),
        "random_minus_baseline_mean": mean(item["random_minus_baseline"] for item in per_task),
        "decode_minus_prefill_mean": mean(item["decode_minus_prefill"] for item in per_task),
        "informative_tasks": [item["task"] for item in informative],
        "inconclusive_tasks": [item["task"] for item in per_task if item["baseline_near_floor"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small reasoning held-out sweep and summarize it.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base_script", type=Path, default=DEFAULT_BASE_SCRIPT)
    parser.add_argument("--raw_dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--summary_json", type=Path, default=Path(__file__).resolve().parent / "quick_reasoning_summary.json")
    parser.add_argument("--summary_md", type=Path, default=Path(__file__).resolve().parent / "quick_reasoning_summary.md")
    parser.add_argument("--reuse_existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_dtype", default="fp16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--tasks", default="gsm8k,commonsenseqa,strategyqa,arc_challenge,openbookqa,qasc,logiqa")
    parser.add_argument("--heldout_tasks", default="gsm8k,logiqa")
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--n_subspace", type=int, default=64)
    parser.add_argument("--n_eval", type=int, default=32)
    parser.add_argument("--pca_var", type=float, default=0.95)
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--m_shared", default="all")
    parser.add_argument("--calib_decode_max_new_tokens", type=int, default=64)
    parser.add_argument("--per_task_max_states", type=int, default=4096)
    parser.add_argument("--alpha_remove", type=float, default=1.0)
    parser.add_argument("--reasoning_tokens", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--template_randomization", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--template_seed", type=int, default=1234)
    parser.add_argument("--shuffle_choices", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add_answer_prefix", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--answer_prefix", default="\nFinal answer:")
    parser.add_argument("--use_forced_choice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fc_warmup_tokens", type=int, default=0)
    parser.add_argument("--fc_prefix_mode", default="auto", choices=["auto", "always", "never"])
    parser.add_argument("--fc_answer_prefix", default="\nFinal answer:")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_prompt_len", type=int, default=512)
    parser.add_argument("--bootstrap_iters", type=int, default=1000)
    parser.add_argument("--perm_iters", type=int, default=2000)
    parser.add_argument("--ci_alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_seed", type=int, default=12345)
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)

    heldout_tasks = parse_csv(args.heldout_tasks)
    if not heldout_tasks:
        raise ValueError("heldout_tasks must not be empty")
    task_set = set(parse_csv(args.tasks))
    bad = [task for task in heldout_tasks if task not in task_set]
    if bad:
        raise ValueError(f"heldout_tasks must be a subset of tasks; got invalid {bad}")

    env = os.environ.copy()
    if args.offline:
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("HF_DATASETS_OFFLINE", "1")

    per_task: List[Dict[str, Any]] = []
    commands: Dict[str, List[str]] = {}
    raw_files: Dict[str, Dict[str, str]] = {}

    for holdout in heldout_tasks:
        out_json = args.raw_dir / f"{holdout}_loto.json"
        out_md = args.raw_dir / f"{holdout}_loto.md"
        cmd = build_command(args, holdout, out_json, out_md)
        commands[holdout] = cmd
        raw_files[holdout] = {"json": str(out_json), "md": str(out_md)}
        if args.reuse_existing and out_json.exists():
            print(f"[reuse] holdout={holdout} json={out_json}")
        else:
            print(f"[run] holdout={holdout}")
            print("[cmd]", " ".join(cmd))
            subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
        payload = load_json(out_json)
        per_task.append(extract_metrics(payload, holdout))

    per_task.sort(key=lambda item: heldout_tasks.index(item["task"]))

    summary = {
        "config": {
            "model": args.model,
            "device": args.device,
            "model_dtype": args.model_dtype,
            "tasks": args.tasks,
            "heldout_tasks": ",".join(heldout_tasks),
            "layer": args.layer,
            "n_subspace": args.n_subspace,
            "n_eval": args.n_eval,
            "use_forced_choice": bool(args.use_forced_choice),
            "do_sample": bool(args.do_sample),
            "offline": bool(args.offline),
        },
        "commands": commands,
        "raw_files": raw_files,
        "per_task": per_task,
        "aggregate": aggregate_metrics(per_task),
    }

    with args.summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    with args.summary_md.open("w", encoding="utf-8") as handle:
        handle.write(build_markdown(summary))

    print(f"[done] summary_json={args.summary_json}")
    print(f"[done] summary_md={args.summary_md}")


if __name__ == "__main__":
    main()
