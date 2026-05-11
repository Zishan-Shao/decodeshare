# -*- coding: utf-8 -*-
"""
exp_A3_eval_saved_basis.py

Reuse a saved A3 basis NPZ and run larger forced-choice evaluations without
recomputing decode-state collections.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start_dir: str) -> str:
    cur = os.path.abspath(start_dir)
    for _ in range(10):
        if os.path.isdir(os.path.join(cur, "src")) and os.path.isdir(os.path.join(cur, "reasoning")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.normpath(os.path.join(start_dir, "..", "..", ".."))


ROOT_DIR = _find_repo_root(THIS_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import eval_perf as EP
from benchmark_dataloaders import load_selected_tasks


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        if o.ndim == 0:
            return float(o.detach().cpu().item())
        return o.detach().cpu().tolist()
    return str(o)


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _atomic_text_dump(text: str, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _fmt_acc(acc: float, lo: float, hi: float) -> str:
    return f"{acc*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def _fmt_diff(stat: Dict[str, Any]) -> str:
    return f"{stat['mean_diff']*100:+.1f} [{stat['ci_low']*100:+.1f}, {stat['ci_high']*100:+.1f}] (p={stat['p_value']:.3g})"


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _load_conditions(npz_path: str, names: List[str]) -> Dict[str, np.ndarray]:
    arrs = np.load(npz_path)
    mapping = {
        "shared": "Q_shared",
        "ctrl_struct": "Q_ctrl_struct",
        "ctrl_energy": "Q_ctrl_energy",
        "rand_struct": "Q_rand_struct",
        "rand_energy": "Q_rand_energy",
    }
    out: Dict[str, np.ndarray] = {}
    for name in names:
        if name == "baseline":
            continue
        key = mapping.get(name)
        if key is None or key not in arrs:
            raise KeyError(f"Condition '{name}' missing in basis npz: key={key!r}")
        out[name] = np.asarray(arrs[key], dtype=np.float32)
    return out


def _config_payload(args: argparse.Namespace, tasks_eval: List[str], cond_names: List[str]) -> Dict[str, Any]:
    return {
        "basis_npz": str(args.basis_npz),
        "model": args.model,
        "device": args.device,
        "dtype": args.dtype,
        "trust_remote_code": bool(args.trust_remote_code),
        "device_map": str(args.device_map),
        "max_memory_per_gpu_gb": float(args.max_memory_per_gpu_gb),
        "max_memory_map": str(args.max_memory_map),
        "cpu_offload_gb": float(args.cpu_offload_gb),
        "layer": int(args.layer),
        "tasks_eval": tasks_eval,
        "conditions": cond_names,
        "protocol": str(args.protocol),
        "eval_n": int(args.eval_n),
        "batch_size": int(args.batch_size),
        "max_prompt_len": int(args.max_prompt_len),
        "seed": int(args.seed),
        "template_seed": int(args.template_seed),
        "template_randomization": bool(args.template_randomization),
        "shuffle_choices": bool(args.shuffle_choices),
        "add_answer_prefix": bool(args.add_answer_prefix),
        "answer_prefix": str(args.answer_prefix),
        "fc_answer_prefix": str(args.fc_answer_prefix),
        "fc_prefix_mode": str(args.fc_prefix_mode),
        "generation": {
            "decoding": str(args.decoding),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "stats": {
            "bootstrap_iters": int(args.bootstrap_iters),
            "perm_iters": int(args.perm_iters),
            "alpha": float(args.alpha),
        },
    }


def _load_resume_payload(json_path: str, expected_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    cfg = obj.get("config", {})
    if cfg != expected_config:
        raise RuntimeError(
            f"Existing resume file has mismatched config: {json_path}\n"
            "Use a different --out_dir/--tag or delete the stale file."
        )
    return obj


def _checkpoint_status(eval_results: Dict[str, Any], tasks_eval: List[str], cond_names: List[str], *, complete: bool) -> Dict[str, Any]:
    completed_conditions = {}
    for task in tasks_eval:
        pt = eval_results.get(task, {}) or {}
        conds = sorted((pt.get("_checkpoint_correct", {}) or {}).keys(), key=lambda x: cond_names.index(x) if x in cond_names else 10**9)
        completed_conditions[task] = conds
    return {
        "complete": bool(complete),
        "tasks_eval": list(tasks_eval),
        "conditions": list(cond_names),
        "completed_conditions": completed_conditions,
    }


def _dump_partial_json(
    *,
    json_path: str,
    config: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    eval_results: Dict[str, Any],
    tasks_eval: List[str],
    cond_names: List[str],
    complete: bool,
) -> None:
    payload = {
        "config": config,
        "dataset_meta": dataset_meta,
        "eval": eval_results,
        "checkpoint": _checkpoint_status(eval_results, tasks_eval, cond_names, complete=complete),
    }
    _atomic_json_dump(payload, json_path)


def _strip_checkpoint_fields(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    for task, pt in eval_results.items():
        pt2 = dict(pt)
        pt2.pop("_checkpoint_correct", None)
        clean[task] = pt2
    return clean


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=0.0)
    ap.add_argument("--max_memory_map", type=str, default="")
    ap.add_argument("--cpu_offload_gb", type=float, default=0.0)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tasks_eval", type=str, required=True)
    ap.add_argument("--conditions", type=str, default="baseline,shared,ctrl_energy,rand_energy")
    ap.add_argument("--protocol", type=str, default="forced_choice", choices=["forced_choice", "generation"])
    ap.add_argument("--eval_n", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_prompt_len", type=int, default=2048)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--fc_save_scores", type=int, default=0, choices=[0, 1])
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--perm_iters", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--resume", type=int, default=1, choices=[0, 1], help="Resume from existing partial JSON if present.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_scaling")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    tasks_eval = _split_csv(args.tasks_eval)
    cond_names = _split_csv(args.conditions)
    if not tasks_eval:
        raise ValueError("Empty --tasks_eval")
    if not cond_names:
        raise ValueError("Empty --conditions")
    if cond_names[0] != "baseline":
        cond_names = ["baseline"] + [c for c in cond_names if c != "baseline"]

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""
    base_name = f"exp_A3_eval_saved_basis_layer{int(args.layer)}{tag}"
    json_path = os.path.join(out_dir, base_name + ".json")
    md_path = os.path.join(out_dir, base_name + ".md")
    config_payload = _config_payload(args, tasks_eval, cond_names)
    resume_obj = _load_resume_payload(json_path, config_payload) if bool(args.resume) else None

    EP.set_global_seed(int(args.seed))
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=(str(args.device_map).strip() or None),
        max_memory_per_gpu_gb=float(args.max_memory_per_gpu_gb),
        max_memory_map=str(args.max_memory_map),
        cpu_offload_gb=float(args.cpu_offload_gb),
    )

    _sub_dummy, eval_by, meta_by = load_selected_tasks(
        tasks=tasks_eval,
        n_subspace=1,
        n_eval=max(1, int(args.eval_n)),
        seed=int(args.seed),
        template_randomization=bool(args.template_randomization),
        template_seed=int(args.template_seed),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
    )

    cond_to_basis = _load_conditions(str(args.basis_npz), cond_names)
    eval_results: Dict[str, Any] = dict((resume_obj or {}).get("eval", {}) or {})
    _dump_partial_json(
        json_path=json_path,
        config=config_payload,
        dataset_meta=meta_by,
        eval_results=eval_results,
        tasks_eval=tasks_eval,
        cond_names=cond_names,
        complete=False,
    )
    for task in tasks_eval:
        examples = eval_by[task]
        existing_task = dict(eval_results.get(task, {}) or {})
        per_task: Dict[str, Any] = {
            "n": int(len(examples)),
            "by_condition": dict(existing_task.get("by_condition", {}) or {}),
            "paired_vs_baseline": dict(existing_task.get("paired_vs_baseline", {}) or {}),
            "_checkpoint_correct": dict(existing_task.get("_checkpoint_correct", {}) or {}),
        }
        if int(per_task["n"]) != int(len(examples)):
            raise RuntimeError(f"Resume mismatch for task={task}: saved n={per_task['n']} current n={len(examples)}")
        corr_by_cond: Dict[str, np.ndarray] = {}
        for name, corr_saved in (per_task.get("_checkpoint_correct", {}) or {}).items():
            corr_by_cond[name] = np.asarray(corr_saved, dtype=np.float32)

        for name in cond_names:
            if name in corr_by_cond and name in per_task["by_condition"]:
                print(f"[Resume] Skipping completed condition task={task} condition={name}")
                continue
            basis = cond_to_basis.get(name)
            alpha = 0.0 if name == "baseline" else 1.0
            if str(args.protocol) == "generation":
                out_eval = EP.generation_eval(
                    model,
                    tok,
                    examples,
                    task,
                    layer_indices=[int(args.layer)],
                    basis_np=basis,
                    alpha=float(alpha),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    max_new_tokens=int(args.max_new_tokens),
                    decoding=str(args.decoding),
                    temperature=float(args.temperature),
                    top_p=float(args.top_p),
                    top_k=int(args.top_k),
                    seed=EP.stable_int_seed(int(args.seed), task, name, str(args.decoding), float(args.temperature), float(args.top_p)),
                )
            else:
                out_eval = EP.forced_choice_logprob_eval(
                    model,
                    tok,
                    examples,
                    task,
                    layer_indices=[int(args.layer)],
                    basis_np=basis,
                    alpha=float(alpha),
                    batch_size=int(args.batch_size),
                    max_prompt_len=int(args.max_prompt_len),
                    warmup_token_ids=None,
                    answer_prefix=str(args.fc_answer_prefix),
                    prefix_mode=str(args.fc_prefix_mode),
                    save_scores=bool(args.fc_save_scores),
                )

            corr = np.asarray(out_eval["correct"], dtype=np.float32)
            acc, lo, hi = EP.bootstrap_ci_mean(
                corr,
                iters=int(args.bootstrap_iters),
                alpha=float(args.alpha),
                seed=int(args.seed) + 11,
            )
            corr_by_cond[name] = corr
            summary_entry = {
                "acc": float(out_eval["acc"]),
                "acc_ci": {"mean": float(acc), "lo": float(lo), "hi": float(hi)},
                "hook_stats": out_eval.get("hook_stats", {}),
            }
            if str(args.protocol) == "generation":
                summary_entry["generation_summary"] = {
                    "extraction_rate": float(out_eval.get("extraction_rate", float("nan"))),
                }
            else:
                summary_entry["metrics_summary"] = out_eval.get("metrics_summary", {})
            per_task["by_condition"][name] = summary_entry
            per_task["_checkpoint_correct"][name] = corr.tolist()
            eval_results[task] = per_task
            _dump_partial_json(
                json_path=json_path,
                config=config_payload,
                dataset_meta=meta_by,
                eval_results=eval_results,
                tasks_eval=tasks_eval,
                cond_names=cond_names,
                complete=False,
            )
            del out_eval
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        base_corr = corr_by_cond["baseline"]
        for name in cond_names:
            if name == "baseline":
                continue
            per_task["paired_vs_baseline"][name] = EP.summarize_paired(
                base_corr,
                corr_by_cond[name],
                label=name,
                bootstrap_iters=int(args.bootstrap_iters),
                perm_iters=int(args.perm_iters),
                alpha=float(args.alpha),
                seed=int(args.seed) + 999,
            )
        eval_results[task] = per_task
        _dump_partial_json(
            json_path=json_path,
            config=config_payload,
            dataset_meta=meta_by,
            eval_results=eval_results,
            tasks_eval=tasks_eval,
            cond_names=cond_names,
            complete=False,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    header = ["Task", "n", "Baseline"]
    if "shared" in cond_names:
        header.extend(["Shared", "ΔShared"])
    if "ctrl_energy" in cond_names:
        header.extend(["Ctrl(E)", "ΔCtrl(E)"])
    if "rand_energy" in cond_names:
        header.extend(["Rand(E)", "ΔRand(E)"])
    if "ctrl_struct" in cond_names:
        header.extend(["Ctrl(k)", "ΔCtrl(k)"])
    if "rand_struct" in cond_names:
        header.extend(["Rand(k)", "ΔRand(k)"])

    rows: List[List[str]] = []
    for task in tasks_eval:
        pt = eval_results[task]
        row = [
            task,
            str(pt["n"]),
            _fmt_acc(
                pt["by_condition"]["baseline"]["acc_ci"]["mean"],
                pt["by_condition"]["baseline"]["acc_ci"]["lo"],
                pt["by_condition"]["baseline"]["acc_ci"]["hi"],
            ),
        ]
        if "shared" in cond_names:
            row.extend([
                _fmt_acc(
                    pt["by_condition"]["shared"]["acc_ci"]["mean"],
                    pt["by_condition"]["shared"]["acc_ci"]["lo"],
                    pt["by_condition"]["shared"]["acc_ci"]["hi"],
                ),
                _fmt_diff(pt["paired_vs_baseline"]["shared"]),
            ])
        if "ctrl_energy" in cond_names:
            row.extend([
                _fmt_acc(
                    pt["by_condition"]["ctrl_energy"]["acc_ci"]["mean"],
                    pt["by_condition"]["ctrl_energy"]["acc_ci"]["lo"],
                    pt["by_condition"]["ctrl_energy"]["acc_ci"]["hi"],
                ),
                _fmt_diff(pt["paired_vs_baseline"]["ctrl_energy"]),
            ])
        if "rand_energy" in cond_names:
            row.extend([
                _fmt_acc(
                    pt["by_condition"]["rand_energy"]["acc_ci"]["mean"],
                    pt["by_condition"]["rand_energy"]["acc_ci"]["lo"],
                    pt["by_condition"]["rand_energy"]["acc_ci"]["hi"],
                ),
                _fmt_diff(pt["paired_vs_baseline"]["rand_energy"]),
            ])
        if "ctrl_struct" in cond_names:
            row.extend([
                _fmt_acc(
                    pt["by_condition"]["ctrl_struct"]["acc_ci"]["mean"],
                    pt["by_condition"]["ctrl_struct"]["acc_ci"]["lo"],
                    pt["by_condition"]["ctrl_struct"]["acc_ci"]["hi"],
                ),
                _fmt_diff(pt["paired_vs_baseline"]["ctrl_struct"]),
            ])
        if "rand_struct" in cond_names:
            row.extend([
                _fmt_acc(
                    pt["by_condition"]["rand_struct"]["acc_ci"]["mean"],
                    pt["by_condition"]["rand_struct"]["acc_ci"]["lo"],
                    pt["by_condition"]["rand_struct"]["acc_ci"]["hi"],
                ),
                _fmt_diff(pt["paired_vs_baseline"]["rand_struct"]),
            ])
        rows.append(row)

    out = {
        "config": config_payload,
        "dataset_meta": meta_by,
        "eval": _strip_checkpoint_fields(eval_results),
        "checkpoint": _checkpoint_status(eval_results, tasks_eval, cond_names, complete=True),
    }

    md = []
    protocol_label = "generation" if str(args.protocol) == "generation" else "forced-choice"
    md.append(f"# Exp-A3 Eval: Reuse saved basis on larger {protocol_label} eval")
    md.append("")
    md.append(f"Basis: `{args.basis_npz}`")
    md.append("")
    md.append(f"## {protocol_label.title()} results")
    md.append(_md_table(rows, header))
    md.append("")
    md.append(f"JSON: `{json_path}`")
    md.append("")

    _atomic_json_dump(out, json_path)
    _atomic_text_dump("\n".join(md), md_path)
    print(f"[Saved] {json_path}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
