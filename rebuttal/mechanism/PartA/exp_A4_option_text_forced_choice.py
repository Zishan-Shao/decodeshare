# -*- coding: utf-8 -*-
"""
exp_A4_option_text_forced_choice.py

Follow-up rebuttal experiment to test whether the decode-shared subspace is
doing more than letter-format routing.

Idea:
  - Reuse the saved shared / matched-control bases from Exp-A3.
  - Evaluate the same MC tasks under:
      1) label-choice forced choice (A/B/C/D...) on the original prompt
      2) option-text forced choice on a prompt whose answer instruction is
         rewritten to request the answer text instead of the answer letter
      3) numeric-label forced choice on a prompt whose labels are rewritten
         from A/B/C/... to 1/2/3/...
  - If shared ablation still hurts under (2), that is stronger evidence
    against a purely formatting-only account.

This script intentionally does not touch basis estimation; it is a pure
evaluation wrapper around existing A3 artifacts.
"""

from __future__ import annotations

import os
import sys
import re
import json
import argparse
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


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

try:
    import eval_perf as EP
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import reasoning/eval_perf.py as module eval_perf.") from e

try:
    from benchmark_dataloaders import Example, load_selected_tasks
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import benchmark_dataloaders.py.") from e


CHOICE_RE = re.compile(r"^\s*([A-H])\)\s*(.+?)\s*$")
ANSWER_SPEC_RE = re.compile(r"Final answer:\s*<[^>\n]+>")


@dataclass
class PreparedExample:
    dataset: str
    ex_id: str
    prompt: str
    gold_label: str
    gold_index: int
    candidate_labels: List[str]
    candidate_strings: List[str]


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


def extract_labeled_choices(prompt: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for line in str(prompt).splitlines():
        m = CHOICE_RE.match(line)
        if not m:
            continue
        lab = str(m.group(1)).strip().upper()
        txt = str(m.group(2)).strip()
        if not txt or lab in seen:
            continue
        seen.add(lab)
        pairs.append((lab, txt))
    return pairs


def rewrite_prompt_answer_instruction(prompt: str) -> str:
    prompt = str(prompt)
    out = ANSWER_SPEC_RE.sub("Final answer: <answer text>", prompt)
    return out


def rewrite_prompt_numeric_labels(prompt: str, pairs: List[Tuple[str, str]]) -> str:
    prompt = str(prompt)
    if not pairs:
        return prompt

    numeric_pairs: List[Tuple[str, str]] = [(str(i + 1), txt) for i, (_lab, txt) in enumerate(pairs)]
    numbered_spec = "/".join([lab for lab, _txt in numeric_pairs])

    out_lines: List[str] = []
    idx = 0
    for line in prompt.splitlines():
        m = CHOICE_RE.match(line)
        if m and idx < len(numeric_pairs):
            num_lab, txt = numeric_pairs[idx]
            out_lines.append(f"{num_lab}) {txt}")
            idx += 1
        else:
            out_lines.append(line)

    out = "\n".join(out_lines)
    out = ANSWER_SPEC_RE.sub(f"Final answer: <{numbered_spec}>", out)
    return out


def prepare_examples(
    task: str,
    examples: List[Example],
    *,
    eval_spec: str,
) -> Tuple[List[PreparedExample], Dict[str, int]]:
    prepared: List[PreparedExample] = []
    stats = {
        "seen": int(len(examples)),
        "kept": 0,
        "skip_no_choices": 0,
        "skip_gold_not_found": 0,
    }

    for ex in examples:
        pairs = extract_labeled_choices(ex.prompt)
        if len(pairs) < 2:
            stats["skip_no_choices"] += 1
            continue

        labels = [lab for lab, _txt in pairs]
        texts = [txt for _lab, txt in pairs]
        gold = str(ex.gold).strip().upper()
        if gold not in labels:
            stats["skip_gold_not_found"] += 1
            continue
        gold_idx = labels.index(gold)

        if eval_spec == "label_original":
            prompt = ex.prompt
            candidates = labels
        elif eval_spec == "text_rewrite":
            prompt = rewrite_prompt_answer_instruction(ex.prompt)
            candidates = texts
        elif eval_spec == "number_rewrite":
            prompt = rewrite_prompt_numeric_labels(ex.prompt, pairs)
            candidates = [str(i + 1) for i in range(len(labels))]
        elif eval_spec == "text_original":
            prompt = ex.prompt
            candidates = texts
        else:
            raise ValueError(f"Unknown eval_spec={eval_spec!r}")

        prepared.append(
            PreparedExample(
                dataset=str(task),
                ex_id=str(ex.ex_id),
                prompt=str(prompt),
                gold_label=gold,
                gold_index=int(gold_idx),
                candidate_labels=labels,
                candidate_strings=[str(x) for x in candidates],
            )
        )

    stats["kept"] = int(len(prepared))
    return prepared, stats


def score_candidates_one(
    model,
    tok,
    prompt: str,
    candidate_strings: List[str],
    *,
    answer_prefix: str,
    prefix_mode: str,
    max_prompt_len: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    prompt = str(prompt)
    answer_prefix = EP.normalize_answer_prefix(answer_prefix)
    prefix_mode = str(prefix_mode).strip().lower()
    if prefix_mode not in {"auto", "always", "never"}:
        raise ValueError(f"Unknown prefix_mode={prefix_mode!r}")

    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    inputs = tok([prompt], return_tensors="pt", padding=True, truncation=True, max_length=int(max_prompt_len)).to(device)
    ids = inputs["input_ids"]
    attn = inputs["attention_mask"]

    past, logits = EP.cache_decode_aligned_boundary(model, ids, attn)

    do_prefix = False
    if prefix_mode == "always":
        do_prefix = bool(answer_prefix)
    elif prefix_mode == "auto":
        do_prefix = bool(answer_prefix) and (not EP.prompt_endswith_prefix(prompt, answer_prefix))

    if do_prefix and answer_prefix:
        prefix_ids = tok.encode(answer_prefix, add_special_tokens=False)
        for pid in prefix_ids:
            inp = torch.full((1, 1), int(pid), dtype=torch.long, device=device)
            attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)
            out = model(input_ids=inp, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

    out_scores = np.full((len(candidate_strings),), np.nan, dtype=np.float32)

    for ci, cand in enumerate(candidate_strings):
        cand_ids = EP.cand_token_ids(tok, str(cand))
        if len(cand_ids) == 0:
            out_scores[ci] = float("-inf")
            continue

        past_c = past
        attn_c = attn
        logits_c = logits
        lp = 0.0

        for ti, tok_id in enumerate(cand_ids):
            logp = torch.log_softmax(logits_c, dim=-1)
            lp += float(logp[0, int(tok_id)].detach().cpu().item())
            if ti < len(cand_ids) - 1:
                inp = torch.full((1, 1), int(tok_id), dtype=torch.long, device=device)
                attn_c = torch.cat([attn_c, torch.ones((1, 1), device=device, dtype=attn_c.dtype)], dim=1)
                out = model(input_ids=inp, attention_mask=attn_c, use_cache=True, past_key_values=past_c)
                logits_c = out.logits[:, -1, :]
                past_c = out.past_key_values

        out_scores[ci] = float(lp)

    return out_scores


@torch.no_grad()
def evaluate_prepared_examples(
    model,
    tok,
    prepared: List[PreparedExample],
    *,
    layer_index: int,
    basis_np: Optional[np.ndarray],
    alpha: float,
    answer_prefix: str,
    prefix_mode: str,
    max_prompt_len: int,
    progress_desc: str,
) -> Dict[str, Any]:
    handles, _hooks, hook_stats, _toggle = EP.register_hooks(
        model,
        layer_indices=[int(layer_index)],
        basis_np=basis_np,
        alpha=float(alpha),
        name=f"a4_fc@{int(layer_index)}",
    )

    N = len(prepared)
    correct = np.zeros(N, dtype=np.float32)
    preds: List[str] = [""] * N
    golds: List[str] = [ex.gold_label for ex in prepared]
    gold_margin = np.full(N, np.nan, dtype=np.float32)
    gold_logprob = np.full(N, np.nan, dtype=np.float32)
    entropy = np.full(N, np.nan, dtype=np.float32)

    try:
        for i, ex in enumerate(tqdm(prepared, desc=progress_desc)):
            scores = score_candidates_one(
                model,
                tok,
                ex.prompt,
                ex.candidate_strings,
                answer_prefix=str(answer_prefix),
                prefix_mode=str(prefix_mode),
                max_prompt_len=int(max_prompt_len),
            )
            pred_idx = int(np.argmax(scores))
            preds[i] = str(ex.candidate_strings[pred_idx])
            correct[i] = 1.0 if pred_idx == int(ex.gold_index) else 0.0

            gi = int(ex.gold_index)
            gold_logprob[i] = float(scores[gi])
            if scores.shape[0] > 1:
                other = np.delete(scores, gi)
                gold_margin[i] = float(scores[gi] - np.max(other))
            probs = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0).cpu().numpy()
            entropy[i] = float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum())
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass

    return {
        "n": int(N),
        "acc": float(correct.mean()) if N > 0 else float("nan"),
        "correct": correct,
        "preds": preds,
        "golds": golds,
        "metrics_summary": {
            "gold_logprob_mean": float(np.nanmean(gold_logprob)) if N > 0 else float("nan"),
            "gold_margin_mean": float(np.nanmean(gold_margin)) if N > 0 else float("nan"),
            "entropy_mean": float(np.nanmean(entropy)) if N > 0 else float("nan"),
        },
        "hook_stats": hook_stats,
    }


def render_md_report(
    result: Dict[str, Any],
    *,
    out_json: str,
) -> str:
    lines: List[str] = []
    lines.append("# Exp-A4: Option-text forced choice under decode-shared ablation")
    lines.append("")
    lines.append("This experiment reuses the Exp-A3 saved bases and evaluates whether the")
    lines.append("shared-subspace effect persists when the answer is scored as option text")
    lines.append("rather than just the answer letter.")
    lines.append("")
    lines.append("JSON: `" + os.path.relpath(out_json, ROOT_DIR) + "`")
    lines.append("")

    for eval_spec in result["eval_spec_order"]:
        block = result["evaluations"][eval_spec]
        lines.append(f"## Eval spec: {eval_spec}")
        lines.append("")
        lines.append("Preparation stats:")
        lines.append("```json")
        lines.append(json.dumps(block["prepare_stats"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

        header = ["Task", "n", "Baseline", "Shared", "Delta Shared", "Ctrl(E)", "Delta Ctrl(E)", "Rand(E)", "Delta Rand(E)"]
        rows: List[List[str]] = []
        for task in block["task_order"]:
            per_task = block["tasks"][task]
            b = per_task["by_condition"]["baseline"]["acc_ci"]
            sh = per_task["by_condition"]["shared"]["acc_ci"]
            ce = per_task["by_condition"]["ctrl_energy"]["acc_ci"]
            re = per_task["by_condition"]["rand_energy"]["acc_ci"]
            dsh = per_task["paired_vs_baseline"]["shared"]
            dce = per_task["paired_vs_baseline"]["ctrl_energy"]
            dre = per_task["paired_vs_baseline"]["rand_energy"]
            rows.append(
                [
                    task,
                    str(per_task["n"]),
                    _fmt_acc(b["mean"], b["lo"], b["hi"]),
                    _fmt_acc(sh["mean"], sh["lo"], sh["hi"]),
                    _fmt_diff(dsh),
                    _fmt_acc(ce["mean"], ce["lo"], ce["hi"]),
                    _fmt_diff(dce),
                    _fmt_acc(re["mean"], re["lo"], re["hi"]),
                    _fmt_diff(dre),
                ]
            )
        lines.append(_md_table(rows, header))
        lines.append("")

        pooled = block["pooled"]
        lines.append("Pooled across tasks:")
        lines.append("")
        pooled_rows = []
        pb = pooled["by_condition"]["baseline"]["acc_ci"]
        for key, label in [("shared", "Shared"), ("ctrl_energy", "Ctrl(E)"), ("rand_energy", "Rand(E)")]:
            pc = pooled["by_condition"][key]["acc_ci"]
            pd = pooled["paired_vs_baseline"][key]
            pooled_rows.append(
                [
                    label,
                    _fmt_acc(pb["mean"], pb["lo"], pb["hi"]),
                    _fmt_acc(pc["mean"], pc["lo"], pc["hi"]),
                    _fmt_diff(pd),
                ]
            )
        lines.append(_md_table(pooled_rows, ["Condition", "Baseline", "Condition", "Delta vs baseline"]))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Exp-A4: option-text forced choice rebuttal experiment.")
    ap.add_argument(
        "--a3_json",
        type=str,
        default=os.path.join(
            ROOT_DIR,
            "rebuttal",
            "mechanism",
            "PartA",
            "results",
            "20260225_205533",
            "A3_causal",
            "exp_A3_causal_layer10.json",
        ),
    )
    ap.add_argument("--basis_npz", type=str, default="")
    ap.add_argument("--eval_specs", type=str, default="label_original,text_rewrite")
    ap.add_argument("--eval_tasks", type=str, default="")
    ap.add_argument("--eval_n", type=int, default=64, help="Override eval_n from A3 config.")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--dtype", type=str, default="")
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=0.0)
    ap.add_argument("--max_memory_map", type=str, default="")
    ap.add_argument("--cpu_offload_gb", type=float, default=0.0)
    ap.add_argument("--max_prompt_len", type=int, default=2048)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--perm_iters", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "rebuttal_mechanism", "a4_option_text"),
    )
    ap.add_argument("--tag", type=str, default="")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    with open(os.path.expanduser(args.a3_json), "r", encoding="utf-8") as f:
        a3 = json.load(f)

    config = dict(a3["config"])
    basis_npz = str(args.basis_npz).strip() or str(a3["saved_basis_npz"])
    basis_npz = os.path.expanduser(basis_npz)
    bases = np.load(basis_npz)

    model_name = str(config["model"])
    device = str(args.device).strip() or str(config["device"])
    dtype = str(args.dtype).strip() or str(config["dtype"])
    device_map = str(args.device_map).strip() or str(config.get("device_map", "") or "")
    max_memory_map = str(args.max_memory_map).strip() or str(config.get("max_memory_map", "") or "")
    max_memory_per_gpu_gb = (
        float(args.max_memory_per_gpu_gb)
        if float(args.max_memory_per_gpu_gb) > 0
        else float(config.get("max_memory_per_gpu_gb", 0.0) or 0.0)
    )
    cpu_offload_gb = (
        float(args.cpu_offload_gb)
        if float(args.cpu_offload_gb) > 0
        else float(config.get("cpu_offload_gb", 0.0) or 0.0)
    )
    layer = int(config["layer"])
    eval_n = int(args.eval_n) if int(args.eval_n) > 0 else int(config["eval_n"])
    eval_tasks = _split_csv(args.eval_tasks) if str(args.eval_tasks).strip() else list(config["tasks_eval"])
    eval_specs = _split_csv(args.eval_specs)
    answer_prefix = str(config["forced_choice"]["fc_answer_prefix"])
    prefix_mode = str(config["forced_choice"]["fc_prefix_mode"])

    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = str(args.tag).strip()
    tag = ("_" + tag) if tag else ""
    out_json = os.path.join(out_dir, f"exp_A4_option_text_layer{layer}{tag}.json")
    out_md = os.path.join(out_dir, f"exp_A4_option_text_layer{layer}{tag}.md")

    print(f"[Load] model={model_name} device={device} dtype={dtype}")
    model, tok = EP.load_model_and_tokenizer(
        model_name,
        device=device,
        dtype=dtype,
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        device_map=(device_map or None),
        max_memory_per_gpu_gb=max_memory_per_gpu_gb,
        max_memory_map=max_memory_map,
        cpu_offload_gb=cpu_offload_gb,
    )

    print(f"[Data] tasks={eval_tasks} eval_n={eval_n}")
    _sub_by, eval_by, meta_by = load_selected_tasks(
        tasks=eval_tasks,
        n_subspace=2,
        n_eval=eval_n,
        seed=int(config["seed"]),
        template_randomization=bool(config["template_randomization"]),
        template_seed=int(config["template_seed"]),
        shuffle_choices=bool(config["shuffle_choices"]),
        add_answer_prefix=bool(config["add_answer_prefix"]),
        answer_prefix=str(config["answer_prefix"]),
    )

    conditions: List[Tuple[str, Optional[np.ndarray], float]] = [
        ("baseline", None, 0.0),
        ("shared", bases["Q_shared"].astype(np.float32), float(config["intervention"]["alpha_remove"])),
        ("ctrl_energy", bases["Q_ctrl_energy"].astype(np.float32), float(config["intervention"]["alpha_remove"])),
        ("rand_energy", bases["Q_rand_energy"].astype(np.float32), float(config["intervention"]["alpha_remove"])),
    ]

    result: Dict[str, Any] = {
        "source_a3_json": os.path.abspath(args.a3_json),
        "source_basis_npz": os.path.abspath(basis_npz),
        "config": {
            "model": model_name,
            "device": device,
            "dtype": dtype,
            "device_map": device_map,
            "max_memory_per_gpu_gb": max_memory_per_gpu_gb,
            "max_memory_map": max_memory_map,
            "cpu_offload_gb": cpu_offload_gb,
            "layer": layer,
            "eval_tasks": eval_tasks,
            "eval_n": eval_n,
            "answer_prefix": answer_prefix,
            "prefix_mode": prefix_mode,
            "eval_specs": eval_specs,
        },
        "dataset_meta": {task: meta_by[task] for task in eval_tasks},
        "eval_spec_order": eval_specs,
        "evaluations": {},
    }

    for eval_spec in eval_specs:
        print(f"[EvalSpec] {eval_spec}")
        spec_block: Dict[str, Any] = {
            "prepare_stats": {},
            "task_order": [],
            "tasks": {},
            "pooled": {},
        }

        pooled_corr: Dict[str, List[np.ndarray]] = {name: [] for name, _Q, _a in conditions}

        for task in eval_tasks:
            prepared, prep_stats = prepare_examples(task, eval_by[task], eval_spec=eval_spec)
            spec_block["prepare_stats"][task] = prep_stats
            if len(prepared) == 0:
                print(f"[Skip] {task} has no valid prepared examples for eval_spec={eval_spec}")
                continue

            per_task: Dict[str, Any] = {"n": int(len(prepared)), "by_condition": {}, "paired_vs_baseline": {}}
            corr_by_cond: Dict[str, np.ndarray] = {}

            for cond_name, Q, alpha in conditions:
                out = evaluate_prepared_examples(
                    model,
                    tok,
                    prepared,
                    layer_index=layer,
                    basis_np=Q,
                    alpha=float(alpha),
                    answer_prefix=answer_prefix,
                    prefix_mode=prefix_mode,
                    max_prompt_len=int(args.max_prompt_len),
                    progress_desc=f"{eval_spec}:{task}:{cond_name}",
                )
                corr = np.asarray(out["correct"], dtype=np.float32)
                acc, lo, hi = EP.bootstrap_ci_mean(
                    corr,
                    iters=int(args.bootstrap_iters),
                    alpha=float(args.alpha),
                    seed=EP.stable_int_seed("a4", eval_spec, task, cond_name, int(config["seed"])),
                )
                corr_by_cond[cond_name] = corr
                pooled_corr[cond_name].append(corr)

                per_task["by_condition"][cond_name] = {
                    "acc": float(out["acc"]),
                    "acc_ci": {"mean": float(acc), "lo": float(lo), "hi": float(hi)},
                    "metrics_summary": out["metrics_summary"],
                    "hook_stats": out["hook_stats"],
                }

            base_corr = corr_by_cond["baseline"]
            for cond_name, _Q, _alpha in conditions:
                if cond_name == "baseline":
                    continue
                per_task["paired_vs_baseline"][cond_name] = EP.summarize_paired(
                    base_corr,
                    corr_by_cond[cond_name],
                    label=cond_name,
                    bootstrap_iters=int(args.bootstrap_iters),
                    perm_iters=int(args.perm_iters),
                    alpha=float(args.alpha),
                    seed=EP.stable_int_seed("a4", eval_spec, task, cond_name, "paired", int(config["seed"])),
                )

            spec_block["task_order"].append(task)
            spec_block["tasks"][task] = per_task

        pooled_block: Dict[str, Any] = {"by_condition": {}, "paired_vs_baseline": {}}
        for cond_name, _Q, _alpha in conditions:
            if not pooled_corr[cond_name]:
                continue
            arr = np.concatenate(pooled_corr[cond_name], axis=0)
            acc, lo, hi = EP.bootstrap_ci_mean(
                arr,
                iters=int(args.bootstrap_iters),
                alpha=float(args.alpha),
                seed=EP.stable_int_seed("a4", eval_spec, cond_name, "pooled", int(config["seed"])),
            )
            pooled_block["by_condition"][cond_name] = {
                "n": int(arr.shape[0]),
                "acc_ci": {"mean": float(acc), "lo": float(lo), "hi": float(hi)},
            }

        if "baseline" in pooled_block["by_condition"]:
            base_arr = np.concatenate(pooled_corr["baseline"], axis=0)
            for cond_name, _Q, _alpha in conditions:
                if cond_name == "baseline" or cond_name not in pooled_block["by_condition"]:
                    continue
                arr = np.concatenate(pooled_corr[cond_name], axis=0)
                pooled_block["paired_vs_baseline"][cond_name] = EP.summarize_paired(
                    base_arr,
                    arr,
                    label=cond_name,
                    bootstrap_iters=int(args.bootstrap_iters),
                    perm_iters=int(args.perm_iters),
                    alpha=float(args.alpha),
                    seed=EP.stable_int_seed("a4", eval_spec, cond_name, "pooled_paired", int(config["seed"])),
                )

        spec_block["pooled"] = pooled_block
        result["evaluations"][eval_spec] = spec_block

    _atomic_json_dump(result, out_json)
    md = render_md_report(result, out_json=out_json)
    _atomic_text_dump(md, out_md)
    print(f"[Done] JSON -> {out_json}")
    print(f"[Done] MD   -> {out_md}")


if __name__ == "__main__":
    main()
