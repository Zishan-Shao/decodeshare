# -*- coding: utf-8 -*-
"""
exp_1d_open_answer_reasoning_probe.py

Open-answer reasoning-style linear probes on saved basis coordinates, intended for
comparing `Q_resid` against `Q_fmt` and a same-width random residual basis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import string
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

import eval_perf as EP  # noqa: E402
from benchmark_dataloaders import load_selected_tasks  # noqa: E402


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(out_path) + ".", suffix=".tmp", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=EP.json_default)
        os.replace(tmp_path, out_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


_STEP_WORDS = {
    "step",
    "first",
    "second",
    "third",
    "fourth",
    "next",
    "then",
    "finally",
}
_EQ_CHARS = set("=+-*/^%×÷")


def _safe_convert_id_to_token(tok, tid: int) -> str:
    try:
        return str(tok.convert_ids_to_tokens([int(tid)])[0])
    except Exception:
        try:
            return str(tok.convert_ids_to_tokens(int(tid)))
        except Exception:
            return f"<id:{int(tid)}>"


def _safe_decode_one(tok, tid: int) -> str:
    try:
        return tok.decode([int(tid)], clean_up_tokenization_spaces=False)
    except Exception:
        try:
            return tok.decode([int(tid)])
        except Exception:
            return ""


def _reasoning_tags(raw_tok: str, decoded: str) -> Dict[str, int]:
    s = decoded
    s_strip = s.strip()
    low = s_strip.lower()
    return {
        "reasoning_marker": int(
            bool(re.search(r"(?i)\btherefore\b|\bthus\b|\bhence\b|\bbecause\b|\bsince\b|\bso\b", s_strip))
        ),
        "digit": int(any(ch.isdigit() for ch in s_strip)),
        "equation_symbol": int(any(ch in _EQ_CHARS for ch in s_strip)),
        "step_marker": int(
            low in _STEP_WORDS
            or bool(re.fullmatch(r"(step|step\s*\d+)", low))
            or bool(re.fullmatch(r"\d+\.", low))
        ),
    }


@dataclass
class ProbeRecords:
    states: np.ndarray
    token_ids: np.ndarray
    prompt_ids: np.ndarray
    tasks: np.ndarray


class DecodeTokenProbeCollector:
    def __init__(self, layer_idx: int):
        self.layer_idx = int(layer_idx)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.pending_prompt_ids: Optional[np.ndarray] = None
        self.pending_token_ids: Optional[np.ndarray] = None
        self.storage: Dict[str, Dict[str, List[Any]]] = {}

    def set_current_task(self, task: str) -> None:
        self._cur_task = str(task)

    def set_capture(
        self,
        enabled: bool,
        *,
        active_mask: Optional[torch.Tensor] = None,
        prompt_ids: Optional[np.ndarray] = None,
        token_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.capture_enabled = bool(enabled)
        self.active_mask = active_mask
        self.pending_prompt_ids = None if prompt_ids is None else np.asarray(prompt_ids, dtype=np.int64)
        self.pending_token_ids = None if token_ids is None else np.asarray(token_ids, dtype=np.int64)

    def make_hook(self):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3 or hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]
            pids = self.pending_prompt_ids
            tids = self.pending_token_ids
            if pids is None or tids is None:
                return output
            if self.active_mask is not None:
                m_dev = self.active_mask.to(device=x.device).bool()
                if int(m_dev.numel()) == int(x.shape[0]):
                    x = x[m_dev]
                m_cpu = self.active_mask.detach().cpu().numpy().astype(bool)
                if int(m_cpu.size) == int(pids.shape[0]):
                    pids = pids[m_cpu]
                    tids = tids[m_cpu]
            if x.numel() == 0:
                return output
            store = self.storage.setdefault(self._cur_task, {"states": [], "prompt_ids": [], "token_ids": []})
            store["states"].append(x.detach().float().cpu().numpy())
            store["prompt_ids"].append(np.asarray(pids, dtype=np.int64))
            store["token_ids"].append(np.asarray(tids, dtype=np.int64))
            return output

        return _hook

    def get(self, task: str) -> ProbeRecords:
        d = self.storage.get(task, {})
        states = np.concatenate(d.get("states", []), axis=0) if d.get("states") else np.zeros((0, 0), dtype=np.float32)
        token_ids = (
            np.concatenate(d.get("token_ids", []), axis=0) if d.get("token_ids") else np.zeros((0,), dtype=np.int64)
        )
        prompt_ids = (
            np.concatenate(d.get("prompt_ids", []), axis=0) if d.get("prompt_ids") else np.zeros((0,), dtype=np.int64)
        )
        tasks = np.array([task] * int(states.shape[0]))
        return ProbeRecords(states=states, token_ids=token_ids, prompt_ids=prompt_ids, tasks=tasks)


@torch.no_grad()
def _collect_decode_token_records(
    *,
    model,
    tok,
    prompts: List[str],
    collector: DecodeTokenProbeCollector,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    prompt_base: int,
) -> int:
    device = EP.infer_model_input_device(model)
    eos = tok.eos_token_id
    model.eval()
    prompt_cursor = int(prompt_base)

    for i in range(0, len(prompts), int(batch_size)):
        batch = prompts[i : i + int(batch_size)]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = int(ids.shape[0])
        batch_prompt_ids = np.arange(prompt_cursor, prompt_cursor + B, dtype=np.int64)
        prompt_cursor += B

        unfinished = torch.ones(B, dtype=torch.bool, device=device)
        collector.set_capture(False)
        past, logits = EP.cache_decode_aligned_boundary(model, ids, attn)

        for _step in range(int(max_new_tokens)):
            next_tok = EP.choose_next_token(
                logits,
                decoding="greedy",
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                eos_token_id=eos,
                ban_eos=False,
            )
            next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
            active = unfinished & (next_tok.squeeze(-1) != eos)
            unfinished = active
            if not bool(active.any().item()):
                break
            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            collector.set_capture(
                True,
                active_mask=active,
                prompt_ids=batch_prompt_ids,
                token_ids=next_tok.squeeze(-1).detach().cpu().numpy(),
            )
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
        collector.set_capture(False)
    return prompt_cursor


def _subsample_task_records(rec: ProbeRecords, n_max: int, seed: int) -> ProbeRecords:
    if int(n_max) <= 0 or int(rec.states.shape[0]) <= int(n_max):
        return rec
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(rec.states.shape[0], size=int(n_max), replace=False)
    idx = np.sort(idx)
    return ProbeRecords(
        states=rec.states[idx],
        token_ids=rec.token_ids[idx],
        prompt_ids=rec.prompt_ids[idx],
        tasks=rec.tasks[idx],
    )


def _split_indices(y: np.ndarray, groups: np.ndarray, seed: int, test_size: float) -> Tuple[np.ndarray, np.ndarray, str]:
    gss = GroupShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    try:
        train_idx, test_idx = next(gss.split(np.zeros_like(y), y, groups=groups))
        y_tr = y[train_idx]
        y_te = y[test_idx]
        if int(y_tr.min()) != int(y_tr.max()) and int(y_te.min()) != int(y_te.max()):
            return np.asarray(train_idx), np.asarray(test_idx), "group"
    except Exception:
        pass
    sss = StratifiedShuffleSplit(n_splits=1, test_size=float(test_size), random_state=int(seed))
    train_idx, test_idx = next(sss.split(np.zeros_like(y), y))
    return np.asarray(train_idx), np.asarray(test_idx), "stratified"


def _fit_eval_binary_probe(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray) -> Dict[str, Any]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
        ]
    )
    pipe.fit(X[train_idx], y[train_idx])
    prob = pipe.predict_proba(X[test_idx])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y[test_idx], prob)),
        "avg_precision": float(average_precision_score(y[test_idx], prob)),
        "balanced_acc": float(balanced_accuracy_score(y[test_idx], pred)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "pos_train": int(y[train_idx].sum()),
        "pos_test": int(y[test_idx].sum()),
    }


def _render_md(config: Dict[str, Any], dataset_summary: Dict[str, Any], probe_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Exp-1d: Open-Answer Reasoning-Style Probe")
    lines.append("")
    lines.append("## Config")
    lines.append("```json")
    lines.append(json.dumps(config, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Dataset")
    lines.append("```json")
    lines.append(json.dumps(dataset_summary, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Probe results")
    lines.append("| tag | basis | k | n_total | n_pos | split | ROC-AUC | AP | BalAcc |")
    lines.append("|---|---|---:|---:|---:|---|---:|---:|---:|")
    for row in probe_rows:
        lines.append(
            "| {tag} | {basis_key} | {k} | {n_total} | {n_pos} | {split_mode} | {roc_auc:.3f} | {avg_precision:.3f} | {balanced_acc:.3f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--basis_keys", type=str, default="Q_resid,Q_fmt,Q_rand_resid")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--tasks", type=str, default="gsm8k,strategyqa,aqua")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--per_task_max_states", type=int, default=4000)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--min_pos", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/reasoning_probe")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    basis_keys = _split_csv(args.basis_keys)
    tasks = _split_csv(args.tasks)
    if not basis_keys:
        raise ValueError("Empty --basis_keys")
    if not tasks:
        raise ValueError("Empty --tasks")

    arrs = np.load(os.path.expanduser(str(args.basis_npz)))
    bases: Dict[str, np.ndarray] = {}
    for key in basis_keys:
        if key not in arrs.files:
            raise KeyError(f"Missing basis {key!r} in {args.basis_npz}")
        bases[key] = np.asarray(arrs[key], dtype=np.float32)

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""

    EP.set_global_seed(int(args.seed))
    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
    )

    sub_by, _eval_dummy, meta_by = load_selected_tasks(
        tasks=tasks,
        n_subspace=max(1, int(args.n_prompts)),
        n_eval=1,
        seed=int(args.seed),
        template_randomization=bool(args.template_randomization),
        template_seed=int(args.template_seed),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
    )

    layers, _ = EP.get_model_layers(model)
    collector = DecodeTokenProbeCollector(int(args.layer))
    handle = layers[int(args.layer)].register_forward_hook(collector.make_hook())

    records_by_task: Dict[str, ProbeRecords] = {}
    prompt_base = 0
    try:
        for task, exs in sub_by.items():
            collector.set_current_task(task)
            prompts = [ex.prompt for ex in exs]
            prompt_base = _collect_decode_token_records(
                model=model,
                tok=tok,
                prompts=prompts,
                collector=collector,
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.max_new_tokens),
                max_prompt_len=int(args.max_prompt_len),
                prompt_base=int(prompt_base),
            )
            rec = collector.get(task)
            rec = _subsample_task_records(rec, int(args.per_task_max_states), seed=EP.stable_int_seed(args.seed, task, "reasoning_probe"))
            records_by_task[task] = rec
            print(f"[Collected] task={task} states={rec.states.shape[0]}")
    finally:
        try:
            handle.remove()
        except Exception:
            pass

    H = np.concatenate([np.asarray(rec.states, dtype=np.float32) for rec in records_by_task.values() if rec.states.size > 0], axis=0)
    groups = np.concatenate([np.asarray(rec.prompt_ids, dtype=np.int64) for rec in records_by_task.values() if rec.states.size > 0], axis=0)
    token_ids = np.concatenate([np.asarray(rec.token_ids, dtype=np.int64) for rec in records_by_task.values() if rec.states.size > 0], axis=0)
    reasoning_rows = [_reasoning_tags(_safe_convert_id_to_token(tok, int(tid)), _safe_decode_one(tok, int(tid))) for tid in token_ids.tolist()]

    probe_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, Dict[str, str]] = {}
    for tag_name in ["reasoning_marker", "digit", "equation_symbol", "step_marker"]:
        y = np.array([int(r[tag_name]) for r in reasoning_rows], dtype=np.int64)
        n_total = int(y.shape[0])
        n_pos = int(y.sum())
        n_neg = int(n_total - n_pos)
        if n_pos < int(args.min_pos) or n_neg < int(args.min_pos):
            skipped[tag_name] = {"all": f"Too few positives/negatives: pos={n_pos} neg={n_neg}"}
            continue
        train_idx, test_idx, split_mode = _split_indices(y, groups, seed=EP.stable_int_seed(args.seed, tag_name), test_size=float(args.test_size))
        skipped.setdefault(tag_name, {})
        for basis_key, Q in bases.items():
            X = H @ Q
            fit = _fit_eval_binary_probe(X, y, train_idx, test_idx)
            probe_rows.append(
                {
                    "tag": tag_name,
                    "basis_key": basis_key,
                    "k": int(Q.shape[1]),
                    "n_total": n_total,
                    "n_pos": n_pos,
                    "split_mode": split_mode,
                    "roc_auc": fit["roc_auc"],
                    "avg_precision": fit["avg_precision"],
                    "balanced_acc": fit["balanced_acc"],
                }
            )

    results = {
        "config": {
            "basis_npz": str(args.basis_npz),
            "basis_keys": basis_keys,
            "model": str(args.model),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "layer": int(args.layer),
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
            "batch_size": int(args.batch_size),
            "max_prompt_len": int(args.max_prompt_len),
            "max_new_tokens": int(args.max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "test_size": float(args.test_size),
            "min_pos": int(args.min_pos),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
        },
        "dataset": {
            "n_samples": int(H.shape[0]),
            "tasks": {task: int(rec.states.shape[0]) for task, rec in records_by_task.items()},
            "meta_by_task": meta_by,
        },
        "probe_rows": probe_rows,
        "skipped": skipped,
    }

    base = f"exp_1d_open_answer_reasoning_probe_layer{int(args.layer)}{tag}"
    json_path = os.path.join(out_dir, base + ".json")
    md_path = os.path.join(out_dir, base + ".md")
    _atomic_json_dump(results, json_path)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_render_md(results["config"], results["dataset"], probe_rows))
    print(f"[Saved] {json_path}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
