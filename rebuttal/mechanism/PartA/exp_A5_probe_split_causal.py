# -*- coding: utf-8 -*-
"""
exp_A5_probe_split_causal.py

Decompose a saved decode-shared basis Q_shared into:
  - Q_fmt   : span of format/readout-predictive probe directions inside Q_shared
  - Q_resid : orthogonal complement of Q_fmt within span(Q_shared)

Then run forced-choice causal ablations for:
  baseline / shared_full / fmt_only / resid_only / rand_fmt_shared / rand_resid_shared

This is designed to quantify whether a substantial residual non-format component
remains causally important after factoring out format/readout-predictive directions.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import string
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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

import eval_perf as EP  # noqa: E402
from benchmark_dataloaders import load_selected_tasks  # noqa: E402


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


_PUNCT_ASCII = set(string.punctuation)
_PUNCT_CJK = set("，。！？；：、…·“”‘’—–")
_BRACKETS = set("()[]{}<>（）【】《》")
_OPTION_LETTERS = set("ABCDE")
_YES_NO = {"yes", "no"}


def _tags_for_token(raw_tok: str, decoded: str) -> List[str]:
    s = decoded
    s_strip = s.strip()
    tags: List[str] = []

    if "\n" in s:
        tags.append("newline")
    if s.startswith(" ") or raw_tok.startswith("▁") or raw_tok.startswith("Ġ"):
        tags.append("leading_space")
    if s_strip == "":
        tags.append("whitespace")
    if any(ch in _BRACKETS for ch in s_strip):
        tags.append("bracket")
    if s_strip in _OPTION_LETTERS:
        tags.append("option_letter")
    if re.fullmatch(r"[A-E][\)\.\:]", s_strip):
        tags.append("option_punct")
    low = s_strip.lower()
    if low in _YES_NO:
        tags.append("yes_no")
    if any(ch.isdigit() for ch in s_strip):
        tags.append("digit")
    # More permissive than exp_1b: allow single-token answer/readout words too.
    if re.search(r"(?i)\bfinal\s*answer\b", s) or re.search(r"(?i)\banswer\s*:", s):
        tags.append("answer_marker")
    elif low in {"answer", "final"}:
        tags.append("answer_marker")
    if re.search(r"(?i)\btherefore\b|\bthus\b|\bhence\b|\bbecause\b", s_strip):
        tags.append("reasoning_marker")
    if s_strip and all((ch in _PUNCT_ASCII) or (ch in _PUNCT_CJK) for ch in s_strip):
        tags.append("punct")
    return tags


def _tag_targets(tags: List[str]) -> Dict[str, int]:
    tag_set = set(tags)
    return {
        "answer_readout": int(
            ("option_letter" in tag_set)
            or ("yes_no" in tag_set)
            or ("answer_marker" in tag_set)
        ),
        "option_letter": int("option_letter" in tag_set),
        "newline": int("newline" in tag_set),
        "answer_marker": int("answer_marker" in tag_set),
        "yes_no": int("yes_no" in tag_set),
        "digit": int("digit" in tag_set),
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
            if int(x.shape[0]) != int(len(pids)) or int(x.shape[0]) != int(len(tids)):
                raise RuntimeError(
                    f"Collector alignment mismatch: states={x.shape[0]} prompt_ids={len(pids)} token_ids={len(tids)}"
                )

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
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> int:
    assert decoding in ["greedy", "sample"]
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

        for _ in range(int(max_new_tokens)):
            next_tok = EP.choose_next_token(
                logits,
                decoding=str(decoding),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
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
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )
    pipe.fit(X[train_idx], y[train_idx])
    prob = pipe.predict_proba(X[test_idx])[:, 1]
    pred = (prob >= 0.5).astype(np.int64)
    clf = pipe.named_steps["clf"]
    scaler = pipe.named_steps["scaler"]
    coef_scaled = np.asarray(clf.coef_[0], dtype=np.float64)
    scale = np.asarray(getattr(scaler, "scale_", np.ones_like(coef_scaled)), dtype=np.float64)
    coef_unscaled = coef_scaled / np.maximum(scale, 1e-12)
    out = {
        "roc_auc": float(roc_auc_score(y[test_idx], prob)),
        "avg_precision": float(average_precision_score(y[test_idx], prob)),
        "balanced_acc": float(balanced_accuracy_score(y[test_idx], pred)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "pos_train": int(y[train_idx].sum()),
        "pos_test": int(y[test_idx].sum()),
        "coef_shared_coords": coef_unscaled.astype(np.float32),
        "intercept": float(clf.intercept_[0]),
    }
    return out


def _orthonormalize_columns(A: np.ndarray, *, tol: float = 1e-8) -> np.ndarray:
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2 or A.shape[1] == 0:
        return np.zeros((int(A.shape[0]), 0), dtype=np.float32)
    Q, R = np.linalg.qr(A)
    keep = np.abs(np.diag(R)) > float(tol)
    if not np.any(keep):
        return np.zeros((int(A.shape[0]), 0), dtype=np.float32)
    return np.asarray(Q[:, keep], dtype=np.float32)


def _shared_complement_coeffs(C_fmt: np.ndarray, k: int) -> np.ndarray:
    if int(C_fmt.shape[1]) == 0:
        return np.eye(int(k), dtype=np.float32)
    U, _, _ = np.linalg.svd(C_fmt.astype(np.float64), full_matrices=True)
    C_res = np.asarray(U[:, int(C_fmt.shape[1]) : int(k)], dtype=np.float32)
    return C_res


def _random_shared_partition(k: int, k_fmt: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    A = rng.standard_normal((int(k), int(k)), dtype=np.float32)
    Q, _ = np.linalg.qr(A)
    return np.asarray(Q[:, : int(k_fmt)], dtype=np.float32), np.asarray(Q[:, int(k_fmt) :], dtype=np.float32)


def _load_shared_basis(path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    arrs = np.load(os.path.expanduser(path))
    if "Q_shared" in arrs.files:
        Q = np.asarray(arrs["Q_shared"], dtype=np.float32)
    elif "Q" in arrs.files:
        Q = np.asarray(arrs["Q"], dtype=np.float32)
    else:
        raise KeyError(f"No Q_shared/Q in {path}")
    meta: Dict[str, Any] = {}
    if "basis_meta" in arrs.files:
        try:
            meta["basis_meta"] = json.loads(str(arrs["basis_meta"].item()))
        except Exception:
            meta["basis_meta"] = str(arrs["basis_meta"])
    if "tasks" in arrs.files:
        try:
            meta["tasks"] = [str(x) for x in arrs["tasks"].tolist()]
        except Exception:
            pass
    return Q, meta


def _collect_probe_dataset(
    *,
    model,
    tok,
    tasks_probe: List[str],
    layer: int,
    seed: int,
    n_probe_prompts: int,
    template_seed: int,
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
    batch_size: int,
    max_prompt_len: int,
    probe_max_new_tokens: int,
    per_task_max_states: int,
) -> Tuple[Dict[str, ProbeRecords], Dict[str, Any]]:
    sub_by, _eval_dummy, meta_by = load_selected_tasks(
        tasks=tasks_probe,
        n_subspace=max(1, int(n_probe_prompts)),
        n_eval=1,
        seed=int(seed),
        template_randomization=bool(template_randomization),
        template_seed=int(template_seed),
        shuffle_choices=bool(shuffle_choices),
        add_answer_prefix=bool(add_answer_prefix),
        answer_prefix=str(answer_prefix),
    )
    layers_mods, _ = EP.get_model_layers(model)
    collector = DecodeTokenProbeCollector(int(layer))
    handle = layers_mods[int(layer)].register_forward_hook(collector.make_hook())
    prompt_base = 0
    records_by_task: Dict[str, ProbeRecords] = {}
    try:
        for task, exs in sub_by.items():
            collector.set_current_task(task)
            prompts = [ex.prompt for ex in exs]
            prompt_base = _collect_decode_token_records(
                model=model,
                tok=tok,
                prompts=prompts,
                collector=collector,
                batch_size=int(batch_size),
                max_new_tokens=int(probe_max_new_tokens),
                max_prompt_len=int(max_prompt_len),
                prompt_base=int(prompt_base),
                decoding="greedy",
                temperature=1.0,
                top_p=1.0,
                top_k=0,
            )
            rec = collector.get(task)
            rec = _subsample_task_records(rec, int(per_task_max_states), seed=EP.stable_int_seed(seed, task, "probe"))
            records_by_task[task] = rec
            print(f"[ProbeCollect] task={task} states={rec.states.shape[0]}")
    finally:
        try:
            handle.remove()
        except Exception:
            pass
    return records_by_task, meta_by


def _fit_probe_directions(
    *,
    tok,
    records_by_task: Dict[str, ProbeRecords],
    Q_shared: np.ndarray,
    probe_tags: List[str],
    probe_min_pos: int,
    seed: int,
    test_size: float,
) -> Dict[str, Any]:
    Z_list: List[np.ndarray] = []
    pid_list: List[np.ndarray] = []
    tag_rows: List[Dict[str, int]] = []
    task_counts: Dict[str, int] = {}

    for task, rec in records_by_task.items():
        if rec.states.size == 0:
            continue
        task_counts[task] = int(rec.states.shape[0])
        Z_list.append(np.asarray(rec.states, dtype=np.float32) @ np.asarray(Q_shared, dtype=np.float32))
        pid_list.append(np.asarray(rec.prompt_ids, dtype=np.int64))
        for tid in rec.token_ids.tolist():
            raw = _safe_convert_id_to_token(tok, int(tid))
            dec = _safe_decode_one(tok, int(tid))
            tag_rows.append(_tag_targets(_tags_for_token(raw, dec)))

    if not Z_list:
        raise RuntimeError("No probe samples collected.")

    Z = np.concatenate(Z_list, axis=0).astype(np.float32, copy=False)
    groups = np.concatenate(pid_list, axis=0).astype(np.int64, copy=False)
    probe_rows: List[Dict[str, Any]] = []
    coef_cols: List[np.ndarray] = []
    selected_tags: List[str] = []
    skipped: Dict[str, str] = {}

    for tag in probe_tags:
        y = np.array([int(r.get(tag, 0)) for r in tag_rows], dtype=np.int64)
        n_total = int(y.shape[0])
        n_pos = int(y.sum())
        n_neg = int(n_total - n_pos)
        if n_pos < int(probe_min_pos) or n_neg < int(probe_min_pos):
            skipped[tag] = f"Too few positives/negatives: pos={n_pos} neg={n_neg}"
            continue
        train_idx, test_idx, split_mode = _split_indices(y, groups, seed=EP.stable_int_seed(seed, tag), test_size=float(test_size))
        fit = _fit_eval_binary_probe(Z, y, train_idx, test_idx)
        coef = np.asarray(fit["coef_shared_coords"], dtype=np.float32)
        if float(np.linalg.norm(coef)) > 1e-8:
            coef_cols.append((coef / np.linalg.norm(coef)).astype(np.float32))
            selected_tags.append(tag)
        probe_rows.append(
            {
                "tag": tag,
                "n_total": n_total,
                "n_pos": n_pos,
                "split_mode": split_mode,
                "roc_auc": float(fit["roc_auc"]),
                "avg_precision": float(fit["avg_precision"]),
                "balanced_acc": float(fit["balanced_acc"]),
                "coef_shared_coords": coef.astype(np.float32),
                "intercept": float(fit["intercept"]),
            }
        )

    if not coef_cols:
        raise RuntimeError("No valid probe directions selected.")
    W = np.stack(coef_cols, axis=1).astype(np.float32, copy=False)
    C_fmt = _orthonormalize_columns(W)
    return {
        "probe_rows": probe_rows,
        "skipped": skipped,
        "selected_tags": selected_tags,
        "task_counts": task_counts,
        "C_fmt": C_fmt,
        "n_probe_samples": int(Z.shape[0]),
    }


def _eval_conditions(
    *,
    model,
    tok,
    tasks_eval: List[str],
    layer: int,
    conditions: Dict[str, Optional[np.ndarray]],
    eval_n: int,
    seed: int,
    template_seed: int,
    template_randomization: bool,
    shuffle_choices: bool,
    add_answer_prefix: bool,
    answer_prefix: str,
    fc_answer_prefix: str,
    fc_prefix_mode: str,
    batch_size: int,
    max_prompt_len: int,
    bootstrap_iters: int,
    perm_iters: int,
    alpha: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    _sub_dummy, eval_by, meta_by = load_selected_tasks(
        tasks=tasks_eval,
        n_subspace=1,
        n_eval=max(1, int(eval_n)),
        seed=int(seed),
        template_randomization=bool(template_randomization),
        template_seed=int(template_seed),
        shuffle_choices=bool(shuffle_choices),
        add_answer_prefix=bool(add_answer_prefix),
        answer_prefix=str(answer_prefix),
    )

    eval_results: Dict[str, Any] = {}
    pooled_corr: Dict[str, List[np.ndarray]] = {name: [] for name in conditions.keys()}

    for task in tasks_eval:
        examples = eval_by[task]
        per_task: Dict[str, Any] = {
            "n": int(len(examples)),
            "by_condition": {},
            "paired_vs_baseline": {},
        }
        corr_by_cond: Dict[str, np.ndarray] = {}
        for name, basis in conditions.items():
            out_fc = EP.forced_choice_logprob_eval(
                model,
                tok,
                examples,
                task,
                layer_indices=[int(layer)],
                basis_np=basis,
                alpha=(0.0 if name == "baseline" else 1.0),
                batch_size=int(batch_size),
                max_prompt_len=int(max_prompt_len),
                warmup_token_ids=None,
                answer_prefix=str(fc_answer_prefix),
                prefix_mode=str(fc_prefix_mode),
                save_scores=False,
            )
            corr = np.asarray(out_fc["correct"], dtype=np.float32)
            corr_by_cond[name] = corr
            pooled_corr[name].append(corr)
            acc, lo, hi = EP.bootstrap_ci_mean(
                corr,
                iters=int(bootstrap_iters),
                alpha=float(alpha),
                seed=int(seed) + 11,
            )
            per_task["by_condition"][name] = {
                "acc": float(out_fc["acc"]),
                "acc_ci": {"mean": float(acc), "lo": float(lo), "hi": float(hi)},
                "hook_stats": out_fc.get("hook_stats", {}),
                "metrics_summary": out_fc.get("metrics_summary", {}),
            }
            del out_fc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        base_corr = corr_by_cond["baseline"]
        for name in conditions.keys():
            if name == "baseline":
                continue
            per_task["paired_vs_baseline"][name] = EP.summarize_paired(
                base_corr,
                corr_by_cond[name],
                label=name,
                bootstrap_iters=int(bootstrap_iters),
                perm_iters=int(perm_iters),
                alpha=float(alpha),
                seed=int(seed) + 999,
            )
        eval_results[task] = per_task

    pooled_stats: Dict[str, Any] = {}
    base_all = np.concatenate(pooled_corr["baseline"], axis=0).astype(np.float32)
    pooled_stats["baseline"] = {}
    acc, lo, hi = EP.bootstrap_ci_mean(
        base_all,
        iters=int(bootstrap_iters),
        alpha=float(alpha),
        seed=int(seed) + 777,
    )
    pooled_stats["baseline"]["acc_ci"] = {"mean": float(acc), "lo": float(lo), "hi": float(hi)}
    pooled_stats["baseline"]["n"] = int(base_all.shape[0])
    for name in conditions.keys():
        if name == "baseline":
            continue
        arr = np.concatenate(pooled_corr[name], axis=0).astype(np.float32)
        acc, lo, hi = EP.bootstrap_ci_mean(
            arr,
            iters=int(bootstrap_iters),
            alpha=float(alpha),
            seed=int(seed) + 778,
        )
        pooled_stats[name] = {
            "acc_ci": {"mean": float(acc), "lo": float(lo), "hi": float(hi)},
            "paired_vs_baseline": EP.summarize_paired(
                base_all,
                arr,
                label=name,
                bootstrap_iters=int(bootstrap_iters),
                perm_iters=int(perm_iters),
                alpha=float(alpha),
                seed=int(seed) + 1001,
            ),
        }
    return eval_results, {"meta_by_task": meta_by, "pooled": pooled_stats}


def _drop_share(full_stat: Dict[str, Any], part_stat: Dict[str, Any]) -> float:
    full = -float(full_stat.get("mean_diff", 0.0))
    part = -float(part_stat.get("mean_diff", 0.0))
    if full <= 1e-9:
        return float("nan")
    return float(part / full)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=0.0)
    ap.add_argument("--max_memory_map", type=str, default="")
    ap.add_argument("--cpu_offload_gb", type=float, default=0.0)
    ap.add_argument("--layer", type=int, required=True)

    ap.add_argument("--tasks_probe", type=str, default="")
    ap.add_argument("--tasks_eval", type=str, required=True)
    ap.add_argument("--n_probe_prompts", type=int, default=128)
    ap.add_argument("--probe_max_new_tokens", type=int, default=32)
    ap.add_argument("--probe_batch_size", type=int, default=4)
    ap.add_argument("--per_task_max_probe_states", type=int, default=4000)
    ap.add_argument("--probe_tags", type=str, default="answer_readout,option_letter,newline")
    ap.add_argument("--probe_min_pos", type=int, default=40)
    ap.add_argument("--probe_test_size", type=float, default=0.2)

    ap.add_argument("--eval_n", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=1)
    ap.add_argument("--max_prompt_len", type=int, default=2048)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=0, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=0, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])

    ap.add_argument("--bootstrap_iters", type=int, default=1000)
    ap.add_argument("--perm_iters", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/a5_probe_split")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    EP.set_global_seed(int(args.seed))
    Q_shared, basis_meta = _load_shared_basis(str(args.basis_npz))
    D, k = int(Q_shared.shape[0]), int(Q_shared.shape[1])

    tasks_probe = _split_csv(args.tasks_probe)
    if not tasks_probe:
        tasks_probe = [str(x) for x in basis_meta.get("tasks", [])]
    tasks_eval = _split_csv(args.tasks_eval)
    probe_tags = _split_csv(args.probe_tags)
    if not tasks_probe:
        raise ValueError("No tasks_probe provided and no tasks found in basis_npz.")
    if not tasks_eval:
        raise ValueError("Empty --tasks_eval")
    if not probe_tags:
        raise ValueError("Empty --probe_tags")

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + str(args.tag).strip()) if str(args.tag).strip() else ""
    base_name = f"exp_A5_probe_split_layer{int(args.layer)}{tag}"
    json_path = os.path.join(out_dir, base_name + ".json")
    md_path = os.path.join(out_dir, base_name + ".md")
    bases_npz_path = os.path.join(out_dir, base_name + "_bases.npz")

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

    records_by_task, probe_meta = _collect_probe_dataset(
        model=model,
        tok=tok,
        tasks_probe=tasks_probe,
        layer=int(args.layer),
        seed=int(args.seed),
        n_probe_prompts=int(args.n_probe_prompts),
        template_seed=int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
        batch_size=int(args.probe_batch_size),
        max_prompt_len=int(args.max_prompt_len),
        probe_max_new_tokens=int(args.probe_max_new_tokens),
        per_task_max_states=int(args.per_task_max_probe_states),
    )

    probe_fit = _fit_probe_directions(
        tok=tok,
        records_by_task=records_by_task,
        Q_shared=Q_shared,
        probe_tags=probe_tags,
        probe_min_pos=int(args.probe_min_pos),
        seed=int(args.seed),
        test_size=float(args.probe_test_size),
    )
    C_fmt = np.asarray(probe_fit["C_fmt"], dtype=np.float32)
    k_fmt = int(C_fmt.shape[1])
    if k_fmt <= 0 or k_fmt >= k:
        raise RuntimeError(f"Bad decomposition rank: k_fmt={k_fmt}, k_shared={k}")
    C_resid = _shared_complement_coeffs(C_fmt, k)
    Q_fmt = np.asarray(Q_shared @ C_fmt, dtype=np.float32)
    Q_resid = np.asarray(Q_shared @ C_resid, dtype=np.float32)

    C_rand_fmt, C_rand_resid = _random_shared_partition(k, k_fmt, seed=EP.stable_int_seed(args.seed, "a5_rand_partition"))
    Q_rand_fmt = np.asarray(Q_shared @ C_rand_fmt, dtype=np.float32)
    Q_rand_resid = np.asarray(Q_shared @ C_rand_resid, dtype=np.float32)

    conditions: Dict[str, Optional[np.ndarray]] = {
        "baseline": None,
        "shared_full": Q_shared,
        "fmt_only": Q_fmt,
        "resid_only": Q_resid,
        "rand_fmt_shared": Q_rand_fmt,
        "rand_resid_shared": Q_rand_resid,
    }

    eval_results, eval_meta = _eval_conditions(
        model=model,
        tok=tok,
        tasks_eval=tasks_eval,
        layer=int(args.layer),
        conditions=conditions,
        eval_n=int(args.eval_n),
        seed=int(args.seed),
        template_seed=int(args.template_seed),
        template_randomization=bool(args.template_randomization),
        shuffle_choices=bool(args.shuffle_choices),
        add_answer_prefix=bool(args.add_answer_prefix),
        answer_prefix=str(args.answer_prefix),
        fc_answer_prefix=str(args.fc_answer_prefix),
        fc_prefix_mode=str(args.fc_prefix_mode),
        batch_size=int(args.eval_batch_size),
        max_prompt_len=int(args.max_prompt_len),
        bootstrap_iters=int(args.bootstrap_iters),
        perm_iters=int(args.perm_iters),
        alpha=float(args.alpha),
    )

    pooled = eval_meta["pooled"]
    pooled_shares = {
        "fmt_vs_full_drop_share": _drop_share(pooled["shared_full"]["paired_vs_baseline"], pooled["fmt_only"]["paired_vs_baseline"]),
        "resid_vs_full_drop_share": _drop_share(pooled["shared_full"]["paired_vs_baseline"], pooled["resid_only"]["paired_vs_baseline"]),
        "rand_fmt_vs_full_drop_share": _drop_share(pooled["shared_full"]["paired_vs_baseline"], pooled["rand_fmt_shared"]["paired_vs_baseline"]),
        "rand_resid_vs_full_drop_share": _drop_share(pooled["shared_full"]["paired_vs_baseline"], pooled["rand_resid_shared"]["paired_vs_baseline"]),
    }

    header = ["Task", "n", "Baseline", "SharedFull", "ΔFull", "FmtOnly", "ΔFmt", "ResidOnly", "ΔResid", "RandFmt", "ΔRandFmt", "RandResid", "ΔRandResid"]
    rows: List[List[str]] = []
    for task in tasks_eval:
        pt = eval_results[task]
        rows.append(
            [
                task,
                str(pt["n"]),
                _fmt_acc(pt["by_condition"]["baseline"]["acc_ci"]["mean"], pt["by_condition"]["baseline"]["acc_ci"]["lo"], pt["by_condition"]["baseline"]["acc_ci"]["hi"]),
                _fmt_acc(pt["by_condition"]["shared_full"]["acc_ci"]["mean"], pt["by_condition"]["shared_full"]["acc_ci"]["lo"], pt["by_condition"]["shared_full"]["acc_ci"]["hi"]),
                _fmt_diff(pt["paired_vs_baseline"]["shared_full"]),
                _fmt_acc(pt["by_condition"]["fmt_only"]["acc_ci"]["mean"], pt["by_condition"]["fmt_only"]["acc_ci"]["lo"], pt["by_condition"]["fmt_only"]["acc_ci"]["hi"]),
                _fmt_diff(pt["paired_vs_baseline"]["fmt_only"]),
                _fmt_acc(pt["by_condition"]["resid_only"]["acc_ci"]["mean"], pt["by_condition"]["resid_only"]["acc_ci"]["lo"], pt["by_condition"]["resid_only"]["acc_ci"]["hi"]),
                _fmt_diff(pt["paired_vs_baseline"]["resid_only"]),
                _fmt_acc(pt["by_condition"]["rand_fmt_shared"]["acc_ci"]["mean"], pt["by_condition"]["rand_fmt_shared"]["acc_ci"]["lo"], pt["by_condition"]["rand_fmt_shared"]["acc_ci"]["hi"]),
                _fmt_diff(pt["paired_vs_baseline"]["rand_fmt_shared"]),
                _fmt_acc(pt["by_condition"]["rand_resid_shared"]["acc_ci"]["mean"], pt["by_condition"]["rand_resid_shared"]["acc_ci"]["lo"], pt["by_condition"]["rand_resid_shared"]["acc_ci"]["hi"]),
                _fmt_diff(pt["paired_vs_baseline"]["rand_resid_shared"]),
            ]
        )

    pooled_row = [
        "Pooled",
        str(int(pooled["baseline"]["n"])),
        _fmt_acc(pooled["baseline"]["acc_ci"]["mean"], pooled["baseline"]["acc_ci"]["lo"], pooled["baseline"]["acc_ci"]["hi"]),
        _fmt_acc(pooled["shared_full"]["acc_ci"]["mean"], pooled["shared_full"]["acc_ci"]["lo"], pooled["shared_full"]["acc_ci"]["hi"]),
        _fmt_diff(pooled["shared_full"]["paired_vs_baseline"]),
        _fmt_acc(pooled["fmt_only"]["acc_ci"]["mean"], pooled["fmt_only"]["acc_ci"]["lo"], pooled["fmt_only"]["acc_ci"]["hi"]),
        _fmt_diff(pooled["fmt_only"]["paired_vs_baseline"]),
        _fmt_acc(pooled["resid_only"]["acc_ci"]["mean"], pooled["resid_only"]["acc_ci"]["lo"], pooled["resid_only"]["acc_ci"]["hi"]),
        _fmt_diff(pooled["resid_only"]["paired_vs_baseline"]),
        _fmt_acc(pooled["rand_fmt_shared"]["acc_ci"]["mean"], pooled["rand_fmt_shared"]["acc_ci"]["lo"], pooled["rand_fmt_shared"]["acc_ci"]["hi"]),
        _fmt_diff(pooled["rand_fmt_shared"]["paired_vs_baseline"]),
        _fmt_acc(pooled["rand_resid_shared"]["acc_ci"]["mean"], pooled["rand_resid_shared"]["acc_ci"]["lo"], pooled["rand_resid_shared"]["acc_ci"]["hi"]),
        _fmt_diff(pooled["rand_resid_shared"]["paired_vs_baseline"]),
    ]
    rows.append(pooled_row)

    probe_table = []
    for r in probe_fit["probe_rows"]:
        probe_table.append(
            [
                r["tag"],
                str(r["n_pos"]),
                r["split_mode"],
                f"{r['roc_auc']:.3f}",
                f"{r['avg_precision']:.3f}",
                f"{r['balanced_acc']:.3f}",
            ]
        )

    np.savez(
        bases_npz_path,
        Q_shared=Q_shared.astype(np.float32),
        Q_fmt=Q_fmt.astype(np.float32),
        Q_resid=Q_resid.astype(np.float32),
        Q_rand_fmt=Q_rand_fmt.astype(np.float32),
        Q_rand_resid=Q_rand_resid.astype(np.float32),
    )

    results = {
        "config": {
            "basis_npz": str(args.basis_npz),
            "model": str(args.model),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "device_map": str(args.device_map),
            "max_memory_per_gpu_gb": float(args.max_memory_per_gpu_gb),
            "max_memory_map": str(args.max_memory_map),
            "cpu_offload_gb": float(args.cpu_offload_gb),
            "layer": int(args.layer),
            "tasks_probe": tasks_probe,
            "tasks_eval": tasks_eval,
            "n_probe_prompts": int(args.n_probe_prompts),
            "probe_max_new_tokens": int(args.probe_max_new_tokens),
            "probe_batch_size": int(args.probe_batch_size),
            "per_task_max_probe_states": int(args.per_task_max_probe_states),
            "probe_tags": probe_tags,
            "probe_min_pos": int(args.probe_min_pos),
            "probe_test_size": float(args.probe_test_size),
            "eval_n": int(args.eval_n),
            "eval_batch_size": int(args.eval_batch_size),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
            "fc_answer_prefix": str(args.fc_answer_prefix),
            "fc_prefix_mode": str(args.fc_prefix_mode),
            "bootstrap_iters": int(args.bootstrap_iters),
            "perm_iters": int(args.perm_iters),
            "alpha": float(args.alpha),
        },
        "basis_meta": basis_meta,
        "probe_meta": probe_meta,
        "probe_fit": {
            "n_probe_samples": int(probe_fit["n_probe_samples"]),
            "task_counts": probe_fit["task_counts"],
            "selected_tags": probe_fit["selected_tags"],
            "skipped": probe_fit["skipped"],
            "probe_rows": probe_fit["probe_rows"],
        },
        "decomposition": {
            "D": int(D),
            "k_shared": int(k),
            "k_fmt": int(k_fmt),
            "k_resid": int(Q_resid.shape[1]),
            "max_overlap_fmt_vs_resid": float(np.max(np.abs(Q_fmt.T @ Q_resid))) if Q_resid.size else 0.0,
            "pooled_drop_shares": pooled_shares,
            "saved_bases_npz": os.path.relpath(bases_npz_path, ROOT_DIR),
        },
        "eval": eval_results,
        "eval_meta": eval_meta,
    }

    md_lines: List[str] = []
    md_lines.append("# Exp-A5: Probe-derived format/readout split of Q_shared")
    md_lines.append("")
    md_lines.append(f"Basis: `{args.basis_npz}`")
    md_lines.append("")
    md_lines.append("## Probe fit")
    md_lines.append(_md_table(probe_table, ["tag", "n_pos", "split", "ROC-AUC", "AP", "BalAcc"]))
    md_lines.append("")
    md_lines.append("Skipped probe tags:")
    md_lines.append("```json")
    md_lines.append(json.dumps(probe_fit["skipped"], ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Decomposition")
    md_lines.append("```json")
    md_lines.append(json.dumps(results["decomposition"], ensure_ascii=False, indent=2, default=_json_default))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## Causal eval")
    md_lines.append(_md_table(rows, header))
    md_lines.append("")
    md_lines.append(f"JSON: `{json_path}`")
    md_lines.append("")

    _atomic_json_dump(results, json_path)
    _atomic_text_dump("\n".join(md_lines), md_path)
    print(f"[Saved] {json_path}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()
