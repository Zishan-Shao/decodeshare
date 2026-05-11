# -*- coding: utf-8 -*-
"""
exp_1b_linear_probe_decode_tags.py

Lightweight linear-probe follow-up to the decode-shared unembedding experiment.

Given a saved shared basis Q (e.g. from exp_1_logit_lens_vocab_signature.py), this
script collects decode-time hidden states at a target layer, projects them into the
shared basis coordinates z = h^T Q, and trains simple held-out logistic probes for
reviewer-friendly token families:

  - option_letter
  - yes_no
  - answer_marker
  - reasoning_marker
  - digit
  - newline
  - answer_readout (union of option_letter / yes_no / answer_marker)

It also reports the same probes in a same-width random subspace baseline.

Typical run
-----------
CUDA_VISIBLE_DEVICES=7 python rebuttal/mechanism/exp_1b_linear_probe_decode_tags.py \
  --basis_npz results/rebuttal_mechanism/logit_lens_l10/basis_layer10_tseed1234.npz \
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp16 \
  --layer 10 \
  --n_prompts 128 --max_new_tokens 32 --per_task_max_states 4000 \
  --template_seed 1234 --template_randomization 1 --shuffle_choices 1 \
  --add_answer_prefix 1 --answer_prefix $'\\nFinal answer:' \
  --out_dir results/rebuttal_mechanism/linear_probe_l10
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


_PUNCT_ASCII = set(string.punctuation)
_PUNCT_CJK = set("，。！？；：、…·“”‘’—–")
_BRACKETS = set("()[]{}<>（）【】《》")
_OPTION_LETTERS = set("ABCDE")
_YES_NO = {"yes", "no"}
_PROBE_TAGS = [
    "answer_readout",
    "option_letter",
    "yes_no",
    "answer_marker",
    "reasoning_marker",
    "digit",
    "newline",
]


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
    if re.search(r"(?i)\bfinal\s*answer\b", s) or re.search(r"(?i)\banswer\s*:", s):
        tags.append("answer_marker")
    if re.search(r"(?i)\btherefore\b|\bthus\b|\bhence\b|\bbecause\b", s_strip):
        tags.append("reasoning_marker")
    if s_strip and all((ch in _PUNCT_ASCII) or (ch in _PUNCT_CJK) for ch in s_strip):
        tags.append("punct")
    return tags


def _tag_targets(tags: List[str]) -> Dict[str, int]:
    tag_set = set(tags)
    out = {t: int(t in tag_set) for t in _PROBE_TAGS}
    out["answer_readout"] = int(
        ("option_letter" in tag_set) or ("yes_no" in tag_set) or ("answer_marker" in tag_set)
    )
    return out


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

        for _step in range(int(max_new_tokens)):
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


def _load_basis(path: str, k_features: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    z = np.load(os.path.expanduser(path), allow_pickle=False)
    Q = np.asarray(z["Q"], dtype=np.float32)
    if int(k_features) > 0 and int(k_features) < int(Q.shape[1]):
        Q = Q[:, : int(k_features)]

    meta: Dict[str, Any] = {}
    for key in ["basis_meta", "meta"]:
        if key in z.files:
            try:
                meta[key] = json.loads(str(z[key].item()))
            except Exception:
                meta[key] = str(z[key])
    if "tasks" in z.files:
        meta["tasks"] = [str(x) for x in z["tasks"].tolist()]
    return Q, meta


def _prepare_probe_dataset(
    *,
    tok,
    records_by_task: Dict[str, ProbeRecords],
    Q_shared: np.ndarray,
    seed: int,
) -> Dict[str, Any]:
    H_list: List[np.ndarray] = []
    Zs_list: List[np.ndarray] = []
    Zr_list: List[np.ndarray] = []
    task_list: List[np.ndarray] = []
    pid_list: List[np.ndarray] = []
    tid_list: List[np.ndarray] = []
    raw_list: List[str] = []
    dec_list: List[str] = []
    tags_list: List[List[str]] = []

    dim, k = int(Q_shared.shape[0]), int(Q_shared.shape[1])
    Q_rand = EP.random_orthonormal_basis_np(dim, k, seed=EP.stable_int_seed(seed, "linear_probe_rand", k))

    for task, rec in records_by_task.items():
        if rec.states.size == 0:
            continue
        H = np.asarray(rec.states, dtype=np.float32)
        H_list.append(H)
        Zs_list.append(H @ Q_shared)
        Zr_list.append(H @ Q_rand)
        task_list.append(np.asarray(rec.tasks))
        pid_list.append(np.asarray(rec.prompt_ids))
        tid_list.append(np.asarray(rec.token_ids))
        for tid in rec.token_ids.tolist():
            raw = _safe_convert_id_to_token(tok, int(tid))
            dec = _safe_decode_one(tok, int(tid))
            raw_list.append(raw)
            dec_list.append(dec)
            tags_list.append(_tags_for_token(raw, dec))

    if not H_list:
        raise RuntimeError("No probe records collected.")

    Z_shared = np.concatenate(Zs_list, axis=0).astype(np.float32, copy=False)
    Z_rand = np.concatenate(Zr_list, axis=0).astype(np.float32, copy=False)
    prompt_ids = np.concatenate(pid_list, axis=0).astype(np.int64, copy=False)
    tasks = np.concatenate(task_list, axis=0)
    token_ids = np.concatenate(tid_list, axis=0).astype(np.int64, copy=False)

    y_by_tag: Dict[str, np.ndarray] = {}
    for tag in _PROBE_TAGS:
        y = np.array([_tag_targets(tags).get(tag, 0) for tags in tags_list], dtype=np.int64)
        y_by_tag[tag] = y

    return {
        "Z_shared": Z_shared,
        "Z_rand": Z_rand,
        "prompt_ids": prompt_ids,
        "tasks": tasks,
        "token_ids": token_ids,
        "raw_tokens": raw_list,
        "decoded_tokens": dec_list,
        "tags": tags_list,
        "y_by_tag": y_by_tag,
        "k_features": int(k),
    }


def _infer_hidden_dim(model) -> int:
    if hasattr(model, "config") and getattr(model.config, "hidden_size", None) is not None:
        return int(model.config.hidden_size)
    if hasattr(model, "lm_head") and getattr(model.lm_head, "weight", None) is not None:
        return int(model.lm_head.weight.shape[1])
    emb = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if emb is not None and getattr(emb, "weight", None) is not None:
        return int(emb.weight.shape[1])
    raise RuntimeError("Could not infer model hidden dimension.")


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
    out = {
        "roc_auc": float(roc_auc_score(y[test_idx], prob)),
        "avg_precision": float(average_precision_score(y[test_idx], prob)),
        "balanced_acc": float(balanced_accuracy_score(y[test_idx], pred)),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "pos_train": int(y[train_idx].sum()),
        "pos_test": int(y[test_idx].sum()),
    }
    return out


def _render_md(config: Dict[str, Any], dataset_summary: Dict[str, Any], probe_rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Exp-1b: Linear Probe On Decode-Shared Coordinates")
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
    lines.append("| tag | n_total | n_pos | split | ROC-AUC shared | ROC-AUC rand | AP shared | AP rand | BalAcc shared | BalAcc rand |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for row in probe_rows:
        lines.append(
            "| {tag} | {n_total} | {n_pos} | {split_mode} | {roc_auc_shared:.3f} | {roc_auc_rand:.3f} | {ap_shared:.3f} | {ap_rand:.3f} | {bal_acc_shared:.3f} | {bal_acc_rand:.3f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis_npz", type=str, required=True)
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])
    ap.add_argument("--device_map", type=str, default="")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=0.0)
    ap.add_argument("--cpu_offload_gb", type=float, default=0.0)

    ap.add_argument("--tasks", type=str, default="")
    ap.add_argument("--n_prompts", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--per_task_max_states", type=int, default=4000)
    ap.add_argument("--k_features", type=int, default=0, help="0 means use all columns from Q.")
    ap.add_argument("--min_pos", type=int, default=40)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/linear_probe")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    EP.set_global_seed(int(args.seed))
    Q_shared, basis_meta = _load_basis(str(args.basis_npz), int(args.k_features))
    hidden_dim = int(Q_shared.shape[0])

    tasks = _split_csv(args.tasks)
    if not tasks:
        tasks = [str(x) for x in basis_meta.get("tasks", [])]
    if not tasks:
        raise ValueError("No tasks provided and no tasks found in basis_npz.")

    out_dir = os.path.expanduser(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    tag = ("_" + args.tag.strip()) if args.tag.strip() else ""

    model, tok = EP.load_model_and_tokenizer(
        args.model,
        args.device,
        args.dtype,
        trust_remote_code=bool(args.trust_remote_code),
        device_map=(None if not str(args.device_map).strip() else str(args.device_map)),
        max_memory_per_gpu_gb=float(args.max_memory_per_gpu_gb),
        cpu_offload_gb=float(args.cpu_offload_gb),
    )
    layers, _ = EP.get_model_layers(model)
    if int(args.layer) < 0 or int(args.layer) >= len(layers):
        raise ValueError(f"layer={args.layer} out of range for model with {len(layers)} layers")
    model_hidden_dim = _infer_hidden_dim(model)
    if hidden_dim != int(model_hidden_dim):
        raise ValueError(f"Basis hidden dim {hidden_dim} does not match model hidden dim {model_hidden_dim}")

    sub_by, _eval_by_dummy, meta_by = load_selected_tasks(
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
                decoding=str(args.decoding),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
            )
            rec = collector.get(task)
            rec = _subsample_task_records(rec, int(args.per_task_max_states), seed=EP.stable_int_seed(args.seed, task, "probe"))
            records_by_task[task] = rec
            print(f"[Collected] task={task} states={rec.states.shape[0]}")
    finally:
        try:
            handle.remove()
        except Exception:
            pass

    data = _prepare_probe_dataset(tok=tok, records_by_task=records_by_task, Q_shared=Q_shared, seed=int(args.seed))
    Z_shared = data["Z_shared"]
    Z_rand = data["Z_rand"]
    prompt_ids = data["prompt_ids"]
    y_by_tag = data["y_by_tag"]

    probe_rows: List[Dict[str, Any]] = []
    skipped: Dict[str, str] = {}
    for tag in _PROBE_TAGS:
        y = np.asarray(y_by_tag[tag], dtype=np.int64)
        n_pos = int(y.sum())
        n_total = int(y.shape[0])
        n_neg = int(n_total - n_pos)
        if n_pos < int(args.min_pos) or n_neg < int(args.min_pos):
            skipped[tag] = f"Too few positives/negatives: pos={n_pos} neg={n_neg}"
            continue

        train_idx, test_idx, split_mode = _split_indices(y, prompt_ids, seed=EP.stable_int_seed(args.seed, tag), test_size=float(args.test_size))
        shared = _fit_eval_binary_probe(Z_shared, y, train_idx, test_idx)
        rand = _fit_eval_binary_probe(Z_rand, y, train_idx, test_idx)
        probe_rows.append(
            {
                "tag": tag,
                "n_total": n_total,
                "n_pos": n_pos,
                "split_mode": split_mode,
                "roc_auc_shared": shared["roc_auc"],
                "roc_auc_rand": rand["roc_auc"],
                "ap_shared": shared["avg_precision"],
                "ap_rand": rand["avg_precision"],
                "bal_acc_shared": shared["balanced_acc"],
                "bal_acc_rand": rand["balanced_acc"],
                "shared_detail": shared,
                "rand_detail": rand,
            }
        )

    results = {
        "config": {
            "basis_npz": str(args.basis_npz),
            "model": str(args.model),
            "device": str(args.device),
            "dtype": str(args.dtype),
            "layer": int(args.layer),
            "tasks": tasks,
            "n_prompts": int(args.n_prompts),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
            "batch_size": int(args.batch_size),
            "max_prompt_len": int(args.max_prompt_len),
            "max_new_tokens": int(args.max_new_tokens),
            "per_task_max_states": int(args.per_task_max_states),
            "k_features": int(data["k_features"]),
            "test_size": float(args.test_size),
            "min_pos": int(args.min_pos),
        },
        "basis_meta": basis_meta,
        "dataset": {
            "n_samples": int(Z_shared.shape[0]),
            "k_features": int(data["k_features"]),
            "tasks": {task: int(rec.states.shape[0]) for task, rec in records_by_task.items()},
            "meta_by_task": meta_by,
        },
        "probe_rows": probe_rows,
        "skipped": skipped,
    }

    out_json = os.path.join(out_dir, f"exp_1b_linear_probe_layer{int(args.layer)}{tag}.json")
    out_md = os.path.join(out_dir, f"exp_1b_linear_probe_layer{int(args.layer)}{tag}.md")
    _atomic_json_dump(results, out_json)
    _atomic_text_dump(_render_md(results["config"], results["dataset"], probe_rows), out_md)
    print(f"[Saved] {out_json}")
    print(f"[Saved] {out_md}")


if __name__ == "__main__":
    main()
