# -*- coding: utf-8 -*-
"""
exp_3_module_attribution_attn_vs_mlp_alignment.py

Mechanism experiment M3-b (minimal attention vs MLP attribution via geometry):
  Collect decode-time activations of:
    - Attn_out (layer.self_attn output)
    - MLP_out  (layer.mlp output)
  Build PCA subspaces Q_attn and Q_mlp, then compare alignment with a decode-shared basis Q_shared
  (e.g., from exp_1_logit_lens_vocab_signature.py outputs).

This is a cheap, reviewer-friendly answer to:
  "Is your shared subspace more associated with attention or MLP?"

Typical run
-----------
CUDA_VISIBLE_DEVICES=7 python rebuttal/mechanism/exp_3_module_attribution_attn_vs_mlp_alignment.py \\
  --model meta-llama/Llama-2-7b-chat-hf --device cuda --dtype fp16 \\
  --layer 28 \\
  --basis_npz results/rebuttal_mechanism/logit_lens_l28/basis_layer28_tseed1234.npz --k_shared 32 \\
  --tasks gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq \\
  --n_prompts 64 --template_seed 1234 --seed 42 \\
  --calib_decode_max_new_tokens 128 --per_task_max_states 20000 \\
  --k_module 32 --out_dir results/rebuttal_mechanism/m3_module_align_l28
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import tempfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Repo-local imports
# -----------------------------------------------------------------------------
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
    return os.path.normpath(os.path.join(start_dir, "..", ".."))


ROOT_DIR = _find_repo_root(THIS_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
REASONING_DIR = os.path.join(ROOT_DIR, "reasoning")

for p in [SRC_DIR, REASONING_DIR, ROOT_DIR]:
    if p not in sys.path:
        sys.path.append(p)

try:
    import eval_perf as EP  # reasoning/eval_perf.py
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import `reasoning/eval_perf.py` as module `eval_perf`.") from e

try:
    from benchmark_dataloaders import load_selected_tasks  # src/benchmark_dataloaders.py
except Exception as e:  # pragma: no cover
    raise RuntimeError("Failed to import `src/benchmark_dataloaders.py` as module `benchmark_dataloaders`.") from e


# -----------------------------------------------------------------------------
# IO helpers
# -----------------------------------------------------------------------------
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


def _md_table(rows: List[List[str]], header: List[str]) -> str:
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{float(x):.{nd}f}"


# -----------------------------------------------------------------------------
# Decode-time module collector (seq_len==1 only)
# -----------------------------------------------------------------------------
class DecodeModuleLastTokenCollector:
    def __init__(self, name: str):
        self.name = str(name)
        self._cur_task: Optional[str] = None
        self.capture_enabled: bool = False
        self.active_mask: Optional[torch.Tensor] = None
        self.storage: Dict[str, List[np.ndarray]] = {}

    def set_current_task(self, task: str) -> None:
        self._cur_task = task

    def set_capture(self, enabled: bool, active_mask: Optional[torch.Tensor] = None) -> None:
        self.capture_enabled = bool(enabled)
        self.active_mask = active_mask

    def make_hook(self):
        def _hook(module, inputs, output):
            if (not self.capture_enabled) or (self._cur_task is None):
                return output
            hs = output[0] if isinstance(output, tuple) else output
            if not isinstance(hs, torch.Tensor) or hs.ndim != 3:
                return output
            if hs.shape[1] != 1:
                return output
            x = hs[:, -1, :]
            if self.active_mask is not None:
                m = self.active_mask.bool()
                if m.numel() == x.shape[0]:
                    x = x[m]
            if x.numel() == 0:
                return output
            self.storage.setdefault(self._cur_task, []).append(x.detach().float().cpu().numpy())
            return output

        return _hook

    def get(self, task: str) -> Optional[np.ndarray]:
        chunks = self.storage.get(task, [])
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)


@torch.no_grad()
def collect_decode_module_states(
    model,
    tok,
    prompts: List[str],
    collectors: Sequence[DecodeModuleLastTokenCollector],
    *,
    batch_size: int,
    max_new_tokens: int,
    max_prompt_len: int,
    decoding: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:
    assert decoding in ["greedy", "sample"]
    device = next(model.parameters()).device
    eos = tok.eos_token_id
    model.eval()

    for i in tqdm(range(0, len(prompts), batch_size), desc="CollectDecode(mod)"):
        batch = prompts[i : i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_prompt_len).to(device)
        ids = inputs["input_ids"]
        attn = inputs["attention_mask"]
        B = ids.shape[0]

        unfinished = torch.ones(B, dtype=torch.bool, device=device)

        for c in collectors:
            c.set_capture(False, None)
        past, logits = EP.cache_decode_aligned_boundary(model, ids, attn)

        for _ in range(int(max_new_tokens)):
            next_tok = EP.choose_next_token(
                logits,
                decoding=decoding,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=eos,
                ban_eos=False,
            )
            next_tok = torch.where(unfinished.unsqueeze(-1), next_tok, torch.full_like(next_tok, eos))
            unfinished = unfinished & (next_tok.squeeze(-1) != eos)
            if not bool(unfinished.any().item()):
                break

            attn = torch.cat([attn, torch.ones((B, 1), device=device, dtype=attn.dtype)], dim=1)
            for c in collectors:
                c.set_capture(True, unfinished)
            out = model(input_ids=next_tok, attention_mask=attn, use_cache=True, past_key_values=past)
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        for c in collectors:
            c.set_capture(False, None)


def _load_basis_from_npz(path: str, *, k: int, key: str = "") -> np.ndarray:
    z = np.load(os.path.expanduser(path))
    if key and key in z:
        Q = z[key]
    elif "Q_shared" in z:
        Q = z["Q_shared"]
    elif "Q" in z:
        Q = z["Q"]
    else:
        raise KeyError(f"Could not find basis array in npz. Available keys: {list(z.keys())}")
    Q = np.asarray(Q, dtype=np.float32)
    if int(k) > 0:
        Q = Q[:, : int(k)]
    return EP.orthonormalize_np(Q)


def _sv_cosines(Qa: np.ndarray, Qb: np.ndarray) -> List[float]:
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return [float(x) for x in s.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()

    # Model
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--trust_remote_code", type=int, default=0, choices=[0, 1])

    # Layer + shared basis
    ap.add_argument("--layer", type=int, default=28)
    ap.add_argument("--basis_npz", type=str, required=True, help="npz path with Q or Q_shared (e.g., exp_1 outputs).")
    ap.add_argument("--basis_key", type=str, default="", help="Optional: npz key override (e.g., Q, Q_shared).")
    ap.add_argument("--k_shared", type=int, default=32, help="Use first k columns of Q_shared (0=all).")

    # Data
    ap.add_argument("--tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,aqua,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--n_prompts", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")

    # Decode collection
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--per_task_max_states", type=int, default=20000)

    # Module PCA
    ap.add_argument("--k_module", type=int, default=32, help="PCA subspace dim for module outputs (fixed).")

    # Output
    ap.add_argument("--out_dir", type=str, default="results/rebuttal_mechanism/m3_module_alignment")
    ap.add_argument("--tag", type=str, default="")

    args = ap.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested (--device cuda) but torch.cuda.is_available()==False.\n"
            "This usually means your NVIDIA driver is missing/too old for your installed torch build.\n"
            f"torch={torch.__version__}  torch.version.cuda={getattr(torch.version, 'cuda', None)}"
        )

    tasks = _split_csv(args.tasks)
    if not tasks:
        raise ValueError("Empty --tasks")
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

    Q_shared = _load_basis_from_npz(str(args.basis_npz), k=int(args.k_shared), key=str(args.basis_key).strip())

    sub_by, _eval_by, meta_by = load_selected_tasks(
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
    if int(args.layer) < 0 or int(args.layer) >= len(layers):
        raise ValueError(f"layer={int(args.layer)} out of range: num_layers={len(layers)}")
    layer_mod = layers[int(args.layer)]
    if not hasattr(layer_mod, "self_attn") or not hasattr(layer_mod, "mlp"):
        raise RuntimeError("Target layer does not expose .self_attn and .mlp (this script currently supports LLaMA-like layers).")

    col_attn = DecodeModuleLastTokenCollector("attn_out")
    col_mlp = DecodeModuleLastTokenCollector("mlp_out")
    h_attn = layer_mod.self_attn.register_forward_hook(col_attn.make_hook())
    h_mlp = layer_mod.mlp.register_forward_hook(col_mlp.make_hook())

    attn_by_task: Dict[str, np.ndarray] = {}
    mlp_by_task: Dict[str, np.ndarray] = {}
    try:
        for task, exs in sub_by.items():
            prompts = [ex.prompt for ex in exs]
            if not prompts:
                continue
            col_attn.set_current_task(task)
            col_mlp.set_current_task(task)

            collect_decode_module_states(
                model,
                tok,
                prompts,
                [col_attn, col_mlp],
                batch_size=int(args.batch_size),
                max_new_tokens=int(args.calib_decode_max_new_tokens),
                max_prompt_len=int(args.max_prompt_len),
                decoding=str(args.decoding),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                top_k=int(args.top_k),
            )

            Xa = col_attn.get(task)
            Xm = col_mlp.get(task)
            if Xa is None or Xa.shape[0] == 0 or Xm is None or Xm.shape[0] == 0:
                print(f"[Warn] No states collected for task={task}. Skipping.")
                continue

            Xa = EP.subsample_rows_np(Xa, int(args.per_task_max_states), seed=EP.stable_int_seed(args.seed, task, "attn"))
            Xm = EP.subsample_rows_np(Xm, int(args.per_task_max_states), seed=EP.stable_int_seed(args.seed, task, "mlp"))
            attn_by_task[task] = Xa
            mlp_by_task[task] = Xm
    finally:
        try:
            h_attn.remove()
        except Exception:
            pass
        try:
            h_mlp.remove()
        except Exception:
            pass
        col_attn.set_capture(False, None)
        col_mlp.set_capture(False, None)

    tasks_used = [t for t in tasks if t in attn_by_task and t in mlp_by_task]
    if len(tasks_used) < 2:
        raise RuntimeError(f"Need >=2 tasks with collected states; got {tasks_used}. Try increasing --n_prompts or --calib_decode_max_new_tokens.")

    # Balance across tasks for PCA to avoid overweighting tasks with more decode steps.
    attn_bal, n_min_a = EP.balance_task_states({t: attn_by_task[t] for t in tasks_used}, seed=EP.stable_int_seed(args.seed, "bal_attn"))
    mlp_bal, n_min_m = EP.balance_task_states({t: mlp_by_task[t] for t in tasks_used}, seed=EP.stable_int_seed(args.seed, "bal_mlp"))
    n_min = min(int(n_min_a), int(n_min_m))
    if n_min <= int(args.k_module) + 1:
        raise RuntimeError(
            f"Too few balanced states per task (n_min={n_min}) for k_module={int(args.k_module)}.\n"
            "Increase --n_prompts and/or --calib_decode_max_new_tokens."
        )

    # Cross-task PCA for each module output (fixed k_module)
    task_dict_attn = {t: {0: attn_bal[t]} for t in tasks_used}
    task_dict_mlp = {t: {0: mlp_bal[t]} for t in tasks_used}
    Q_attn_raw, k_a, contrib_a = EP.compute_cross_task_subspace(
        task_dict_attn,
        variance_threshold=0.999,
        min_dim=int(args.k_module),
        max_dim=int(args.k_module),
        return_full_pca=False,
    )
    Q_mlp_raw, k_m, contrib_m = EP.compute_cross_task_subspace(
        task_dict_mlp,
        variance_threshold=0.999,
        min_dim=int(args.k_module),
        max_dim=int(args.k_module),
        return_full_pca=False,
    )
    if Q_attn_raw is None or Q_mlp_raw is None:
        raise RuntimeError("PCA failed for module outputs.")
    if int(k_a) != int(args.k_module) or int(k_m) != int(args.k_module):
        raise RuntimeError(f"Unexpected PCA dims: k_attn={k_a}, k_mlp={k_m}, expected k_module={int(args.k_module)}")

    Q_attn = EP.orthonormalize_np(np.asarray(Q_attn_raw, dtype=np.float32))
    Q_mlp = EP.orthonormalize_np(np.asarray(Q_mlp_raw, dtype=np.float32))

    # Similarity / angles
    sim_attn = EP.subspace_similarity(Q_shared, Q_attn)
    sim_mlp = EP.subspace_similarity(Q_shared, Q_mlp)
    cos_attn = _sv_cosines(Q_shared, Q_attn)
    cos_mlp = _sv_cosines(Q_shared, Q_mlp)

    # Energy ratios: how much of module output lies in Q_shared
    def _energy_stats_by_task(states_by_task: Dict[str, np.ndarray]) -> Dict[str, Any]:
        out = {}
        for t in tasks_used:
            out[t] = EP.energy_ratio_stats(states_by_task[t], Q_shared)
        pooled = np.concatenate([states_by_task[t] for t in tasks_used], axis=0)
        out["_pooled"] = EP.energy_ratio_stats(pooled, Q_shared)
        return out

    er_attn = _energy_stats_by_task(attn_bal)
    er_mlp = _energy_stats_by_task(mlp_bal)

    # Save bases for reuse
    bases_npz = os.path.join(out_dir, f"exp_3_bases_layer{int(args.layer)}{tag}.npz")
    np.savez(
        bases_npz,
        Q_shared=Q_shared.astype(np.float32),
        Q_attn=Q_attn.astype(np.float32),
        Q_mlp=Q_mlp.astype(np.float32),
        tasks=np.array(tasks_used, dtype=object),
    )

    out = {
        "config": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "trust_remote_code": bool(args.trust_remote_code),
            "layer": int(args.layer),
            "basis_npz": str(args.basis_npz),
            "basis_key": str(args.basis_key),
            "k_shared": int(args.k_shared),
            "k_module": int(args.k_module),
            "tasks": tasks_used,
            "n_prompts": int(args.n_prompts),
            "seed": int(args.seed),
            "template_seed": int(args.template_seed),
            "template_randomization": bool(args.template_randomization),
            "shuffle_choices": bool(args.shuffle_choices),
            "add_answer_prefix": bool(args.add_answer_prefix),
            "answer_prefix": str(args.answer_prefix),
            "decode_collect": {
                "batch_size": int(args.batch_size),
                "max_prompt_len": int(args.max_prompt_len),
                "calib_decode_max_new_tokens": int(args.calib_decode_max_new_tokens),
                "decoding": str(args.decoding),
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "per_task_max_states": int(args.per_task_max_states),
            },
            "balanced_states_per_task": int(n_min),
        },
        "dataset_meta": meta_by,
        "n_states_raw": {t: {"attn": int(attn_by_task[t].shape[0]), "mlp": int(mlp_by_task[t].shape[0])} for t in tasks_used},
        "similarity": {"shared_vs_attn": sim_attn, "shared_vs_mlp": sim_mlp},
        "principal_cosines": {"shared_vs_attn": cos_attn, "shared_vs_mlp": cos_mlp},
        "energy_ratio_to_shared": {"attn": er_attn, "mlp": er_mlp},
        "module_pca_contrib": {"attn": contrib_a, "mlp": contrib_m},
        "saved_bases_npz": os.path.relpath(bases_npz, ROOT_DIR),
    }

    out_json = os.path.join(out_dir, f"exp_3_module_alignment_layer{int(args.layer)}{tag}.json")
    _atomic_json_dump(out, out_json)

    md = []
    md.append("# Exp-3 (M3-b): Module attribution via alignment (Attn_out vs MLP_out)")
    md.append("")
    md.append("We compare a decode-shared basis `Q_shared` against PCA subspaces built from decode-time module outputs:")
    md.append("- `Attn_out`: layer.self_attn output")
    md.append("- `MLP_out` : layer.mlp output")
    md.append("")
    md.append("## Similarity (subspace principal angles)")
    rows = []
    for name, sim in [("shared vs attn_out", sim_attn), ("shared vs mlp_out", sim_mlp)]:
        rows.append([name, _fmt(sim.get("max_cos", float("nan"))), _fmt(sim.get("mean_cos", float("nan"))), _fmt(sim.get("min_cos", float("nan"))), _fmt(sim.get("fro_norm", float("nan")))])
    md.append(_md_table(rows, ["Pair", "max cos", "mean cos", "min cos", "fro"]))
    md.append("")
    md.append("## Energy ratio to shared subspace (pooled)")
    rows = []
    rows.append(["attn_out", _fmt(er_attn["_pooled"]["mean"]), _fmt(er_attn["_pooled"]["p50"]), _fmt(er_attn["_pooled"]["p95"])])
    rows.append(["mlp_out", _fmt(er_mlp["_pooled"]["mean"]), _fmt(er_mlp["_pooled"]["p50"]), _fmt(er_mlp["_pooled"]["p95"])])
    md.append(_md_table(rows, ["Module", "mean", "p50", "p95"]))
    md.append("")
    md.append("## Energy ratio to shared subspace (per-task mean)")
    rows = []
    for t in tasks_used:
        rows.append([t, _fmt(er_attn[t]["mean"]), _fmt(er_mlp[t]["mean"])])
    md.append(_md_table(rows, ["Task", "attn_out mean", "mlp_out mean"]))
    md.append("")
    md.append("JSON: `" + os.path.relpath(out_json, ROOT_DIR) + "`")
    md.append("Bases: `" + os.path.relpath(bases_npz, ROOT_DIR) + "`")
    md_path = os.path.join(out_dir, f"exp_3_module_alignment_layer{int(args.layer)}{tag}.md")
    _atomic_text_dump("\n".join(md).rstrip() + "\n", md_path)

    print(f"[Saved] {out_json}")
    print(f"[Saved] {md_path}")


if __name__ == "__main__":
    main()

