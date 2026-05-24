"""Open-answer patchback provenance experiment for math and code-style tasks."""

from __future__ import annotations

import os
import sys
import json
import re
import argparse
import importlib.util
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


try:
    from datasets import load_dataset  # type: ignore
except Exception:
    load_dataset = None


def import_module_from_path(module_name: str, file_path: str):
    file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {module_name} from {file_path}")

    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in (s or "").split(",") if x.strip()]


def _maybe_call_model(model: AutoModelForCausalLM, **kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("return_dict", True)
    try:
        return model(**kwargs, return_legacy_cache=True)
    except TypeError:
        return model(**kwargs)


def orthonormalize_np_fallback(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=np.float32)
    q, _ = np.linalg.qr(M)
    return q.astype(np.float32, copy=False)


def sample_random_orthonormal_basis(d: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, k)).astype(np.float32)
    return orthonormalize_np_fallback(A)


def sample_random_orthonormal_complement(Q_shared: np.ndarray, k: int, seed: int) -> np.ndarray:
    Qs = orthonormalize_np_fallback(Q_shared)
    d = Qs.shape[0]
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, k + 16)).astype(np.float32)
    A = A - Qs @ (Qs.T @ A)
    Q, _ = np.linalg.qr(A)
    return Q[:, :k].astype(np.float32, copy=False)


def energy_matched_random_vector_in_subspace(Q_sub: np.ndarray, target_norms: torch.Tensor, seed: int) -> torch.Tensor:
    Q = torch.tensor(Q_sub, dtype=torch.float32, device="cpu")
    B = int(target_norms.shape[0])
    k = int(Q.shape[1])
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((B, k)).astype(np.float32)
    z = torch.tensor(z, dtype=torch.float32, device="cpu")
    eps = 1e-12
    z = z / (torch.linalg.norm(z, dim=1)[:, None] + eps)
    z = z * target_norms[:, None]
    return z @ Q.T


def shuffle_coeffs_in_subspace(p_cpu: torch.Tensor, Q_sub: np.ndarray, seed: int, mode: str = "permute") -> torch.Tensor:
    Q = torch.tensor(Q_sub, dtype=torch.float32, device="cpu")
    c = p_cpu @ Q
    k = int(c.shape[1])
    rng = np.random.default_rng(seed)
    if mode == "permute":
        perm = torch.tensor(rng.permutation(k), dtype=torch.long)
        c2 = c[:, perm]
    elif mode == "signflip":
        signs = torch.tensor(rng.choice([-1.0, 1.0], size=(k,)).astype(np.float32))
        c2 = c * signs[None, :]
    else:
        raise ValueError(f"Unknown mode={mode}")
    return c2 @ Q.T


@dataclass
class OAExample:
    ex_id: str
    prompt: str
    gold: str
    meta: Dict[str, Any]


_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


def _strip_final_answer_prefix(s: str) -> str:
    s = (s or "").strip()

    for tag in ["Final answer (number only):", "Final answer:", "Final Answer:", "Answer:", "answer:", "FINAL ANSWER:"]:
        if tag in s:
            s = s.split(tag)[-1].strip()
    return s


def extract_boxed(s: str) -> Optional[str]:
    m = re.search(r"\\boxed\{([^}]*)\}", s)
    if m:
        return m.group(1).strip()
    return None


def extract_last_number(s: str) -> Optional[str]:
    xs = _NUM_RE.findall(s)
    if not xs:
        return None
    return xs[-1].strip()


def normalize_gold_answer_text(gold: str) -> str:
    g = (gold or "").strip()
    g = _strip_final_answer_prefix(g)
    bx = extract_boxed(g)
    if bx is not None:
        g = bx
    if "####" in g:
        g = g.split("####")[-1].strip()
    n = extract_last_number(g)
    if n is not None:
        return n
    return g.strip()


def normalize_pred_answer_text(pred: str) -> str:
    p = (pred or "").strip()
    p = _strip_final_answer_prefix(p)
    bx = extract_boxed(p)
    if bx is not None:
        p = bx
    n = extract_last_number(p)
    if n is not None:
        return n
    return p.strip()


def answers_match(gold: str, pred: str) -> bool:
    g = normalize_gold_answer_text(gold)
    p = normalize_pred_answer_text(pred)
    try:
        if re.fullmatch(r"[-+]?\d+", g) and re.fullmatch(r"[-+]?\d+", p):
            return int(g) == int(p)
    except Exception:
        pass
    try:
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", g) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", p):
            return abs(float(g) - float(p)) <= 1e-6
    except Exception:
        pass
    return g == p


@dataclass
class LogprobResult:
    total_logprob: float
    mean_logprob: float
    n_tokens: int


@torch.no_grad()
def decode_aligned_logprob(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    continuation_text: str,
    *,
    layer_module: torch.nn.Module,
    removal_hook: Optional[Any] = None,
    patch_hook: Optional[Any] = None,
    capture_hook: Optional[Any] = None,
    add_special_tokens_prompt: bool = True,
    max_cont_tokens: int = 0,
) -> LogprobResult:
    model.eval()
    device = next(model.parameters()).device

    toks = tok(prompt, return_tensors="pt", add_special_tokens=add_special_tokens_prompt)
    input_ids = toks["input_ids"].to(device)
    attn = torch.ones_like(input_ids)

    cont_ids = tok.encode(continuation_text, add_special_tokens=False)
    if max_cont_tokens and max_cont_tokens > 0:
        cont_ids = cont_ids[: int(max_cont_tokens)]
    if len(cont_ids) == 0:
        return LogprobResult(total_logprob=float("-inf"), mean_logprob=float("-inf"), n_tokens=0)

    cont = torch.tensor(cont_ids, dtype=torch.long, device=device)
    L = int(cont.shape[0])

    handles = []
    if removal_hook is not None:
        handles.append(layer_module.register_forward_hook(removal_hook))
    if patch_hook is not None:
        patch_hook.reset()
        handles.append(layer_module.register_forward_hook(patch_hook))
    if capture_hook is not None:
        capture_hook.reset()
        handles.append(layer_module.register_forward_hook(capture_hook))

    try:
        total_lp = 0.0

        if input_ids.shape[1] > 1:
            out_pre = _maybe_call_model(
                model,
                input_ids=input_ids[:, :-1],
                attention_mask=attn[:, :-1],
                use_cache=True,
            )
            past = out_pre.past_key_values
            out0 = _maybe_call_model(
                model,
                input_ids=input_ids[:, -1:],
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
        else:
            out0 = _maybe_call_model(
                model,
                input_ids=input_ids,
                attention_mask=attn,
                use_cache=True,
            )
            past = out0.past_key_values

        attn_cur = attn

        logits0 = out0.logits[:, -1, :]
        logp0 = torch.log_softmax(logits0, dim=-1)
        total_lp += float(logp0[0, cont[0]].item())

        for j in range(L - 1):
            tid = cont[j].view(1, 1)
            next_tid = cont[j + 1].item()
            attn_cur = torch.cat([attn_cur, torch.ones((1, 1), device=device, dtype=attn_cur.dtype)], dim=1)

            outj = _maybe_call_model(
                model,
                input_ids=tid,
                attention_mask=attn_cur,
                past_key_values=past,
                use_cache=True,
            )
            past = outj.past_key_values
            logpj = torch.log_softmax(outj.logits[:, -1, :], dim=-1)
            total_lp += float(logpj[0, next_tid].item())

        return LogprobResult(total_logprob=float(total_lp), mean_logprob=float(total_lp / float(L)), n_tokens=L)

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


@dataclass
class PairResult:
    pred: str
    correct: bool
    margin: float
    lp_gold: float
    lp_dist: float
    n_tokens_gold: int
    n_tokens_dist: int


def pairwise_logprob_decision(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    gold_text: str,
    dist_text: str,
    *,
    layer_module: torch.nn.Module,
    removal_hook: Optional[Any] = None,
    patch_hook: Optional[Any] = None,
    capture_hook: Optional[Any] = None,
    add_special_tokens_prompt: bool = True,
    gold_prefix: str = " ",
    dist_prefix: str = " ",
    max_answer_tokens: int = 0,
) -> PairResult:
    lg = decode_aligned_logprob(
        model, tok, prompt, gold_prefix + gold_text,
        layer_module=layer_module,
        removal_hook=removal_hook,
        patch_hook=patch_hook,
        capture_hook=capture_hook,
        add_special_tokens_prompt=add_special_tokens_prompt,
        max_cont_tokens=max_answer_tokens,
    )
    ld = decode_aligned_logprob(
        model, tok, prompt, dist_prefix + dist_text,
        layer_module=layer_module,
        removal_hook=removal_hook,
        patch_hook=patch_hook,
        capture_hook=None,
        add_special_tokens_prompt=add_special_tokens_prompt,
        max_cont_tokens=max_answer_tokens,
    )

    margin = float(lg.total_logprob - ld.total_logprob)
    pred = "gold" if margin >= 0 else "dist"
    correct = (pred == "gold")
    return PairResult(
        pred=pred,
        correct=correct,
        margin=margin,
        lp_gold=float(lg.total_logprob),
        lp_dist=float(ld.total_logprob),
        n_tokens_gold=int(lg.n_tokens),
        n_tokens_dist=int(ld.n_tokens),
    )


@torch.no_grad()
def greedy_generate_decode_aligned(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompt: str,
    *,
    layer_module: torch.nn.Module,
    max_new_tokens: int,
    removal_hook: Optional[Any] = None,
    patch_hook: Optional[Any] = None,
    capture_hook: Optional[Any] = None,
    add_special_tokens_prompt: bool = True,
    stop_on_eos: bool = True,
) -> Tuple[List[int], str, bool]:
    model.eval()
    device = next(model.parameters()).device

    toks = tok(prompt, return_tensors="pt", add_special_tokens=add_special_tokens_prompt)
    input_ids = toks["input_ids"].to(device)
    attn_full = torch.ones_like(input_ids)

    handles = []
    if removal_hook is not None:
        handles.append(layer_module.register_forward_hook(removal_hook))
    if patch_hook is not None:
        patch_hook.reset()
        handles.append(layer_module.register_forward_hook(patch_hook))
    if capture_hook is not None:
        capture_hook.reset()
        handles.append(layer_module.register_forward_hook(capture_hook))

    gen_ids: List[int] = []
    stopped_eos = False

    try:
        if input_ids.shape[1] > 1:
            out_pre = _maybe_call_model(
                model,
                input_ids=input_ids[:, :-1],
                attention_mask=attn_full[:, :-1],
                use_cache=True,
            )
            past = out_pre.past_key_values
            cur = input_ids[:, -1:]
            attn = attn_full
        else:
            past = None
            cur = input_ids
            attn = attn_full

        for _ in range(int(max_new_tokens)):
            out = _maybe_call_model(
                model,
                input_ids=cur,
                attention_mask=attn,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]
            next_id = int(torch.argmax(logits, dim=-1).item())
            gen_ids.append(next_id)

            if stop_on_eos and tok.eos_token_id is not None and next_id == int(tok.eos_token_id):
                stopped_eos = True
                break

            cur = torch.tensor([[next_id]], device=device, dtype=input_ids.dtype)
            attn = torch.cat([attn, torch.ones((1, 1), device=device, dtype=attn.dtype)], dim=1)

        full_ids = torch.cat([input_ids, torch.tensor([gen_ids], device=device, dtype=input_ids.dtype)], dim=1)
        gen_text = tok.decode(full_ids[0], skip_special_tokens=True)
        return gen_ids, gen_text, stopped_eos

    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass


def eval_gen_math(gold: str, gen_text: str) -> Tuple[str, bool]:
    tail = _strip_final_answer_prefix(gen_text)
    pred_ans = normalize_pred_answer_text(tail)
    ok = answers_match(gold, pred_ans)
    return pred_ans, ok


def eval_gen_code_compile(gen_text: str) -> Tuple[str, bool]:
    s = gen_text
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[-2]
    code = s.strip()
    try:
        compile(code, "<gen>", "exec")
        return code, True
    except Exception:
        return code, False


def summarize_rescue_pair(flip_rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    n = len(flip_rows)
    if n == 0:
        return {"n": 0}
    rescued = int(sum(1 for r in flip_rows if bool(r[key]["correct"])))
    dms = []
    for r in flip_rows:
        dms.append(float(r[key]["margin"]) - float(r["ablated"]["margin"]))
    dms = np.array(dms, dtype=np.float32)
    return {
        "n": n,
        "rescued": rescued,
        "rescued_pct": 100.0 * rescued / n,
        "mean_delta_margin_vs_ablated": float(dms.mean()),
        "median_delta_margin_vs_ablated": float(np.median(dms)),
    }


def summarize_rescue_gen(flip_rows: List[Dict[str, Any]], key: str, field: str) -> Dict[str, Any]:
    n = len(flip_rows)
    if n == 0:
        return {"n": 0}
    rescued = int(sum(1 for r in flip_rows if bool(r[key][field])))
    return {
        "n": n,
        "rescued": rescued,
        "rescued_pct": 100.0 * rescued / n,
    }


def load_examples_from_benchmark(
    base_mod: Any,
    dl: Any,
    *,
    task: str,
    n_eval: int,
    seed: int,
    answer_prefix: str,
    template_randomization: bool,
    shuffle_choices: bool,
) -> Tuple[List[OAExample], Dict[str, Any]]:
    _, eval_by, meta_by = base_mod.load_selected_tasks_eval_only(
        dl,
        task=task,
        n_eval=n_eval,
        seed=seed,
        template_randomization=template_randomization,
        template_seed=seed + 999,
        shuffle_choices=shuffle_choices,
        answer_prefix=answer_prefix,
    )
    exs = eval_by[task]
    meta = meta_by.get(task, {})
    out: List[OAExample] = []
    for ex in exs:
        out.append(OAExample(ex_id=ex.ex_id, prompt=ex.prompt, gold=ex.gold or "", meta={"source": "benchmark"}))
    return out, meta


def load_examples_from_hf(
    *,
    hf_id: str,
    split: str,
    n_eval: int,
    seed: int,
    task_hint: str,
    answer_prefix: str,
) -> Tuple[List[OAExample], Dict[str, Any]]:
    if load_dataset is None:
        raise RuntimeError("datasets is not installed; cannot use HF loader.")

    ds = load_dataset(hf_id, split=split)

    rng = np.random.default_rng(seed)
    idxs = np.arange(len(ds))
    rng.shuffle(idxs)
    idxs = idxs[: int(min(n_eval, len(ds)))]

    out: List[OAExample] = []

    for idx in idxs.tolist():
        row = ds[int(idx)]
        ex_id = f"{hf_id}-{split}-{idx}"

        if "humaneval" in task_hint.lower() or "humaneval" in hf_id.lower():
            prompt = str(row.get("prompt") or "")
            gold = str(row.get("canonical_solution") or row.get("solution") or "")
            meta = {
                "source": "hf",
                "test": row.get("test", ""),
                "entry_point": row.get("entry_point", ""),
            }
            out.append(OAExample(ex_id=ex_id, prompt=prompt, gold=gold, meta=meta))
            continue

        q = row.get("question") or row.get("problem") or row.get("prompt") or ""
        a = row.get("answer") or row.get("solution") or row.get("gold") or ""
        prompt = (str(q).strip() + (answer_prefix or "")).strip()
        gold = str(a)
        out.append(OAExample(ex_id=ex_id, prompt=prompt, gold=gold, meta={"source": "hf"}))

    meta_out = {"hf_id": hf_id, "split": split, "n_total": len(ds)}
    return out, meta_out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--base_script_path", type=str, required=True,
                    help="Path to your existing exp_subspace_patching_transfer.py")

    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--seed", type=int, default=123)


    ap.add_argument("--Qs_path", type=str, default="")
    ap.add_argument("--compute_Qs", type=int, default=0)
    ap.add_argument("--Qs_out", type=str, default="Q_shared_openanswer.npy")
    ap.add_argument("--basis_tasks", type=str, default="gsm8k,commonsenseqa,strategyqa,openbookqa,qasc,boolq,piqa")
    ap.add_argument("--basis_n_subspace", type=int, default=128)
    ap.add_argument("--calib_batch_size", type=int, default=8)
    ap.add_argument("--calib_max_new_tokens", type=int, default=128)
    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--variance_threshold", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=8)
    ap.add_argument("--max_dim", type=int, default=1024)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")


    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--n_eval", type=int, default=256)
    ap.add_argument("--max_flips", type=int, default=64)

    ap.add_argument("--use_benchmark_loader", type=int, default=1)
    ap.add_argument("--hf_id", type=str, default="", help="HF dataset id (used when --use_benchmark_loader=0)")
    ap.add_argument("--hf_split", type=str, default="test", help="HF dataset split (used when --use_benchmark_loader=0)")

    ap.add_argument("--loto8_path", type=str, default="exp_patchback_loto.py")
    ap.add_argument("--dataloaders_path", type=str, default="")


    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--template_randomization", type=int, default=1)
    ap.add_argument("--shuffle_choices", type=int, default=1)


    ap.add_argument("--eval_mode", type=str, required=True,
                    choices=["pair_logprob", "gen_math", "gen_code_compile"])


    ap.add_argument("--gold_text_prefix", type=str, default=" ")
    ap.add_argument("--dist_text_prefix", type=str, default=" ")
    ap.add_argument("--gold_max_tokens", type=int, default=0)
    ap.add_argument("--distractor_mode", type=str, default="next_gold", choices=["next_gold", "random_gold"])


    ap.add_argument("--patch_n_steps", type=int, default=1)


    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument("--run_coeff_controls", type=int, default=0)
    ap.add_argument("--add_special_tokens_prompt", type=int, default=1)

    ap.add_argument("--out_json", type=str, default="openanswer_patching_results.json")

    args = ap.parse_args()
    seed_everything(args.seed)


    answer_prefix_effective = args.answer_prefix
    max_new_tokens_effective = int(args.max_new_tokens)

    if args.eval_mode == "gen_math":

        if args.answer_prefix.strip() == "Final answer:" or args.answer_prefix.strip() == "\nFinal answer:".strip():
            answer_prefix_effective = "\nLet's think step by step.\nFinal answer (number only):"
            print(f"[Info] gen_math: auto-upgrade answer_prefix -> {answer_prefix_effective!r}")
        if max_new_tokens_effective < 64:
            print(f"[Info] gen_math: max_new_tokens {max_new_tokens_effective} too small; bumping to 64 for stability.")
            max_new_tokens_effective = 64


    base_mod = import_module_from_path("base_subspace_patching_transfer_for_openanswer", args.base_script_path)
    loto8, dl = base_mod.load_aux_modules(args.loto8_path, args.dataloaders_path)


    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch_dtype,
            device_map=None,
        ).to(args.device)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=None,
        ).to(args.device)
    model.eval()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token


    layers, path_used = base_mod.get_transformer_layers(model)
    if args.layer < 0 or args.layer >= len(layers):
        raise ValueError(f"--layer {args.layer} out of range for layers at {path_used} (n={len(layers)})")
    layer_module = layers[args.layer]
    print(f"[Info] Hooking layer={args.layer} at path {path_used}")


    if args.Qs_path:
        Qs = base_mod.orthonormalize_np(np.load(args.Qs_path).astype(np.float32)) if hasattr(base_mod, "orthonormalize_np") \
             else orthonormalize_np_fallback(np.load(args.Qs_path).astype(np.float32))
        print(f"[Info] Loaded Q_shared from {args.Qs_path} shape={Qs.shape}")
    elif int(args.compute_Qs) == 1:
        tasks = parse_csv_list(args.basis_tasks)
        Qs = base_mod.maybe_compute_Qs(
            loto8=loto8,
            dl=dl,
            model=model,
            tokenizer=tok,
            layer_idx=args.layer,
            seed=args.seed,
            tasks=tasks,
            n_subspace=args.basis_n_subspace,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
            answer_prefix=answer_prefix_effective,
            calib_batch_size=args.calib_batch_size,
            calib_max_new_tokens=args.calib_max_new_tokens,
            per_task_max_states=args.per_task_max_states,
            max_prompt_len=args.max_prompt_len,
            variance_threshold=args.variance_threshold,
            min_dim=args.min_dim,
            max_dim=args.max_dim,
            tau=args.tau,
            m_shared=args.m_shared,
            out_path=args.Qs_out,
        )
        print(f"[Info] Computed Q_shared -> {args.Qs_out} shape={Qs.shape}")
    else:
        raise RuntimeError("Provide --Qs_path or set --compute_Qs=1")

    d, k = Qs.shape


    if bool(args.use_benchmark_loader):
        examples, eval_meta = load_examples_from_benchmark(
            base_mod, dl,
            task=args.task,
            n_eval=args.n_eval,
            seed=args.seed,
            answer_prefix=answer_prefix_effective,
            template_randomization=bool(args.template_randomization),
            shuffle_choices=bool(args.shuffle_choices),
        )
        print(f"[Info] Loaded eval examples from benchmark_dataloaders: task={args.task} n={len(examples)} meta={eval_meta}")
    else:
        if not args.hf_id:
            raise RuntimeError("--use_benchmark_loader=0 but --hf_id is empty.")
        examples, eval_meta = load_examples_from_hf(
            hf_id=args.hf_id,
            split=args.hf_split,
            n_eval=args.n_eval,
            seed=args.seed,
            task_hint=args.task,
            answer_prefix=answer_prefix_effective,
        )
        print(f"[Info] Loaded eval examples from HF: hf_id={args.hf_id} split={args.hf_split} n={len(examples)} meta={eval_meta}")

    if len(examples) == 0:
        raise RuntimeError("No eval examples loaded.")


    patch_steps: Set[int] = set(range(int(max(1, args.patch_n_steps))))
    print(f"[Info] patch_steps={sorted(list(patch_steps))}  (narrow-window)")


    Q_nonshared = sample_random_orthonormal_complement(Qs, k=k, seed=args.seed + 2024)


    rng = np.random.default_rng(args.seed + 999)
    gold_texts_norm = [normalize_gold_answer_text(ex.gold) for ex in examples]
    dist_map: Dict[str, str] = {}
    if args.eval_mode == "pair_logprob":
        if args.distractor_mode == "next_gold":
            for i, ex in enumerate(examples):
                dist_map[ex.ex_id] = gold_texts_norm[(i + 1) % len(gold_texts_norm)]
        else:
            for i, ex in enumerate(examples):
                j = int(rng.integers(0, len(gold_texts_norm)))
                if j == i:
                    j = (j + 1) % len(gold_texts_norm)
                dist_map[ex.ex_id] = gold_texts_norm[j]


    scan_rows: List[Dict[str, Any]] = []
    flip_examples: List[OAExample] = []

    for ex in examples:
        prompt = ex.prompt
        gold_raw = ex.gold
        ex_id = ex.ex_id

        if args.eval_mode == "pair_logprob":
            gold_text = normalize_gold_answer_text(gold_raw)
            dist_text = dist_map[ex_id]

            base = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            remove = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer"))
            ablt = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=remove,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            scan_rows.append({
                "ex_id": ex_id,
                "gold_norm": gold_text,
                "dist_norm": dist_text,
                "baseline": base.__dict__,
                "ablated": ablt.__dict__,
            })
            if base.correct and (not ablt.correct):
                flip_examples.append(ex)

        elif args.eval_mode == "gen_math":
            gen_ids, gen_text, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_ans, ok_base = eval_gen_math(gold_raw, gen_text)

            remove = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer"))
            gen_ids_a, gen_text_a, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=remove,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_ans_a, ok_ablt = eval_gen_math(gold_raw, gen_text_a)

            scan_rows.append({
                "ex_id": ex_id,
                "gold_raw": gold_raw,
                "baseline": {"pred_answer": pred_ans, "correct": bool(ok_base), "n_gen_tokens": len(gen_ids)},
                "ablated": {"pred_answer": pred_ans_a, "correct": bool(ok_ablt), "n_gen_tokens": len(gen_ids_a)},
            })
            if ok_base and (not ok_ablt):
                flip_examples.append(ex)

        else:
            gen_ids, gen_text, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_base = eval_gen_code_compile(gen_text)

            remove = loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer"))
            gen_ids_a, gen_text_a, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=remove,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_ablt = eval_gen_code_compile(gen_text_a)

            scan_rows.append({
                "ex_id": ex_id,
                "baseline": {"compile_ok": bool(ok_base), "n_gen_tokens": len(gen_ids)},
                "ablated": {"compile_ok": bool(ok_ablt), "n_gen_tokens": len(gen_ids_a)},
            })
            if ok_base and (not ok_ablt):
                flip_examples.append(ex)

    n_scanned = len(scan_rows)
    n_flips_total = len(flip_examples)
    n_flips_used = int(min(n_flips_total, args.max_flips))
    flip_used = flip_examples[:n_flips_used]

    if args.eval_mode == "gen_code_compile":
        base_acc = float(np.mean([1.0 if r["baseline"]["compile_ok"] else 0.0 for r in scan_rows])) if scan_rows else float("nan")
        ablt_acc = float(np.mean([1.0 if r["ablated"]["compile_ok"] else 0.0 for r in scan_rows])) if scan_rows else float("nan")
    else:
        base_acc = float(np.mean([1.0 if r["baseline"]["correct"] else 0.0 for r in scan_rows])) if scan_rows else float("nan")
        ablt_acc = float(np.mean([1.0 if r["ablated"]["correct"] else 0.0 for r in scan_rows])) if scan_rows else float("nan")

    print(f"[Scan] n_scanned={n_scanned} base_acc={base_acc:.3f} ablt_acc={ablt_acc:.3f} flips_total={n_flips_total} flips_used={n_flips_used}")

    if n_flips_used == 0:
        out = {
            "meta": {
                "note": "No flips found under this eval_mode. Increase n_eval or choose different task/layer/seed.",
                "model": args.model,
                "layer": args.layer,
                "task": args.task,
                "eval_mode": args.eval_mode,
                "n_scanned": n_scanned,
                "base_acc_scan": base_acc,
                "ablt_acc_scan": ablt_acc,
                "flips_total": 0,
                "patch_steps": sorted(list(patch_steps)),
                "Qs_path": args.Qs_path or args.Qs_out,
                "Qs_shape": [int(d), int(k)],
                "layers_path": path_used,
                "eval_meta": eval_meta,
                "answer_prefix_effective": answer_prefix_effective,
                "max_new_tokens_effective": max_new_tokens_effective,
            },
            "scan_rows": scan_rows,
            "flip_rows": [],
        }
        ensure_dir(args.out_json)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"[Done] Wrote {args.out_json}")
        return


    donor_bank: List[Dict[int, torch.Tensor]] = []

    for ex in flip_used:
        cap = base_mod.DecodeStepHiddenCaptureHook(capture_steps=patch_steps)

        if args.eval_mode == "pair_logprob":
            gold_text = normalize_gold_answer_text(ex.gold)
            dist_text = dist_map[ex.ex_id]
            _ = pairwise_logprob_decision(
                model, tok, ex.prompt, gold_text, dist_text,
                layer_module=layer_module,
                capture_hook=cap,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )
        else:
            _ = greedy_generate_decode_aligned(
                model, tok, ex.prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                capture_hook=cap,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )

        donor_shared: Dict[int, torch.Tensor] = {}
        for t, h in cap.hidden_by_step.items():
            donor_shared[int(t)] = base_mod.project_cpu(h, Qs)
        donor_bank.append(donor_shared)


    flip_rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(flip_used):
        prompt = ex.prompt
        gold_raw = ex.gold
        ex_id = ex.ex_id
        donor_self = donor_bank[i]
        donor_other = donor_bank[(i + 1) % len(donor_bank)]


        donor0 = donor_self.get(0, None)
        if donor0 is None:
            donor0 = donor_self[sorted(list(donor_self.keys()))[0]]
        target_norms = torch.linalg.norm(donor0.cpu().float(), dim=1).cpu().float()

        if args.eval_mode == "pair_logprob":
            gold_text = normalize_gold_answer_text(gold_raw)
            dist_text = dist_map[ex_id]

            base = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )
            ablt = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            patched_self = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_self, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            ctrl_time = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_other, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            r_shared = energy_matched_random_vector_in_subspace(Qs, target_norms, seed=args.seed + 9200 + i)
            ctrl_shared_randvec = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step={0: r_shared}, patch_steps={0}),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            Q_rand = sample_random_orthonormal_basis(d=d, k=k, seed=args.seed + 9000 + i)
            r_rand = energy_matched_random_vector_in_subspace(Q_rand, target_norms, seed=args.seed + 9100 + i)
            ctrl_rand_subspace = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Q_rand, donor_by_step={0: r_rand}, patch_steps={0}),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            cap_ns = base_mod.DecodeStepHiddenCaptureHook(capture_steps=patch_steps)
            _ = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                capture_hook=cap_ns,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )
            donor_ns = {int(t): base_mod.project_cpu(h, Q_nonshared) for t, h in cap_ns.hidden_by_step.items()}
            ctrl_nonshared = pairwise_logprob_decision(
                model, tok, prompt, gold_text, dist_text,
                layer_module=layer_module,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Q_nonshared, donor_by_step=donor_ns, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                gold_prefix=args.gold_text_prefix,
                dist_prefix=args.dist_text_prefix,
                max_answer_tokens=args.gold_max_tokens,
            )

            row = {
                "ex_id": ex_id,
                "gold_norm": gold_text,
                "dist_norm": dist_text,
                "baseline": base.__dict__,
                "ablated": ablt.__dict__,
                "patched_self": patched_self.__dict__,
                "control_time_shuffled": ctrl_time.__dict__,
                "control_shared_randvec": ctrl_shared_randvec.__dict__,
                "control_rand_subspace": ctrl_rand_subspace.__dict__,
                "control_patch_nonshared": ctrl_nonshared.__dict__,
            }

            if bool(args.run_coeff_controls):
                p0 = donor0.cpu().float()
                perm = shuffle_coeffs_in_subspace(p0, Qs, seed=args.seed + 9300 + i, mode="permute")
                sign = shuffle_coeffs_in_subspace(p0, Qs, seed=args.seed + 9400 + i, mode="signflip")
                ctrl_perm = pairwise_logprob_decision(
                    model, tok, prompt, gold_text, dist_text,
                    layer_module=layer_module,
                    removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                    patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step={0: perm}, patch_steps={0}),
                    add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                    gold_prefix=args.gold_text_prefix,
                    dist_prefix=args.dist_text_prefix,
                    max_answer_tokens=args.gold_max_tokens,
                )
                ctrl_sign = pairwise_logprob_decision(
                    model, tok, prompt, gold_text, dist_text,
                    layer_module=layer_module,
                    removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                    patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step={0: sign}, patch_steps={0}),
                    add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
                    gold_prefix=args.gold_text_prefix,
                    dist_prefix=args.dist_text_prefix,
                    max_answer_tokens=args.gold_max_tokens,
                )
                row["control_shared_perm"] = ctrl_perm.__dict__
                row["control_shared_signflip"] = ctrl_sign.__dict__

            flip_rows.append(row)
            print(f"[Flip {i+1}/{len(flip_used)}] {ex_id} base={base.pred}({base.correct}) ablt={ablt.pred}({ablt.correct}) patched={patched_self.pred}({patched_self.correct})")

        elif args.eval_mode == "gen_math":
            gen_ids, gen_text, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_base, ok_base = eval_gen_math(gold_raw, gen_text)

            gen_ids_a, gen_text_a, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_ablt, ok_ablt = eval_gen_math(gold_raw, gen_text_a)

            _, text_p, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_self, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_patch, ok_patch = eval_gen_math(gold_raw, text_p)

            _, text_t, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_other, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_time, ok_time = eval_gen_math(gold_raw, text_t)

            r_shared = energy_matched_random_vector_in_subspace(Qs, target_norms, seed=args.seed + 9200 + i)
            _, text_r, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step={0: r_shared}, patch_steps={0}),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_rand, ok_rand = eval_gen_math(gold_raw, text_r)

            Q_rand = sample_random_orthonormal_basis(d=d, k=k, seed=args.seed + 9000 + i)
            r_rand = energy_matched_random_vector_in_subspace(Q_rand, target_norms, seed=args.seed + 9100 + i)
            _, text_rs, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Q_rand, donor_by_step={0: r_rand}, patch_steps={0}),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_rsub, ok_rsub = eval_gen_math(gold_raw, text_rs)

            cap_ns = base_mod.DecodeStepHiddenCaptureHook(capture_steps=patch_steps)
            _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                capture_hook=cap_ns,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            donor_ns = {int(t): base_mod.project_cpu(h, Q_nonshared) for t, h in cap_ns.hidden_by_step.items()}
            _, text_ns, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Q_nonshared, donor_by_step=donor_ns, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            pred_ns, ok_ns = eval_gen_math(gold_raw, text_ns)

            flip_rows.append({
                "ex_id": ex_id,
                "gold_raw": gold_raw,
                "baseline": {"pred_answer": pred_base, "correct": bool(ok_base)},
                "ablated": {"pred_answer": pred_ablt, "correct": bool(ok_ablt)},
                "patched_self": {"pred_answer": pred_patch, "correct": bool(ok_patch)},
                "control_time_shuffled": {"pred_answer": pred_time, "correct": bool(ok_time)},
                "control_shared_randvec": {"pred_answer": pred_rand, "correct": bool(ok_rand)},
                "control_rand_subspace": {"pred_answer": pred_rsub, "correct": bool(ok_rsub)},
                "control_patch_nonshared": {"pred_answer": pred_ns, "correct": bool(ok_ns)},
            })
            print(f"[Flip {i+1}/{len(flip_used)}] {ex_id} patched_ok={ok_patch} time_ok={ok_time} sharedrand_ok={ok_rand}")

        else:
            _, gen_text, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_base = eval_gen_code_compile(gen_text)

            _, gen_text_a, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_ablt = eval_gen_code_compile(gen_text_a)

            _, text_p, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_self, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_patch = eval_gen_code_compile(text_p)

            _, text_t, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step=donor_other, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_time = eval_gen_code_compile(text_t)

            r_shared = energy_matched_random_vector_in_subspace(Qs, target_norms, seed=args.seed + 9200 + i)
            _, text_r, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Qs, donor_by_step={0: r_shared}, patch_steps={0}),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_rand = eval_gen_code_compile(text_r)

            Q_rand = sample_random_orthonormal_basis(d=d, k=k, seed=args.seed + 9000 + i)
            r_rand = energy_matched_random_vector_in_subspace(Q_rand, target_norms, seed=args.seed + 9100 + i)
            _, text_rs, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Q_rand, donor_by_step={0: r_rand}, patch_steps={0}),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_rsub = eval_gen_code_compile(text_rs)

            cap_ns = base_mod.DecodeStepHiddenCaptureHook(capture_steps=patch_steps)
            _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                capture_hook=cap_ns,
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            donor_ns = {int(t): base_mod.project_cpu(h, Q_nonshared) for t, h in cap_ns.hidden_by_step.items()}
            _, text_ns, _ = greedy_generate_decode_aligned(
                model, tok, prompt,
                layer_module=layer_module,
                max_new_tokens=max_new_tokens_effective,
                removal_hook=loto8.LastTokenRemovalHook(Qs, alpha=1.0, stats=loto8.HookStats("remove_shared_openanswer")),
                patch_hook=base_mod.SubspacePatchHook(Q_nonshared, donor_by_step=donor_ns, patch_steps=patch_steps),
                add_special_tokens_prompt=bool(args.add_special_tokens_prompt),
            )
            _, ok_ns = eval_gen_code_compile(text_ns)

            flip_rows.append({
                "ex_id": ex_id,
                "baseline": {"compile_ok": bool(ok_base)},
                "ablated": {"compile_ok": bool(ok_ablt)},
                "patched_self": {"compile_ok": bool(ok_patch)},
                "control_time_shuffled": {"compile_ok": bool(ok_time)},
                "control_shared_randvec": {"compile_ok": bool(ok_rand)},
                "control_rand_subspace": {"compile_ok": bool(ok_rsub)},
                "control_patch_nonshared": {"compile_ok": bool(ok_ns)},
            })
            print(f"[Flip {i+1}/{len(flip_used)}] {ex_id} patched_ok={ok_patch} time_ok={ok_time} sharedrand_ok={ok_rand}")


    if args.eval_mode == "pair_logprob":
        summary = {
            "patched_self": summarize_rescue_pair(flip_rows, "patched_self"),
            "control_time_shuffled": summarize_rescue_pair(flip_rows, "control_time_shuffled"),
            "control_shared_randvec": summarize_rescue_pair(flip_rows, "control_shared_randvec"),
            "control_rand_subspace": summarize_rescue_pair(flip_rows, "control_rand_subspace"),
            "control_patch_nonshared": summarize_rescue_pair(flip_rows, "control_patch_nonshared"),
        }
        if bool(args.run_coeff_controls):
            if all("control_shared_perm" in r for r in flip_rows):
                summary["control_shared_perm"] = summarize_rescue_pair(flip_rows, "control_shared_perm")
            if all("control_shared_signflip" in r for r in flip_rows):
                summary["control_shared_signflip"] = summarize_rescue_pair(flip_rows, "control_shared_signflip")
    elif args.eval_mode == "gen_math":
        summary = {
            "patched_self": summarize_rescue_gen(flip_rows, "patched_self", "correct"),
            "control_time_shuffled": summarize_rescue_gen(flip_rows, "control_time_shuffled", "correct"),
            "control_shared_randvec": summarize_rescue_gen(flip_rows, "control_shared_randvec", "correct"),
            "control_rand_subspace": summarize_rescue_gen(flip_rows, "control_rand_subspace", "correct"),
            "control_patch_nonshared": summarize_rescue_gen(flip_rows, "control_patch_nonshared", "correct"),
        }
    else:
        summary = {
            "patched_self": summarize_rescue_gen(flip_rows, "patched_self", "compile_ok"),
            "control_time_shuffled": summarize_rescue_gen(flip_rows, "control_time_shuffled", "compile_ok"),
            "control_shared_randvec": summarize_rescue_gen(flip_rows, "control_shared_randvec", "compile_ok"),
            "control_rand_subspace": summarize_rescue_gen(flip_rows, "control_rand_subspace", "compile_ok"),
            "control_patch_nonshared": summarize_rescue_gen(flip_rows, "control_patch_nonshared", "compile_ok"),
        }

    print("\n[Summary on flips_used]")
    for kname, sval in summary.items():
        print(f"  {kname:>22s}: rescued={sval.get('rescued',0)}/{sval.get('n',0)} ({sval.get('rescued_pct',0):.1f}%)")

    out = {
        "meta": {
            "model": args.model,
            "device": args.device,
            "dtype": args.dtype,
            "layer": args.layer,
            "layers_path": path_used,
            "seed": args.seed,

            "task": args.task,
            "eval_mode": args.eval_mode,
            "eval_meta": eval_meta,

            "n_eval_loaded": len(examples),
            "n_scanned": n_scanned,
            "base_acc_scan": base_acc,
            "ablt_acc_scan": ablt_acc,
            "flips_total": n_flips_total,
            "flips_used": n_flips_used,

            "patch_steps": sorted(list(patch_steps)),
            "patch_n_steps": int(args.patch_n_steps),

            "Qs_path": args.Qs_path or args.Qs_out,
            "Qs_shape": [int(d), int(k)],

            "gold_text_prefix": args.gold_text_prefix,
            "dist_text_prefix": args.dist_text_prefix,
            "gold_max_tokens": int(args.gold_max_tokens),
            "distractor_mode": args.distractor_mode,

            "answer_prefix_effective": answer_prefix_effective,
            "max_new_tokens_effective": int(max_new_tokens_effective),

            "run_coeff_controls": bool(args.run_coeff_controls),

            "use_benchmark_loader": bool(args.use_benchmark_loader),
            "hf_id": args.hf_id,
            "hf_split": args.hf_split,
        },
        "summary_on_flips": summary,
        "scan_rows": scan_rows,
        "flip_rows": flip_rows,
    }

    ensure_dir(args.out_json)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n[Done] Wrote {args.out_json}")


if __name__ == "__main__":
    main()
