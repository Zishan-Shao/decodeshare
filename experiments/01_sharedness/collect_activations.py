# -*- coding: utf-8 -*-
"""
collect_activations.py

一次性采集：每个 task 的 decode-phase (seq_len==1) last-token hidden states，
并做与 decodeshare.sharedness 一致的公平预处理：
  - per_task_max_states cap
  - balance_to="min"（所有 tasks 统一到同样的 state 数）
  - task-wise centering

输出：
  out_dir/
    meta.json
    <task>.npy   (float16 或 float32)

用法示例：
  CUDA_VISIBLE_DEVICES=0 python collect_activations.py \
    --model meta-llama/Llama-2-7b-chat-hf --device cuda --model_dtype fp32 \
    --layer 10 --n_prompts 128 --calib_max_new_tokens 256 --max_prompt_len 512 \
    --per_task_max_states 20000 --seed 42 \
    --out_dir results/acts/llama2_layer10_n128_new256_maxlen512_states20000_seed42 \
    --save_dtype fp16 --out_txt results/acts/.../collect.txt
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional

import numpy as np
import torch

from decodeshare.activations import (
    _should_write_txt,
    TeeStdout,
    set_global_seed,
    load_calib_prompts,
    load_model_and_tokenizer,
    get_model_layers,
    DecodeLastTokenActivationCollector,
    collect_decode_last_token_states,
    center_and_balance,
    to_py,
)

def _parse_tasks_arg(s: str) -> Optional[List[str]]:
    if s is None:
        return None
    ss = str(s).strip()
    if ss == "" or ss.lower() in {"all", "none", "null"}:
        return None
    return [t.strip() for t in ss.split(",") if t.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model_dtype", type=str, default="fp32", choices=["fp32", "fp16"])

    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n_prompts", type=int, default=128)

    ap.add_argument("--max_prompt_len", type=int, default=512)
    ap.add_argument("--calib_max_new_tokens", type=int, default=256)
    ap.add_argument("--calib_decoding", type=str, default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=4)

    ap.add_argument("--per_task_max_states", type=int, default=20000)
    ap.add_argument("--balance_to", type=str, default="min")  # 建议保持 min，便于后续不同子集可复用
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tasks", type=str, default="all",
                    help="逗号分隔 task 列表；默认 all 表示 load_calib_prompts() 能加载到的全部")
    ap.add_argument("--save_dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--out_txt", type=str, default="",
                    help='tee stdout 到这个 txt；传 "" 或 "none" 关闭')

    args = ap.parse_args()

    # tee stdout
    orig_stdout = sys.stdout
    txt_f = None
    if _should_write_txt(args.out_txt):
        out_dir = os.path.dirname(os.path.abspath(args.out_txt))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        txt_f = open(args.out_txt, "w", encoding="utf-8")
        sys.stdout = TeeStdout(orig_stdout, txt_f)

    try:
        print(f"[Cmd] {' '.join(sys.argv)}")
        set_global_seed(args.seed)

        os.makedirs(args.out_dir, exist_ok=True)

        # 1) load prompts (same loader as decodeshare.sharedness)
        prompts_by_task = load_calib_prompts(args.n_prompts, args.seed)
        want = _parse_tasks_arg(args.tasks)
        if want is not None:
            missing = [t for t in want if t not in prompts_by_task]
            if missing:
                print(f"[Warn] tasks not loaded and will be skipped: {missing}")
            prompts_by_task = {t: prompts_by_task[t] for t in want if t in prompts_by_task}

        tasks = list(prompts_by_task.keys())
        if len(tasks) == 0:
            raise RuntimeError("No tasks available after filtering. Check --tasks or dataset loading.")

        print(f"[Data] tasks={tasks}")
        for t in tasks:
            print(f"[Data] task={t} prompts={len(prompts_by_task[t])}")

        # 2) load model + hooks
        model, tok = load_model_and_tokenizer(args.model, args.device, args.model_dtype)
        layers, _ = get_model_layers(model)
        if args.layer >= len(layers):
            raise RuntimeError(f"layer={args.layer} out of range, num_layers={len(layers)}")

        collector = DecodeLastTokenActivationCollector([int(args.layer)])
        h = layers[int(args.layer)].register_forward_hook(collector.make_hook(int(args.layer)))

        # 3) collect
        try:
            with torch.inference_mode():
                for task in tasks:
                    print(f"[Collect] task={task}")
                    collector.set_current_task(task)
                    collect_decode_last_token_states(
                        model=model,
                        tokenizer=tok,
                        prompts=prompts_by_task[task],
                        collector=collector,
                        batch_size=int(args.batch_size),
                        max_prompt_len=int(args.max_prompt_len),
                        calib_max_new_tokens=int(args.calib_max_new_tokens),
                        decoding=str(args.calib_decoding),
                        temperature=float(args.temperature),
                        top_p=float(args.top_p),
                        top_k=int(args.top_k),
                    )
        finally:
            try:
                h.remove()
            except Exception:
                pass
            collector.set_capture(False, None)

        # 4) assemble raw
        X_raw: Dict[str, np.ndarray] = {}
        raw_counts: Dict[str, int] = {}
        for task in tasks:
            X = collector.get(task, int(args.layer))
            if X is None or X.shape[0] == 0:
                raise RuntimeError(f"No activations collected for task={task}")
            X_raw[task] = X
            raw_counts[task] = int(X.shape[0])
            print(f"[Collect] task={task} raw_states={X.shape[0]} x {X.shape[1]}")

        # 5) fair preprocessing (cap + balance + center)
        X_bal, n0 = center_and_balance(
            X_raw,
            per_task_max_states=int(args.per_task_max_states),
            balance_to=str(args.balance_to),
            seed=int(args.seed) + 999,
        )
        print(f"[Fair] balanced states per task = {n0}")

        # 6) save
        save_np_dtype = np.float16 if args.save_dtype == "fp16" else np.float32

        files = {}
        for task, X in X_bal.items():
            path = os.path.join(args.out_dir, f"{task}.npy")
            if os.path.exists(path) and (not args.overwrite):
                raise FileExistsError(f"{path} exists. Use --overwrite to overwrite.")
            np.save(path, X.astype(save_np_dtype, copy=False))
            files[task] = os.path.basename(path)

        meta = {
            "config": {
                "model": args.model,
                "device": args.device,
                "model_dtype": args.model_dtype,
                "layer": int(args.layer),
                "n_prompts": int(args.n_prompts),
                "max_prompt_len": int(args.max_prompt_len),
                "calib_max_new_tokens": int(args.calib_max_new_tokens),
                "calib_decoding": args.calib_decoding,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "top_k": int(args.top_k),
                "batch_size": int(args.batch_size),
                "per_task_max_states": int(args.per_task_max_states),
                "balance_to": str(args.balance_to),
                "seed": int(args.seed),
                "save_dtype": args.save_dtype,
            },
            "tasks": tasks,
            "raw_counts": raw_counts,
            "balanced_states_per_task": int(n0),
            "files": files,
        }

        meta_path = os.path.join(args.out_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=to_py)

        print("[Done]")
        print(f"Saved meta: {meta_path}")
        print(f"Saved npy : {args.out_dir}/*.npy")

    finally:
        try:
            sys.stdout.flush()
        except Exception:
            pass
        sys.stdout = orig_stdout
        if txt_f is not None:
            try:
                txt_f.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()
