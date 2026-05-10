#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

"""
Usage examples:

1) Run step1 (cached) + default eval ppl (step4)
python run_svdllm_whitening_only.py \
  --svdllm_root ./svdllm_vendor \
  --model meta-llama/Llama-2-7b-chat-hf \
  --ratio 0.2 \
  --dataset wikitext2 \
  --whitening_nsamples 128 \
  --model_seq_len 2048 \
  --save_path ./outputs/svdllm_whiten_r0.2

2) Force recompute (ignore cache) + eval ppl
python run_svdllm_whitening_only.py ... --force_recompute

3) Reuse ckpt but force re-run eval logs
python run_svdllm_whitening_only.py ... --force_eval

4) Run efficiency eval too
python run_svdllm_whitening_only.py ... --eval_eff
"""

# ----------------------------
# Sentinel + checkpoint heuristics
# ----------------------------

_STEP1_SENTINEL = ".svdllm_step1.done"

# HF-ish checkpoint signatures (optional; step1 may NOT produce HF checkpoint)
_WEIGHT_GLOBS = [
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model-*.bin",     # sharded
    "model-*.safetensors",     # sharded
]
_CONFIG_FILES = ["config.json"]

# Non-HF SVDLLM step1 artifact hints (very permissive)
_SVDLLM_STEP1_HINT_GLOBS = [
    "*.pt", "*.pth", "*.npz", "*.npy", "*.pkl",
    "*whiten*", "*whitening*", "*svd*", "*U*", "*V*", "*S*",
    "*singular*", "*decomp*",
]

_TQDM_RE = re.compile(r"(?P<cur>\d+)\s*/\s*(?P<tot>\d+)")


def _looks_like_hf_checkpoint(d: Path) -> bool:
    if not d.is_dir():
        return False
    if not all((d / c).exists() for c in _CONFIG_FILES):
        return False
    for g in _WEIGHT_GLOBS:
        if list(d.glob(g)):
            return True
    if list(d.glob("*.bin")) or list(d.glob("*.safetensors")):
        return True
    return False


def _looks_like_svdllm_step1_dir(d: Path) -> bool:
    if not d.is_dir():
        return False
    if (d / _STEP1_SENTINEL).exists():
        return True
    for g in _SVDLLM_STEP1_HINT_GLOBS:
        if list(d.glob(g)):
            return True
    # sometimes outputs land in a subfolder
    try:
        for sub in d.iterdir():
            if sub.is_dir() and (sub / _STEP1_SENTINEL).exists():
                return True
    except Exception:
        pass
    return False


def _looks_like_any_checkpoint_or_artifacts(d: Path) -> bool:
    return _looks_like_hf_checkpoint(d) or _looks_like_svdllm_step1_dir(d)


def _ratio_aliases(r: float) -> set[str]:
    aliases: set[str] = set()
    fmts = ["{:.6f}", "{:.4f}", "{:.3f}", "{:.2f}"]
    for f in fmts:
        s = f.format(r).rstrip("0").rstrip(".")
        if s:
            aliases.add(s)
            aliases.add("r" + s)
    s0 = str(r).rstrip("0").rstrip(".")
    if s0:
        aliases.add(s0)
        aliases.add("r" + s0)
    return aliases


def _sanitize_model_id(model: str) -> str:
    p = Path(model)
    if p.exists():
        return p.name
    return model.split("/")[-1]


def mark_step1_done(save_path: Path, meta: dict) -> None:
    save_path.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["done_time_unix"] = time.time()
    (save_path / _STEP1_SENTINEL).write_text(json.dumps(meta, indent=2))


def find_cached_by_ratio(outputs_dir: Path, ratio: float) -> Optional[Path]:
    """
    Scan outputs_dir for any subdir that matches the ratio in its name and
    contains either:
      - step1 sentinel OR
      - HF checkpoint OR
      - SVDLLM step1 artifacts
    Return the best candidate (prefer sentinel, then newest).
    """
    outputs_dir = outputs_dir.resolve()
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        return None

    aliases = _ratio_aliases(ratio)
    candidates: list[Path] = []
    for d in outputs_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name.lower()
        if any(a in name for a in aliases):
            if _looks_like_any_checkpoint_or_artifacts(d):
                candidates.append(d)

    if not candidates:
        return None

    def score(d: Path) -> tuple[int, float]:
        s = 0
        if (d / _STEP1_SENTINEL).exists():
            s += 10
        if _looks_like_hf_checkpoint(d):
            s += 5
        # tie-break newest
        return (s, d.stat().st_mtime)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def best_model_path_for_eval(save_path: Path) -> Path:
    """
    SVDLLM step4/5 expects --model_path.
    If an HF checkpoint exists in save_path or its direct subdirs, prefer it;
    otherwise use save_path itself.
    """
    save_path = save_path.resolve()
    if _looks_like_hf_checkpoint(save_path):
        return save_path

    sub_ckpts = []
    if save_path.exists() and save_path.is_dir():
        for d in save_path.iterdir():
            if d.is_dir() and _looks_like_hf_checkpoint(d):
                sub_ckpts.append(d)

    if len(sub_ckpts) == 1:
        return sub_ckpts[0]
    if sub_ckpts:
        sub_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return sub_ckpts[0]

    return save_path


# ----------------------------
# Streaming subprocess runner with tqdm
# ----------------------------

def _run_cmd(
    cmd: list[str],
    cwd: Path,
    env: dict,
    log_file: Optional[Path] = None,
    desc: str = "",
) -> str:
    print("\n[CMD]")
    print("  " + " ".join(cmd))
    print(f"[CWD] {cwd}\n")

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )

    all_lines: list[str] = []
    f = open(log_file, "w") if log_file is not None else None

    pbar: Optional[tqdm] = None
    last_tot: Optional[int] = None

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            all_lines.append(line)
            if f:
                f.write(line)

            # echo stdout (comment out if you want less spam)
            print(line, end="")

            m = _TQDM_RE.search(line)
            if m:
                cur = int(m.group("cur"))
                tot = int(m.group("tot"))

                if pbar is None or (last_tot is not None and tot != last_tot):
                    if pbar is not None:
                        pbar.close()
                    pbar = tqdm(total=tot, desc=(desc or "progress"), leave=True)
                    last_tot = tot

                if pbar is not None:
                    if cur >= pbar.n:
                        pbar.update(cur - pbar.n)
                    else:
                        pbar.n = cur
                        pbar.refresh()

        ret = proc.wait()
    finally:
        if pbar is not None:
            # try to complete bar
            try:
                if last_tot is not None and pbar.n < last_tot:
                    pbar.update(last_tot - pbar.n)
            except Exception:
                pass
            pbar.close()
        if f:
            f.close()

    out = "".join(all_lines)
    if ret != 0:
        raise RuntimeError(f"Command failed with exit code {ret}.\n--- Output ---\n{out}")
    return out


def _parse_ppl(output: str) -> Optional[float]:
    patterns = [
        r"\bppl\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
        r"\bperplexity\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, output, flags=re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    return None


@dataclass
class EvalResult:
    ppl: Optional[float] = None
    ppl_log: Optional[str] = None
    eff_log: Optional[str] = None


# ----------------------------
# SVDLLM runner + eval runners
# ----------------------------

def run_svdllm_step1(
    svdllm_root: Path,
    model: str,
    save_path: Path,
    ratio: float,
    dataset: str,
    whitening_nsamples: int,
    seed: int,
    model_seq_len: int,
    extra_args: Optional[list[str]] = None,
) -> str:
    cmd = [
        sys.executable, "SVDLLM.py",
        "--step", "1",
        "--ratio", str(ratio),
        "--model", model,
        "--whitening_nsamples", str(whitening_nsamples),
        "--dataset", dataset,
        "--seed", str(seed),
        "--model_seq_len", str(model_seq_len),
        "--save_path", str(save_path),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(svdllm_root) + os.pathsep + env.get("PYTHONPATH", "")

    step1_log = save_path / "step1.log"
    out = _run_cmd(cmd, cwd=svdllm_root, env=env, log_file=step1_log, desc="step1")

    # Mark done for reliable caching even if output isn't HF-style
    mark_step1_done(save_path, {
        "model": model,
        "ratio": ratio,
        "dataset": dataset,
        "whitening_nsamples": whitening_nsamples,
        "seed": seed,
        "model_seq_len": model_seq_len,
    })
    return out


def run_eval_steps(
    svdllm_root: Path,
    model_path: Path,
    run_ppl: bool,
    run_eff: bool,
    force_eval: bool,
) -> EvalResult:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(svdllm_root) + os.pathsep + env.get("PYTHONPATH", "")

    res = EvalResult()

    if run_ppl:
        ppl_log = model_path / "eval_step4_ppl.log"
        ppl_json = model_path / "eval_step4_ppl.json"

        if ppl_json.exists() and not force_eval:
            try:
                data = json.loads(ppl_json.read_text())
                res.ppl = data.get("ppl", None)
                res.ppl_log = str(ppl_log) if ppl_log.exists() else None
            except Exception:
                pass
        else:
            cmd = [sys.executable, "SVDLLM.py", "--step", "4", "--model_path", str(model_path)]
            out = _run_cmd(cmd, cwd=svdllm_root, env=env, log_file=ppl_log, desc="eval ppl (step4)")
            res.ppl = _parse_ppl(out)
            res.ppl_log = str(ppl_log)
            ppl_json.write_text(json.dumps({"ppl": res.ppl}, indent=2))

    if run_eff:
        eff_log = model_path / "eval_step5_eff.log"
        if eff_log.exists() and not force_eval:
            res.eff_log = str(eff_log)
        else:
            cmd = [sys.executable, "SVDLLM.py", "--step", "5", "--model_path", str(model_path)]
            _run_cmd(cmd, cwd=svdllm_root, env=env, log_file=eff_log, desc="eval eff (step5)")
            res.eff_log = str(eff_log)

    return res


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Run SVD-LLM Step1 (whitening+SVD) with caching + automatic eval (step4/5)."
    )
    p.add_argument("--svdllm_root", type=str, required=True,
                   help="Path containing SVDLLM.py, component/, utils/")
    p.add_argument("--model", type=str, required=True,
                   help="HF model repo or local path")

    p.add_argument("--save_path", type=str, required=True,
                   help="Where SVD-LLM saves step1 outputs (and where sentinel/logs are stored).")

    # NEW: scan outputs_dir for existing same-ratio runs (default: save_path.parent)
    p.add_argument("--outputs_dir", type=str, default="",
                   help="Directory to scan for cached runs at same ratio (default: parent of save_path).")

    p.add_argument("--ratio", type=float, default=0.2,
                   help="Compression ratio (parameter reduction fraction). Default=0.2")
    p.add_argument("--dataset", type=str, default="wikitext2",
                   help="Whitening dataset name (used in step 1)")
    p.add_argument("--whitening_nsamples", type=int, default=128,
                   help="Number of whitening samples")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model_seq_len", type=int, default=2048)
    p.add_argument("--extra", type=str, default="",
                   help="Extra args appended verbatim to step 1, e.g. '--some_flag 1'")

    # Caching controls
    p.add_argument("--force_recompute", action="store_true",
                   help="Ignore cached artifacts and rerun step 1.")
    p.add_argument("--force_eval", action="store_true",
                   help="Ignore cached eval logs and rerun eval steps.")

    # Eval controls
    p.add_argument("--eval_ppl", action="store_true",
                   help="Run perplexity evaluation (SVDLLM.py --step 4).")
    p.add_argument("--eval_eff", action="store_true",
                   help="Run efficiency evaluation (SVDLLM.py --step 5).")
    p.add_argument("--no_eval", action="store_true",
                   help="Skip all evaluation (overrides eval flags).")

    args = p.parse_args()

    svdllm_root = Path(args.svdllm_root).resolve()
    if not (svdllm_root / "SVDLLM.py").exists():
        raise FileNotFoundError(f"Cannot find SVDLLM.py under {svdllm_root}")

    save_path = Path(args.save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    outputs_dir = Path(args.outputs_dir).resolve() if args.outputs_dir.strip() else save_path.parent.resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Decide whether to reuse cached run (by ratio inside outputs_dir)
    # ----------------------------
    chosen_save_path = save_path
    if not args.force_recompute:
        # Prefer exact save_path if it already looks like step1 outputs
        if _looks_like_any_checkpoint_or_artifacts(save_path):
            chosen_save_path = save_path
            print(f"[CACHE] Using existing artifacts at save_path:\n  {chosen_save_path}")
        else:
            cached = find_cached_by_ratio(outputs_dir, args.ratio)
            if cached is not None:
                chosen_save_path = cached
                print(f"[CACHE] Found cached run with same ratio in outputs_dir, reusing:\n  {chosen_save_path}")
            else:
                print("[CACHE] No cached run found; will run step1.")

    else:
        print("[FORCE] --force_recompute set; will rerun step1.")

    # ----------------------------
    # Run Step 1 if needed
    # ----------------------------
    if args.force_recompute or not _looks_like_any_checkpoint_or_artifacts(chosen_save_path):
        print(f"[RUN] Running step1 into:\n  {chosen_save_path}")
        extra_args = args.extra.split() if args.extra.strip() else None
        run_svdllm_step1(
            svdllm_root=svdllm_root,
            model=args.model,
            save_path=chosen_save_path,
            ratio=args.ratio,
            dataset=args.dataset,
            whitening_nsamples=args.whitening_nsamples,
            seed=args.seed,
            model_seq_len=args.model_seq_len,
            extra_args=extra_args,
        )
    else:
        print("[RUN] Step1 already present; skipping recompute (use --force_recompute to override).")

    if not _looks_like_any_checkpoint_or_artifacts(chosen_save_path):
        raise RuntimeError(
            "Step 1 finished (or was skipped) but no artifacts were detected.\n"
            f"Checked: {chosen_save_path}\n"
            "Inspect step1.log to see where SVD-LLM wrote outputs."
        )

    # ----------------------------
    # Eval default behavior:
    #   - if --no_eval: skip
    #   - else if user set any eval flag: honor them
    #   - else: default to eval ppl (step4)
    # ----------------------------
    if args.no_eval:
        run_ppl = run_eff = False
    else:
        any_flags = args.eval_ppl or args.eval_eff
        if any_flags:
            run_ppl = args.eval_ppl
            run_eff = args.eval_eff
        else:
            run_ppl = True
            run_eff = False

    model_path = best_model_path_for_eval(chosen_save_path)
    print(f"\n[INFO] Using model_path for eval:\n  {model_path}")

    if run_ppl or run_eff:
        print("\n[EVAL] Running requested evaluations...")
        res = run_eval_steps(
            svdllm_root=svdllm_root,
            model_path=model_path,
            run_ppl=run_ppl,
            run_eff=run_eff,
            force_eval=args.force_eval,
        )

        print("\n[RESULTS]")
        if run_ppl:
            print(f"  perplexity (step4): {res.ppl}  (log: {res.ppl_log})")
        if run_eff:
            print(f"  efficiency (step5): log saved at {res.eff_log}")
    else:
        print("\n[NOTE] Evaluation skipped (--no_eval).")

    print(f"\n[DONE] artifacts/checkpoint dir:\n  {chosen_save_path}")


if __name__ == "__main__":
    main()
