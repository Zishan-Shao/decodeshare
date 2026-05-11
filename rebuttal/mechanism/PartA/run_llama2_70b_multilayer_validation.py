#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_llama2_70b_multilayer_validation.py

Orchestrate a stronger Llama-2-70B validation sweep around the existing PartA
artifacts:
  1) A3 basis construction / smoke causal test
  2) A3 saved-basis larger eval
  3) A4 answer-format confound checks
  4) A5 probe-derived fmt/residual split
  5) summary aggregation

The goal is to make the 70B story reviewer-harder without modifying the core
experiment scripts themselves.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def _split_csv(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


def _atomic_json_dump(obj: Any, out_path: str) -> None:
    out_path = os.path.expanduser(out_path)
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out_path)


def _shell_join(argv: Sequence[str]) -> str:
    return shlex.join([str(x) for x in argv])


def _parse_layer_groups(spec: str) -> Tuple[List[int], Dict[int, str], Dict[str, List[int]]]:
    ordered_layers: List[int] = []
    layer_to_group: Dict[int, str] = {}
    groups: Dict[str, List[int]] = {}

    raw = str(spec).strip()
    if not raw:
        raise ValueError("Empty --layer_groups")

    for block in raw.split(";"):
        block = block.strip()
        if not block:
            continue
        if ":" not in block:
            raise ValueError(
                "Each layer-group block must have the form 'group:l1,l2,...'. "
                f"Bad block: {block!r}"
            )
        group, layers_csv = block.split(":", 1)
        group = group.strip()
        layers = []
        for part in _split_csv(layers_csv):
            layer = int(part)
            layers.append(layer)
            if layer not in layer_to_group:
                layer_to_group[layer] = group
                ordered_layers.append(layer)
        if not layers:
            raise ValueError(f"Group {group!r} has no layers.")
        groups[group] = layers

    if not ordered_layers:
        raise ValueError("No layers parsed from --layer_groups.")
    return ordered_layers, layer_to_group, groups


@dataclass
class StageRecord:
    status: str
    log_path: str
    output_path: str
    command: List[str]


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.python_bin = str(args.python_bin)
        self.run_dir = os.path.join(os.path.expanduser(args.out_base), str(args.run_id))
        self.logs_dir = os.path.join(self.run_dir, "logs")
        self.commands_path = os.path.join(self.run_dir, "commands.sh")
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.summary_dir = os.path.join(self.run_dir, "summary")
        self.a3_basis_dir = os.path.join(self.run_dir, "a3_basis")
        self.a3_eval_dir = os.path.join(self.run_dir, "a3_eval")
        self.a4_dir = os.path.join(self.run_dir, "a4_option_text")
        self.a5_dir = os.path.join(self.run_dir, "a5_probe_split")
        self.layer_entries: Dict[str, Dict[str, Any]] = {}

    def ensure_dirs(self) -> None:
        for path in [
            self.run_dir,
            self.logs_dir,
            self.summary_dir,
            self.a3_basis_dir,
            self.a3_eval_dir,
            self.a4_dir,
            self.a5_dir,
        ]:
            os.makedirs(path, exist_ok=True)

    def append_command(self, argv: Sequence[str]) -> None:
        with open(self.commands_path, "a", encoding="utf-8") as f:
            f.write(_shell_join(argv))
            f.write("\n")

    def write_manifest(self, final_status: Optional[str] = None) -> None:
        payload = {
            "config": {
                "model": self.args.model,
                "device": self.args.device,
                "dtype": self.args.dtype,
                "device_map": self.args.device_map,
                "max_memory_per_gpu_gb": self.args.max_memory_per_gpu_gb,
                "max_memory_map": self.args.max_memory_map,
                "cpu_offload_gb": self.args.cpu_offload_gb,
                "layer_groups": self.args.layer_groups,
                "tag_prefix": self.args.tag_prefix,
                "skip_existing": bool(self.args.skip_existing),
                "run_a4": bool(self.args.run_a4),
                "run_a5": bool(self.args.run_a5),
                "summary_main_task": self.args.summary_main_task,
                "out_base": self.args.out_base,
                "run_id": self.args.run_id,
                "dry_run": bool(self.args.dry_run),
            },
            "paths": {
                "run_dir": self.run_dir,
                "commands_sh": self.commands_path,
                "summary_dir": self.summary_dir,
                "a3_basis_dir": self.a3_basis_dir,
                "a3_eval_dir": self.a3_eval_dir,
                "a4_dir": self.a4_dir,
                "a5_dir": self.a5_dir,
            },
            "layers": self.layer_entries,
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        if final_status is not None:
            payload["final_status"] = str(final_status)
        _atomic_json_dump(payload, self.manifest_path)

    def run_step(
        self,
        *,
        name: str,
        argv: List[str],
        log_path: str,
        output_path: str,
    ) -> StageRecord:
        self.append_command(argv)

        if bool(self.args.skip_existing) and os.path.exists(output_path):
            print(f"[Skip] {name} -> {output_path}")
            return StageRecord(status="skipped_existing", log_path=log_path, output_path=output_path, command=argv)

        if bool(self.args.dry_run):
            print(f"[DryRun] {name}")
            print("  " + _shell_join(argv))
            return StageRecord(status="dry_run", log_path=log_path, output_path=output_path, command=argv)

        print(f"[Run] {name}")
        print(f"  log={log_path}")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as logf:
            logf.write("$ " + _shell_join(argv) + "\n\n")
            logf.flush()
            proc = subprocess.run(argv, cwd=ROOT_DIR, stdout=logf, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{name} failed with exit code {proc.returncode}. See {log_path}")
        return StageRecord(status="completed", log_path=log_path, output_path=output_path, command=argv)


def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run a stronger Llama-2-70B multi-layer PartA validation sweep.")
    ap.add_argument("--python_bin", type=str, default=sys.executable)
    ap.add_argument("--model", type=str, default="meta-llama/Llama-2-70b-chat-hf")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--max_memory_per_gpu_gb", type=float, default=55.0)
    ap.add_argument("--max_memory_map", type=str, default="")
    ap.add_argument("--cpu_offload_gb", type=float, default=120.0)

    ap.add_argument(
        "--layer_groups",
        type=str,
        default="early:10,14,18;mid:22,25,28;late:58,62,66",
        help="Semicolon-separated spec: group:l1,l2,...;group:l3,l4,...",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--template_seed", type=int, default=1234)
    ap.add_argument("--template_randomization", type=int, default=1, choices=[0, 1])
    ap.add_argument("--shuffle_choices", type=int, default=1, choices=[0, 1])
    ap.add_argument("--add_answer_prefix", type=int, default=1, choices=[0, 1])
    ap.add_argument("--answer_prefix", type=str, default="\nFinal answer:")
    ap.add_argument("--fc_prefix_mode", type=str, default="auto", choices=["auto", "always", "never"])

    ap.add_argument("--tasks_subspace", type=str, default="commonsenseqa,arc_challenge,openbookqa,qasc,logiqa,boolq")
    ap.add_argument("--tasks_smoke_eval", type=str, default="commonsenseqa,arc_challenge")
    ap.add_argument("--tasks_main_eval", type=str, default="commonsenseqa")
    ap.add_argument("--tasks_a4_eval", type=str, default="commonsenseqa,arc_challenge")
    ap.add_argument("--tasks_a5_probe", type=str, default="")
    ap.add_argument("--tasks_a5_eval", type=str, default="commonsenseqa,arc_challenge")

    ap.add_argument("--a3_n_prompts", type=int, default=32)
    ap.add_argument("--a3_smoke_eval_n", type=int, default=16)
    ap.add_argument("--a3_main_eval_n", type=int, default=64)
    ap.add_argument("--a3_batch_size", type=int, default=1)
    ap.add_argument("--a3_max_prompt_len", type=int, default=512)
    ap.add_argument("--a3_calib_decode_max_new_tokens", type=int, default=128)
    ap.add_argument("--a3_per_task_max_states", type=int, default=12000)
    ap.add_argument("--pca_var", type=float, default=0.95)
    ap.add_argument("--min_dim", type=int, default=8)
    ap.add_argument("--max_dim", type=int, default=256)
    ap.add_argument("--tau", type=float, default=0.001)
    ap.add_argument("--m_shared", type=str, default="all")
    ap.add_argument("--k_eval", type=int, default=128)
    ap.add_argument("--alpha_remove", type=float, default=1.0)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)
    ap.add_argument("--perm_iters", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05)

    ap.add_argument("--run_a4", type=int, default=1, choices=[0, 1])
    ap.add_argument("--run_a5", type=int, default=1, choices=[0, 1])
    ap.add_argument("--a4_eval_specs", type=str, default="number_rewrite,text_rewrite")
    ap.add_argument("--a4_eval_n", type=int, default=64)
    ap.add_argument("--a4_max_prompt_len", type=int, default=2048)
    ap.add_argument("--a4_bootstrap_iters", type=int, default=2000)
    ap.add_argument("--a4_perm_iters", type=int, default=5000)

    ap.add_argument("--a5_n_probe_prompts", type=int, default=64)
    ap.add_argument("--a5_probe_max_new_tokens", type=int, default=32)
    ap.add_argument("--a5_probe_batch_size", type=int, default=2)
    ap.add_argument("--a5_per_task_max_probe_states", type=int, default=3000)
    ap.add_argument("--a5_probe_tags", type=str, default="answer_readout,option_letter,newline")
    ap.add_argument("--a5_probe_min_pos", type=int, default=20)
    ap.add_argument("--a5_probe_test_size", type=float, default=0.2)
    ap.add_argument("--a5_eval_n", type=int, default=32)
    ap.add_argument("--a5_eval_batch_size", type=int, default=1)
    ap.add_argument("--a5_max_prompt_len", type=int, default=2048)

    ap.add_argument("--summary_main_task", type=str, default="commonsenseqa")
    ap.add_argument("--tag_prefix", type=str, default="llama2_70b_multilayer")
    ap.add_argument(
        "--out_base",
        type=str,
        default=os.path.join(ROOT_DIR, "results", "rebuttal_scaling", "llama2_70b_multilayer_validation"),
    )
    ap.add_argument("--run_id", type=str, default=dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    ap.add_argument("--skip_existing", type=int, default=1, choices=[0, 1])
    ap.add_argument("--dry_run", type=int, default=0, choices=[0, 1])
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    ordered_layers, layer_to_group, groups = _parse_layer_groups(args.layer_groups)

    if not str(args.tasks_a5_probe).strip():
        args.tasks_a5_probe = args.tasks_subspace

    runner = Runner(args)
    runner.ensure_dirs()

    with open(runner.commands_path, "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n\n")

    a3_script = os.path.join(THIS_DIR, "exp_A3_causal_decode_only_controls.py")
    a3_eval_script = os.path.join(THIS_DIR, "exp_A3_eval_saved_basis.py")
    a4_script = os.path.join(THIS_DIR, "exp_A4_option_text_forced_choice.py")
    a5_script = os.path.join(THIS_DIR, "exp_A5_probe_split_causal.py")
    summary_script = os.path.join(THIS_DIR, "summarize_llama2_70b_multilayer_validation.py")

    if not os.path.exists(summary_script):
        raise FileNotFoundError(f"Missing summary script: {summary_script}")

    for layer in ordered_layers:
        group = layer_to_group[int(layer)]
        stage_tag = f"{args.tag_prefix}_{group}_l{int(layer)}"

        a3_json = os.path.join(runner.a3_basis_dir, f"exp_A3_causal_layer{int(layer)}_{stage_tag}.json")
        a3_basis_npz = os.path.join(runner.a3_basis_dir, f"exp_A3_bases_layer{int(layer)}_{stage_tag}.npz")
        a3_eval_json = os.path.join(runner.a3_eval_dir, f"exp_A3_eval_saved_basis_layer{int(layer)}_{stage_tag}.json")
        a4_json = os.path.join(runner.a4_dir, f"exp_A4_option_text_layer{int(layer)}_{stage_tag}.json")
        a5_json = os.path.join(runner.a5_dir, f"exp_A5_probe_split_layer{int(layer)}_{stage_tag}.json")

        layer_key = str(int(layer))
        runner.layer_entries[layer_key] = {
            "group": group,
            "paths": {
                "a3_json": a3_json,
                "a3_basis_npz": a3_basis_npz,
                "a3_eval_json": a3_eval_json,
                "a4_json": a4_json,
                "a5_json": a5_json,
            },
            "stages": {},
        }
        runner.write_manifest()

        a3_cmd = [
            runner.python_bin,
            a3_script,
            "--model", args.model,
            "--device", args.device,
            "--dtype", args.dtype,
            "--device_map", args.device_map,
            "--max_memory_per_gpu_gb", str(args.max_memory_per_gpu_gb),
            "--max_memory_map", str(args.max_memory_map),
            "--cpu_offload_gb", str(args.cpu_offload_gb),
            "--layer", str(int(layer)),
            "--tasks_subspace", args.tasks_subspace,
            "--tasks_eval", args.tasks_smoke_eval,
            "--n_prompts", str(args.a3_n_prompts),
            "--eval_n", str(args.a3_smoke_eval_n),
            "--seed", str(args.seed),
            "--template_seed", str(args.template_seed),
            "--template_randomization", str(args.template_randomization),
            "--shuffle_choices", str(args.shuffle_choices),
            "--add_answer_prefix", str(args.add_answer_prefix),
            "--answer_prefix", args.answer_prefix,
            "--batch_size", str(args.a3_batch_size),
            "--max_prompt_len", str(args.a3_max_prompt_len),
            "--calib_decode_max_new_tokens", str(args.a3_calib_decode_max_new_tokens),
            "--per_task_max_states", str(args.a3_per_task_max_states),
            "--pca_var", str(args.pca_var),
            "--min_dim", str(args.min_dim),
            "--max_dim", str(args.max_dim),
            "--tau", str(args.tau),
            "--m_shared", str(args.m_shared),
            "--k_eval", str(args.k_eval),
            "--alpha_remove", str(args.alpha_remove),
            "--fc_prefix_mode", args.fc_prefix_mode,
            "--fc_answer_prefix", args.answer_prefix,
            "--bootstrap_iters", str(args.bootstrap_iters),
            "--perm_iters", str(args.perm_iters),
            "--alpha", str(args.alpha),
            "--out_dir", runner.a3_basis_dir,
            "--tag", stage_tag,
        ]
        a3_record = runner.run_step(
            name=f"A3 basis/smoke layer={layer} group={group}",
            argv=a3_cmd,
            log_path=os.path.join(runner.logs_dir, f"a3_layer{int(layer)}.log"),
            output_path=a3_json,
        )
        runner.layer_entries[layer_key]["stages"]["a3_basis"] = asdict(a3_record)
        runner.write_manifest()

        a3_eval_cmd = [
            runner.python_bin,
            a3_eval_script,
            "--basis_npz", a3_basis_npz,
            "--model", args.model,
            "--device", args.device,
            "--dtype", args.dtype,
            "--device_map", args.device_map,
            "--max_memory_per_gpu_gb", str(args.max_memory_per_gpu_gb),
            "--max_memory_map", str(args.max_memory_map),
            "--cpu_offload_gb", str(args.cpu_offload_gb),
            "--layer", str(int(layer)),
            "--tasks_eval", args.tasks_main_eval,
            "--conditions", "baseline,shared,ctrl_energy,rand_energy",
            "--eval_n", str(args.a3_main_eval_n),
            "--batch_size", "1",
            "--max_prompt_len", str(max(args.a3_max_prompt_len, 2048)),
            "--template_randomization", str(args.template_randomization),
            "--template_seed", str(args.template_seed),
            "--shuffle_choices", str(args.shuffle_choices),
            "--add_answer_prefix", str(args.add_answer_prefix),
            "--answer_prefix", args.answer_prefix,
            "--fc_answer_prefix", args.answer_prefix,
            "--fc_prefix_mode", args.fc_prefix_mode,
            "--bootstrap_iters", str(args.bootstrap_iters),
            "--perm_iters", str(args.perm_iters),
            "--alpha", str(args.alpha),
            "--seed", str(args.seed),
            "--out_dir", runner.a3_eval_dir,
            "--tag", stage_tag,
        ]
        a3_eval_record = runner.run_step(
            name=f"A3 saved-basis eval layer={layer} group={group}",
            argv=a3_eval_cmd,
            log_path=os.path.join(runner.logs_dir, f"a3_eval_layer{int(layer)}.log"),
            output_path=a3_eval_json,
        )
        runner.layer_entries[layer_key]["stages"]["a3_eval"] = asdict(a3_eval_record)
        runner.write_manifest()

        if bool(args.run_a4):
            a4_cmd = [
                runner.python_bin,
                a4_script,
                "--a3_json", a3_json,
                "--basis_npz", a3_basis_npz,
                "--eval_specs", args.a4_eval_specs,
                "--eval_tasks", args.tasks_a4_eval,
                "--eval_n", str(args.a4_eval_n),
                "--device", args.device,
                "--dtype", args.dtype,
                "--device_map", args.device_map,
                "--max_memory_per_gpu_gb", str(args.max_memory_per_gpu_gb),
                "--max_memory_map", str(args.max_memory_map),
                "--cpu_offload_gb", str(args.cpu_offload_gb),
                "--max_prompt_len", str(args.a4_max_prompt_len),
                "--bootstrap_iters", str(args.a4_bootstrap_iters),
                "--perm_iters", str(args.a4_perm_iters),
                "--alpha", str(args.alpha),
                "--out_dir", runner.a4_dir,
                "--tag", stage_tag,
            ]
            a4_record = runner.run_step(
                name=f"A4 confound check layer={layer} group={group}",
                argv=a4_cmd,
                log_path=os.path.join(runner.logs_dir, f"a4_layer{int(layer)}.log"),
                output_path=a4_json,
            )
            runner.layer_entries[layer_key]["stages"]["a4"] = asdict(a4_record)
            runner.write_manifest()

        if bool(args.run_a5):
            a5_cmd = [
                runner.python_bin,
                a5_script,
                "--basis_npz", a3_basis_npz,
                "--model", args.model,
                "--device", args.device,
                "--dtype", args.dtype,
                "--device_map", args.device_map,
                "--max_memory_per_gpu_gb", str(args.max_memory_per_gpu_gb),
                "--max_memory_map", str(args.max_memory_map),
                "--cpu_offload_gb", str(args.cpu_offload_gb),
                "--layer", str(int(layer)),
                "--tasks_probe", args.tasks_a5_probe,
                "--tasks_eval", args.tasks_a5_eval,
                "--n_probe_prompts", str(args.a5_n_probe_prompts),
                "--probe_max_new_tokens", str(args.a5_probe_max_new_tokens),
                "--probe_batch_size", str(args.a5_probe_batch_size),
                "--per_task_max_probe_states", str(args.a5_per_task_max_probe_states),
                "--probe_tags", args.a5_probe_tags,
                "--probe_min_pos", str(args.a5_probe_min_pos),
                "--probe_test_size", str(args.a5_probe_test_size),
                "--eval_n", str(args.a5_eval_n),
                "--eval_batch_size", str(args.a5_eval_batch_size),
                "--max_prompt_len", str(args.a5_max_prompt_len),
                "--seed", str(args.seed),
                "--template_seed", str(args.template_seed),
                "--template_randomization", str(args.template_randomization),
                "--shuffle_choices", str(args.shuffle_choices),
                "--add_answer_prefix", str(args.add_answer_prefix),
                "--answer_prefix", args.answer_prefix,
                "--fc_answer_prefix", args.answer_prefix,
                "--fc_prefix_mode", args.fc_prefix_mode,
                "--bootstrap_iters", str(args.bootstrap_iters),
                "--perm_iters", str(args.perm_iters),
                "--alpha", str(args.alpha),
                "--out_dir", runner.a5_dir,
                "--tag", stage_tag,
            ]
            a5_record = runner.run_step(
                name=f"A5 probe split layer={layer} group={group}",
                argv=a5_cmd,
                log_path=os.path.join(runner.logs_dir, f"a5_layer{int(layer)}.log"),
                output_path=a5_json,
            )
            runner.layer_entries[layer_key]["stages"]["a5"] = asdict(a5_record)
            runner.write_manifest()

    summary_cmd = [
        runner.python_bin,
        summary_script,
        "--root_dirs", runner.run_dir,
        "--layer_groups", args.layer_groups,
        "--main_task", args.summary_main_task,
        "--a4_eval_specs", args.a4_eval_specs,
        "--out_dir", runner.summary_dir,
        "--tag", args.tag_prefix,
    ]
    summary_record = runner.run_step(
        name="summary aggregation",
        argv=summary_cmd,
        log_path=os.path.join(runner.logs_dir, "summary.log"),
        output_path=os.path.join(runner.summary_dir, f"llama2_70b_multilayer_summary_{args.tag_prefix}.md"),
    )
    runner.layer_entries["_summary"] = {"stages": {"summary": asdict(summary_record)}, "group": "summary", "paths": {}}
    runner.write_manifest(final_status="completed" if not bool(args.dry_run) else "dry_run")

    print(f"[Done] run_dir={runner.run_dir}")
    print(f"[Done] manifest={runner.manifest_path}")


if __name__ == "__main__":
    main()
