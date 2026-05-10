
# -*- coding: utf-8 -*-
"""
data_utils.py

封装你提供的 benchmark_dataloaders_aqua_prefix_default.py：
- 按任务名加载 subspace/eval 样本
- 返回 prompt 列表

你可以在 config 或 CLI 里指定任务列表，例如：
  aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa,gsm8k

注意：本文件假设 benchmark_dataloaders_aqua_prefix_default.py 与本文件同目录，
或者已加入 PYTHONPATH。
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

# 你的 attached 脚本（文件名作为 module）
import benchmark_dataloaders_aqua_prefix_default as bdl


TASK_TO_LOADER = {
    "gsm8k": bdl.load_gsm8k,
    "commonsenseqa": bdl.load_commonsenseqa,
    "strategyqa": bdl.load_strategyqa,
    "aqua": bdl.load_aqua,
    "arc_challenge": bdl.load_arc_challenge,
    "openbookqa": bdl.load_openbookqa,
}


def list_supported_tasks() -> List[str]:
    return sorted(TASK_TO_LOADER.keys())


def load_task(
    task: str,
    *,
    n_subspace: int,
    n_eval: int,
    seed: int,
    template_randomization: bool = True,
    template_seed: int = 0,
    shuffle_choices: bool = True,
    add_answer_prefix: bool = True,
    answer_prefix: str = "Final answer:",
) -> Tuple[List[bdl.Example], List[bdl.Example], Dict[str, Any]]:
    """
    返回 (subspace_examples, eval_examples, meta)
    """
    task = task.strip().lower()
    if task not in TASK_TO_LOADER:
        raise ValueError(f"Unknown task: {task}. Supported: {list_supported_tasks()}")

    loader = TASK_TO_LOADER[task]
    sub_exs, eval_exs, meta = loader(
        n_subspace=int(n_subspace),
        n_eval=int(n_eval),
        seed=int(seed),
        template_randomization=bool(template_randomization),
        template_seed=int(template_seed),
        shuffle_choices=bool(shuffle_choices),
        add_answer_prefix=bool(add_answer_prefix),
        answer_prefix=str(answer_prefix),
    )
    return sub_exs, eval_exs, meta


def get_prompts(exs: Sequence[bdl.Example]) -> List[str]:
    return [ex.prompt for ex in exs]


def build_mixture_prompts(
    tasks: Sequence[str],
    *,
    n_per_task: int,
    seed: int,
    split: str = "subspace",
    template_randomization: bool = True,
    template_seed: int = 0,
    shuffle_choices: bool = True,
    add_answer_prefix: bool = True,
    answer_prefix: str = "Final answer:",
) -> List[str]:
    """
    从多个任务各取 n_per_task 条 prompt，拼成一个 mixture（用于轨迹/回归）。
    split: "subspace" 或 "eval"
    """
    split = split.strip().lower()
    if split not in {"subspace", "eval"}:
        raise ValueError("split must be 'subspace' or 'eval'")

    prompts: List[str] = []
    # 为了稳定性：每个 task 单独 seed 偏移
    for i, task in enumerate(tasks):
        sub_exs, eval_exs, _ = load_task(
            task,
            n_subspace=n_per_task,
            n_eval=n_per_task,
            seed=seed + 1000 * i,
            template_randomization=template_randomization,
            template_seed=template_seed,
            shuffle_choices=shuffle_choices,
            add_answer_prefix=add_answer_prefix,
            answer_prefix=answer_prefix,
        )
        exs = sub_exs if split == "subspace" else eval_exs
        prompts.extend(get_prompts(exs[:n_per_task]))
    return prompts
