"""Shared-subspace construction and intervention helpers.

This module is the maintained home for the small API that paper experiments
share across H1/H2/H3 and downstream provenance scripts. The historical
``decodeshare.disturb_cross_task_all_shared`` module now re-exports from here
for compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


HIDDEN_DIM = 4096

__all__ = [
    "HIDDEN_DIM",
    "find_fully_shared_basis_improved",
    "infer_hidden_dim_from_model",
    "get_model_layers",
    "compute_cross_task_subspace",
    "validate_subspace_basis",
    "JointSubspaceRemovalHook",
]


def find_fully_shared_basis_improved(
    task_contributions: Dict[str, Dict[str, Any]],
    all_tasks: Sequence[str],
    cross_subspace_dim: int,
    min_tasks_shared: Optional[int] = None,
    relative_threshold: float = 0.001,
    top_k_components: Optional[int] = None,
) -> List[int]:
    """Return component indices whose variance contribution is shared by many tasks.

    A component is counted for a task when its raw variance exceeds
    ``relative_threshold * task_total_variance``. By default a component must be
    significant for every task in ``all_tasks``.
    """
    if min_tasks_shared is None:
        min_tasks_shared = len(all_tasks)
    if top_k_components is None:
        top_k_components = cross_subspace_dim

    num_components_to_check = min(cross_subspace_dim, top_k_components)
    print(f"Finding components shared by at least {min_tasks_shared} tasks...")
    print(f"  Relative threshold: {relative_threshold * 100:.3f}% of task variance")
    print(f"  Components checked: {num_components_to_check}")

    task_total_variances: Dict[str, float] = {}
    for task_name in all_tasks:
        contrib = task_contributions.get(task_name)
        if not contrib:
            print(f"  Warning: task {task_name} has no contribution data")
            return []
        if "total_variance" in contrib:
            task_total_variances[task_name] = float(contrib["total_variance"])
        elif "raw_variances" in contrib:
            task_total_variances[task_name] = float(np.sum(contrib["raw_variances"]))
        else:
            print(f"  Warning: task {task_name} has no variance data")
            return []

    component_task_significance: Dict[int, List[str]] = {}
    for comp_idx in range(num_components_to_check):
        significant_tasks: List[str] = []
        for task_name in all_tasks:
            contrib = task_contributions.get(task_name, {})
            raw_variances = contrib.get("raw_variances")
            if raw_variances is None or len(raw_variances) <= comp_idx:
                continue
            total_variance = task_total_variances.get(task_name, 1.0)
            if total_variance > 0 and raw_variances[comp_idx] > total_variance * relative_threshold:
                significant_tasks.append(task_name)

        if significant_tasks:
            component_task_significance[comp_idx] = significant_tasks

    shared_indices = [
        comp_idx
        for comp_idx, tasks in component_task_significance.items()
        if len(tasks) >= min_tasks_shared
    ]

    print(
        f"  Found {len(shared_indices)}/{num_components_to_check} components "
        f"shared by at least {min_tasks_shared} tasks"
    )

    if shared_indices:
        basis_info = []
        for idx in shared_indices:
            avg_var = 0.0
            avg_rel = 0.0
            task_count = 0
            for task_name in all_tasks:
                raw_variances = task_contributions.get(task_name, {}).get("raw_variances")
                if raw_variances is None or len(raw_variances) <= idx:
                    continue
                total_variance = task_total_variances.get(task_name, 1.0)
                if total_variance <= 0:
                    continue
                var = float(raw_variances[idx])
                avg_var += var
                avg_rel += var / total_variance
                task_count += 1

            if task_count:
                shared_tasks = component_task_significance[idx]
                missing_tasks = [task for task in all_tasks if task not in shared_tasks]
                basis_info.append(
                    {
                        "idx": idx,
                        "avg_variance": avg_var / task_count,
                        "avg_relative": avg_rel / task_count,
                        "shared_task_count": len(shared_tasks),
                        "missing_tasks": missing_tasks,
                    }
                )

        basis_info.sort(key=lambda item: item["avg_variance"], reverse=True)
        print(f"  Top {min(10, len(basis_info))} shared components:")
        for info in basis_info[:10]:
            print(
                f"    component #{info['idx']}: shared_by={info['shared_task_count']}, "
                f"missing={info['missing_tasks']}, "
                f"avg_var={info['avg_variance']:.4e}, "
                f"avg_rel={info['avg_relative']:.4f}"
            )

    return shared_indices


def _get_attr_chain(obj: Any, chain: Iterable[str]) -> Any:
    """Resolve ``obj.a.b.c`` style chains, returning None if any link is absent."""
    cur = obj
    for attr in chain:
        if cur is None or not hasattr(cur, attr):
            return None
        cur = getattr(cur, attr)
    return cur


def _auto_find_transformer_layers(model: nn.Module) -> Tuple[Optional[nn.ModuleList], Optional[str]]:
    """Locate the most likely transformer block ModuleList inside an HF model."""
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue

        lname = name.lower()
        if any(k in lname for k in ["embed", "embedding", "token", "position", "rotary", "vision", "image", "patch"]):
            continue

        elem = module[0]
        elem_name = elem.__class__.__name__.lower()
        has_attn = any(hasattr(elem, k) for k in ["self_attn", "attn", "attention", "self_attention"])
        has_mlp = any(hasattr(elem, k) for k in ["mlp", "ffn", "feed_forward", "feedforward", "dense_h_to_4h"])
        if not (has_attn or has_mlp or "block" in elem_name or "layer" in elem_name or "decoder" in elem_name):
            continue

        score = len(module) * 10
        if "decoder" in lname:
            score += 80
        if "language_model" in lname or "text_model" in lname or "lm" in lname:
            score += 40
        if "layers" in lname or lname.endswith(".h") or lname.endswith(".blocks"):
            score += 30
        if "encoder" in lname:
            score -= 50

        candidates.append((score, len(module), name, module))

    if not candidates:
        return None, None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, name, module = candidates[0]
    return module, name


def infer_hidden_dim_from_model(model: nn.Module, fallback: int = HIDDEN_DIM) -> int:
    """Infer hidden size from common HF config fields."""
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for attr in ["hidden_size", "n_embd", "d_model", "model_dim", "dim", "hidden_dim"]:
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val

        for nested in ["text_config", "language_config", "decoder_config"]:
            sub = getattr(cfg, nested, None)
            if sub is None:
                continue
            for attr in ["hidden_size", "n_embd", "d_model", "dim", "hidden_dim"]:
                val = getattr(sub, attr, None)
                if isinstance(val, int) and val > 0:
                    return val

    return fallback


def get_model_layers(model: nn.Module) -> Tuple[Any, str]:
    """Return transformer block layers for common causal-LM architectures."""
    known_paths = [
        (("model", "decoder", "layers"), "opt"),
        (("model", "layers"), "llama_like"),
        (("transformer", "h"), "gpt2_like"),
        (("gpt_neox", "layers"), "gpt_neox"),
        (("transformer", "blocks"), "blocks_like"),
        (("model", "transformer", "h"), "gpt2_like_wrapped"),
        (("model", "transformer", "blocks"), "blocks_like_wrapped"),
    ]
    for chain, arch in known_paths:
        layers = _get_attr_chain(model, chain)
        if layers is not None:
            return layers, arch

    wrapped_paths = [
        (("language_model", "model", "layers"), "wrapped_language_model"),
        (("language_model", "layers"), "wrapped_language_model"),
        (("model", "language_model", "model", "layers"), "model_language_model"),
        (("model", "language_model", "layers"), "model_language_model"),
        (("text_model", "model", "layers"), "wrapped_text_model"),
        (("text_model", "layers"), "wrapped_text_model"),
        (("model", "text_model", "model", "layers"), "model_text_model"),
        (("model", "text_model", "layers"), "model_text_model"),
        (("model", "language_model", "language_model", "model", "layers"), "double_wrapped_language_model"),
        (("language_model", "language_model", "model", "layers"), "double_wrapped_language_model"),
    ]
    for chain, arch in wrapped_paths:
        layers = _get_attr_chain(model, chain)
        if layers is not None:
            return layers, arch

    layers, name = _auto_find_transformer_layers(model)
    if layers is not None:
        print(f"[get_model_layers] Auto-detected transformer layers at: {name} (len={len(layers)})")
        return layers, f"auto:{name}"

    raise ValueError(f"Could not find layers in model. Model type: {type(model)}")


def validate_subspace_basis(subspace: np.ndarray, name: str = "subspace") -> np.ndarray:
    """Print basic basis diagnostics and return a normalized basis when needed."""
    k = subspace.shape[1]
    print(f"\n[VALIDATE] {name}:")
    print(f"  shape: {subspace.shape}")

    norms = np.linalg.norm(subspace, axis=0)
    print(f"  column norm range: [{np.min(norms):.6f}, {np.max(norms):.6f}]")

    if k > 1:
        ortho_matrix = subspace.T @ subspace
        np.fill_diagonal(ortho_matrix, 0)
        max_off_diag = np.max(np.abs(ortho_matrix))
        print(f"  max off-diagonal overlap: {max_off_diag:.6e}")

    norm_errors = np.abs(norms - 1.0)
    avg_norm_error = np.mean(norm_errors)
    print(f"  average norm error: {avg_norm_error:.6e}")

    if avg_norm_error > 0.1:
        safe_norms = np.where(norms < 1e-12, 1.0, norms)
        return subspace / safe_norms
    return subspace


def _empty_cross_task_result(return_full_pca: bool):
    if return_full_pca:
        return None, 0, {}, {}
    return None, 0, {}


def compute_cross_task_subspace(
    task_activations_dict: Dict[str, Dict[int, np.ndarray]],
    variance_threshold: float = 0.95,
    min_dim: int = 1,
    max_dim: int = 2000,
    return_full_pca: bool = False,
):
    """Compute a pooled cross-task PCA subspace and per-task component variance."""
    all_activations = []
    task_sample_counts: Dict[str, int] = {}
    task_start_indices: Dict[str, int] = {}

    print(f"\nCombining activations from {len(task_activations_dict)} tasks...")
    current_idx = 0
    for task_name, layer_activations in task_activations_dict.items():
        task_activations = [
            acts for acts in layer_activations.values()
            if acts is not None and acts.shape[0] > 0
        ]
        if not task_activations:
            continue

        x_task = np.vstack(task_activations)
        all_activations.append(x_task)
        task_sample_counts[task_name] = x_task.shape[0]
        task_start_indices[task_name] = current_idx
        current_idx += x_task.shape[0]
        print(f"  {task_name}: {x_task.shape[0]} samples")

    if not all_activations:
        return _empty_cross_task_result(return_full_pca)

    x_combined = np.vstack(all_activations).astype(np.float64)
    n_samples_total, hidden_dim = x_combined.shape
    print(f"\nTotal samples for cross-task PCA: {n_samples_total}")

    x_mean = np.mean(x_combined, axis=0, dtype=np.float64)
    x_centered = x_combined - x_mean
    feature_scales = np.std(x_centered, axis=0, dtype=np.float64)
    feature_scales = np.where(feature_scales < 1e-12, 1.0, feature_scales)
    x_scaled = x_centered / feature_scales

    max_components = min(n_samples_total - 1, hidden_dim, max_dim)
    if max_components < 1:
        return _empty_cross_task_result(return_full_pca)

    print("Computing cross-task PCA...")
    try:
        pca = PCA(n_components=max_components)
        pca.fit(x_scaled)
    except Exception as exc:
        print(f"Cross-task PCA failed: {exc}")
        return _empty_cross_task_result(return_full_pca)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    k = np.argmax(cumsum >= variance_threshold) + 1
    if k == 0 or cumsum[-1] < variance_threshold:
        k = len(cumsum)
    k = min(max(k, min_dim), max_dim, max_components)

    joint_subspace_scaled = pca.components_[:k].T
    joint_subspace_centered = joint_subspace_scaled * feature_scales.reshape(-1, 1)
    cross_task_subspace = joint_subspace_centered.astype(np.float32)

    task_variance_contributions: Dict[str, Dict[str, Any]] = {}
    print("\nAnalyzing task contributions to cross-task subspace...")
    for task_name, start_idx in task_start_indices.items():
        task_count = task_sample_counts[task_name]
        end_idx = start_idx + task_count
        x_task_scaled = x_scaled[start_idx:end_idx, :]
        projection = x_task_scaled @ joint_subspace_scaled
        task_variances = np.var(projection, axis=0)
        total_task_variance = np.sum(task_variances)
        normalized = task_variances / total_task_variance if total_task_variance > 0 else np.zeros(k)

        task_variance_contributions[task_name] = {
            "raw_variances": task_variances,
            "normalized": normalized,
            "total_variance": total_task_variance,
            "sample_count": task_count,
        }

        top_5_idx = np.argsort(task_variances)[-5:][::-1]
        print(f"  {task_name}:")
        print(f"    Samples: {task_count}")
        print(f"    Total variance in subspace: {total_task_variance:.2e}")
        print(f"    Top 5 components by variance: {top_5_idx.tolist()}")
        print(f"    Top 5 variances: {task_variances[top_5_idx].round(6)}")

    explained_var = cumsum[k - 1] * 100 if k > 0 and k <= len(cumsum) else 0.0
    print(
        f"\nCross-task subspace: {k}/{hidden_dim} dim ({k / hidden_dim * 100:.1f}%), "
        f"explains {explained_var:.1f}% variance"
    )

    norms = np.linalg.norm(cross_task_subspace, axis=0, keepdims=True)
    cross_task_subspace = cross_task_subspace / (norms + 1e-12)
    cross_task_subspace = validate_subspace_basis(cross_task_subspace, "cross-task subspace")

    if return_full_pca:
        full_pca_info = {
            "components": pca.components_.T,
            "feature_scales": feature_scales,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "max_components": max_components,
        }
        return cross_task_subspace, k, task_variance_contributions, full_pca_info

    return cross_task_subspace, k, task_variance_contributions


class JointSubspaceRemovalHook:
    """Forward hook that removes a supplied subspace from hidden states."""

    def __init__(
        self,
        layer_idx: int,
        joint_subspace: np.ndarray,
        enabled: bool = True,
        track_stats: bool = False,
        eps: float = 1e-6,
        preserve_statistics: bool = True,
        strength: float = 1.0,
    ):
        self.layer_idx = layer_idx
        self.joint_subspace_np = joint_subspace
        self.enabled = enabled
        self.track_stats = track_stats
        self.eps = eps
        self.preserve_statistics = preserve_statistics
        self.strength = strength
        self.stats = {"original_variances": [], "removed_variances": [], "variance_ratios": []}

    def __call__(self, module: nn.Module, input: Tuple[Any, ...], output: Any) -> Any:
        if not self.enabled:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = ()

        dtype = hidden_states.dtype
        device = hidden_states.device
        joint_subspace = torch.tensor(self.joint_subspace_np, dtype=dtype, device=device)
        original_shape = hidden_states.shape

        if len(hidden_states.shape) == 3:
            _, _, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_dim)
        elif len(hidden_states.shape) == 2:
            hidden_flat = hidden_states
        else:
            return output

        hidden_flat_original = hidden_flat.clone()
        if self.preserve_statistics:
            original_mean = hidden_flat.mean(dim=0, keepdim=True)
            original_std = hidden_flat.std(dim=0, keepdim=True)
            std_mask = original_std < self.eps
            safe_std = original_std.clone()
            safe_std[std_mask] = 1.0
            hidden_basis_input = (hidden_flat - original_mean) / safe_std
        else:
            original_mean = None
            safe_std = None
            std_mask = None
            hidden_basis_input = hidden_flat

        try:
            u_fp32 = joint_subspace.float()
            u_svd, _, _ = torch.linalg.svd(u_fp32, full_matrices=False)
            u_orth = u_svd.to(dtype=dtype)
        except Exception as exc:
            print(f"Warning: SVD failed in JointSubspaceRemovalHook: {exc}")
            u_orth = joint_subspace + torch.randn_like(joint_subspace) * self.eps

        with torch.no_grad():
            u_check = u_orth.float()
            ortho_check = u_check.T @ u_check
            identity = torch.eye(ortho_check.shape[0], dtype=ortho_check.dtype, device=ortho_check.device)
            max_deviation = (ortho_check - identity).abs().max().item()
            if max_deviation > 1e-3:
                print(
                    "  Warning: joint subspace columns are not properly orthonormal. "
                    f"Max deviation: {max_deviation:.6f}"
                )

        projection_coeffs = hidden_basis_input @ u_orth
        subspace_component = projection_coeffs @ u_orth.T
        orthogonal_complement = hidden_basis_input - self.strength * subspace_component

        if self.preserve_statistics:
            orthogonal_complement = orthogonal_complement * safe_std + original_mean
            if std_mask is not None and std_mask.any():
                orthogonal_complement[:, std_mask.squeeze()] = hidden_flat_original[:, std_mask.squeeze()]

        if torch.isnan(orthogonal_complement).any() or torch.isinf(orthogonal_complement).any():
            print(f"Warning: NaN or Inf in orthogonal complement for layer {self.layer_idx}")
            orthogonal_complement = torch.nan_to_num(orthogonal_complement, nan=0.0, posinf=0.0, neginf=0.0)
            if torch.isnan(orthogonal_complement).any() or torch.isinf(orthogonal_complement).any():
                orthogonal_complement = hidden_flat_original

        perturbed_states = orthogonal_complement.reshape(original_shape)
        if isinstance(output, tuple):
            return (perturbed_states,) + other_outputs
        return perturbed_states
