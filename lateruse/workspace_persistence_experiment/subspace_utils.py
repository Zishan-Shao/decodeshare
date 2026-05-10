
# -*- coding: utf-8 -*-
"""
subspace_utils.py

用于“跨层共享子空间”实验的一组通用数学工具：
- PCA 基（每个任务/层）
- 多任务共享子空间估计（基于平均投影矩阵的低秩 SVD）
- 子空间相似度（principal angles / canonical correlations）
- Ridge 线性映射与 R²

本文件不依赖具体模型结构，可被多个脚本 import。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ----------------------------
# PCA
# ----------------------------

def pca_basis(
    X: torch.Tensor,
    k: int,
    *,
    center: bool = True,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 PCA 的前 k 个主方向。

    参数
    ----
    X: [N, d]  float tensor (CPU/GPU)
    k: 需要的主方向数
    center: 是否做去均值
    eps: 数值稳定

    返回
    ----
    Q: [d, k]  正交列向量（主方向）
    S: [k]     奇异值（与解释方差相关）
    """
    if X.dim() != 2:
        raise ValueError(f"X must be 2D [N,d], got {tuple(X.shape)}")
    N, d = X.shape
    if N < 2:
        raise ValueError("Need at least 2 samples for PCA.")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive.")
    k = min(k, min(N - 1, d))

    Xc = X
    if center:
        Xc = X - X.mean(dim=0, keepdim=True)

    # torch.pca_lowrank 是随机化算法，适合 d 很大时（常见 hidden_size 4096/8192）
    q = min(k + 16, min(N - 1, d))
    U, S, V = torch.pca_lowrank(Xc, q=q, center=False)  # 我们已经手动 center
    Q = V[:, :k]  # [d, k]
    Sk = S[:k].clamp_min(eps)
    return Q, Sk


# ----------------------------
# Shared subspace estimation
# ----------------------------

@dataclass
class SharedSubspaceResult:
    Q_shared: torch.Tensor          # [d, k_shared]
    eigvals: torch.Tensor           # [r_union]  P = mean(Q_i Q_i^T) 的特征值（在 union 空间内）
    k_shared: int
    Q_control_pool: torch.Tensor    # [d, k_pool]  非共享控制方向（尽量从 union 的“剩余方向”取）


def _svd_shared_from_task_bases(
    Q_tasks: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    给定多个任务的 PCA 基 Q_i (每个 [d,p] 且列正交)，
    计算 B = [Q_1 ... Q_T] 的薄 SVD：B = U S V^T。
    返回 U, S。
    """
    if len(Q_tasks) < 2:
        raise ValueError("Need at least 2 tasks to estimate shared subspace.")

    d = Q_tasks[0].shape[0]
    for i, Q in enumerate(Q_tasks):
        if Q.dim() != 2:
            raise ValueError(f"Q_tasks[{i}] must be 2D, got {tuple(Q.shape)}")
        if Q.shape[0] != d:
            raise ValueError("All Q_tasks must have same hidden dim d.")
    B = torch.cat(list(Q_tasks), dim=1)  # [d, T*p]
    # 这里的 SVD 规模是 d x (T*p)，T*p 通常 < 1024，所以比 dxd 特征分解轻很多
    U, S, Vh = torch.linalg.svd(B, full_matrices=False)
    return U, S


def _pick_k_from_eigvals(
    eigvals: torch.Tensor,
    *,
    tau: Optional[float],
    max_k: int,
    min_k: int,
) -> int:
    if tau is None:
        # 经验默认：如果一个方向在 >= ~20% 的任务子空间里“重复出现”，eigval 往往会明显高于随机水平
        # 这里给一个保守的阈值。用户可通过 CLI 覆盖。
        tau = 0.20
    k = int((eigvals > float(tau)).sum().item())
    k = max(k, int(min_k))
    k = min(k, int(max_k))
    return k


def random_orthonormal(
    d: int,
    k: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    """
    生成随机正交基 [d,k]。
    """
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    X = torch.randn(d, k, generator=g, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(X, mode="reduced")
    return Q[:, :k]


def complete_with_orth_complement(
    Q_existing: torch.Tensor,
    k_more: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """
    在 Q_existing 的正交补中补齐 k_more 个随机正交方向，返回 [d, k_more]。
    """
    device = Q_existing.device
    dtype = Q_existing.dtype
    d = Q_existing.shape[0]
    if k_more <= 0:
        return torch.empty(d, 0, device=device, dtype=dtype)

    # 生成过采样随机向量，然后做 Gram-Schmidt（通过投影消除 Q_existing 分量）
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    oversample = max(k_more * 2, k_more + 8)

    X = torch.randn(d, oversample, generator=g, device=device, dtype=dtype)
    # 去掉 Q_existing 分量
    if Q_existing.numel() > 0:
        X = X - Q_existing @ (Q_existing.T @ X)

    # QR 得到正交基
    Qnew, _ = torch.linalg.qr(X, mode="reduced")
    return Qnew[:, :k_more]


def estimate_shared_subspace(
    Q_tasks: Sequence[torch.Tensor],
    *,
    k_shared: Optional[int] = None,
    tau_shared: Optional[float] = None,
    max_k: int = 64,
    min_k: int = 1,
    control_pool_dim: int = 64,
    control_seed: int = 0,
) -> SharedSubspaceResult:
    """
    估计共享子空间 Q_shared：

    1) 对每个任务有 PCA 子空间 Q_i (d x p)。
    2) 平均投影矩阵：P = mean_i (Q_i Q_i^T)。
       P 的主特征向量对应“在多个任务中重复出现”的方向。
    3) 用 B=[Q_1..Q_T] 做薄 SVD，避免对 dxd 做特征分解。

    返回：
      - Q_shared: d x k
      - eigvals: union 空间内的 P 特征值（0~1）
      - 控制方向池 Q_control_pool：用于结构相似度的 baseline
    """
    U, S = _svd_shared_from_task_bases(Q_tasks)
    T = len(Q_tasks)
    # P = (1/T) B B^T; eigenvalues = (S^2)/T
    eigvals = (S**2) / float(T)

    if k_shared is None:
        k_shared = _pick_k_from_eigvals(eigvals, tau=tau_shared, max_k=max_k, min_k=min_k)
    else:
        k_shared = int(k_shared)
        k_shared = max(k_shared, int(min_k))
        k_shared = min(k_shared, int(max_k))

    Q_shared = U[:, :k_shared].contiguous()

    # control pool：尽量用 union 中“剩下的方向”
    start = k_shared
    end = start + int(control_pool_dim)
    if end <= U.shape[1]:
        Q_ctrl = U[:, start:end].contiguous()
    else:
        # union 的维数不足，先用剩余 union 方向，再补随机正交补
        Q_part = U[:, start:].contiguous()
        need = int(control_pool_dim) - Q_part.shape[1]
        Q_extra = complete_with_orth_complement(
            torch.cat([Q_shared, Q_part], dim=1),
            need,
            seed=control_seed,
        )
        Q_ctrl = torch.cat([Q_part, Q_extra], dim=1).contiguous()

    return SharedSubspaceResult(
        Q_shared=Q_shared,
        eigvals=eigvals.detach().contiguous(),
        k_shared=int(k_shared),
        Q_control_pool=Q_ctrl.detach().contiguous(),
    )


# ----------------------------
# Subspace similarity (principal angles)
# ----------------------------

def subspace_similarity(
    Qa: np.ndarray,
    Qb: np.ndarray,
    r: Optional[int] = None,
) -> Tuple[float, np.ndarray]:
    """
    子空间相似度（主角/典型相关）：

      M = Qa^T Qb
      s = svd(M)
      sim = mean(s_i^2) over i=1..r

    Qa: [d, ka], columns orthonormal
    Qb: [d, kb], columns orthonormal

    返回 (sim, s_all)
    """
    if Qa.ndim != 2 or Qb.ndim != 2:
        raise ValueError("Qa/Qb must be 2D arrays.")
    if Qa.shape[0] != Qb.shape[0]:
        raise ValueError("Qa and Qb must have same d.")
    ka = Qa.shape[1]
    kb = Qb.shape[1]
    if r is None:
        r = min(ka, kb)
    r = int(min(r, ka, kb))
    if r <= 0:
        raise ValueError("r must be positive.")
    M = Qa.T @ Qb
    s = np.linalg.svd(M, compute_uv=False)
    s = s[:r]
    sim = float(np.mean(s ** 2))
    return sim, s


# ----------------------------
# Linear mapping + R2
# ----------------------------

def fit_linear_map_ridge(
    Z_src: np.ndarray,
    Z_tgt: np.ndarray,
    ridge: float = 1e-3,
) -> np.ndarray:
    """
    Ridge 回归拟合 A: k_tgt x k_src

    最小化 ||Z_tgt - Z_src @ A^T||^2 + ridge||A||^2

    返回 A: [k_tgt, k_src]
    """
    X = np.asarray(Z_src, dtype=np.float64)
    Y = np.asarray(Z_tgt, dtype=np.float64)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("Z_src/Z_tgt must be 2D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError("Z_src and Z_tgt must have same N.")
    k1 = X.shape[1]
    k2 = Y.shape[1]
    XtX = X.T @ X + float(ridge) * np.eye(k1)
    XtY = X.T @ Y
    A_T = np.linalg.solve(XtX, XtY)   # [k1, k2]
    A = A_T.T                         # [k2, k1]
    return A


def r2_score(Y: np.ndarray, Yhat: np.ndarray, eps: float = 1e-12) -> float:
    Y = np.asarray(Y, dtype=np.float64)
    Yhat = np.asarray(Yhat, dtype=np.float64)
    ss_res = np.sum((Y - Yhat) ** 2)
    ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2)
    return float(1.0 - ss_res / (ss_tot + eps))
