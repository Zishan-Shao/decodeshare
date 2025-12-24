#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
act_rank_distribution.py — Compute and visualize FULL SVD rank distribution of model activations

What it does:
- Loads multiple models (OPT-125M, OPT-6.7B, LLaMA-2-7B, Qwen2.5-7B, Mixtral-8x7B)
- Collects activations from forward passes on multiple datasets
- Computes FULL SVD for all activation matrices (all singular values)
- Focuses on: Q-preRoPE, K-preRoPE, Q-postRoPE, K-postRoPE, V, O, MLP (up_proj, gate_proj, down_proj)
- Saves detailed per-layer activation rank distributions to CSV files
- Creates visualizations showing singular value accumulation

   For each dataset:
     1. Collect activations → buffers (CPU)
     2. Save buffers to disk
     3. Clear GPU cache (model still in GPU)
   
   After all datasets:
     1. Remove model from GPU (del model)
     2. Clear GPU cache
     3. Buffers remain on CPU for SVD computation

Environment variables:
- HF_ENDPOINT: Set to hf-mirror URL if needed (e.g., https://hf-mirror.com)
- DEVICE: Device to run forward pass and SVD on (default: auto - auto-detects GPU)
- NSAMPLES: Number of samples to collect (default: 64)
- SEQLEN: Sequence length (default: 1024)
- DATASETS: Comma-separated list of datasets (default: wikitext2,ptb,c4)
- PLOT_DIR: Directory to save plots (default: plot)
- CSV_DIR: Directory to save CSV files (default: .)
- LIMIT_LAYERS: If >0, only analyze first N layers (default: 0, analyze all)
- MAX_TOKENS: Maximum tokens to collect per activation type (default: 65536)
- SKIP_ACT_SAVE: If set to 1 or true, skip saving activations to disk (default: 0, saves activations)
- FORCE_RECOMPUTE: If set to 1 or true, force recompute even if cached activations exist (default: 0, uses cache if available)

Examples:
# Basic usage
HF_ENDPOINT=https://hf-mirror.com python3 motivations/act_rank_distribution.py

# Custom datasets and samples
DATASETS=wikitext2,ptb,gsm8k,commonsenseqa,humaneval,aqua,strategyqa,multiarith NSAMPLES=32 SEQLEN=1024 python3 motivations/act_rank_distribution.py

FORCE_RECOMPUTE=1 CUDA_VISIBLE_DEVICES=1 DATASETS=wikitext2,ptb,gsm8k,commonsenseqa,aqua,strategyqa NSAMPLES=16 SEQLEN=512 python3 motivations/act_rank_distribution.py

FORCE_RECOMPUTE=1 CUDA_VISIBLE_DEVICES=1 DATASETS=gsm8k,commonsenseqa,strategyqa NSAMPLES=16 SEQLEN=512 python3 motivations/act_rank_distribution.py


# Skip saving activations to disk (reduces IO overhead)
SKIP_ACT_SAVE=1 FORCE_RECOMPUTE=1 CUDA_VISIBLE_DEVICES=1 BATCH_SIZE=1 DATASETS=gsm8k,commonsenseqa,strategyqa NSAMPLES=16 SEQLEN=512 python3 motivations/act_rank_distribution.py


# Force CPU usage
DEVICE=cpu python3 motivations/act_rank_distribution.py

"""

import os
import sys
import csv
import gc
import pickle
import statistics
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Add repo root to path for data utils
current_path = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_path)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

try:
    from utils.data_utils import get_calib_train_data
except ImportError:
    print("Warning: Could not import data_utils. Dataset loading may be limited.")
    get_calib_train_data = None


# Model configurations: (display_name, hf_model_name)
MODELS = [
    ("OPT-125M", "facebook/opt-125m"),
    ("OPT-6.7B", "facebook/opt-6.7b"),
    # ("LLaMA-2-7B", "meta-llama/Llama-2-7b-hf"),
    # ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
    # ("Mixtral-8x7B", "mistralai/Mixtral-8x7B-v0.1"),
]

# Available datasets
AVAILABLE_DATASETS = [
    'wikitext2', 'ptb', 'c4', 'gsm8k', 'commonsenseqa', 
    'humaneval', 'aqua', 'strategyqa', 'multiarith'
]


def _setup_matplotlib():
    """Setup matplotlib with non-interactive backend if needed."""
    try:
        import matplotlib
        if os.getenv('MPLBACKEND') is None:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return matplotlib, plt
    except Exception as e:
        print(f"[plot] matplotlib unavailable: {e}")
        return None, None


@torch.no_grad()
def compute_full_svd_analysis(
    activation_matrix: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    use_gpu_for_large: bool = True,
    gpu_threshold: int = 100,  # Lower threshold to use GPU more aggressively
) -> Optional[Dict]:
    """
    Compute full SVD analysis for an activation matrix.
    
    Args:
        activation_matrix: Activation tensor [N_tokens, D_features]
        device: Device to use for computation
        use_gpu_for_large: Whether to use GPU for large matrices
        gpu_threshold: Minimum matrix size to use GPU
    
    Returns:
        Dictionary with:
        - singular_values: All singular values
        - theoretical_max_rank: min(N_tokens, D_features)
        - actual_rank: Number of non-zero singular values
        - energy_cumulative: Cumulative energy fraction
        - total_energy: Total energy (sum of squared singular values)
    """
    if activation_matrix is None or activation_matrix.numel() == 0:
        return None
    
    # Ensure 2D
    if activation_matrix.dim() > 2:
        # Reshape: [B, T, D] -> [B*T, D] or [T, D]
        if activation_matrix.dim() == 3:
            B, T, D = activation_matrix.shape
            activation_matrix = activation_matrix.reshape(B * T, D)
    else:
            activation_matrix = activation_matrix.reshape(-1, activation_matrix.shape[-1])
    
    if activation_matrix.dim() != 2:
        return None
    
    N, D = activation_matrix.shape
    
    if N == 0 or D == 0:
            return None
    
    # Theoretical maximum rank
    theoretical_max_rank = min(N, D)
    
    # Decide whether to use GPU
    matrix_size = N * D
    use_gpu = use_gpu_for_large and device.type == 'cuda' and matrix_size > gpu_threshold
    
    # Move to appropriate device
    if use_gpu:
        try:
            # If matrix is already on GPU, use it directly; otherwise move to GPU
            if activation_matrix.is_cuda:
                A_gpu = activation_matrix.float()
            else:
                A_gpu = activation_matrix.float().to(device)
            # Compute SVD on GPU
            U, S, Vh = torch.linalg.svd(A_gpu, full_matrices=False)
            # Move singular values to CPU only when needed (for numpy conversion)
            singular_values = S.cpu().numpy()
            del A_gpu, U, Vh, S
            # Don't clear cache after every SVD - let GPU manage memory more efficiently
            # Only clear cache periodically or when memory pressure is detected
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Fallback to CPU
                A_cpu = activation_matrix.float().cpu()
                U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
                singular_values = S.numpy()
                del A_cpu, U, Vh, S
            else:
                raise
    else:
        # Use CPU
        A_cpu = activation_matrix.float().cpu()
        U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
        singular_values = S.numpy()
        del A_cpu, U, Vh, S
    
    # Compute actual rank (number of non-zero singular values)
    # Use a small threshold to account for numerical precision
    threshold = 1e-6 * singular_values.max() if len(singular_values) > 0 else 1e-10
    actual_rank = int(np.sum(singular_values > threshold))
    
    # Compute cumulative energy
    energy_squared = singular_values ** 2
    total_energy = float(np.sum(energy_squared))
    
    if total_energy > 0:
        energy_cumulative = np.cumsum(energy_squared) / total_energy
    else:
        energy_cumulative = np.zeros_like(singular_values)
    
    return {
        'singular_values': singular_values.tolist(),
        'theoretical_max_rank': theoretical_max_rank,
        'actual_rank': actual_rank,
        'energy_cumulative': energy_cumulative.tolist(),
        'total_energy': total_energy,
    }


class ActivationBuffer:
    """Buffer to collect activations across forward passes."""
    
    def __init__(self, max_tokens: int, center: bool = False, device: torch.device = None):
        self.max_tokens = int(max_tokens)
        self.center = bool(center)
        self.device = device  # Keep activations on this device
        self._items: List[torch.Tensor] = []
        self.count = 0
    
    def add(self, activation: torch.Tensor):
        """Add activation tensor. Expected shape: [B, T, D] or [T, D]."""
        if activation is None:
            return
        
        # Ensure 3D: [B, T, D]
        if activation.dim() == 2:
            activation = activation.unsqueeze(0)  # [1, T, D]
        if activation.dim() != 3:
            return
        
        B, T, D = activation.shape
        remaining = max(0, self.max_tokens - self.count)
        if remaining == 0:
            return
        
        take = min(remaining, B * T)
        X = activation.reshape(B * T, D)[:take].to(torch.float32)
        
        if self.center:
            X = X - X.mean(dim=0, keepdim=True)
        
        # Keep on GPU if device is GPU, otherwise move to CPU
        if self.device is not None and self.device.type == 'cuda':
            self._items.append(X)  # Keep on GPU
        else:
            self._items.append(X.cpu())  # Move to CPU
        self.count += take
    
    def finalize(self) -> Optional[torch.Tensor]:
        """Concatenate all collected activations."""
        if not self._items:
            return None
        result = torch.cat(self._items, dim=0)  # [N_total, D]
        # Ensure result is on the correct device
        if self.device is not None and self.device.type == 'cuda' and not result.is_cuda:
            result = result.to(self.device)
        return result
    
    def is_full(self) -> bool:
        return self.count >= self.max_tokens


def hook_attention_for_rope(model, layer_idx: int, buffers: Dict, max_tokens: int, device: torch.device = None):
    """
    Hook attention module to capture pre-RoPE and post-RoPE Q/K activations.
    This is model-specific and may need adjustment for different architectures.
    """
    handles = []
    layer = None
    
    # Find the layer (handle different architectures)
    layer = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA/Qwen/Mixtral architecture
        layers = model.model.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT architecture (newer)
        layers = model.model.decoder.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        # OPT architecture (alternative)
        layers = model.decoder.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # OPT architecture (older)
        layers = model.transformer.h
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    
    if layer is None:
        return handles
    
    # Initialize buffers for this layer
    layer_buf = buffers.get(layer_idx, {})
    
    # Hook Q/K/V projections (pre-RoPE)
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
    elif hasattr(layer, 'attn'):
        attn = layer.attn
    else:
        return handles
    
    # Pre-RoPE buffers
    if hasattr(attn, 'q_proj'):
        q_pre_buf = ActivationBuffer(max_tokens, center=False, device=device)
        k_pre_buf = ActivationBuffer(max_tokens, center=False, device=device)
        layer_buf['Q_preRoPE'] = q_pre_buf
        layer_buf['K_preRoPE'] = k_pre_buf
        
        def hook_q_pre(m, i, o):
            q_pre_buf.add(o.detach())
        
        def hook_k_pre(m, i, o):
            k_pre_buf.add(o.detach())
        
        handles.append(attn.q_proj.register_forward_hook(hook_q_pre))
        if hasattr(attn, 'k_proj'):
            handles.append(attn.k_proj.register_forward_hook(hook_k_pre))
    
    # V and O buffers
    if hasattr(attn, 'v_proj'):
        v_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['V'] = v_buf
        
        def hook_v(m, i, o):
            v_buf.add(o.detach())
        
        handles.append(attn.v_proj.register_forward_hook(hook_v))
    
    # Output projection (OPT uses out_proj, LLaMA uses o_proj)
    if hasattr(attn, 'o_proj'):
        o_buf = ActivationBuffer(max_tokens, center=False, device=device)
        layer_buf['O'] = o_buf
        
        def hook_o(m, i, o):
            o_buf.add(o.detach())
        
        handles.append(attn.o_proj.register_forward_hook(hook_o))
    elif hasattr(attn, 'out_proj'):
        # OPT architecture uses out_proj
        o_buf = ActivationBuffer(max_tokens, center=False, device=device)
        layer_buf['O'] = o_buf
        
        def hook_o(m, i, o):
            o_buf.add(o.detach())
        
        handles.append(attn.out_proj.register_forward_hook(hook_o))
    
    # Post-RoPE Q/K: Hook the attention forward method
    # This is tricky and model-specific. We'll capture Q/K after RoPE is applied.
    # For LLaMA/Qwen: RoPE is applied in the forward method
    # For OPT: No RoPE, so post-RoPE = pre-RoPE
    
    # Try to hook the attention forward to capture post-RoPE
    # We'll need to modify the forward temporarily or use a wrapper
    # For now, we'll capture pre-RoPE and note that post-RoPE may be the same for OPT
    
    buffers[layer_idx] = layer_buf
    
    return handles


def hook_mlp_activations(model, layer_idx: int, buffers: Dict, max_tokens: int, device: torch.device = None):
    """Hook MLP activations: up_proj, gate_proj (if exists), down_proj."""
    handles = []
    layer = None
    
    # Find the layer (handle different architectures)
    layer = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA/Qwen/Mixtral architecture
        layers = model.model.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT architecture (newer)
        layers = model.model.decoder.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        # OPT architecture (alternative)
        layers = model.decoder.layers
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # OPT architecture (older)
        layers = model.transformer.h
        if layer_idx < len(layers):
            layer = layers[layer_idx]
    
    if layer is None:
        return handles
    
    layer_buf = buffers.get(layer_idx, {})
    
    # Hook MLP projections (OPT uses ffn with fc1/fc2, LLaMA uses mlp with gate_proj/up_proj/down_proj)
    mlp = None
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
    elif hasattr(layer, 'ffn'):
        mlp = layer.ffn
    
    if mlp is None:
        return handles
    
    # LLaMA-style MLP: gate_proj, up_proj, down_proj
    if hasattr(mlp, 'up_proj'):
        up_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['up_proj'] = up_buf
        
        def hook_up(m, i, o):
            up_buf.add(o.detach())
        
        handles.append(mlp.up_proj.register_forward_hook(hook_up))
    
    if hasattr(mlp, 'gate_proj'):
        gate_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['gate_proj'] = gate_buf
        
        def hook_gate(m, i, o):
            gate_buf.add(o.detach())
        
        handles.append(mlp.gate_proj.register_forward_hook(hook_gate))
    
    if hasattr(mlp, 'down_proj'):
        down_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['down_proj'] = down_buf
        
        def hook_down(m, i, o):
            down_buf.add(o.detach())
        
        handles.append(mlp.down_proj.register_forward_hook(hook_down))
    
    # OPT-style MLP: fc1 (equivalent to gate_proj+up_proj), fc2 (equivalent to down_proj)
    if hasattr(mlp, 'fc1'):
        fc1_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['fc1'] = fc1_buf  # OPT's first MLP layer
        
        def hook_fc1(m, i, o):
            fc1_buf.add(o.detach())
        
        handles.append(mlp.fc1.register_forward_hook(hook_fc1))
    
    if hasattr(mlp, 'fc2'):
        fc2_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['fc2'] = fc2_buf  # OPT's second MLP layer
        
        def hook_fc2(m, i, o):
            fc2_buf.add(o.detach())
        
        handles.append(mlp.fc2.register_forward_hook(hook_fc2))
    
    buffers[layer_idx] = layer_buf
    
    return handles


def check_activations_cached(
    model_name: str,
    datasets: List[str],
    save_dir: str = 'checkpoints/motivation_activations'
) -> Dict[str, bool]:
    """
    Check which datasets have cached activations.
    
    Returns:
        Dictionary mapping dataset names to boolean (True if cached)
    """
    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    cached_status = {}
    
    for dataset_name in datasets:
        safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
        save_path = os.path.join(save_dir, safe_model_name, safe_dataset_name)
        activation_file = os.path.join(save_path, 'activations.pkl')
        cached_status[dataset_name] = os.path.exists(activation_file)
    
    return cached_status


def load_cached_activations(
    model_name: str,
    datasets: List[str],
    save_dir: str = 'checkpoints/motivation_activations',
    max_tokens: int = 65536,
    device: torch.device = None,
) -> Optional[Dict]:
    """
    Load cached activations from disk and convert to ActivationBuffer format.
    
    Returns:
        Dictionary with activation buffers organized by layer and activation type,
        or None if loading fails
    """
    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    all_dataset_buffers = {}
    
    for dataset_name in datasets:
        safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
        save_path = os.path.join(save_dir, safe_model_name, safe_dataset_name)
        activation_file = os.path.join(save_path, 'activations.pkl')
        
        if not os.path.exists(activation_file):
            print(f"  Warning: Cached activations not found for {dataset_name}")
            return None
        
        try:
            with open(activation_file, 'rb') as f:
                saved_buffers = pickle.load(f)
            
            # Convert string keys to integers if necessary
            # This handles the case where keys were saved as strings
            converted_buffers = {}
            for key, value in saved_buffers.items():
                if isinstance(key, str) and key.isdigit():
                    converted_buffers[int(key)] = value
                else:
                    converted_buffers[key] = value
            
            # Convert saved buffers back to ActivationBuffer format
            dataset_buffers = {}
            for layer_idx, layer_buf in converted_buffers.items():
                dataset_buffers[layer_idx] = {}
                for act_type, saved_data in layer_buf.items():
                    if isinstance(saved_data, dict) and 'data' in saved_data:
                        # Convert numpy array back to tensor and create ActivationBuffer
                        activation_data = torch.from_numpy(saved_data['data'])
                        buf = ActivationBuffer(max_tokens, center=False, device=device)
                        # Add the data as a single item (already concatenated)
                        if activation_data.dim() == 2:
                            activation_data = activation_data.unsqueeze(0)  # [1, N, D]
                        buf.add(activation_data)
                        buf.count = saved_data.get('count', activation_data.shape[0] * activation_data.shape[1])
                        dataset_buffers[layer_idx][act_type] = buf
            
            all_dataset_buffers[dataset_name] = dataset_buffers
            print(f"  Loaded cached activations for {dataset_name}")
            
        except Exception as e:
            print(f"  Error loading cached activations for {dataset_name}: {e}")
            return None
    
    # Combine all dataset buffers into a single structure for SVD computation
    # Find max layer index
    # max_layer_idx = max(
    #     max(layer_buf.keys()) if layer_buf else -1
    #     for dataset_buf in all_dataset_buffers.values()
    #     for layer_buf in dataset_buf.values()
    # )
    # 找到 max layer index
    # 关键：将可能为字符串类型的键（如'0', '1'）转换为整数后再取最大值
    max_layer_idx = max(
        max(int(k) if isinstance(k, str) and k.isdigit() else k for k in layer_buf.keys()) if layer_buf else -1
        for dataset_buf in all_dataset_buffers.values()
        for layer_buf in dataset_buf.values()
    )

    if max_layer_idx < 0:
        return None
    
    combined_buffers: Dict[int, Dict[str, ActivationBuffer]] = {}
    for layer_idx in range(max_layer_idx + 1):
        combined_buffers[layer_idx] = {}
        # For each activation type, combine across datasets
        for dataset_name, dataset_buf in all_dataset_buffers.items():
            if layer_idx in dataset_buf:
                for act_type, buf in dataset_buf[layer_idx].items():
                    if isinstance(buf, ActivationBuffer):
                        if act_type not in combined_buffers[layer_idx]:
                            # Create new buffer for combined data
                            combined_buffers[layer_idx][act_type] = ActivationBuffer(max_tokens, center=False, device=device)
                        # Add activations from this dataset
                        activation_matrix = buf.finalize()
                        if activation_matrix is not None:
                            # Reshape to [B, T, D] for add() method
                            if activation_matrix.dim() == 2:
                                activation_matrix = activation_matrix.unsqueeze(0)
                            combined_buffers[layer_idx][act_type].add(activation_matrix)
                            del activation_matrix
    
    return combined_buffers


def save_activations_to_disk(
    buffers: Dict,
    model_name: str,
    dataset_name: str,
    save_dir: str = 'checkpoints/motivation_activations'
) -> str:
    """
    Save activation buffers to disk.
    
    Args:
        buffers: Dictionary of activation buffers organized by layer and activation type
        model_name: Name of the model (for directory structure)
        dataset_name: Name of the dataset (for directory structure)
        save_dir: Base directory for saving activations
    
    Returns:
        Path where activations were saved
    """
    # Create directory structure: checkpoints/motivation_activations/{model}/{dataset}/
    safe_model_name = model_name.replace('/', '_').replace('-', '_')
    safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
    save_path = os.path.join(save_dir, safe_model_name, safe_dataset_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Count total activations to save for progress bar
    total_activations = 0
    for layer_idx, layer_buf in buffers.items():
        for act_type, buf in layer_buf.items():
            if isinstance(buf, ActivationBuffer) and buf._items:
                total_activations += 1
    
    # Convert buffers to CPU tensors and save
    # Note: We create a copy of the data for saving, but keep buffers intact for later SVD
    saved_buffers = {}
    pbar = tqdm(total=total_activations, desc=f"Saving {dataset_name}", unit="activation",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for layer_idx, layer_buf in buffers.items():
        saved_buffers[layer_idx] = {}
        for act_type, buf in layer_buf.items():
            if isinstance(buf, ActivationBuffer):
                # Create a temporary copy for saving without modifying the original buffer
                # We'll concatenate items manually to avoid calling finalize() which might clear the buffer
                if buf._items:
                    # Update progress bar
                    pbar.set_description(f"Saving {dataset_name}: L{layer_idx}-{act_type}")
                    
                    # Concatenate items for saving
                    temp_items = [item.cpu() if isinstance(item, torch.Tensor) and item.is_cuda else item 
                                 for item in buf._items]
                    activation_matrix = torch.cat(temp_items, dim=0)
                    if activation_matrix is not None:
                        # Convert to numpy for efficient storage
                        activation_matrix_cpu = activation_matrix.cpu().numpy()
                        saved_buffers[layer_idx][act_type] = {
                            'data': activation_matrix_cpu,
                            'shape': list(activation_matrix_cpu.shape),
                            'count': buf.count,
                        }
                        del activation_matrix_cpu
                        del activation_matrix
                        del temp_items
                    pbar.update(1)
    
    # Save to pickle file
    save_file = os.path.join(save_path, 'activations.pkl')
    pbar.set_description(f"Saving {dataset_name}: Writing pickle file")
    with open(save_file, 'wb') as f:
        pickle.dump(saved_buffers, f)
    
    pbar.close()
    print(f"Saved activations to {save_file}")
    return save_path


@torch.no_grad()
def collect_activations(
    model: nn.Module,
    tokenizer,
    datasets: List[str],
    nsamples: int = 64,
    seqlen: int = 1024,
    batch_size: int = 4,  # Increased default for better GPU utilization
    max_tokens: int = 65536,
    device: torch.device = torch.device('cpu'),
    limit_layers: int = 0,
    model_name: str = '',
    save_activations: bool = True,
    save_dir: str = 'checkpoints/motivation_activations',
) -> Dict:
    """
    Collect activations from model forward passes on multiple datasets.
    
    Returns:
        Dictionary with activation buffers organized by layer and activation type
    """
    model.eval()

    # Determine number of layers (handle different architectures)
    num_layers = 0
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA/Qwen/Mixtral architecture
        num_layers = len(model.model.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        # OPT architecture (newer)
        num_layers = len(model.model.decoder.layers)
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        # OPT architecture (alternative)
        num_layers = len(model.decoder.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # OPT architecture (older)
        num_layers = len(model.transformer.h)
    else:
        print("Warning: Could not determine number of layers")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
        num_layers = 0
    
    if limit_layers > 0:
        num_layers = min(num_layers, limit_layers)
    
    print(f"Found {num_layers} layers to hook")
    
    # Initialize buffers for all layers
    buffers: Dict[int, Dict[str, ActivationBuffer]] = {}
    all_handles = []
    
    # Register hooks for all layers
    hooks_registered = 0
    for layer_idx in range(num_layers):
        buffers[layer_idx] = {}
        # Hook attention activations
        handles_attn = hook_attention_for_rope(model, layer_idx, buffers, max_tokens, device=device)
        all_handles.extend(handles_attn)
        hooks_registered += len(handles_attn)
        # Hook MLP activations
        handles_mlp = hook_mlp_activations(model, layer_idx, buffers, max_tokens, device=device)
        all_handles.extend(handles_mlp)
        hooks_registered += len(handles_mlp)
    
    print(f"Registered {hooks_registered} hooks across {num_layers} layers")
    
    # Collect data from datasets - each dataset gets its own separate buffers
    total_samples_collected = 0
    all_dataset_buffers = {}  # Store buffers per dataset for later SVD computation
    
    for dataset_name in datasets:
        if get_calib_train_data is None:
            print(f"Warning: Skipping {dataset_name} (data_utils not available)")
            continue
        
        # Reset buffers for this dataset (clear previous dataset's data)
        for layer_idx, layer_buf in buffers.items():
            for act_type, buf in layer_buf.items():
                if isinstance(buf, ActivationBuffer):
                    buf._items = []
                    buf.count = 0
        
        try:
            print(f"\nCollecting activations from {dataset_name}...")
            calib_loader = get_calib_train_data(
                dataset_name, tokenizer, 
                nsamples=nsamples, 
                seqlen=seqlen, 
                batch_size=batch_size
            )
            
            # Forward passes
            samples_this_dataset = 0
            batch_count = 0
            for batch in tqdm(calib_loader, desc=f"Processing {dataset_name}"):
                batch_count += 1
                if isinstance(batch, dict):
                    input_ids = batch.get('input_ids', None)
                elif isinstance(batch, (list, tuple)):
                    input_ids = batch[0] if len(batch) > 0 else None
                else:
                    input_ids = batch
                
                if input_ids is None:
                    if batch_count == 1:
                        print(f"Warning: First batch from {dataset_name} has no input_ids")
                    continue
                
                # Move to device with non_blocking transfer for better GPU utilization
                if isinstance(input_ids, torch.Tensor):
                    if device.type == 'cuda' and not input_ids.is_cuda:
                        input_ids = input_ids.to(device, non_blocking=True)
                    else:
                        input_ids = input_ids.to(device)
                    if input_ids.numel() == 0:
                        if batch_count == 1:
                            print(f"Warning: First batch from {dataset_name} has empty input_ids")
                        continue
                else:
                    if batch_count == 1:
                        print(f"Warning: First batch from {dataset_name} is not a tensor")
                    continue
                
                # Check if all buffers are full
                all_full = all(
                    all(buf.is_full() for buf in layer_buf.values() if isinstance(buf, ActivationBuffer))
                    for layer_buf in buffers.values()
                )
                if all_full:
                    print(f"All buffers full, stopping collection from {dataset_name}")
                    break

                # Forward pass - use async execution for better GPU utilization
                try:
                    if device.type == 'cuda':
                        # Use CUDA stream for async execution
                        with torch.cuda.stream(torch.cuda.Stream()):
                            _ = model(input_ids)
                        torch.cuda.synchronize()  # Sync only once after forward
                    else:
                        _ = model(input_ids)
                    samples_this_dataset += 1
                    total_samples_collected += 1
                    
                    # Debug: Check if any activations were collected after first sample
                    if samples_this_dataset == 1:
                        total_activations = sum(
                            sum(buf.count for buf in layer_buf.values() if isinstance(buf, ActivationBuffer))
                            for layer_buf in buffers.values()
                        )
                        print(f"After first sample: {total_activations} activation tokens collected")
                except Exception as e:
                    print(f"Error in forward pass: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # Check if we have enough samples
                if samples_this_dataset >= nsamples:
                    break
            
            if samples_this_dataset == 0 and batch_count > 0:
                print(f"Warning: Processed {batch_count} batches but collected 0 samples from {dataset_name}")
            
            print(f"Collected {samples_this_dataset} samples from {dataset_name}")
            
            # Save activations for this dataset separately (if enabled)
            if save_activations and model_name:
                print(f"\nSaving activations for {dataset_name}...")
                # Create a deep copy of buffers for saving (so we can reset them for next dataset)
                dataset_buffers_copy = {}
                for layer_idx, layer_buf in buffers.items():
                    dataset_buffers_copy[layer_idx] = {}
                    for act_type, buf in layer_buf.items():
                        if isinstance(buf, ActivationBuffer):
                            # Create a copy of the buffer with its current state
                            copy_buf = ActivationBuffer(max_tokens, center=buf.center, device=device)
                            copy_buf._items = [item.clone() if isinstance(item, torch.Tensor) else item 
                                               for item in buf._items]
                            copy_buf.count = buf.count
                            dataset_buffers_copy[layer_idx][act_type] = copy_buf
                
                try:
                    save_activations_to_disk(dataset_buffers_copy, model_name, dataset_name, save_dir)
                    
                    # Store dataset buffers copy for later SVD computation
                    all_dataset_buffers[dataset_name] = dataset_buffers_copy
                    
                    # Clear GPU and CPU memory after saving
                    # Note: We keep dataset_buffers_copy in all_dataset_buffers for SVD, but clear the reference
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    gc.collect()
                    print(f"Cleared GPU and CPU memory after saving {dataset_name} activations")
                except Exception as e:
                    print(f"Warning: Failed to save activations for {dataset_name}: {e}")
                    print(f"  Continuing without saving (will still compute SVD)...")
                    # Create copy for SVD computation even if save failed
                    dataset_buffers_copy = {}
                    for layer_idx, layer_buf in buffers.items():
                        dataset_buffers_copy[layer_idx] = {}
                        for act_type, buf in layer_buf.items():
                            if isinstance(buf, ActivationBuffer):
                                copy_buf = ActivationBuffer(max_tokens, center=buf.center, device=device)
                                copy_buf._items = [item.clone() if isinstance(item, torch.Tensor) else item 
                                                   for item in buf._items]
                                copy_buf.count = buf.count
                                dataset_buffers_copy[layer_idx][act_type] = copy_buf
                    all_dataset_buffers[dataset_name] = dataset_buffers_copy
            else:
                # Create copy for SVD computation (even if not saving to disk)
                dataset_buffers_copy = {}
                for layer_idx, layer_buf in buffers.items():
                    dataset_buffers_copy[layer_idx] = {}
                    for act_type, buf in layer_buf.items():
                        if isinstance(buf, ActivationBuffer):
                            copy_buf = ActivationBuffer(max_tokens, center=buf.center, device=device)
                            copy_buf._items = [item.clone() if isinstance(item, torch.Tensor) else item 
                                               for item in buf._items]
                            copy_buf.count = buf.count
                            dataset_buffers_copy[layer_idx][act_type] = copy_buf
                all_dataset_buffers[dataset_name] = dataset_buffers_copy
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all dataset buffers into a single structure for SVD computation
    # This allows analyzing all datasets together while keeping them separate on disk
    combined_buffers: Dict[int, Dict[str, ActivationBuffer]] = {}
    for layer_idx in range(num_layers):
        combined_buffers[layer_idx] = {}
        # For each activation type, combine across datasets
        for dataset_name, dataset_buf in all_dataset_buffers.items():
            if layer_idx in dataset_buf:
                for act_type, buf in dataset_buf[layer_idx].items():
                    if isinstance(buf, ActivationBuffer):
                        if act_type not in combined_buffers[layer_idx]:
                            # Create new buffer for combined data
                            combined_buffers[layer_idx][act_type] = ActivationBuffer(max_tokens, center=False, device=device)
                        # Add activations from this dataset
                        activation_matrix = buf.finalize()
                        if activation_matrix is not None:
                            # Reshape to [B, T, D] for add() method
                            if activation_matrix.dim() == 2:
                                activation_matrix = activation_matrix.unsqueeze(0)
                            combined_buffers[layer_idx][act_type].add(activation_matrix)
                            del activation_matrix
    
    # Return combined buffers for SVD computation
    buffers = combined_buffers
    
    # Remove all hooks
    for handle in all_handles:
        handle.remove()
    
    print(f"\nTotal samples collected: {total_samples_collected}")
    
    return buffers


@torch.no_grad()
def analyze_model_activations(
    display_name: str,
    model_name: str,
    datasets: List[str],
    nsamples: int = 64,
    seqlen: int = 1024,
    batch_size: int = 4,  # Increased default for better GPU utilization
    max_tokens: int = 65536,
    device_str: str = 'auto',
    limit_layers: int = 0,
) -> Optional[Dict]:
    """
    Analyze activation ranks for a single model.
    
    Returns:
        Dictionary with activation rank analysis results
    """
    # Determine device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device_str == 'cuda':
        if not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Using CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device('cuda')
    else:
        device = torch.device(device_str)
    
    print(f"\n{'='*80}")
    print(f"Analyzing Activations: {display_name} ({model_name})")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Using device: {device} ({'GPU' if device.type == 'cuda' else 'CPU'})")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*80}")
    
    # Check for cached activations first (before loading model)
    force_recompute = os.getenv('FORCE_RECOMPUTE', '0').lower() in ('1', 'true', 'yes')
    skip_act_save = os.getenv('SKIP_ACT_SAVE', '0').lower() in ('1', 'true', 'yes')
    
    buffers = None
    model = None
    tokenizer = None
    
    if not force_recompute:
        print(f"\nChecking for cached activations...")
        cached_status = check_activations_cached(model_name, datasets, 'checkpoints/motivation_activations')
        all_cached = all(cached_status.values())
        some_cached = any(cached_status.values())
        
        if all_cached:
            print(f"  Found cached activations for all datasets: {', '.join(datasets)}")
            print(f"  Loading from cache (set FORCE_RECOMPUTE=1 to recompute)...")
            buffers = load_cached_activations(
                model_name, datasets, 'checkpoints/motivation_activations', max_tokens, device=device
            )
            if buffers is None:
                print(f"  Failed to load cached activations, will recompute...")
            else:
                print(f"  Successfully loaded cached activations, skipping model loading and forward passes")
        elif some_cached:
            cached_datasets = [d for d, cached in cached_status.items() if cached]
            missing_datasets = [d for d, cached in cached_status.items() if not cached]
            print(f"  Found cached activations for: {', '.join(cached_datasets)}")
            print(f"  Missing cached activations for: {', '.join(missing_datasets)}")
            print(f"  Will recompute all datasets (set FORCE_RECOMPUTE=1 to force recompute cached ones)")
    
    # Load model and tokenizer only if we need to collect activations
    if buffers is None:
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print(f"\nLoading model...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            model.eval()
            model.to(device)
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        # Collect activations
        print(f"\nCollecting activations...")
        if skip_act_save:
            print(f"  Note: Skipping activation save to disk (SKIP_ACT_SAVE=1)")
        
        buffers = collect_activations(
            model, tokenizer, datasets, nsamples, seqlen, batch_size,
            max_tokens, device, limit_layers,
            model_name=model_name,
            save_activations=not skip_act_save,
            save_dir='checkpoints/motivation_activations'
        )
        
        # After collecting all datasets, remove model from GPU memory
        print(f"\nRemoving model from GPU memory...")
        del model
        del tokenizer
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Aggressively clear CPU memory
        gc.collect()
        gc.collect()  # Call twice for thorough cleanup
        print(f"Model removed from GPU memory, CPU memory cleared")
    
    # After collecting all datasets, remove model from GPU memory (if not already removed)
    if 'model' in locals():
        print(f"\nRemoving model from GPU memory...")
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print(f"Model removed from GPU memory")
    
    # Compute SVD for all activations
    print(f"\nComputing SVD for all activations...")
    activation_ranks = []
    
    # Determine device for SVD (may differ from forward pass device)
    svd_device = device if device.type == 'cuda' else torch.device('cpu')
    
    # Count total activations to process for progress bar (without finalizing)
    total_activations = 0
    for layer_idx in sorted(buffers.keys()):
        layer_buf = buffers[layer_idx]
        for act_type, buf in layer_buf.items():
            if isinstance(buf, ActivationBuffer) and buf.count > 0:
                total_activations += 1
    
    # Process each layer with progress bar
    pbar = tqdm(total=total_activations, desc="Computing SVD", unit="matrix", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Batch process matrices to keep GPU memory utilized
    # Only clear cache periodically (every N matrices) instead of after each one
    cache_clear_interval = 50  # Clear cache every 10 matrices
    matrices_processed = 0
    
    for layer_idx in sorted(buffers.keys()):
        layer_buf = buffers[layer_idx]
        
        # Process each activation type
        for act_type, buf in layer_buf.items():
            if not isinstance(buf, ActivationBuffer):
                continue
            
            if buf.count == 0:
                continue
            
            activation_matrix = buf.finalize()
            if activation_matrix is None:
                pbar.update(1)
                continue
            
            # Update progress bar with current task
            matrix_shape = activation_matrix.shape
            matrix_size = matrix_shape[0] * matrix_shape[1] if len(matrix_shape) >= 2 else 0
            device_info = f"GPU" if svd_device.type == 'cuda' and matrix_size > 100 else "CPU"
            pbar.set_description(f"Computing SVD: L{layer_idx}-{act_type} ({device_info})")
            
            # Compute SVD - use GPU more aggressively for better utilization
            svd_result = compute_full_svd_analysis(
                activation_matrix, svd_device, 
                use_gpu_for_large=True, gpu_threshold=100  # Lower threshold for better GPU utilization
            )
            
            if svd_result is None:
                pbar.update(1)
                del activation_matrix
                matrices_processed += 1
                continue
            
            activation_ranks.append({
                'layer_index': layer_idx,
                'activation_type': act_type,
                'shape': list(activation_matrix.shape),
                'theoretical_max_rank': svd_result['theoretical_max_rank'],
                'actual_rank': svd_result['actual_rank'],
                'singular_values': svd_result['singular_values'],
                'energy_cumulative': svd_result['energy_cumulative'],
                'total_energy': svd_result['total_energy'],
            })
            
            # Clean up activation matrix immediately
            del activation_matrix
            matrices_processed += 1
            pbar.update(1)
            
            # Periodically clear CPU memory for intermediate SVD results
            if matrices_processed % (cache_clear_interval * 2) == 0:
                # Clear singular values from memory (they're already saved in activation_ranks)
                gc.collect()
            
            # Periodically clear GPU cache and CPU memory to prevent memory fragmentation
            # But not after every matrix to keep GPU memory utilized
            if matrices_processed % cache_clear_interval == 0:
                if svd_device.type == 'cuda':
                    torch.cuda.empty_cache()
                # Also clear CPU memory periodically
                gc.collect()
    
    pbar.close()
    
    # Model was already removed from GPU after collecting activations
    # Aggressively clear CPU memory after SVD computation
    print(f"\nClearing CPU memory after SVD computation...")
    del buffers
    gc.collect()
    gc.collect()  # Call twice to ensure cleanup
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"CPU memory cleared")
    
    if not activation_ranks:
        print(f"No activations collected for {display_name}")
        return None
    
    # Print summary
    print(f"\nModel: {display_name}")
    print(f"Total activation matrices analyzed: {len(activation_ranks)}")
    
    actual_ranks = [a['actual_rank'] for a in activation_ranks]
    theoretical_max_ranks = [a['theoretical_max_rank'] for a in activation_ranks]
    
    print(f"\nRank Statistics:")
    print(f"  Actual Rank - Mean: {statistics.mean(actual_ranks):.2f}, "
          f"Min: {min(actual_ranks)}, Max: {max(actual_ranks)}")
    print(f"  Theoretical Max Rank - Mean: {statistics.mean(theoretical_max_ranks):.2f}, "
          f"Min: {min(theoretical_max_ranks)}, Max: {max(theoretical_max_ranks)}")
    print(f"  Rank Utilization: {statistics.mean(actual_ranks) / statistics.mean(theoretical_max_ranks) * 100:.2f}%")
    
    return {
        'display_name': display_name,
        'model_name': model_name,
        'datasets': datasets,
        'nsamples': nsamples,
        'seqlen': seqlen,
        'activation_ranks': activation_ranks,
    }


def save_to_csv(results: Dict, output_dir: str = '.'):
    """
    Save full SVD and rank distribution to CSV files.
    Similar to weight_rank_distribution.py, saves detailed per-activation singular values.
    
    Args:
        results: Analysis results dictionary
        output_dir: Directory to save CSV files
    """
    if results is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model name
    safe_name = results['display_name'].replace('/', '_').replace('-', '_')
    
    activation_ranks = results.get('activation_ranks', [])
    if not activation_ranks:
        return
    
    # 1. Save detailed rank distribution (all activation matrices with full SVD info)
    csv_path = os.path.join(output_dir, f"{safe_name}_activation_rank_distribution.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'layer_index',
            'activation_type',
            'shape_0',
            'shape_1',
            'theoretical_max_rank',
            'actual_rank',
            'rank_utilization_pct',
            'total_energy',
            'singular_values_count',
        ])
        
        for act in activation_ranks:
            shape = act['shape']
            theo_max = act['theoretical_max_rank']
            actual = act['actual_rank']
            sv_count = len(act['singular_values'])
            util_pct = (actual / theo_max * 100) if theo_max > 0 else 0.0
            
            writer.writerow([
                act['layer_index'],
                act['activation_type'],
                shape[0] if len(shape) > 0 else 0,
                shape[1] if len(shape) > 1 else 0,
                theo_max,
                actual,
                f"{util_pct:.4f}",
                f"{act['total_energy']:.6e}",
                sv_count,
            ])
    
    print(f"\nSaved detailed rank distribution to: {csv_path}")
    
    # 2. Save singular values for each activation (separate file per activation for detailed analysis)
    svd_dir = os.path.join(output_dir, f"{safe_name}_singular_values")
    os.makedirs(svd_dir, exist_ok=True)
    
    # Also create a consolidated file with all singular values
    consolidated_path = os.path.join(output_dir, f"{safe_name}_all_singular_values.csv")
    
    with open(consolidated_path, 'w', newline='') as f_consolidated:
        writer_consolidated = csv.writer(f_consolidated)
        writer_consolidated.writerow([
            'layer_index',
            'activation_type',
            'rank_index',
            'singular_value',
            'cumulative_energy',
            'theoretical_max_rank',
        ])
        
        for act in activation_ranks:
            layer_idx = act['layer_index']
            act_type = act['activation_type']
            singular_values = np.array(act['singular_values'])
            energy_cumulative = np.array(act['energy_cumulative'])
            theoretical_max = act['theoretical_max_rank']
            
            # Create safe filename from layer index and activation type
            act_safe = act_type.replace('.', '_').replace('/', '_').replace('-', '_')
            svd_path = os.path.join(svd_dir, f"layer_{layer_idx}_{act_safe}_singular_values.csv")
            
            # Ensure we have values up to theoretical_max_rank
            if len(singular_values) < theoretical_max:
                # Pad with zeros (shouldn't happen normally, but handle gracefully)
                padded_sv = np.zeros(theoretical_max)
                padded_sv[:len(singular_values)] = singular_values
                singular_values = padded_sv
                
                padded_energy = np.ones(theoretical_max)
                if len(energy_cumulative) > 0:
                    padded_energy[:len(energy_cumulative)] = energy_cumulative
                energy_cumulative = padded_energy
            elif len(singular_values) > theoretical_max:
                # Truncate (shouldn't happen)
                singular_values = singular_values[:theoretical_max]
                energy_cumulative = energy_cumulative[:theoretical_max]
            
            # Save individual file for this activation - ALL singular values up to theoretical_max_rank
            with open(svd_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'rank_index',
                    'singular_value',
                    'cumulative_energy',
                    'theoretical_max_rank',
                ])
                # Write ALL singular values (up to theoretical_max_rank)
                for i in range(theoretical_max):
                    sv = float(singular_values[i])
                    cum_energy = float(energy_cumulative[i])
                    writer.writerow([
                        i,
                        f"{sv:.12e}",
                        f"{cum_energy:.12f}",
                        theoretical_max,
                    ])
            
            # Add to consolidated file - ALL singular values for all activations
            for i in range(theoretical_max):
                sv = float(singular_values[i])
                cum_energy = float(energy_cumulative[i])
                writer_consolidated.writerow([
                    layer_idx,
                    act_type,
                    i,
                    f"{sv:.12e}",
                    f"{cum_energy:.12f}",
                    theoretical_max,
                ])
    
    print(f"Saved singular values to: {svd_dir}/")
    print(f"Saved consolidated singular values to: {consolidated_path}")
    
    # 3. Save per-layer detailed summary
    per_layer_summary = {}
    for act in activation_ranks:
        layer_idx = act['layer_index']
        if layer_idx not in per_layer_summary:
            per_layer_summary[layer_idx] = []
        per_layer_summary[layer_idx].append(act)
    
    if per_layer_summary:
        summary_path = os.path.join(output_dir, f"{safe_name}_layer_details.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'layer_index',
                'activation_type',
                'shape_0',
                'shape_1',
                'theoretical_max_rank',
                'actual_rank',
                'rank_utilization_pct',
            ])
            
            for layer_idx in sorted(per_layer_summary.keys()):
                for act_info in per_layer_summary[layer_idx]:
                    shape = act_info['shape']
                    theo_max = act_info['theoretical_max_rank']
                    actual = act_info['actual_rank']
                    util_pct = (actual / theo_max * 100) if theo_max > 0 else 0.0
                    writer.writerow([
                        layer_idx,
                        act_info['activation_type'],
                        shape[0] if len(shape) > 0 else 0,
                        shape[1] if len(shape) > 1 else 0,
                        theo_max,
                        actual,
                        f"{util_pct:.4f}",
                    ])
        
        print(f"Saved per-layer details to: {summary_path}")
    
    return csv_path


def plot_rank_distribution(results: Dict, output_dir: str = 'plot'):
    """Create visualizations of activation rank distribution."""
    matplotlib, plt = _setup_matplotlib()
    if plt is None:
        return
    
    if results is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = results['display_name'].replace('/', '_').replace('-', '_')
    
    activation_ranks = results.get('activation_ranks', [])
    if not activation_ranks:
        return
    
    # Color scheme: green and orange
    color_green = '#2ecc71'
    color_orange = '#e67e22'
    color_green_light = '#a8e6cf'
    
    _plot_singular_value_accumulation(results, output_dir, safe_name,
                                       color_green, color_orange, color_green_light)
    
    print(f"Saved plots to {output_dir}/")


def _plot_singular_value_accumulation(results: Dict, output_dir: str, safe_name: str,
                                      color_green: str, color_orange: str, color_green_light: str):
    """Create plot showing singular value accumulation."""
    matplotlib, plt = _setup_matplotlib()
    if plt is None:
        return
    
    activation_ranks = results.get('activation_ranks', [])
    if not activation_ranks:
        return
    
    # Find maximum theoretical rank
    max_theoretical = max([a['theoretical_max_rank'] for a in activation_ranks]) if activation_ranks else 1000
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=150)
    
    # Plot 1: Segmented lines showing cumulative energy accumulation
    for act in activation_ranks:
        energy_cumulative = np.array(act['energy_cumulative'])
        theoretical_max = act['theoretical_max_rank']
        act_type = act['activation_type']
        layer_idx = act['layer_index']
        
        # Create rank indices
        rank_indices = np.arange(theoretical_max)
        
        # Plot segmented line
        label = f"L{layer_idx}-{act_type}" if len(activation_ranks) <= 50 else None
        ax1.plot(rank_indices, energy_cumulative[:theoretical_max],
                linewidth=1.5, alpha=0.6, label=label)
    
    ax1.set_xlabel('Rank Index (0 → Theoretical Maximum Rank)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Energy Fraction', fontsize=13, fontweight='bold')
    ax1.set_title('Singular Value Accumulation: Cumulative Energy per Activation',
                  fontsize=14, fontweight='bold')
    ax1.set_xlim([0, max_theoretical])
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle='--')
    if len(activation_ranks) <= 50:
        ax1.legend(fontsize=7, loc='lower right', ncol=2, framealpha=0.9)
    else:
        ax1.text(0.02, 0.98, f'{len(activation_ranks)} activation matrices',
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Histogram showing distribution
    num_bins = min(100, max_theoretical)
    rank_positions = np.linspace(0, max_theoretical, num_bins)
    
    energies_by_rank = []
    rank_indices_for_hist = []
    
    for rank_pos in rank_positions:
        rank_idx = int(rank_pos)
        energies_at_rank = []
        
        for act in activation_ranks:
            energy_cumulative = np.array(act['energy_cumulative'])
            theoretical_max = act['theoretical_max_rank']
            
            if rank_idx < theoretical_max:
                if rank_idx < len(energy_cumulative):
                    energies_at_rank.append(energy_cumulative[rank_idx])
                else:
                    if len(energy_cumulative) > 0:
                        energies_at_rank.append(energy_cumulative[-1])
        
        if energies_at_rank:
            energies_by_rank.append(energies_at_rank)
            rank_indices_for_hist.append(rank_idx)
    
    if energies_by_rank:
        all_ranks_flat = []
        all_energies_flat = []
        for i, energies in enumerate(energies_by_rank):
            all_ranks_flat.extend([rank_indices_for_hist[i]] * len(energies))
            all_energies_flat.extend(energies)
        
        hist_2d, x_edges, y_edges = np.histogram2d(
            all_ranks_flat, all_energies_flat,
            bins=[min(50, max_theoretical), 50],
            range=[[0, max_theoretical], [0, 1]]
        )
        
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        im = ax2.contourf(X, Y, hist_2d.T, levels=20, cmap='YlGn', alpha=0.8)
        
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Count', fontsize=11)
        
        mean_energies = [np.mean(energies) if energies else 0 for energies in energies_by_rank]
        if len(mean_energies) == len(rank_indices_for_hist):
            ax2.plot(rank_indices_for_hist, mean_energies,
                    color=color_orange, linewidth=2, linestyle='--',
                    label='Mean Cumulative Energy', zorder=10)
            ax2.legend(fontsize=10, loc='lower right')
    
    ax2.set_xlabel('Rank Index (0 → Theoretical Maximum Rank)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Energy Fraction', fontsize=13, fontweight='bold')
    ax2.set_title('Distribution of Cumulative Energy Across All Activations',
                  fontsize=14, fontweight='bold')
    ax2.set_xlim([0, max_theoretical])
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f"Activation Singular Value Accumulation Analysis\n{results['display_name']}",
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{safe_name}_activation_singular_value_accumulation.png")
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)


def main():
    """Main function to analyze activation ranks for all models."""
    # Parse environment variables
    device_str = os.getenv('DEVICE', 'auto')
    nsamples = int(os.getenv('NSAMPLES', '64'))
    seqlen = int(os.getenv('SEQLEN', '1024'))
    # Default batch size: use larger batches for better GPU utilization
    # Can be overridden with BATCH_SIZE env var
    default_batch_size = 4 if torch.cuda.is_available() else 1
    batch_size = int(os.getenv('BATCH_SIZE', str(default_batch_size)))
    max_tokens = int(os.getenv('MAX_TOKENS', '65536'))
    limit_layers = int(os.getenv('LIMIT_LAYERS', '0'))
    
    datasets_str = os.getenv('DATASETS', 'wikitext2,ptb,c4')
    datasets = [d.strip() for d in datasets_str.split(',') if d.strip() in AVAILABLE_DATASETS]
    
    if not datasets:
        print("Warning: No valid datasets specified. Using default: wikitext2")
        datasets = ['wikitext2']
    
    plot_dir = os.getenv('PLOT_DIR', 'motivations/act_rank_distribution/plot')
    csv_dir = os.getenv('CSV_DIR', 'motivations/act_rank_distribution/csv_data')
    
    # Set HF endpoint if specified
    hf_endpoint = os.getenv('HF_ENDPOINT', None)
    if hf_endpoint:
        os.environ['HF_ENDPOINT'] = hf_endpoint
    
    print(f"\n{'='*80}")
    print(f"Activation Rank Distribution Analysis")
    print(f"{'='*80}")
    skip_act_save = os.getenv('SKIP_ACT_SAVE', '0').lower() in ('1', 'true', 'yes')
    
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"NSamples: {nsamples}, SeqLen: {seqlen}, BatchSize: {batch_size}")
    if torch.cuda.is_available() and batch_size == 1:
        print(f"  Note: Consider increasing BATCH_SIZE (e.g., BATCH_SIZE=4) for better GPU utilization")
    print(f"MaxTokens per activation: {max_tokens}")
    print(f"Device: {device_str}")
    print(f"Skip activation save: {skip_act_save}")
    print(f"Output directories: Plot={plot_dir}, CSV={csv_dir}")
    print(f"{'='*80}\n")
    
    # Process each model
    for display_name, model_name in MODELS:
        try:
            results = analyze_model_activations(
                display_name, model_name, datasets, nsamples, seqlen, batch_size,
                max_tokens, device_str, limit_layers
            )
            
            if results is not None:
                save_to_csv(results, csv_dir)
                plot_rank_distribution(results, plot_dir)
            
            # Aggressively clean up after each model
            print(f"\nClearing memory after processing {display_name}...")
            del results
            gc.collect()
            gc.collect()  # Call twice for thorough cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"Memory cleared for {display_name}")
            
        except Exception as e:
            print(f"\nError processing {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*80}")
    print(f"Analysis complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
