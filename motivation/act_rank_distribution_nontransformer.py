#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
act_rank_distribution_nontransformer.py 

similar to act_rank_distribution.py, but for non-transformer models
we should include mamba, rwkv, etc.

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
# Basic usage (for non-transformer models: Mamba, RWKV, etc.)
HF_ENDPOINT=https://hf-mirror.com python3 motivations/act_rank_distribution_nontransformer.py

# Custom datasets and samples
DATASETS=wikitext2,ptb,gsm8k,commonsenseqa,humaneval,aqua,strategyqa,multiarith NSAMPLES=16 SEQLEN=512 python3 motivations/act_rank_distribution_nontransformer.py

HF_ENDPOINT=https://hf-mirror.com FORCE_RECOMPUTE=1 SKIP_ACT_SAVE=1 CUDA_VISIBLE_DEVICES=1 DATASETS=wikitext2,ptb,gsm8k,commonsenseqa,aqua,strategyqa NSAMPLES=16 SEQLEN=512 python3 motivations/act_rank_distribution_nontransformer.py 

FORCE_RECOMPUTE=1 CUDA_VISIBLE_DEVICES=1 DATASETS=wikitext2,ptb,gsm8k,commonsenseqa,aqua,strategyqa NSAMPLES=16 SEQLEN=512 python3 motivations/act_rank_distribution_nontransformer.py 


# Analyze specific non-transformer model
MODELS="Mamba-130M" DATASETS=wikitext2,ptb NSAMPLES=16 SEQLEN=256 python3 motivations/act_rank_distribution_nontransformer.py

# Skip saving activations to disk (reduces IO overhead)
SKIP_ACT_SAVE=1 CUDA_VISIBLE_DEVICES=3 DATASETS=wikitext2,ptb,gsm8k,commonsenseqa,aqua,strategyqa NSAMPLES=16 SEQLEN=256 python3 motivations/act_rank_distribution_nontransformer.py

# Use cached activations if available (default behavior)
# If activations are cached, will skip forward passes and compute SVD directly
python3 motivations/act_rank_distribution_nontransformer.py

# Force recompute even if cached activations exist
FORCE_RECOMPUTE=1 python3 motivations/act_rank_distribution_nontransformer.py

# Force CPU usage
DEVICE=cpu python3 motivations/act_rank_distribution_nontransformer.py

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
# Non-transformer models: Mamba, RWKV, etc.
# All models can be loaded via hf-mirror by setting HF_ENDPOINT=https://hf-mirror.com
MODELS = [
    # Mamba models
    # ("Mamba-130M", "state-spaces/mamba-130m"),
    # ("Mamba-370M", "state-spaces/mamba-370m"),
    # ("Mamba-790M", "state-spaces/mamba-790m"),
    # ("Mamba-1.4B", "state-spaces/mamba-1.4b"),
    ("Mamba2-2.7B", "AntonV/mamba2-2.7b-hf"),
    # ("Falcon-Mamba-1.3B", "tiiuae/falcon-mamba-1.3b"),  # 2.8B parameters
    ("Falcon-Mamba-7B", "tiiuae/falcon-mamba-7b"),  # 7B parameters - verified
    # RWKV models - Note: RWKV models may use different naming conventions
    # Common patterns: rwkv-raven-*, rwkv-world-*, rwkv-*-*
    # ("RWKV-Raven-1B5", "RWKV/rwkv-raven-1b5"),  # Alternative naming - verify
    # RWKV-4 World 系列 (已转换为标准HF格式，推荐使用)[citation:8]
    # ("RWKV-4-World-0.1B", "RWKV/rwkv-4-world-169m"),
    #("RWKV-4-World-0.4B", "RWKV/rwkv-4-world-430m"),
    # ("RWKV-4-World-1.5B", "RWKV/rwkv-4-world-1b5"),
    #("RWKV-4-World-3B", "RWKV/rwkv-4-world-3b"),
    # RWKV-5 World 系列 (新一代架构)[citation:4]
    # ("RWKV-5-World-0.1B", "RWKV/rwkv-5-world-169m"),
    # Note: To use hf-mirror, set: HF_ENDPOINT=https://hf-mirror.com
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
    gpu_threshold: int = 10000,
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
            # Ensure matrix is on CPU first (it might already be), then move to GPU
            if activation_matrix.is_cuda:
                A_gpu = activation_matrix.float()
            else:
                A_gpu = activation_matrix.float().to(device)
            # Compute SVD on GPU
            U, S, Vh = torch.linalg.svd(A_gpu, full_matrices=False)
            singular_values = S.cpu().numpy()
            del A_gpu, U, Vh
            # Don't clear cache after every SVD - let GPU manage memory more efficiently
            # Only clear cache periodically or when memory pressure is detected
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Fallback to CPU
                A_cpu = activation_matrix.float().cpu()
                U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
                singular_values = S.numpy()
                del A_cpu, U, Vh
            else:
                raise
    else:
        # Use CPU
        A_cpu = activation_matrix.float().cpu()
        U, S, Vh = torch.linalg.svd(A_cpu, full_matrices=False)
        singular_values = S.numpy()
        del A_cpu, U, Vh
    
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
    
    def __init__(self, max_tokens: int, center: bool = False):
        self.max_tokens = int(max_tokens)
        self.center = bool(center)
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
        
        self._items.append(X.cpu())
        self.count += take
    
    def finalize(self) -> Optional[torch.Tensor]:
        """Concatenate all collected activations."""
        if not self._items:
            return None
        return torch.cat(self._items, dim=0)  # [N_total, D]
    
    def is_full(self) -> bool:
        return self.count >= self.max_tokens


def find_layer(model, layer_idx: int):
    """
    Find layer by index across different architectures.
    Supports: Transformer (LLaMA, OPT, etc.), Mamba, RWKV.
    """
    # Transformer architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        if layer_idx < len(layers):
            return layers[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        layers = model.model.decoder.layers
        if layer_idx < len(layers):
            return layers[layer_idx]
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        layers = model.decoder.layers
        if layer_idx < len(layers):
            return layers[layer_idx]
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
        if layer_idx < len(layers):
            return layers[layer_idx]
    
    # Mamba architecture
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        layers = model.backbone.layers
        if layer_idx < len(layers):
            return layers[layer_idx]
    elif hasattr(model, 'layers'):
        layers = model.layers
        if layer_idx < len(layers):
            return layers[layer_idx]
    
    # RWKV architecture
    # RWKV models typically have model.rwkv.blocks or rwkv.blocks
    if hasattr(model, 'rwkv') and hasattr(model.rwkv, 'blocks'):
        blocks = model.rwkv.blocks
        if layer_idx < len(blocks):
            return blocks[layer_idx]
    elif hasattr(model, 'blocks'):
        blocks = model.blocks
        if layer_idx < len(blocks):
            return blocks[layer_idx]
    elif hasattr(model, 'model') and hasattr(model.model, 'rwkv') and hasattr(model.model.rwkv, 'blocks'):
        blocks = model.model.rwkv.blocks
        if layer_idx < len(blocks):
            return blocks[layer_idx]
    
    return None


def hook_attention_for_rope(model, layer_idx: int, buffers: Dict, max_tokens: int):
    """
    Hook attention/SSM module to capture activations.
    Supports: Transformer (Q/K/V/O), Mamba (in_proj/out_proj/ssm), RWKV (att/ffn).
    """
    handles = []
    layer = find_layer(model, layer_idx)
    
    if layer is None:
        return handles
    
    # Initialize buffers for this layer
    layer_buf = buffers.get(layer_idx, {})
    
    # ===== Transformer architectures =====
    attn = None
    if hasattr(layer, 'self_attn'):
        attn = layer.self_attn
    elif hasattr(layer, 'attn'):
        attn = layer.attn
    
    if attn is not None:
        # Pre-RoPE buffers
        if hasattr(attn, 'q_proj'):
            q_pre_buf = ActivationBuffer(max_tokens, center=False)
            k_pre_buf = ActivationBuffer(max_tokens, center=False)
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
        
        # Output projection
        if hasattr(attn, 'o_proj'):
            o_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['O'] = o_buf
            
            def hook_o(m, i, o):
                o_buf.add(o.detach())
            
            handles.append(attn.o_proj.register_forward_hook(hook_o))
        elif hasattr(attn, 'out_proj'):
            o_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['O'] = o_buf
            
            def hook_o(m, i, o):
                o_buf.add(o.detach())
            
            handles.append(attn.out_proj.register_forward_hook(hook_o))
    
    # ===== Mamba architecture =====
    if hasattr(layer, 'mixer'):
        # Mamba SSM layer
        mixer = layer.mixer
        if hasattr(mixer, 'in_proj'):
            in_proj_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['in_proj'] = in_proj_buf
            
            def hook_in_proj(m, i, o):
                in_proj_buf.add(o.detach())
            
            handles.append(mixer.in_proj.register_forward_hook(hook_in_proj))
        
        if hasattr(mixer, 'out_proj'):
            out_proj_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['out_proj'] = out_proj_buf
            
            def hook_out_proj(m, i, o):
                out_proj_buf.add(o.detach())
            
            handles.append(mixer.out_proj.register_forward_hook(hook_out_proj))
        
        # SSM state (if accessible)
        if hasattr(mixer, 'ssm'):
            # Hook SSM output if possible
            ssm_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['ssm'] = ssm_buf
            
            def hook_ssm(m, i, o):
                if isinstance(o, torch.Tensor):
                    ssm_buf.add(o.detach())
            
            # Try to hook SSM forward if it's a module
            if isinstance(mixer.ssm, nn.Module):
                handles.append(mixer.ssm.register_forward_hook(hook_ssm))
    
    # ===== RWKV architecture =====
    # RWKV blocks typically have: pre_ln, att, ffn, post_ln
    # Hook individual components for better activation capture
    if hasattr(layer, 'att'):
        # RWKV attention-like mechanism
        att = layer.att
        att_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['att'] = att_buf
        
        def hook_att(m, i, o):
            # RWKV att may return tuple (output, state) or just output
            if isinstance(o, tuple):
                # Take the first element (output tensor)
                o = o[0]
            if isinstance(o, torch.Tensor):
                att_buf.add(o.detach())
        
        handles.append(att.register_forward_hook(hook_att))
    
    if hasattr(layer, 'ffn'):
        # RWKV feed-forward
        ffn = layer.ffn
        ffn_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['ffn'] = ffn_buf
        
        def hook_ffn(m, i, o):
            # RWKV ffn may return tuple (output, state) or just output
            if isinstance(o, tuple):
                # Take the first element (output tensor)
                o = o[0]
            if isinstance(o, torch.Tensor):
                ffn_buf.add(o.detach())
        
        handles.append(ffn.register_forward_hook(hook_ffn))
    
    # Hook RWKV layer's forward output directly (this captures the final output)
    # This is important as RWKV layers process sequentially
    if hasattr(layer, 'forward'):
        layer_output_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['layer_output'] = layer_output_buf
        
        def hook_layer_output(m, i, o):
            # RWKV layer forward may return tuple (output, state) or just output
            if isinstance(o, tuple):
                o = o[0]
            if isinstance(o, torch.Tensor):
                layer_output_buf.add(o.detach())
        
        handles.append(layer.register_forward_hook(hook_layer_output))
    
    # Debug: Print layer structure for first layer
    if layer_idx == 0:
        print(f"RWKV Layer 0 structure: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
        if hasattr(layer, 'att'):
            print(f"  - layer.att type: {type(layer.att)}")
        if hasattr(layer, 'ffn'):
            print(f"  - layer.ffn type: {type(layer.ffn)}")
    
    buffers[layer_idx] = layer_buf
    
    return handles


def hook_mlp_activations(model, layer_idx: int, buffers: Dict, max_tokens: int):
    """
    Hook MLP/FFN activations across different architectures.
    Supports: Transformer (gate_proj/up_proj/down_proj, fc1/fc2), Mamba, RWKV.
    """
    handles = []
    layer = find_layer(model, layer_idx)
    
    if layer is None:
        return handles
    
    layer_buf = buffers.get(layer_idx, {})
    
    # ===== Transformer architectures =====
    mlp = None
    if hasattr(layer, 'mlp'):
        mlp = layer.mlp
    elif hasattr(layer, 'ffn'):
        mlp = layer.ffn
    
    if mlp is not None:
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
        
        # OPT-style MLP: fc1, fc2
        if hasattr(mlp, 'fc1'):
            fc1_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['fc1'] = fc1_buf
            
            def hook_fc1(m, i, o):
                fc1_buf.add(o.detach())
            
            handles.append(mlp.fc1.register_forward_hook(hook_fc1))
        
        if hasattr(mlp, 'fc2'):
            fc2_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['fc2'] = fc2_buf
            
            def hook_fc2(m, i, o):
                fc2_buf.add(o.detach())
            
            handles.append(mlp.fc2.register_forward_hook(hook_fc2))
    
    # ===== Mamba architecture =====
    if hasattr(layer, 'mlp'):
        # Mamba may have MLP after SSM
        mamba_mlp = layer.mlp
        if hasattr(mamba_mlp, 'fc1'):
            mamba_fc1_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['mamba_fc1'] = mamba_fc1_buf
            
            def hook_mamba_fc1(m, i, o):
                mamba_fc1_buf.add(o.detach())
            
            handles.append(mamba_mlp.fc1.register_forward_hook(hook_mamba_fc1))
        
        if hasattr(mamba_mlp, 'fc2'):
            mamba_fc2_buf = ActivationBuffer(max_tokens, center=False)
            layer_buf['mamba_fc2'] = mamba_fc2_buf
            
            def hook_mamba_fc2(m, i, o):
                mamba_fc2_buf.add(o.detach())
            
            handles.append(mamba_mlp.fc2.register_forward_hook(hook_mamba_fc2))
    
    # ===== RWKV architecture =====
    # RWKV ffn is already handled in hook_attention_for_rope
    # But we can also hook individual components if needed
    if hasattr(layer, 'ffn') and 'ffn' not in layer_buf:
        rwkv_ffn = layer.ffn
        rwkv_ffn_buf = ActivationBuffer(max_tokens, center=False)
        layer_buf['rwkv_ffn'] = rwkv_ffn_buf
        
        def hook_rwkv_ffn(m, i, o):
            if isinstance(o, torch.Tensor):
                rwkv_ffn_buf.add(o.detach())
        
        handles.append(rwkv_ffn.register_forward_hook(hook_rwkv_ffn))
    
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
            
            # Convert saved buffers back to ActivationBuffer format
            dataset_buffers = {}
            for layer_idx, layer_buf in saved_buffers.items():
                dataset_buffers[layer_idx] = {}
                for act_type, saved_data in layer_buf.items():
                    if isinstance(saved_data, dict) and 'data' in saved_data:
                        # Convert numpy array back to tensor and create ActivationBuffer
                        activation_data = torch.from_numpy(saved_data['data'])
                        buf = ActivationBuffer(max_tokens, center=False)
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
    max_layer_idx = max(
        max(layer_buf.keys()) if layer_buf else -1
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
                            combined_buffers[layer_idx][act_type] = ActivationBuffer(max_tokens, center=False)
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
    batch_size: int = 1,
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
    
    # Transformer architectures
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        num_layers = len(model.model.layers)
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        num_layers = len(model.model.decoder.layers)
    elif hasattr(model, 'decoder') and hasattr(model.decoder, 'layers'):
        num_layers = len(model.decoder.layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        num_layers = len(model.transformer.h)
    # Mamba architecture
    elif hasattr(model, 'backbone') and hasattr(model.backbone, 'layers'):
        num_layers = len(model.backbone.layers)
    elif hasattr(model, 'layers'):
        num_layers = len(model.layers)
    # RWKV architecture
    # RWKV models typically have model.rwkv.blocks or rwkv.blocks
    elif hasattr(model, 'rwkv') and hasattr(model.rwkv, 'blocks'):
        num_layers = len(model.rwkv.blocks)
        print(f"Detected RWKV architecture: model.rwkv.blocks with {num_layers} layers")
    elif hasattr(model, 'blocks'):
        num_layers = len(model.blocks)
        print(f"Detected RWKV architecture: model.blocks with {num_layers} layers")
    elif hasattr(model, 'model') and hasattr(model.model, 'rwkv') and hasattr(model.model.rwkv, 'blocks'):
        num_layers = len(model.model.rwkv.blocks)
        print(f"Detected RWKV architecture: model.model.rwkv.blocks with {num_layers} layers")
    else:
        print("Warning: Could not determine number of layers")
        print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")
        if hasattr(model, 'model'):
            print(f"model.model attributes: {[attr for attr in dir(model.model) if not attr.startswith('_')]}")
        if hasattr(model, 'rwkv'):
            print(f"model.rwkv attributes: {[attr for attr in dir(model.rwkv) if not attr.startswith('_')]}")
            if hasattr(model.rwkv, 'blocks'):
                print(f"model.rwkv.blocks type: {type(model.rwkv.blocks)}, length: {len(model.rwkv.blocks) if hasattr(model.rwkv.blocks, '__len__') else 'N/A'}")
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
        handles_attn = hook_attention_for_rope(model, layer_idx, buffers, max_tokens)
        all_handles.extend(handles_attn)
        hooks_registered += len(handles_attn)
        # Hook MLP activations
        handles_mlp = hook_mlp_activations(model, layer_idx, buffers, max_tokens)
        all_handles.extend(handles_mlp)
        hooks_registered += len(handles_mlp)
    
    print(f"Registered {hooks_registered} hooks across {num_layers} layers")
    
    # Debug: Print what activation types were registered for first layer
    if num_layers > 0 and 0 in buffers:
        layer_0_types = list(buffers[0].keys())
        print(f"Layer 0 activation types registered: {layer_0_types}")
    
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
                
                # Move to device
                if isinstance(input_ids, torch.Tensor):
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
                
                # Forward pass
                try:
                    # RWKV models may need special handling
                    # Try standard forward first
                    output = model(input_ids)
                    # Handle tuple outputs (some models return (logits, state) or similar)
                    if isinstance(output, tuple):
                        output = output[0]
                    samples_this_dataset += 1
                    total_samples_collected += 1
                    
                    # Debug: Check if any activations were collected after first sample
                    if samples_this_dataset == 1:
                        total_activations = sum(
                            sum(buf.count for buf in layer_buf.values() if isinstance(buf, ActivationBuffer))
                            for layer_buf in buffers.values()
                        )
                        print(f"After first sample: {total_activations} activation tokens collected")
                        # Debug: Print which activation types have data
                        activation_types_with_data = []
                        for layer_idx, layer_buf in buffers.items():
                            for act_type, buf in layer_buf.items():
                                if isinstance(buf, ActivationBuffer) and buf.count > 0:
                                    activation_types_with_data.append(f"L{layer_idx}-{act_type}({buf.count})")
                        if activation_types_with_data:
                            print(f"  Activation types with data: {', '.join(activation_types_with_data[:10])}")
                        else:
                            print(f"  Warning: No activations collected! Check if hooks are working.")
                            # Print layer structure for debugging
                            if num_layers > 0:
                                sample_layer = find_layer(model, 0)
                                if sample_layer is not None:
                                    print(f"  Sample layer (0) attributes: {[attr for attr in dir(sample_layer) if not attr.startswith('_')]}")
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
                            copy_buf = ActivationBuffer(max_tokens, center=buf.center)
                            copy_buf._items = [item.clone() if isinstance(item, torch.Tensor) else item 
                                               for item in buf._items]
                            copy_buf.count = buf.count
                            dataset_buffers_copy[layer_idx][act_type] = copy_buf
                
                try:
                    save_activations_to_disk(dataset_buffers_copy, model_name, dataset_name, save_dir)
                    
                    # Store dataset buffers copy for later SVD computation
                    all_dataset_buffers[dataset_name] = dataset_buffers_copy
                    
                    # Clear GPU memory after saving
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        gc.collect()
                        print(f"Cleared GPU memory after saving {dataset_name} activations")
                except Exception as e:
                    print(f"Warning: Failed to save activations for {dataset_name}: {e}")
                    print(f"  Continuing without saving (will still compute SVD)...")
                    # Create copy for SVD computation even if save failed
                    dataset_buffers_copy = {}
                    for layer_idx, layer_buf in buffers.items():
                        dataset_buffers_copy[layer_idx] = {}
                        for act_type, buf in layer_buf.items():
                            if isinstance(buf, ActivationBuffer):
                                copy_buf = ActivationBuffer(max_tokens, center=buf.center)
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
                            copy_buf = ActivationBuffer(max_tokens, center=buf.center)
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
                            combined_buffers[layer_idx][act_type] = ActivationBuffer(max_tokens, center=False)
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
    batch_size: int = 1,
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
                model_name, datasets, 'checkpoints/motivation_activations', max_tokens
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
            
            print(f"Loading model...")
            # Try fast tokenizer first, fallback to slow if needed (e.g., for Mamba models requiring tiktoken)
            tokenizer = None
            
            # Check if tiktoken is available (required for some Mamba models)
            try:
                import tiktoken
                has_tiktoken = True
            except ImportError:
                has_tiktoken = False
                print(f"  Note: tiktoken not installed. Some models may require it.")
            
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
            except (ValueError, ModuleNotFoundError, OSError) as e:
                error_msg = str(e).lower()
                if "tiktoken" in error_msg or "fast" in error_msg or "converting" in error_msg:
                    print(f"  Fast tokenizer failed: {str(e)[:150]}")
                    
                    if not has_tiktoken and "tiktoken" in error_msg:
                        print(f"  Installing tiktoken to resolve tokenizer issue...")
                        import subprocess
                        import sys
                        try:
                            subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken", "-q"])
                            print(f"  ✓ tiktoken installed successfully")
                            # Retry with fast tokenizer now that tiktoken is installed
                            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
                        except Exception as install_error:
                            print(f"  Failed to install tiktoken: {install_error}")
                            print(f"  Falling back to slow tokenizer...")
                            # Continue to slow tokenizer fallback
                    
                    # If still no tokenizer, try slow tokenizer
                    if tokenizer is None:
                        try:
                            # For Mamba models (GPT-NeoX based), try to import slow tokenizer from correct module
                            # Try different import paths depending on transformers version
                            try:
                                from transformers.models.gpt_neox import GPTNeoXTokenizer
                            except ImportError:
                                try:
                                    from transformers.models.gpt_neox.tokenization_gpt_neox import GPTNeoXTokenizer
                                except ImportError:
                                    # Try to get it from AutoTokenizer's registry
                                    from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING
                                    # Find GPTNeoX slow tokenizer in mapping
                                    GPTNeoXTokenizer = None
                                    for config_class, (slow_tokenizer_class, fast_tokenizer_class) in TOKENIZER_MAPPING.items():
                                        if 'GPTNeoX' in config_class.__name__ and slow_tokenizer_class is not None:
                                            GPTNeoXTokenizer = slow_tokenizer_class
                                            break
                                    
                                    if GPTNeoXTokenizer is None:
                                        raise ImportError("Could not find GPTNeoXTokenizer class")
                            
                            print(f"  Attempting to load slow GPTNeoX tokenizer...")
                            tokenizer = GPTNeoXTokenizer.from_pretrained(model_name, trust_remote_code=True)
                            print(f"  ✓ Successfully loaded slow GPTNeoX tokenizer")
                        except Exception as e2:
                            print(f"  Slow GPTNeoXTokenizer failed: {str(e2)[:150]}")
                            # Final fallback: try to manually load tokenizer files and construct slow tokenizer
                            try:
                                print(f"  Attempting manual tokenizer construction...")
                                from transformers.utils import cached_file
                                import json
                                
                                # Get tokenizer config
                                tokenizer_config_file = cached_file(
                                    model_name,
                                    "tokenizer_config.json",
                                    _raise_exceptions_for_missing_entries=False,
                                )
                                
                                # Try to load vocab file directly
                                vocab_file = cached_file(
                                    model_name,
                                    "vocab.json",
                                    _raise_exceptions_for_missing_entries=False,
                                )
                                
                                merges_file = cached_file(
                                    model_name,
                                    "merges.txt",
                                    _raise_exceptions_for_missing_entries=False,
                                )
                                
                                if vocab_file and merges_file:
                                    # Manually construct GPT2-style tokenizer (GPT-NeoX uses GPT2 tokenizer)
                                    from transformers import GPT2Tokenizer
                                    tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file)
                                    print(f"  ✓ Successfully constructed tokenizer from vocab files")
                                else:
                                    # Last resort: try AutoTokenizer with use_fast=False
                                    # Force it by temporarily modifying transformers behavior
                                    import transformers.models.auto.tokenization_auto as auto_tokenization
                                    original_from_pretrained = auto_tokenization.AutoTokenizer.from_pretrained
                                    
                                    def patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                                        # Force use_fast=False
                                        kwargs['use_fast'] = False
                                        return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                                    
                                    # Temporarily patch
                                    auto_tokenization.AutoTokenizer.from_pretrained = classmethod(patched_from_pretrained)
                                    try:
                                        tokenizer = AutoTokenizer.from_pretrained(
                                            model_name,
                                            trust_remote_code=True,
                                        )
                                    finally:
                                        # Restore original
                                        auto_tokenization.AutoTokenizer.from_pretrained = original_from_pretrained
                            except Exception as e3:
                                # Check if the error is specifically about None tiktoken URL
                                error_str = str(e3).lower()
                                if "nonetype" in error_str or "none" in error_str or "encode" in error_str:
                                    print(f"\n  ERROR: Tokenizer config has invalid tiktoken BPE file URL (None)")
                                    print(f"  This is a known issue with some Mamba models on Hugging Face.")
                                    print(f"  Attempting workaround: using GPT2Tokenizer as fallback...")
                                    
                                    try:
                                        # Mamba models often use GPT2-style tokenization
                                        # Try to use GPT2Tokenizer with the model's vocab
                                        from transformers import GPT2Tokenizer
                                        
                                        # Try to get vocab files
                                        from transformers.utils import cached_file
                                        vocab_file = cached_file(
                                            model_name,
                                            "vocab.json",
                                            _raise_exceptions_for_missing_entries=False,
                                        )
                                        merges_file = cached_file(
                                            model_name,
                                            "merges.txt",
                                            _raise_exceptions_for_missing_entries=False,
                                        )
                                        
                                        if vocab_file and merges_file:
                                            tokenizer = GPT2Tokenizer(vocab_file=vocab_file, merges_file=merges_file)
                                            print(f"  ✓ Successfully loaded GPT2Tokenizer as fallback")
                                        else:
                                            # Last resort: use a generic GPT2 tokenizer
                                            print(f"  Using generic GPT2Tokenizer (may not match model's vocab exactly)")
                                            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                                            print(f"  ⚠ Warning: Using GPT2 tokenizer may cause tokenization mismatches")
                                    except Exception as e4:
                                        print(f"\n  ERROR: All tokenizer loading methods failed for {model_name}")
                                        print(f"  Final error: {str(e4)[:200]}")
                                        print(f"  This model's tokenizer configuration appears to be incomplete.")
                                        print(f"  Consider skipping this model or using a different Mamba model.")
                                        raise RuntimeError(
                                            f"Tokenizer loading failed for {model_name}. "
                                            f"The tokenizer config has invalid tiktoken settings. "
                                            f"Try a different model or contact the model maintainers."
                                        )
                                else:
                                    print(f"\n  ERROR: All tokenizer loading methods failed for {model_name}")
                                    print(f"  Error: {str(e3)[:200]}")
                                    raise RuntimeError(
                                        f"Tokenizer loading failed for {model_name}: {str(e3)[:200]}"
                                    )
                else:
                    raise
            
            if tokenizer is None:
                raise RuntimeError(f"Failed to load tokenizer for {model_name}")
            
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
    cache_clear_interval = 10  # Clear cache every 10 matrices
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
            device_info = f"GPU" if svd_device.type == 'cuda' and matrix_size > 5000 else "CPU"
            pbar.set_description(f"Computing SVD: L{layer_idx}-{act_type} ({device_info})")
            
            # Compute SVD
            svd_result = compute_full_svd_analysis(
                activation_matrix, svd_device, 
                use_gpu_for_large=True, gpu_threshold=5000
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
            
            # Clean up
            del activation_matrix
            matrices_processed += 1
            pbar.update(1)
            
            # Periodically clear GPU cache to prevent memory fragmentation
            # But not after every matrix to keep GPU memory utilized
            if svd_device.type == 'cuda' and matrices_processed % cache_clear_interval == 0:
                torch.cuda.empty_cache()
    
    pbar.close()
    
    # Model was already removed from GPU after collecting activations
    # Just ensure CPU memory is clean
    gc.collect()
    
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
    batch_size = int(os.getenv('BATCH_SIZE', '1'))
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
        print(f"Using HF endpoint: {hf_endpoint}")
    
    skip_act_save = os.getenv('SKIP_ACT_SAVE', '0').lower() in ('1', 'true', 'yes')
    
    print(f"\n{'='*80}")
    print(f"Activation Rank Distribution Analysis (Non-Transformer Models)")
    print(f"{'='*80}")
    print(f"Models: {len(MODELS)}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"NSamples: {nsamples}, SeqLen: {seqlen}, BatchSize: {batch_size}")
    print(f"MaxTokens per activation: {max_tokens}")
    print(f"Device: {device_str}")
    print(f"Skip activation save: {skip_act_save}")
    print(f"Output directories: Plot={plot_dir}, CSV={csv_dir}")
    if hf_endpoint:
        print(f"HF Endpoint: {hf_endpoint}")
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
            
            # Clean up
            del results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
