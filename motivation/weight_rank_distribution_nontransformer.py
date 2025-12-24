#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weight_rank_distribution.py — Compute and visualize FULL SVD rank distribution of model weights

What it does:
- Loads multiple models (OPT-125M, OPT-6.7B, LLaMA-2-7B, Qwen2.5-7B, Mixtral-8x7B)
- Computes FULL SVD for all 2D weight matrices (all singular values, no energy threshold)
- Provides theoretical maximum rank and actual rank for each linear layer
- Saves detailed per-layer rank distributions to CSV files with model-specific filenames
- Creates visualizations showing rank distribution up to theoretical maximum

Environment variables:
- HF_ENDPOINT: Set to hf-mirror URL if needed (e.g., https://hf-mirror.com)
- DEVICE: Device to run SVD on (default: auto - auto-detects GPU if available)
  Options: 'auto', 'cpu', 'cuda'
- PLOT_DIR: Directory to save plots (default: plot)
- CSV_DIR: Directory to save CSV files (default: .)
- LIMIT_LAYERS: If >0, only analyze first N layers (default: 0, analyze all)

Examples:

# Basic usage (auto-detect GPU, use hf-mirror)
HF_ENDPOINT=https://hf-mirror.com python3 motivations/weight_rank_distribution.py

# Force CPU usage
DEVICE=cpu python3 motivations/weight_rank_distribution.py

# Use GPU explicitly
DEVICE=cuda python3 motivations/weight_rank_distribution.py

# With custom output directories
HF_ENDPOINT=https://hf-mirror.com PLOT_DIR=plot CSV_DIR=csv_data python3 motivations/weight_rank_distribution.py

# Analyze only first 5 layers (for testing)
LIMIT_LAYERS=5 DEVICE=cuda PLOT_DIR=motivations/weight_rank_distribution/plot CSV_DIR=motivations/weight_rank_distribution/csv_data python3 motivations/weight_rank_distribution.py
  
Output files:
- {Model}_rank_distribution.csv - Summary of all weight matrices
- {Model}_layer_details.csv - Detailed per-layer breakdown
- {Model}_singular_values/ - Directory with singular values CSV for each parameter
- {Model}_layer_wise_distribution.png - Heatmap and violin plots
- {Model}_rank_distribution_full.png - Full rank distribution plot
- {Model}_rank_per_layer_details.png - Per-layer detailed visualization

"""

import os
import csv
import statistics
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from tqdm import tqdm


# Model configurations: (display_name, hf_model_name)
MODELS = [
    ("OPT-125M", "facebook/opt-125m"),
    ("OPT-6.7B", "facebook/opt-6.7b"),
    ("LLaMA-2-7B", "meta-llama/Llama-2-7b-hf"),
    ("Qwen2.5-7B", "Qwen/Qwen2.5-7B"),
    ("Mixtral-8x7B", "mistralai/Mixtral-8x7B-v0.1"),  # 8 experts of 7B each (45B total)
]


@torch.no_grad()
def _effective_rank_from_singulars(s: torch.Tensor, energy: float) -> int:
    """
    Compute effective rank from singular values using energy fraction.
    
    Args:
        s: Singular values in descending order
        energy: Target energy fraction (0 < energy <= 1)
    
    Returns:
        Effective rank r such that top-r singular values capture energy fraction
    """
    if s is None or s.numel() == 0:
        return 0
    e2 = s.float().pow(2)
    tot = e2.sum()
    if tot <= 0:
        return 0
    csum = torch.cumsum(e2, dim=0)
    target = float(energy) * tot
    r = int(torch.searchsorted(csum, target, right=False).item() + 1)
    return max(1, min(r, s.numel()))


@torch.no_grad()
def compute_full_svd_analysis(
    weight: torch.Tensor,
    device: torch.device = torch.device('cpu'),
    use_gpu_for_large: bool = True,
    gpu_threshold: int = 10000,  # Use GPU only if matrix size > this
) -> Dict:
    """
    Compute full SVD analysis of a weight matrix.
    Returns all singular values and rank information up to theoretical maximum.
    Memory-efficient: uses GPU only for large matrices, CPU for smaller ones.
    
    Args:
        weight: 2D weight tensor
        device: Device preference (but may use CPU for small matrices)
        use_gpu_for_large: Whether to use GPU for large matrices
        gpu_threshold: Minimum matrix dimension to use GPU
    
    Returns:
        Dictionary with:
        - singular_values: all singular values (numpy array)
        - theoretical_max_rank: min(shape[0], shape[1])
        - actual_rank: number of non-zero singular values (above threshold)
        - energy_cumulative: cumulative energy fraction for each rank
    """
    if weight.dim() != 2:
        return None
    
    theoretical_max_rank = min(weight.shape[0], weight.shape[1])
    
    # Decide whether to use GPU or CPU based on matrix size
    # For very large matrices, use GPU; for smaller ones, use CPU to save memory
    max_dim = max(weight.shape[0], weight.shape[1])
    use_gpu = (device.type == 'cuda' and use_gpu_for_large and 
               max_dim > gpu_threshold and torch.cuda.is_available())
    
    # Get weight data (detach to avoid gradients)
    weight_cpu = weight.detach().cpu()
    
    if use_gpu:
        # For large matrices: use GPU
        try:
            W = weight_cpu.to(torch.float32).to(device, non_blocking=True)
            s = torch.linalg.svdvals(W)
            s_cpu = s.cpu().numpy()
            del W, s
            if device.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except RuntimeError as e:
            # Fallback to CPU if GPU fails
            W = weight_cpu.to(torch.float32)
            s = torch.linalg.svdvals(W)
            s_cpu = s.numpy()
            del W, s
    else:
        # For smaller matrices: use CPU (more memory efficient)
        W = weight_cpu.to(torch.float32)
        s = torch.linalg.svdvals(W)
        s_cpu = s.numpy()
        del W, s
    
    # Compute cumulative energy on CPU (numpy)
    s_squared = s_cpu ** 2
    total_energy = s_squared.sum()
    if total_energy > 0:
        cumulative_energy = np.cumsum(s_squared) / total_energy
    else:
        cumulative_energy = np.zeros_like(s_cpu)
    
    # Compute actual rank (number of singular values above threshold)
    threshold = max(1e-7, s_cpu[0] * 1e-6) if len(s_cpu) > 0 and s_cpu[0] > 0 else 1e-7
    actual_rank = int(np.sum(s_cpu > threshold))
    
    return {
        'singular_values': s_cpu,
        'theoretical_max_rank': theoretical_max_rank,
        'actual_rank': actual_rank,
        'energy_cumulative': cumulative_energy,
        'total_energy': float(total_energy),
    }


def analyze_model_weights(
    model_name: str,
    display_name: str,
    limit_layers: int = 0,
    device: str = 'cpu',
) -> Dict:
    """
    Analyze weight rank distribution for a model.
    Computes full SVD for all linear layers and provides detailed rank information.
    
    Args:
        model_name: HuggingFace model identifier
        display_name: Display name for the model
        limit_layers: If >0, only analyze first N layers
        device: Device to run computations on ('cpu', 'cuda', or 'auto' for auto-detect)
    
    Returns:
        Dictionary with analysis results including full SVD details for each layer
    """
    # Auto-detect GPU if available and device is 'auto' or 'cuda'
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif device == 'cuda' and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Using CPU.")
        device = 'cpu'
    
    dev = torch.device(device)
    
    print(f"\n{'='*80}")
    print(f"Analyzing: {display_name} ({model_name})")
    print(f"Computing FULL SVD for all linear layers")
    print(f"Using device: {dev} ({'GPU' if dev.type == 'cuda' else 'CPU'})")
    if dev.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*80}")
    
    # Load model to CPU first to avoid GPU memory issues
    # We'll move individual weights to GPU only when needed for SVD
    try:
        # Clear GPU cache before loading new model
        if dev.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Always load to CPU first - we'll process weights individually
        # This prevents loading entire model to GPU at once
        print(f"Loading model to CPU (will process weights individually)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.eval()
        
        # Move model to CPU explicitly (in case it was loaded elsewhere)
        model = model.cpu()
        
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Collect all 2D weight matrices first (for progress bar)
    weight_params = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() == 2:
            # Extract layer information from parameter name
            parts = name.split('.')
            layer_idx = None
            
            # Try to extract layer index based on common patterns
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                        break
                    except ValueError:
                        pass
            
            weight_params.append({
                'name': name,
                'param': param,
                'layer_idx': layer_idx,
            })
    
    print(f"Found {len(weight_params)} 2D weight matrices to analyze")
    
    # Process weights with progress bar
    weight_ranks = []
    layer_groups = defaultdict(list)
    
    # Use tqdm for progress bar
    pbar = tqdm(weight_params, desc="Computing SVD ranks", unit="matrix", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    # Process weights with periodic GPU cache clearing
    clear_cache_interval = 10  # Clear GPU cache every N matrices
    
    for idx, item in enumerate(pbar):
        name = item['name']
        param = item['param']
        layer_idx = item['layer_idx']
        
        # Update progress bar description with current matrix info
        shape_str = f"{param.shape[0]}x{param.shape[1]}"
        pbar.set_postfix({'matrix': name.split('.')[-1][:20], 'shape': shape_str})
        
        # Compute full SVD analysis (handles GPU/CPU selection internally)
        svd_result = compute_full_svd_analysis(param, dev, use_gpu_for_large=True, gpu_threshold=5000)
        if svd_result is None:
            continue
        
        # Periodic GPU cache clearing to prevent memory accumulation
        if dev.type == 'cuda' and (idx + 1) % clear_cache_interval == 0:
            torch.cuda.empty_cache()
        
        theoretical_max_rank = svd_result['theoretical_max_rank']
        actual_rank = svd_result['actual_rank']
        singular_values = svd_result['singular_values']
        energy_cumulative = svd_result['energy_cumulative']
        
        weight_ranks.append({
            'parameter_name': name,
            'layer_index': layer_idx if layer_idx is not None else -1,
            'shape': list(param.shape),
            'theoretical_max_rank': theoretical_max_rank,
            'actual_rank': actual_rank,
            'singular_values': singular_values,
            'energy_cumulative': energy_cumulative,
            'total_energy': svd_result['total_energy'],
        })
        
        if layer_idx is not None and (limit_layers == 0 or layer_idx < limit_layers):
            layer_groups[layer_idx].append({
                'name': name,
                'theoretical_max_rank': theoretical_max_rank,
                'actual_rank': actual_rank,
                'shape': list(param.shape),
                'singular_values': singular_values,
                'energy_cumulative': energy_cumulative,
            })
    
    pbar.close()
    
    if not weight_ranks:
        print(f"No 2D weight matrices found in {display_name}")
        return None
    
    # Collect all ranks for statistics
    actual_ranks = [w['actual_rank'] for w in weight_ranks]
    theoretical_max_ranks = [w['theoretical_max_rank'] for w in weight_ranks]
    
    # Per-layer detailed information
    per_layer_details = {}  # Store full details per layer
    for layer_idx in sorted(layer_groups.keys()):
        layer_weights = layer_groups[layer_idx]
        per_layer_details[layer_idx] = layer_weights
    
    # Print summary
    print(f"\nModel: {display_name}")
    print(f"Total 2D weight matrices: {len(weight_ranks)}")
    print(f"\nRank Statistics:")
    print(f"  Actual Rank - Mean: {statistics.mean(actual_ranks):.2f}, "
          f"Min: {min(actual_ranks)}, Max: {max(actual_ranks)}")
    print(f"  Theoretical Max Rank - Mean: {statistics.mean(theoretical_max_ranks):.2f}, "
          f"Min: {min(theoretical_max_ranks)}, Max: {max(theoretical_max_ranks)}")
    print(f"  Rank Utilization: {statistics.mean(actual_ranks) / statistics.mean(theoretical_max_ranks) * 100:.2f}%")
    
    # Print detailed per-layer information
    if per_layer_details:
        print(f"\nDetailed Per-Layer Information:")
        print(f"{'Layer':>6} | {'Parameter':<40} | {'Shape':>15} | {'TheoMax':>8} | {'Actual':>8} | {'Util%':>8}")
        print("-" * 110)
        for layer_idx in sorted(per_layer_details.keys()):
            for weight_info in per_layer_details[layer_idx]:
                param_name = weight_info['name'].split('.')[-1]
                shape_str = f"{weight_info['shape'][0]}x{weight_info['shape'][1]}"
                theo_max = weight_info['theoretical_max_rank']
                actual = weight_info['actual_rank']
                util_pct = (actual / theo_max * 100) if theo_max > 0 else 0.0
                print(f"{layer_idx:6d} | {param_name:<40} | {shape_str:>15} | "
                      f"{theo_max:8d} | {actual:8d} | {util_pct:7.2f}%")
    
    # Clean up model from GPU memory before returning
    print(f"\nCleaning up model from memory...")
    del model
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"GPU memory cleared")
    
    return {
        'model_name': model_name,
        'display_name': display_name,
        'weight_ranks': weight_ranks,
        'per_layer_details': per_layer_details,
        'actual_ranks': actual_ranks,
        'theoretical_max_ranks': theoretical_max_ranks,
    }


def save_to_csv(results: Dict, output_dir: str = '.'):
    """
    Save full SVD and rank distribution to CSV files.
    
    Args:
        results: Analysis results dictionary
        output_dir: Directory to save CSV files
    """
    if results is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from model name
    safe_name = results['display_name'].replace('/', '_').replace('-', '_')
    
    # Save detailed rank distribution (all weight matrices with full SVD info)
    csv_path = os.path.join(output_dir, f"{safe_name}_rank_distribution.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'parameter_name',
            'layer_index',
            'shape_0',
            'shape_1',
            'theoretical_max_rank',
            'actual_rank',
            'rank_utilization_pct',
            'total_energy',
            'singular_values_count',
        ])
        
        for w in results['weight_ranks']:
            sv_count = len(w['singular_values'])
            util_pct = (w['actual_rank'] / w['theoretical_max_rank'] * 100) if w['theoretical_max_rank'] > 0 else 0.0
            writer.writerow([
                w['parameter_name'],
                w['layer_index'],
                w['shape'][0],
                w['shape'][1],
                w['theoretical_max_rank'],
                w['actual_rank'],
                f"{util_pct:.4f}",
                f"{w['total_energy']:.6e}",
                sv_count,
            ])
    
    print(f"\nSaved detailed rank distribution to: {csv_path}")
    
    # Save singular values for each layer (separate file per parameter for detailed analysis)
    svd_dir = os.path.join(output_dir, f"{safe_name}_singular_values")
    os.makedirs(svd_dir, exist_ok=True)
    
    # Also create a consolidated file with all singular values
    consolidated_path = os.path.join(output_dir, f"{safe_name}_all_singular_values.csv")
    
    with open(consolidated_path, 'w', newline='') as f_consolidated:
        writer_consolidated = csv.writer(f_consolidated)
        writer_consolidated.writerow([
            'parameter_name',
            'layer_index',
            'rank_index',
            'singular_value',
            'cumulative_energy',
            'theoretical_max_rank',
        ])
        
        for w in results['weight_ranks']:
            # Create safe filename from parameter name
            param_safe = w['parameter_name'].replace('.', '_').replace('/', '_')
            svd_path = os.path.join(svd_dir, f"{param_safe}_singular_values.csv")
            
            # Get all singular values (SVD returns exactly theoretical_max_rank = min(shape[0], shape[1]) values)
            singular_values = np.array(w['singular_values'])
            energy_cumulative = np.array(w['energy_cumulative'])
            theoretical_max = w['theoretical_max_rank']
            
            # Verify we have the expected number of singular values
            # SVD should return exactly theoretical_max_rank values
            if len(singular_values) != theoretical_max:
                print(f"Warning: {w['parameter_name']} has {len(singular_values)} singular values but theoretical_max_rank={theoretical_max}")
                # Ensure we record exactly theoretical_max_rank values
                if len(singular_values) < theoretical_max:
                    # Pad with zeros (shouldn't happen normally)
                    padded_sv = np.zeros(theoretical_max)
                    padded_sv[:len(singular_values)] = singular_values
                    singular_values = padded_sv
                    
                    padded_energy = np.ones(theoretical_max)
                    if len(energy_cumulative) > 0:
                        padded_energy[:len(energy_cumulative)] = energy_cumulative
                    energy_cumulative = padded_energy
                else:
                    # Truncate (shouldn't happen)
                    singular_values = singular_values[:theoretical_max]
                    energy_cumulative = energy_cumulative[:theoretical_max]
            
            # Save individual file for this parameter - ALL singular values
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
            
            # Add to consolidated file - ALL singular values for all parameters
            for i in range(theoretical_max):
                sv = float(singular_values[i])
                cum_energy = float(energy_cumulative[i])
                writer_consolidated.writerow([
                    w['parameter_name'],
                    w['layer_index'],
                    i,
                    f"{sv:.12e}",
                    f"{cum_energy:.12f}",
                    theoretical_max,
                ])
    
    print(f"Saved singular values to: {svd_dir}/")
    print(f"Saved consolidated singular values to: {consolidated_path}")
    
    # Save per-layer detailed summary
    if results.get('per_layer_details'):
        summary_path = os.path.join(output_dir, f"{safe_name}_layer_details.csv")
        with open(summary_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'layer_index',
                'parameter_name',
                'shape_0',
                'shape_1',
                'theoretical_max_rank',
                'actual_rank',
                'rank_utilization_pct',
            ])
            
            for layer_idx in sorted(results['per_layer_details'].keys()):
                for weight_info in results['per_layer_details'][layer_idx]:
                    util_pct = (weight_info['actual_rank'] / weight_info['theoretical_max_rank'] * 100) if weight_info['theoretical_max_rank'] > 0 else 0.0
                    writer.writerow([
                        layer_idx,
                        weight_info['name'],
                        weight_info['shape'][0],
                        weight_info['shape'][1],
                        weight_info['theoretical_max_rank'],
                        weight_info['actual_rank'],
                        f"{util_pct:.4f}",
                    ])
        
        print(f"Saved per-layer details to: {summary_path}")
    
    return csv_path


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


def plot_rank_distribution(results: Dict, output_dir: str = 'plot'):
    """
    Create visualization showing singular value accumulation for each linear layer.
    Shows how singular values accumulate up to theoretical maximum rank.
    
    Args:
        results: Analysis results dictionary
        output_dir: Directory to save plots
    """
    _, plt = _setup_matplotlib()
    if plt is None:
        return
    
    if results is None:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = results['display_name'].replace('/', '_').replace('-', '_')
    
    if not results.get('weight_ranks'):
        return
    
    # Color scheme: green and orange
    color_green = '#2ecc71'  # Beautiful green
    color_orange = '#e67e22'  # Beautiful orange
    color_green_light = '#a8e6cf'  # Light green
    color_orange_light = '#ffd3a5'  # Light orange
    
    # Plot singular value accumulation
    _plot_singular_value_accumulation(results, output_dir, safe_name,
                                     color_green, color_orange, color_green_light, color_orange_light)
    
    print(f"Saved plot to {output_dir}/")


def _plot_singular_value_accumulation(results: Dict, output_dir: str, safe_name: str,
                                     color_green: str, color_orange: str,
                                     color_green_light: str, color_orange_light: str):
    """
    Plot singular value accumulation for each linear layer.
    Shows how singular values accumulate up to theoretical maximum rank.
    Creates two plots: segmented lines and histogram.
    """
    _, plt = _setup_matplotlib()
    if plt is None:
        return
    
    weight_ranks = results.get('weight_ranks', [])
    if not weight_ranks:
        return
    
    # Find maximum theoretical rank to set x-axis limit
    max_theoretical = max([w['theoretical_max_rank'] for w in weight_ranks]) if weight_ranks else 1000
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), dpi=150)
    
    # Collect data for plotting
    all_cumulative_energies = []
    all_rank_indices = []
    param_names = []
    
    # Plot 1: Segmented lines showing cumulative energy accumulation
    for w in weight_ranks:
        singular_values = np.array(w['singular_values'])
        energy_cumulative = np.array(w['energy_cumulative'])
        theoretical_max = w['theoretical_max_rank']
        param_name = w['parameter_name'].split('.')[-1]  # Get last part of parameter name
        
        # Ensure we have values up to theoretical_max_rank
        if len(energy_cumulative) < theoretical_max:
            padded_energy = np.ones(theoretical_max)
            padded_energy[:len(energy_cumulative)] = energy_cumulative
            energy_cumulative = padded_energy
        
        # Create rank indices (0 to theoretical_max_rank)
        rank_indices = np.arange(theoretical_max)
        
        # Plot segmented line (step plot) for this layer
        ax1.plot(rank_indices, energy_cumulative[:theoretical_max], 
                linewidth=1.5, alpha=0.6, label=param_name if len(param_names) < 20 else None)
        
        # Store for histogram
        all_cumulative_energies.extend(energy_cumulative[:theoretical_max])
        all_rank_indices.extend(rank_indices)
        param_names.append(param_name)
    
    ax1.set_xlabel('Rank Index (0 → Theoretical Maximum Rank)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Cumulative Energy Fraction', fontsize=13, fontweight='bold')
    ax1.set_title('Singular Value Accumulation: Cumulative Energy per Linear Layer', 
                 fontsize=14, fontweight='bold')
    ax1.set_xlim([0, max_theoretical])
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle='--')
    if len(param_names) <= 20:
        ax1.legend(fontsize=8, loc='lower right', ncol=2, framealpha=0.9)
    else:
        ax1.text(0.02, 0.98, f'{len(weight_ranks)} linear layers', 
                transform=ax1.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Histogram showing distribution of cumulative energy at different ranks
    # Collect all cumulative energy values at each rank position
    num_bins = min(100, max_theoretical)
    rank_positions = np.linspace(0, max_theoretical, num_bins)
    
    # For each rank position, collect cumulative energy values from all layers
    energies_by_rank = []
    rank_indices_for_hist = []
    
    for rank_pos in rank_positions:
        rank_idx = int(rank_pos)
        energies_at_rank = []
        
        for w in weight_ranks:
            energy_cumulative = np.array(w['energy_cumulative'])
            theoretical_max = w['theoretical_max_rank']
            
            if rank_idx < theoretical_max:
                if rank_idx < len(energy_cumulative):
                    energies_at_rank.append(energy_cumulative[rank_idx])
                else:
                    # If we have fewer values, use the last one
                    if len(energy_cumulative) > 0:
                        energies_at_rank.append(energy_cumulative[-1])
        
        if energies_at_rank:
            energies_by_rank.append(energies_at_rank)
            rank_indices_for_hist.append(rank_idx)
    
    # Create histogram: for each rank, show distribution of cumulative energies
    if energies_by_rank:
        # Create 2D histogram: rank position vs cumulative energy
        all_ranks_flat = []
        all_energies_flat = []
        for i, energies in enumerate(energies_by_rank):
            all_ranks_flat.extend([rank_indices_for_hist[i]] * len(energies))
            all_energies_flat.extend(energies)
        
        # Create 2D histogram
        hist_2d, x_edges, y_edges = np.histogram2d(
            all_ranks_flat, all_energies_flat,
            bins=[min(50, max_theoretical), 50],
            range=[[0, max_theoretical], [0, 1]]
        )
        
        # Plot as filled contour/heatmap
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])
        im = ax2.contourf(X, Y, hist_2d.T, levels=20, cmap='YlGn', alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Count', fontsize=11)
        
        # Overlay mean line
        mean_energies = [np.mean(energies) if energies else 0 for energies in energies_by_rank]
        if len(mean_energies) == len(rank_indices_for_hist):
            ax2.plot(rank_indices_for_hist, mean_energies, 
                    color=color_orange, linewidth=2, linestyle='--', 
                    label='Mean Cumulative Energy', zorder=10)
            ax2.legend(fontsize=10, loc='lower right')
    
    ax2.set_xlabel('Rank Index (0 → Theoretical Maximum Rank)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Cumulative Energy Fraction', fontsize=13, fontweight='bold')
    ax2.set_title('Distribution of Cumulative Energy Across All Linear Layers', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlim([0, max_theoretical])
    ax2.set_ylim([0, 1.1])
    ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle(f"Singular Value Accumulation Analysis\n{results['display_name']}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{safe_name}_singular_value_accumulation.png")
    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
    plt.close(fig)




def parse_energy_values(energy_str: str) -> List[float]:
    """
    Parse comma-separated energy values from environment variable.
    
    Args:
        energy_str: Comma-separated string of energy values (e.g., "0.9,0.95,0.99")
    
    Returns:
        List of energy values as floats
    """
    if not energy_str:
        return [0.99]
    
    try:
        energies = [float(e.strip()) for e in energy_str.split(',')]
        # Validate energy values
        valid_energies = [e for e in energies if 0.0 < e <= 1.0]
        if not valid_energies:
            print(f"Warning: No valid energy values found. Using default 0.99")
            return [0.99]
        return valid_energies
    except ValueError as e:
        print(f"Warning: Error parsing energy values '{energy_str}': {e}. Using default 0.99")
        return [0.99]


def main():
    """Main function to analyze all models with full SVD computation."""
    torch.set_grad_enabled(False)
    
    # Configuration from environment
    device = os.getenv("DEVICE", "auto").lower()  # Default to auto-detect GPU
    limit_layers = int(os.getenv("LIMIT_LAYERS", "0"))
    plot_dir = os.getenv("PLOT_DIR", "plot")
    csv_dir = os.getenv("CSV_DIR", ".")
    
    # Set HF endpoint for mirror if provided
    hf_endpoint = os.getenv("HF_ENDPOINT")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"Using HF endpoint: {hf_endpoint}")
    
    print(f"\n{'='*80}")
    print(f"Configuration:")
    print(f"  Computing FULL SVD for all linear layers")
    print(f"  Device: {device}")
    print(f"  Limit layers: {limit_layers if limit_layers > 0 else 'all'}")
    print(f"  Plot directory: {plot_dir}")
    print(f"  CSV directory: {csv_dir}")
    print(f"{'='*80}")
    
    # Process all models one at a time to avoid GPU memory issues
    all_results = []
    for display_name, model_name in MODELS:
        try:
            # Analyze model (loads, processes, and cleans up model)
            results = analyze_model_weights(
                model_name,
                display_name,
                limit_layers=limit_layers,
                device=device,
            )
            
            if results:
                # Save CSV files (doesn't need model in memory)
                save_to_csv(results, csv_dir)
                
                # Generate plots (doesn't need model in memory)
                plot_rank_distribution(results, plot_dir)
                
                # Store results (only metadata, no model weights)
                all_results.append(results)
                
                # Clear results from memory (they contain numpy arrays which can be large)
                del results
                if device == 'cuda' or device == 'auto':
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
            
            # Extra cleanup between models
            import gc
            gc.collect()
            if device == 'cuda' or device == 'auto':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError processing {display_name}: {e}")
            import traceback
            traceback.print_exc()
            
            # Clean up on error too
            if device == 'cuda' or device == 'auto':
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            continue
    
    # Summary across all models
    if all_results:
        print(f"\n{'='*80}")
        print("Summary across all models:")
        print(f"{'='*80}")
        print(f"{'Model':<20} | {'Avg Actual':>12} | {'Avg TheoMax':>12} | {'Util%':>8} | {'Count':>6}")
        print("-" * 80)
        for r in all_results:
            actual_ranks = r['actual_ranks']
            theo_max_ranks = r['theoretical_max_ranks']
            avg_actual = statistics.mean(actual_ranks)
            avg_theo = statistics.mean(theo_max_ranks)
            util_pct = (avg_actual / avg_theo * 100) if avg_theo > 0 else 0.0
            print(f"{r['display_name']:<20} | {avg_actual:12.2f} | {avg_theo:12.2f} | {util_pct:7.2f}% | {len(actual_ranks):6d}")


if __name__ == "__main__":
    main()

