#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
activations_by_datasets.py — Analyze activation patterns across all models and datasets

What it does:
- Automatically discovers all models and datasets from saved activations
- Computes SVD and singular values for each activation across all datasets
- Analyzes energy concentration patterns per dataset and across datasets
- Provides detailed statistics and cross-dataset comparisons

Usage:
    python3 motivations/activations_by_datasets.py

Output:
    Prints detailed analysis of activation patterns across all datasets for all models.
"""

import pickle
import csv
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

def load_activations_from_pickle(pickle_path: Path) -> Optional[Dict]:
    """Load activations from pickle file."""
    try:
        with open(pickle_path, 'rb') as f:
            activations = pickle.load(f)
        return activations
    except Exception as e:
        print(f"Error loading {pickle_path}: {e}")
        return None

def compute_svd_for_activation(data_dict: Dict) -> Optional[Dict]:
    """Compute SVD for a single activation matrix."""
    if not isinstance(data_dict, dict) or 'data' not in data_dict:
        return None
    
    data = data_dict['data']
    if not isinstance(data, np.ndarray):
        return None
    
    # Ensure 2D
    if data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])
    
    if data.ndim != 2:
        return None
    
    N, D = data.shape
    if N == 0 or D == 0:
        return None
    
    theoretical_max_rank = min(N, D)
    
    # Compute SVD
    try:
        U, s, Vh = np.linalg.svd(data.astype(np.float32), full_matrices=False)
    except Exception as e:
        print(f"Error computing SVD: {e}")
        return None
    
    # Ensure we have exactly theoretical_max_rank singular values
    if len(s) < theoretical_max_rank:
        # Pad with zeros
        s_padded = np.zeros(theoretical_max_rank)
        s_padded[:len(s)] = s
        s = s_padded
    elif len(s) > theoretical_max_rank:
        # Truncate
        s = s[:theoretical_max_rank]
    
    # Compute cumulative energy
    energy_squared = s ** 2
    total_energy = float(np.sum(energy_squared))
    
    if total_energy > 0:
        cumulative_energy = np.cumsum(energy_squared) / total_energy
    else:
        cumulative_energy = np.ones_like(s)
    
    # Compute actual rank
    threshold = 1e-6 * s.max() if len(s) > 0 and s.max() > 0 else 1e-10
    actual_rank = int(np.sum(s > threshold))
    
    return {
        'singular_values': s,
        'cumulative_energy': cumulative_energy,
        'theoretical_max_rank': theoretical_max_rank,
        'actual_rank': actual_rank,
        'total_energy': total_energy,
        'shape': (N, D),
    }

def analyze_energy_concentration(svd_result: Dict) -> Optional[Dict]:
    """Analyze energy concentration from SVD result."""
    if svd_result is None:
        return None
    
    svs = svd_result['singular_values']
    cum_energy = svd_result['cumulative_energy']
    theo_max = svd_result['theoretical_max_rank']
    
    # Find rank at which each energy threshold is reached
    energy_thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    rank_at_threshold = {}
    
    for threshold in energy_thresholds:
        idx = np.searchsorted(cum_energy, threshold)
        if idx < len(cum_energy):
            rank_at_threshold[threshold] = idx + 1
        else:
            rank_at_threshold[threshold] = len(cum_energy)
    
    # Calculate rank ratios
    rank_ratios = {k: v / theo_max for k, v in rank_at_threshold.items()}
    
    # Calculate effective rank (trace norm / max singular value)
    if len(svs) > 0 and svs[0] > 0:
        trace_norm = np.sum(svs)
        max_sv = svs[0]
        effective_rank_trace = trace_norm / max_sv
    else:
        effective_rank_trace = 0
    
    return {
        'theoretical_max_rank': theo_max,
        'rank_at_50pct': rank_at_threshold[0.5],
        'rank_at_80pct': rank_at_threshold[0.8],
        'rank_at_90pct': rank_at_threshold[0.9],
        'rank_at_95pct': rank_at_threshold[0.95],
        'rank_at_99pct': rank_at_threshold[0.99],
        'rank_ratio_50pct': rank_ratios[0.5],
        'rank_ratio_80pct': rank_ratios[0.8],
        'rank_ratio_90pct': rank_ratios[0.9],
        'rank_ratio_95pct': rank_ratios[0.95],
        'rank_ratio_99pct': rank_ratios[0.99],
        'effective_rank_trace': effective_rank_trace,
        'effective_rank_ratio': effective_rank_trace / theo_max if theo_max > 0 else 0,
        'max_singular_value': float(svs[0]) if len(svs) > 0 else 0.0,
        'min_singular_value': float(svs[-1]) if len(svs) > 0 else 0.0,
        'sv_decay_ratio': float(svs[-1] / svs[0]) if len(svs) > 0 and svs[0] > 0 else 0.0,
    }

def discover_models_and_datasets(base_dir: str = 'checkpoints/motivation_activations') -> Tuple[List[str], Dict[str, List[str]]]:
    """Discover all available models and their datasets."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return [], {}
    
    models = []
    model_datasets = {}
    
    for model_dir in sorted(base_path.iterdir()):
        if not model_dir.is_dir():
            continue
        
        model_name = model_dir.name
        datasets = []
        
        for dataset_dir in sorted(model_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            
            activation_file = dataset_dir / 'activations.pkl'
            if activation_file.exists():
                datasets.append(dataset_dir.name)
        
        if datasets:
            models.append(model_name)
            model_datasets[model_name] = sorted(datasets)
    
    return models, model_datasets

def analyze_dataset(model_name: str, dataset_name: str, base_dir: str = 'checkpoints/motivation_activations') -> Optional[List[Dict]]:
    """Analyze activations for a specific model and dataset."""
    pickle_path = Path(base_dir) / model_name / dataset_name / 'activations.pkl'
    
    if not pickle_path.exists():
        return None
    
    activations = load_activations_from_pickle(pickle_path)
    if activations is None:
        return None
    
    results = []
    
    for layer_idx in sorted(activations.keys()):
        layer_data = activations[layer_idx]
        
        for act_type in sorted(layer_data.keys()):
            data_dict = layer_data[act_type]
            
            # Compute SVD
            svd_result = compute_svd_for_activation(data_dict)
            if svd_result is None:
                continue
            
            # Analyze energy concentration
            analysis = analyze_energy_concentration(svd_result)
            if analysis is None:
                continue
            
            results.append({
                'layer': layer_idx,
                'activation_type': act_type,
                'dataset': dataset_name,
                'singular_values': svd_result['singular_values'],
                'cumulative_energy': svd_result['cumulative_energy'],
                **analysis
            })
    
    return results

def save_singular_values_by_dataset(
    all_results: Dict[str, Dict[str, List[Dict]]],
    output_dir: str = 'motivations/activations_by_datasets'
):
    """Save singular values for each activation across all datasets."""
    os.makedirs(output_dir, exist_ok=True)
    
    for model_name, model_results in all_results.items():
        # Create directory for this model
        model_output_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Collect all unique (layer, activation_type) pairs
        activation_keys = set()
        for dataset_results in model_results.values():
            for result in dataset_results:
                activation_keys.add((result['layer'], result['activation_type']))
        
        # Save singular values for each activation across all datasets
        for layer_idx, act_type in sorted(activation_keys):
            # Create safe filename
            act_safe = act_type.replace('.', '_').replace('/', '_').replace('-', '_')
            csv_path = os.path.join(model_output_dir, f"layer_{layer_idx}_{act_safe}_by_dataset.csv")
            
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'dataset',
                    'rank_index',
                    'singular_value',
                    'cumulative_energy',
                    'theoretical_max_rank',
                    'rank_at_90pct',
                    'rank_at_99pct',
                    'effective_rank_ratio',
                ])
                
                # Write data for each dataset
                for dataset_name in sorted(model_results.keys()):
                    dataset_results = model_results[dataset_name]
                    
                    # Find matching activation
                    matching_result = None
                    for result in dataset_results:
                        if result['layer'] == layer_idx and result['activation_type'] == act_type:
                            matching_result = result
                            break
                    
                    if matching_result is None:
                        continue
                    
                    svs = matching_result['singular_values']
                    cum_energy = matching_result['cumulative_energy']
                    theo_max = matching_result['theoretical_max_rank']
                    rank_90 = matching_result['rank_at_90pct']
                    rank_99 = matching_result['rank_at_99pct']
                    eff_rank_ratio = matching_result['effective_rank_ratio']
                    
                    # Write all singular values
                    for rank_idx in range(theo_max):
                        sv = float(svs[rank_idx])
                        cum_e = float(cum_energy[rank_idx])
                        writer.writerow([
                            dataset_name,
                            rank_idx,
                            f"{sv:.12e}",
                            f"{cum_e:.12f}",
                            theo_max,
                            rank_90,
                            rank_99,
                            f"{eff_rank_ratio:.6f}",
                        ])
            
            print(f"  Saved: {csv_path}")

def print_dataset_summary(model_name: str, dataset_name: str, results: List[Dict]):
    """Print summary for a dataset."""
    if not results:
        print(f"    {dataset_name}: No results")
        return
    
    # Group by activation type
    by_act_type = defaultdict(list)
    for r in results:
        by_act_type[r['activation_type']].append(r)
    
    # Calculate averages
    avg_rank_90 = np.mean([r['rank_ratio_90pct'] for r in results])
    avg_rank_99 = np.mean([r['rank_ratio_99pct'] for r in results])
    avg_eff_rank_ratio = np.mean([r['effective_rank_ratio'] for r in results])
    avg_theo_max = np.mean([r['theoretical_max_rank'] for r in results])
    
    print(f"    {dataset_name}:")
    print(f"      Count: {len(results)} activations")
    print(f"      Rank for 90% energy: {avg_rank_90:.2%} ({avg_rank_90 * avg_theo_max:.1f} / {avg_theo_max:.1f})")
    print(f"      Rank for 99% energy: {avg_rank_99:.2%} ({avg_rank_99 * avg_theo_max:.1f} / {avg_theo_max:.1f})")
    print(f"      Effective rank ratio: {avg_eff_rank_ratio:.2%}")

def print_model_summary(model_name: str, model_results: Dict[str, List[Dict]]):
    """Print summary for a model across all datasets."""
    print(f"\n{'='*80}")
    print(f"{model_name}")
    print(f"{'='*80}")
    
    if not model_results:
        print("  No datasets found")
        return
    
    # Aggregate across all datasets
    all_model_results = []
    for dataset_results in model_results.values():
        all_model_results.extend(dataset_results)
    
    if not all_model_results:
        print("  No results")
        return
    
    avg_rank_90 = np.mean([r['rank_ratio_90pct'] for r in all_model_results])
    avg_rank_99 = np.mean([r['rank_ratio_99pct'] for r in all_model_results])
    avg_eff_rank_ratio = np.mean([r['effective_rank_ratio'] for r in all_model_results])
    avg_theo_max = np.mean([r['theoretical_max_rank'] for r in all_model_results])
    
    print(f"  Overall (all datasets):")
    print(f"    Total activations: {len(all_model_results)}")
    print(f"    Rank for 90% energy: {avg_rank_90:.2%} ({avg_rank_90 * avg_theo_max:.1f} / {avg_theo_max:.1f})")
    print(f"    Rank for 99% energy: {avg_rank_99:.2%} ({avg_rank_99 * avg_theo_max:.1f} / {avg_theo_max:.1f})")
    print(f"    Effective rank ratio: {avg_eff_rank_ratio:.2%}")
    
    # Per dataset
    print(f"\n  Per dataset:")
    for dataset_name in sorted(model_results.keys()):
        print_dataset_summary(model_name, dataset_name, model_results[dataset_name])
    
    # Consistency check
    if len(model_results) > 1:
        dataset_rank_90s = [np.mean([r['rank_ratio_90pct'] for r in model_results[d]]) 
                           for d in sorted(model_results.keys())]
        std_rank_90 = np.std(dataset_rank_90s)
        
        print(f"\n  Consistency across datasets:")
        print(f"    Std dev of 90% energy rank: {std_rank_90:.4f}")
        if std_rank_90 < 0.05:
            print(f"    ✓ HIGHLY CONSISTENT")
        elif std_rank_90 < 0.10:
            print(f"    ✓ MODERATELY CONSISTENT")
        else:
            print(f"    ⚠ VARIABLE")

def main():
    base_dir = 'checkpoints/motivation_activations'
    
    # Discover all models and datasets
    print(f"\n{'='*80}")
    print(f"Activation Pattern Analysis by Dataset")
    print(f"{'='*80}")
    
    models, model_datasets = discover_models_and_datasets(base_dir)
    
    if not models:
        print("No models found in checkpoints/motivation_activations")
        return
    
    print(f"Found {len(models)} model(s): {', '.join(models)}")
    
    all_results = {}
    
    # Analyze each model
    for model_name in models:
        datasets = model_datasets.get(model_name, [])
        if not datasets:
            print(f"\n{model_name}: No datasets found")
            continue
        
        print(f"\n{model_name}: Analyzing {len(datasets)} dataset(s)...")
        
        model_results = {}
        
        for dataset_name in datasets:
            results = analyze_dataset(model_name, dataset_name, base_dir)
            if results:
                model_results[dataset_name] = results
        
        if model_results:
            all_results[model_name] = model_results
            print_model_summary(model_name, model_results)
    
    # Save singular values by dataset
    if all_results:
        print(f"\n{'='*80}")
        print(f"Saving singular values by dataset...")
        print(f"{'='*80}")
        save_singular_values_by_dataset(all_results)
        print(f"\nSaved singular values to: motivations/activations_by_datasets/")
    
    # Cross-model comparison
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"Cross-Model Comparison")
        print(f"{'='*80}")
        
        for model_name in sorted(all_results.keys()):
            model_results = all_results[model_name]
            all_model_results = []
            for dataset_results in model_results.values():
                all_model_results.extend(dataset_results)
            
            if all_model_results:
                avg_rank_90 = np.mean([r['rank_ratio_90pct'] for r in all_model_results])
                avg_eff_rank_ratio = np.mean([r['effective_rank_ratio'] for r in all_model_results])
                print(f"{model_name}: 90% energy at {avg_rank_90:.2%}, eff rank {avg_eff_rank_ratio:.2%}")

if __name__ == '__main__':
    main()

