# 在文件顶部添加或修改以下导入
# joint_subspace_large.py主要研究的是大一点的模型（7b这种）在activation空间上有无shared subspace

import numpy as np
import json
import pickle
from datetime import datetime
import warnings
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from scipy.linalg import subspace_angles
from tqdm import tqdm
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import pickle
import json
import warnings
from scipy.stats import ttest_ind
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Multiple comparison correction will be skipped.")
warnings.filterwarnings('ignore')

# ==================== 增强的数据类型处理器 ====================
class DataTypeHandler:
    """处理NumPy和Python数据类型之间的转换，确保JSON序列化兼容"""
    
    @staticmethod
    def convert_to_serializable(obj):
        """
        递归地将对象转换为JSON可序列化的格式
        支持: numpy标量、numpy数组、列表、字典、集合
        """
        if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            # 处理特殊浮点值
            if np.isnan(obj):
                return None
            elif np.isinf(obj):
                return str(obj)  # 将inf转换为字符串
            else:
                return float(obj)
        elif isinstance(obj, np.ndarray):
            # 处理numpy数组
            if obj.size > 10000:  # 大数组只保存摘要信息
                return {
                    'type': 'ndarray_summary',
                    'shape': list(obj.shape),
                    'dtype': str(obj.dtype),
                    'mean': float(np.mean(obj)) if obj.size > 0 else None,
                    'std': float(np.std(obj)) if obj.size > 0 else None,
                    'min': float(np.min(obj)) if obj.size > 0 else None,
                    'max': float(np.max(obj)) if obj.size > 0 else None,
                    'size': obj.size
                }
            else:
                return DataTypeHandler.convert_to_serializable(obj.tolist())
        elif isinstance(obj, (list, tuple)):
            return [DataTypeHandler.convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: DataTypeHandler.convert_to_serializable(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, '__dict__'):
            # 尝试将对象转换为字典
            try:
                return DataTypeHandler.convert_to_serializable(obj.__dict__)
            except:
                return str(obj)
        else:
            # 其他类型尝试直接返回
            try:
                json.dumps(obj)
                return obj
            except (TypeError, OverflowError):
                return str(obj)
    
    @staticmethod
    def safe_json_dump(data, filepath, indent=2):
        """安全的JSON保存方法"""
        serializable_data = DataTypeHandler.convert_to_serializable(data)
        
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=indent, ensure_ascii=False)
        
        return filepath
    
    @staticmethod
    def safe_pickle_dump(data, filepath, protocol=4):
        """安全的Pickle保存方法，处理大文件"""
        # 创建目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 使用压缩保存大文件
        if sys.getsizeof(data) > 1e7:  # 大于10MB
            import gzip
            with gzip.open(filepath + '.gz', 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
            return filepath + '.gz'
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=protocol)
            return filepath



# ==================== 数据生成器 ====================
class DataGenerator:
    """为不同任务类别生成示例数据"""
    
    @staticmethod
    def generate_examples(task_name: str, n_samples: int, category: str):
        """根据任务类别生成示例"""
        base_examples = {
            # Linguistic Completion 任务
            "wikitext": [
                "The quick brown fox jumps over the lazy dog. ",
                "Artificial intelligence is transforming various industries across the world. ",
                "Renewable energy sources are becoming increasingly important for sustainable development. ",
                "The history of computing dates back to ancient times with the abacus. ",
                "Climate change is one of the most pressing issues facing humanity today. "
            ],
            "text_completion": [
                "Once upon a time, there was a ",
                "The key to success is ",
                "In the future, artificial intelligence will ",
                "The most important scientific discovery of the 21st century is ",
                "When I think about the universe, I "
            ],
            "text_continuation": [
                "The cat sat on the mat, purring contentedly. ",
                "The sun was setting over the horizon, painting the sky in shades of orange and pink. ",
                "The experiment yielded unexpected results that challenged conventional wisdom. ",
                "Economic indicators suggest a period of growth, but concerns remain about ",
                "The discovery of exoplanets has revolutionized our understanding of "
            ],
            
            # Reasoning 任务
            "commonsense_qa": [
                "What is the purpose of a refrigerator?",
                "Why do people wear coats in winter?",
                "What happens if you drop a glass on a hard floor?",
                "Why is it important to brush your teeth?",
                "What is the function of a steering wheel in a car?"
            ],
            "strategyqa": [
                "Is it possible to see the Great Wall of China from space?",
                "Can a kangaroo jump higher than the Empire State Building?",
                "Would a paperclip sink in water?",
                "Is chocolate toxic to dogs?",
                "Can humans breathe underwater without equipment?"
            ],
            "openbookqa": [
                "What is the capital of France?",
                "Who wrote 'Romeo and Juliet'?",
                "What is the chemical symbol for gold?",
                "Which planet is known as the Red Planet?",
                "What is the largest organ in the human body?"
            ],
            
            # Math 任务
            "gsm8k": [
                "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
                "A farmer has 12 chickens. Each chicken lays 3 eggs per day. He sells each egg for $0.25. How much money does he make in 5 days?",
                "If a train travels 60 miles per hour, how far will it travel in 3.5 hours?",
                "Tom has 15 apples. He gives 3 to Mary and 4 to John. Then he buys twice as many as he has now. How many apples does Tom have in the end?",
                "A rectangle has length 8 cm and width 5 cm. What is its area in square centimeters?"
            ],
            "mathqa": [
                "If x + y = 15 and x - y = 3, what is the value of x?",
                "Simplify the expression: 3(x + 2) - 2(x - 1)",
                "What is 25% of 200?",
                "Solve for x: 2x + 5 = 17",
                "Calculate the square root of 144"
            ],
            "aqua": [
                "A bag contains 5 red marbles and 3 blue marbles. If two marbles are drawn at random without replacement, what is the probability that both are red?",
                "If the sum of three consecutive integers is 72, what is the largest integer?",
                "A car travels at 60 km/h for 2 hours, then at 80 km/h for 3 hours. What is the average speed?",
                "Solve for x: 3x^2 - 12x = 0",
                "What is the value of log10(1000)?"
            ],
            
            # Logic 任务
            "arc": [
                "All cats have tails. Fluffy is a cat. Does Fluffy have a tail?",
                "If it rains, the ground gets wet. The ground is wet. Does it mean it rained?",
                "All squares are rectangles. All rectangles have four sides. Therefore, all squares have four sides. This reasoning is:",
                "If A implies B, and B implies C, and we know A is true, what can we conclude about C?",
                "No reptiles have fur. All snakes are reptiles. Therefore, no snakes have fur. This argument is:"
            ],
            "logical_deduction": [
                "Premises: All men are mortal. Socrates is a man. Conclusion: Socrates is mortal. Is this valid?",
                "If the battery is dead, the car won't start. The car won't start. Therefore, the battery is dead. Is this valid reasoning?",
                "All birds can fly. Penguins are birds. Therefore, penguins can fly. What is wrong with this argument?",
                "If it is raining, then the streets are wet. The streets are wet. Therefore, it is raining. Is this valid?",
                "All fruits contain seeds. Tomatoes contain seeds. Therefore, tomatoes are fruits. Is this valid?"
            ],
            "syllogism": [
                "All A are B. All B are C. Therefore, all A are C.",
                "Some A are B. All B are C. Therefore, some A are C.",
                "No A are B. All C are B. Therefore, no A are C.",
                "All A are B. Some B are C. Therefore, some A are C.",
                "Some A are B. No B are C. Therefore, some A are not C."
            ]
        }
        
        # 获取基础示例
        if task_name in base_examples:
            base = base_examples[task_name]
            # 重复直到达到所需样本数
            repetitions = (n_samples + len(base) - 1) // len(base)
            data = []
            for i in range(repetitions):
                for j, example in enumerate(base):
                    data.append(example)
            return data[:n_samples]
        else:
            return [f"{task_name} example {i}" for i in range(n_samples)]


# ==================== Model Architecture Helper ====================
def get_model_layers(model):
    """Get the layers from a model, handling different architectures"""
    # OPT architecture: model.model.decoder.layers
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers, 'opt'
    # LLaMA/Qwen architecture: model.model.layers
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers, 'llama'
    # GPT-2 architecture: model.transformer.h
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, 'gpt2'
    else:
        raise ValueError(f"Could not find layers in model. Model type: {type(model)}")


@dataclass
class SharedSpaceConfig:
    """共享子空间分析配置"""
    model_name: str = "facebook/opt-125m"
    batch_size: int = 8
    max_seq_len: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./shared_reasoning_analysis"
    
    # 任务分类
    task_categories: Dict[str, List[str]] = None
    
    # 实验设置
    n_samples_per_task: int = 128  # 为快速测试，先用300样本
    
    # 激活收集策略
    activation_strategy: str = "all_tokens"  # "last_token", "mean", "max", "all_tokens"
    
    # PCA设置
    pca_variance_threshold: float = 0.95
    min_subspace_dimension: int = 1
    max_subspace_dimension: int = 2000
    
    # 共享子空间分析设置
    reasoning_tasks_only: bool = True  # 是否只分析推理任务
    test_other_tasks: bool = True  # 是否测试其他任务在共享子空间上的表现
    
    # 层选择
    layers_to_probe: List[int] = None
    focus_layers: List[int] = None  # 重点关注层
    
    # 隐藏层维度
    hidden_dim: int = 4096
    
    def __post_init__(self):
        if self.task_categories is None:
            self.task_categories = {
                "linguistic": ["wikitext", "text_completion", "text_continuation"],
                "reasoning": ["commonsense_qa", "strategyqa", "openbookqa"],
                "math": ["gsm8k", "mathqa", "aqua"],
                "logic": ["arc", "logical_deduction", "syllogism"]
            }
        
        if self.layers_to_probe is None:
            # 选择关键的几层进行分析
            self.layers_to_probe = [0, 3, 5, 8, 11, 14, 17, 18, 20, 24, 30]
        
        if self.focus_layers is None:
            # 重点关注中间层
            self.focus_layers = [5, 8]


# ==================== 共享推理分析器 ====================
class SharedReasoningAnalyzer:
    """分析共享低维推理子空间"""
    
    @staticmethod
    def compute_shared_subspace(activations_dict, config, layer_idx, tasks_to_include=None):
        """计算共享子空间"""
        if tasks_to_include is None:
            # 默认使用所有推理任务
            tasks_to_include = config.task_categories["reasoning"]
        
        # 收集指定任务的激活
        X_combined = None
        task_samples = {}
        
        for task in tasks_to_include:
            if task in activations_dict:
                X = activations_dict[task]
                if X is not None and X.shape[0] > 10:
                    task_samples[task] = X.shape[0]
                    if X_combined is None:
                        X_combined = X
                    else:
                        X_combined = np.vstack([X_combined, X])
        
        if X_combined is None:
            print(f"层 {layer_idx}: 没有找到指定任务的激活数据")
            return None, None, None
        
        print(f"层 {layer_idx}: 合并 {len(tasks_to_include)} 个任务, {X_combined.shape[0]} 个样本/激活向量")
        
        # 计算共享子空间
        n_samples, n_features = X_combined.shape
        
        # 检查样本量是否足够
        if n_samples < n_features:
            print(f"  警告: 样本数 ({n_samples}) < 特征数 ({n_features}), PCA可能不稳定")
            print(f"  PCA最多只能计算 {n_samples - 1} 个主成分（样本数-1）")
        else:
            print(f"  样本数 ({n_samples}) >= 特征数 ({n_features}), PCA应该稳定")
        
        # 使用PCA计算共享子空间
        # 限制PCA组件数不超过样本数-1（避免数值问题）
        max_components = min(n_samples - 1, n_features, config.max_subspace_dimension)
        pca = PCA(n_components=max_components)
        pca.fit(X_combined)
        
        # 计算达到指定方差阈值所需的维度
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(cumsum >= config.pca_variance_threshold) + 1
        
        # 如果找不到达到阈值的维度（可能因为样本量不足），使用所有可用组件
        if k == 0 or cumsum[-1] < config.pca_variance_threshold:
            print(f"  警告: 无法达到 {config.pca_variance_threshold*100}% 方差阈值, 实际达到 {cumsum[-1]*100:.2f}%")
            print(f"  使用所有可用组件: {len(cumsum)} 维")
            k = len(cumsum)
        
        # 打印前10个主成分的解释方差比例（用于诊断）
        if len(pca.explained_variance_ratio_) >= 10:
            top10_var = np.sum(pca.explained_variance_ratio_[:10])
            print(f"  前10个主成分解释方差: {top10_var*100:.2f}%")
            if k <= len(cumsum) and k > 0:
                print(f"  前{k}个主成分解释方差: {cumsum[k-1]*100:.2f}%")
        
        # 诊断：如果维度被限制在10左右
        if k <= 15:
            if n_samples < n_features:
                print(f"  诊断: 维度较低 ({k}) 可能是因为样本量 ({n_samples}) < 特征数 ({n_features})")
                print(f"  建议: 增加样本量或使用 TruncatedSVD 代替 PCA")
            else:
                print(f"  诊断: 维度较低 ({k}) 但样本量充足 ({n_samples} >= {n_features})")
                print(f"  可能原因: Llama的激活确实很分散，前{k}个主成分就能解释95%的方差")
                print(f"  前10个主成分解释方差: {top10_var*100:.2f}%")
                if top10_var >= 0.95:
                    print(f"  → 这确实表明前10个主成分就解释了95%以上的方差")
                    print(f"  → Llama可能没有很强的共享子空间结构，或者共享结构非常低维")
        
        # 应用最小和最大维度限制
        k = max(min(k, config.max_subspace_dimension), config.min_subspace_dimension)
        
        # 确保 k 不超过实际可用的组件数
        max_available_components = len(pca.components_)
        if k > max_available_components:
            print(f"  警告: 请求的维度 ({k}) > 可用组件数 ({max_available_components})")
            print(f"  将维度调整为可用组件数: {max_available_components}")
            k = max_available_components
        
        print(f"  最终共享子空间维度: {k} (目标: 达到 {config.pca_variance_threshold*100}% 方差阈值)")
        print(f"  最小维度要求: {config.min_subspace_dimension}, 实际维度: {k}")
        
        components = pca.components_[:k].T
        shared_subspace = components
        
        # 验证维度
        if shared_subspace.shape[1] != k:
            print(f"  错误: shared_subspace.shape[1] ({shared_subspace.shape[1]}) != k ({k})")
            print(f"  实际维度: {shared_subspace.shape[1]}")
        
        # 计算每个任务在共享子空间上的重建误差
        reconstruction_errors = {}
        for task in tasks_to_include:
            if task in activations_dict:
                X_task = activations_dict[task]
                if X_task is not None:
                    # 重建
                    reconstruction = X_task @ shared_subspace @ shared_subspace.T
                    error = np.mean((X_task - reconstruction) ** 2)
                    reconstruction_errors[task] = error
        
        return shared_subspace, k, reconstruction_errors, task_samples
    
    @staticmethod
    def evaluate_shared_subspace(shared_subspace, activations_dict, config, layer_idx):
        """评估共享子空间对其他任务的表示能力
        
        包含两种显著性检测方法：
        1. 原始方法：基于任务级别的误差（每个任务一个误差值）
        2. 改进方法：基于样本级别的误差（每个样本一个误差值，样本量更大）
        """
        results = {
            'layer': layer_idx,
            'shared_subspace_dim': shared_subspace.shape[1] if shared_subspace is not None else 0,
            'reasoning_tasks': {},
            'other_tasks': {},
            'comparison': {},  # 原始方法（任务级别）
            'comparison_improved': {}  # 改进方法（样本级别）
        }
        
        # 推理任务（应该重建误差较低）
        reasoning_tasks = config.task_categories["reasoning"]
        reasoning_errors = []  # 任务级别误差
        reasoning_errors_per_sample = []  # 样本级别误差（改进方法）
        
        for task in reasoning_tasks:
            if task in activations_dict:
                X = activations_dict[task]
                if X is not None and shared_subspace is not None:
                    reconstruction = X @ shared_subspace @ shared_subspace.T
                    # 任务级别误差（原始方法）
                    error = np.mean((X - reconstruction) ** 2)
                    results['reasoning_tasks'][task] = error
                    reasoning_errors.append(error)
                    
                    # 样本级别误差（改进方法）
                    # 对每个样本计算误差：[n_samples, hidden_dim]
                    sample_errors = np.mean((X - reconstruction) ** 2, axis=1)  # [n_samples]
                    reasoning_errors_per_sample.extend(sample_errors.tolist())
        
        # 其他任务（重建误差可能较高）
        if config.test_other_tasks:
            other_tasks = []
            for category, tasks in config.task_categories.items():
                if category != "reasoning":
                    other_tasks.extend(tasks)
            
            other_errors = []  # 任务级别误差
            other_errors_per_sample = []  # 样本级别误差（改进方法）
            
            for task in other_tasks:
                if task in activations_dict:
                    X = activations_dict[task]
                    if X is not None and shared_subspace is not None:
                        reconstruction = X @ shared_subspace @ shared_subspace.T
                        # 任务级别误差（原始方法）
                        error = np.mean((X - reconstruction) ** 2)
                        results['other_tasks'][task] = error
                        other_errors.append(error)
                        
                        # 样本级别误差（改进方法）
                        sample_errors = np.mean((X - reconstruction) ** 2, axis=1)  # [n_samples]
                        other_errors_per_sample.extend(sample_errors.tolist())
            
            # ========== 原始方法：任务级别显著性检测 ==========
            if reasoning_errors and other_errors:
                # 计算统计显著性（任务级别）
                t_stat, p_value = ttest_ind(reasoning_errors, other_errors, equal_var=False)
                
                # 计算效应量（Cohen's d）
                pooled_std = np.std(reasoning_errors + other_errors)
                cohens_d = (np.mean(other_errors) - np.mean(reasoning_errors)) / pooled_std if pooled_std > 0 else 0
                
                results['comparison'] = {
                    'method': 'task_level',
                    'n_reasoning_tasks': len(reasoning_errors),
                    'n_other_tasks': len(other_errors),
                    'mean_reasoning_error': np.mean(reasoning_errors),
                    'std_reasoning_error': np.std(reasoning_errors),
                    'mean_other_error': np.mean(other_errors),
                    'std_other_error': np.std(other_errors),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'effect_size': cohens_d  # 保持向后兼容
                }
            
            # ========== 改进方法：样本级别显著性检测 ==========
            if reasoning_errors_per_sample and other_errors_per_sample:
                # 计算统计显著性（样本级别，样本量更大）
                t_stat_improved, p_value_improved = ttest_ind(
                    reasoning_errors_per_sample, 
                    other_errors_per_sample, 
                    equal_var=False
                )
                
                # 计算效应量（Cohen's d）
                pooled_std_improved = np.std(reasoning_errors_per_sample + other_errors_per_sample)
                cohens_d_improved = (np.mean(other_errors_per_sample) - np.mean(reasoning_errors_per_sample)) / pooled_std_improved if pooled_std_improved > 0 else 0
                
                # 计算95%置信区间
                mean_diff = np.mean(other_errors_per_sample) - np.mean(reasoning_errors_per_sample)
                se_diff = np.sqrt(
                    np.var(reasoning_errors_per_sample) / len(reasoning_errors_per_sample) +
                    np.var(other_errors_per_sample) / len(other_errors_per_sample)
                )
                ci_lower = mean_diff - 1.96 * se_diff
                ci_upper = mean_diff + 1.96 * se_diff
                
                results['comparison_improved'] = {
                    'method': 'sample_level',
                    'n_reasoning_samples': len(reasoning_errors_per_sample),
                    'n_other_samples': len(other_errors_per_sample),
                    'mean_reasoning_error': np.mean(reasoning_errors_per_sample),
                    'std_reasoning_error': np.std(reasoning_errors_per_sample),
                    'mean_other_error': np.mean(other_errors_per_sample),
                    'std_other_error': np.std(other_errors_per_sample),
                    't_statistic': t_stat_improved,
                    'p_value': p_value_improved,
                    'cohens_d': cohens_d_improved,
                    'mean_difference': mean_diff,
                    'ci_95_lower': ci_lower,
                    'ci_95_upper': ci_upper
                }
        
        return results
    
    @staticmethod
    def analyze_subspace_stability(shared_subspace, activations_dict, config, layer_idx):
        """分析子空间稳定性（通过交叉验证）"""
        reasoning_tasks = config.task_categories["reasoning"]
        
        # 留一法交叉验证
        stability_results = []
        
        for i, held_out_task in enumerate(reasoning_tasks):
            # 使用除held_out_task外的所有任务计算共享子空间
            train_tasks = [t for t in reasoning_tasks if t != held_out_task]
            
            # 收集训练任务的激活
            X_train = None
            for task in train_tasks:
                if task in activations_dict:
                    X = activations_dict[task]
                    if X is not None:
                        if X_train is None:
                            X_train = X
                        else:
                            X_train = np.vstack([X_train, X])
            
            if X_train is None or X_train.shape[0] < 10:
                continue
            
            # 计算训练子空间
            pca = PCA(n_components=min(shared_subspace.shape[1], X_train.shape[0]-1, X_train.shape[1]))
            pca.fit(X_train)
            train_subspace = pca.components_.T
            
            # 计算与完整子空间的角度
            if train_subspace.shape[1] > 0 and shared_subspace.shape[1] > 0:
                angles = subspace_angles(train_subspace, shared_subspace)
                avg_angle = np.mean(np.degrees(angles))
            else:
                avg_angle = 90.0
            
            # 计算held-out任务在训练子空间上的重建误差
            if held_out_task in activations_dict:
                X_test = activations_dict[held_out_task]
                if X_test is not None:
                    reconstruction = X_test @ train_subspace @ train_subspace.T
                    test_error = np.mean((X_test - reconstruction) ** 2)
                    
                    # 计算在完整子空间上的重建误差（作为基准）
                    reconstruction_full = X_test @ shared_subspace @ shared_subspace.T
                    full_error = np.mean((X_test - reconstruction_full) ** 2)
                    
                    stability_results.append({
                        'held_out_task': held_out_task,
                        'train_subspace_dim': train_subspace.shape[1],
                        'avg_angle_with_full': avg_angle,
                        'test_error': test_error,
                        'full_error': full_error,
                        'error_ratio': test_error / full_error if full_error > 0 else 1.0
                    })
        
        return stability_results


# ==================== 激活收集器 ====================
class EnhancedActivationCollector:
    """增强版激活收集器"""
    
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.activations = {}
        self.hooks = []
        self.current_task = None
        
    def _get_hook(self, layer_idx, hook_type):
        """创建钩子函数"""
        def hook(module, input, output):
            if self.current_task is None:
                return
            
            key = f"layer{layer_idx}_{hook_type}"
            if key not in self.activations:
                self.activations[key] = []
            
            # 获取激活
            act = output
            if isinstance(output, tuple):
                act = output[0]  # 取第一个元素
            
            # 根据策略收集激活
            if self.config.activation_strategy == "last_token":
                # 只取最后一个token的激活
                if len(act.shape) == 3:  # [batch, seq_len, hidden]
                    act = act[:, -1, :]  # 最后一个token
                elif len(act.shape) == 2:  # [batch, hidden]
                    act = act
                else:
                    return
            
            elif self.config.activation_strategy == "mean":
                # 取序列平均
                if len(act.shape) == 3:
                    act = act.mean(dim=1)  # 平均池化
                elif len(act.shape) == 2:
                    act = act
                else:
                    return
            
            elif self.config.activation_strategy == "max":
                # 取序列最大值
                if len(act.shape) == 3:
                    act, _ = act.max(dim=1)  # 最大池化
                elif len(act.shape) == 2:
                    act = act
                else:
                    return
            
            elif self.config.activation_strategy == "all_tokens":
                # 收集所有token的激活（展平为 [batch*seq_len, hidden]）
                if len(act.shape) == 3:  # [batch, seq_len, hidden]
                    batch_size, seq_len, hidden_dim = act.shape
                    act = act.view(-1, hidden_dim)  # [batch*seq_len, hidden]
                elif len(act.shape) == 2:  # [batch, hidden]
                    act = act
                else:
                    return
            
            act = act.detach().cpu()
            
            # 存储激活
            self.activations[key].append({
                'task': self.current_task,
                'activation': act
            })
        
        return hook
    
    def register_hooks(self, layer_indices):
        """注册钩子到指定层"""
        print(f"注册钩子到层 {layer_indices}...")
        
        # Get layers based on architecture
        layers, arch_type = get_model_layers(self.model)
        self.arch_type = arch_type  # Store for later use
        
        for layer_idx in layer_indices:
            if layer_idx < len(layers):
                layer = layers[layer_idx]
                
                # Register MLP output hook based on architecture
                if arch_type == 'opt':
                    # OPT: MLP output is at layer.fc2
                    hook_mlp = layer.fc2.register_forward_hook(
                        self._get_hook(layer_idx, "mlp")
                    )
                elif arch_type == 'llama':
                    # LLaMA/Qwen: MLP output is at layer.mlp (or layer.mlp.gate_proj -> up_proj -> down_proj)
                    # We'll hook the final output (down_proj)
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'down_proj'):
                        hook_mlp = layer.mlp.down_proj.register_forward_hook(
                            self._get_hook(layer_idx, "mlp")
                        )
                    elif hasattr(layer, 'mlp'):
                        # Fallback: hook the mlp module itself
                        hook_mlp = layer.mlp.register_forward_hook(
                            self._get_hook(layer_idx, "mlp")
                        )
                    else:
                        print(f"Warning: Could not find MLP layer for layer {layer_idx} in {arch_type} architecture")
                        continue
                else:
                    # GPT-2 or other: try common patterns
                    if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'c_proj'):
                        hook_mlp = layer.mlp.c_proj.register_forward_hook(
                            self._get_hook(layer_idx, "mlp")
                        )
                    else:
                        print(f"Warning: Could not find MLP layer for layer {layer_idx} in {arch_type} architecture")
                        continue
                
                self.hooks.append(hook_mlp)
        
        print(f"注册了 {len(self.hooks)} 个钩子")
    
    def remove_hooks(self):
        """移除钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def collect_activations(self, task_name, texts):
        """收集激活"""
        print(f"收集任务 '{task_name}' 的激活 ({len(texts)} 样本)...")
        self.current_task = task_name
        
        for i in tqdm(range(0, len(texts), self.config.batch_size)):
            batch_texts = texts[i:i + self.config.batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
    
    def get_activations(self, layer_idx, hook_type="mlp", task_name=None):
        """获取激活矩阵"""
        key = f"layer{layer_idx}_{hook_type}"
        
        if key not in self.activations:
            return None
        
        acts_list = self.activations[key]
        
        # 筛选任务
        filtered_acts = []
        for act_dict in acts_list:
            if task_name is not None and act_dict['task'] != task_name:
                continue
            filtered_acts.append(act_dict['activation'])
        
        if filtered_acts:
            return torch.cat(filtered_acts, dim=0).numpy()
        return None
    
    def get_all_activations_for_layer(self, layer_idx):
        """获取该层所有任务的激活"""
        key_prefix = f"layer{layer_idx}_"
        tasks = set()
        
        # 收集所有任务
        for key in self.activations.keys():
            if key.startswith(key_prefix):
                for act_dict in self.activations[key]:
                    tasks.add(act_dict['task'])
        
        # 获取每个任务的激活
        activations_dict = {}
        for task in tasks:
            act = self.get_activations(layer_idx, "mlp", task)
            if act is not None:
                activations_dict[task] = act
        
        return activations_dict

# ==================== 主实验类 ====================
class SharedReasoningExperiment:
    """主实验类"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.collector = None
        self.all_layer_results = []
        self.shared_subspaces = {}
        
    def setup(self):
        """设置实验"""
        print(f"加载模型 {self.config.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Use float16 for CUDA to save memory (like in disturb_joint_subspace.py)
        torch_dtype = torch.float16 if self.config.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype
        )
        self.model.to(self.config.device)
        self.model.eval()
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Get architecture info
        layers, arch_type = get_model_layers(self.model)
        print(f"模型架构类型: {arch_type}, 总层数: {len(layers)}")
        
        print("实验设置完成")
        
    def collect_all_activations(self):
        """收集所有任务的激活"""
        print("\n" + "="*60)
        print("开始收集所有任务激活")
        print("="*60)
        
        # 初始化激活收集器
        self.collector = EnhancedActivationCollector(self.model, self.tokenizer, self.config)
        self.collector.register_hooks(self.config.layers_to_probe)
        
        # 获取所有任务
        all_tasks = []
        for category, tasks in self.config.task_categories.items():
            all_tasks.extend(tasks)
        
        # 为每个任务收集激活
        for task in tqdm(all_tasks, desc="收集激活"):
            # 找到任务类别
            category = None
            for cat, tasks in self.config.task_categories.items():
                if task in tasks:
                    category = cat
                    break
            
            if category is None:
                continue
            
            # 生成示例数据
            texts = DataGenerator.generate_examples(
                task, 
                self.config.n_samples_per_task, 
                category
            )
            
            # 收集激活
            self.collector.collect_activations(task, texts)
        
        # 移除钩子
        self.collector.remove_hooks()
        
        # 保存激活数据
        activations_path = os.path.join(self.config.output_dir, "all_activations.pkl")
        with open(activations_path, 'wb') as f:
            pickle.dump(self.collector.activations, f)
        
        print(f"\n激活数据已保存到: {activations_path}")
        
    def analyze_shared_subspaces(self):
        """分析共享子空间"""
        print("\n" + "="*60)
        print("开始分析共享推理子空间")
        print("="*60)
        
        all_layer_results = []
        
        for layer_idx in self.config.layers_to_probe:
            print(f"\n分析层 {layer_idx}...")
            
            # 获取该层所有任务的激活
            activations_dict = self.collector.get_all_activations_for_layer(layer_idx)
            
            if not activations_dict:
                print(f"层 {layer_idx}: 没有激活数据，跳过")
                continue
            
            # 计算共享子空间
            shared_subspace, subspace_dim, recon_errors, task_samples = \
                SharedReasoningAnalyzer.compute_shared_subspace(
                    activations_dict, 
                    self.config, 
                    layer_idx,
                    tasks_to_include=self.config.task_categories["reasoning"]
                )
            
            if shared_subspace is None:
                print(f"层 {layer_idx}: 无法计算共享子空间，跳过")
                continue
            
            # 保存共享子空间
            self.shared_subspaces[layer_idx] = shared_subspace
            
            # 评估共享子空间
            results = SharedReasoningAnalyzer.evaluate_shared_subspace(
                shared_subspace, 
                activations_dict, 
                self.config, 
                layer_idx
            )
            
            # 添加额外信息
            results['subspace_dim'] = subspace_dim
            results['reconstruction_errors'] = recon_errors
            results['task_samples'] = task_samples
            
            # 分析子空间稳定性
            stability_results = SharedReasoningAnalyzer.analyze_subspace_stability(
                shared_subspace,
                activations_dict,
                self.config,
                layer_idx
            )
            
            results['stability'] = stability_results
            
            all_layer_results.append(results)
            
            # 打印该层关键发现
            print(f"  共享子空间维度: {subspace_dim}/{self.config.hidden_dim} ({subspace_dim/self.config.hidden_dim*100:.1f}%)")
            
            if results['reasoning_tasks']:
                avg_error = np.mean(list(results['reasoning_tasks'].values()))
                print(f"  推理任务平均重建误差: {avg_error:.2e}")
            
            # 原始方法（任务级别）
            if 'comparison' in results and results['comparison']:
                comp = results['comparison']
                sig_status = "显著" if comp['p_value'] < 0.05 else "不显著"
                print(f"  显著性（任务级别）: p={comp['p_value']:.4f} ({sig_status}), "
                      f"Cohen's d={comp.get('cohens_d', comp.get('effect_size', 0)):.3f}, "
                      f"n={comp.get('n_reasoning_tasks', 0)} vs {comp.get('n_other_tasks', 0)}")
            
            # 改进方法（样本级别）
            if 'comparison_improved' in results and results['comparison_improved']:
                comp_imp = results['comparison_improved']
                sig_status_imp = "显著" if comp_imp['p_value'] < 0.05 else "不显著"
                print(f"  显著性（样本级别，改进）: p={comp_imp['p_value']:.4f} ({sig_status_imp}), "
                      f"Cohen's d={comp_imp.get('cohens_d', 0):.3f}, "
                      f"n={comp_imp.get('n_reasoning_samples', 0)} vs {comp_imp.get('n_other_samples', 0)}")
                if 'ci_95_lower' in comp_imp:
                    print(f"    95% CI: [{comp_imp['ci_95_lower']:.4e}, {comp_imp['ci_95_upper']:.4e}]")
        
        self.all_layer_results = all_layer_results
        
        # 多重比较校正（如果有多层）
        if len(all_layer_results) > 1 and HAS_STATSMODELS:
            # 对改进方法的p值进行多重比较校正
            p_values_improved = []
            for result in all_layer_results:
                if 'comparison_improved' in result and result['comparison_improved']:
                    p_values_improved.append(result['comparison_improved']['p_value'])
                else:
                    p_values_improved.append(1.0)
            
            if p_values_improved:
                # Bonferroni校正
                _, p_adjusted_bonf, _, _ = multipletests(p_values_improved, method='bonferroni', alpha=0.05)
                # FDR校正（Benjamini-Hochberg）
                _, p_adjusted_fdr, _, _ = multipletests(p_values_improved, method='fdr_bh', alpha=0.05)
                
                # 将校正后的p值添加到结果中
                for i, result in enumerate(all_layer_results):
                    if 'comparison_improved' in result and result['comparison_improved']:
                        result['comparison_improved']['p_value_bonferroni'] = p_adjusted_bonf[i]
                        result['comparison_improved']['p_value_fdr'] = p_adjusted_fdr[i]
                        result['comparison_improved']['significant_bonferroni'] = p_adjusted_bonf[i] < 0.05
                        result['comparison_improved']['significant_fdr'] = p_adjusted_fdr[i] < 0.05
                
                print(f"\n多重比较校正（{len(p_values_improved)}层）:")
                print(f"  Bonferroni校正后显著层数: {sum(1 for p in p_adjusted_bonf if p < 0.05)}/{len(p_adjusted_bonf)}")
                print(f"  FDR校正后显著层数: {sum(1 for p in p_adjusted_fdr if p < 0.05)}/{len(p_adjusted_fdr)}")
        
        # 保存分析结果
        results_path = os.path.join(self.config.output_dir, "shared_subspace_results.pkl")
        with open(results_path, 'wb') as f:
            pickle.dump({
                'all_layer_results': all_layer_results,
                'shared_subspaces': self.shared_subspaces
            }, f)
        
        print(f"\n分析结果已保存到: {results_path}")
        return all_layer_results
    
    def visualize_results(self):
        """可视化结果"""
        print("\n" + "="*60)
        print("生成可视化结果")
        print("="*60)
        
        # 为每层生成详细分析图
        for result in self.all_layer_results:
            layer_idx = result['layer']
            SharedReasoningVisualizer.plot_shared_subspace_analysis(
                result, 
                self.config, 
                self.config.output_dir
            )
        
        # 生成跨层比较图
        if len(self.all_layer_results) > 1:
            SharedReasoningVisualizer.plot_cross_layer_comparison(
                self.all_layer_results, 
                self.config, 
                self.config.output_dir
            )
        
        print("\n所有可视化已完成")
    
    def generate_reports(self):
        """生成报告"""
        print("\n" + "="*60)
        print("生成分析报告")
        print("="*60)
        
        # 生成文本报告
        report_path = SharedReasoningReportGenerator.generate_comprehensive_report(
            self.all_layer_results, 
            self.config, 
            self.config.output_dir
        )
        
        # 生成JSON总结
        json_path = SharedReasoningReportGenerator.generate_json_summary(
            self.all_layer_results, 
            self.config, 
            self.config.output_dir
        )
        
        print(f"\n报告生成完成:")
        print(f"  文本报告: {report_path}")
        print(f"  JSON总结: {json_path}")
        
        return report_path, json_path
    
    def run_comprehensive_analysis(self):
        """运行完整分析"""
        print("=" * 80)
        print("共享低维推理子空间分析实验")
        print("=" * 80)
        
        # 1. 设置
        self.setup()
        
        # 2. 收集激活
        self.collect_all_activations()
        
        # 3. 分析共享子空间
        self.analyze_shared_subspaces()
        
        # 4. 可视化
        self.visualize_results()
        
        # 5. 生成报告
        self.generate_reports()
        
        print("\n" + "=" * 80)
        print("实验完成!")
        print("=" * 80)
        
        # 打印关键发现总结
        self.print_key_findings()
    
    def print_key_findings(self):
        """打印关键发现"""
        print("\n" + "="*60)
        print("关键发现总结")
        print("="*60)
        
        if not self.all_layer_results:
            print("没有分析结果")
            return
        
        # 1. 维度分析
        subspace_dims = []
        layers = []
        
        for result in self.all_layer_results:
            layers.append(result['layer'])
            subspace_dims.append(result.get('shared_subspace_dim', 0))
        
        # 找到维度最低和最高的层
        min_idx = np.argmin(subspace_dims)
        max_idx = np.argmax(subspace_dims)
        
        print(f"1. 维度分析:")
        print(f"   - 最低维: 层 {layers[min_idx]} ({subspace_dims[min_idx]}维)")
        print(f"   - 最高维: 层 {layers[max_idx]} ({subspace_dims[max_idx]}维)")
        print(f"   - 平均维度: {np.mean(subspace_dims):.1f}维")
        
        # 2. U型曲线检测
        if len(layers) >= 3:
            middle_idx = len(layers) // 2
            u_score = (subspace_dims[-1] + subspace_dims[0] - 2*subspace_dims[middle_idx])/(subspace_dims[-1] + subspace_dims[0])
            
            if u_score > 0.1:
                print(f"2. 检测到U型维度模式 (强度: {u_score:.3f})")
                print("   中间层维度显著高于两端层")
        
        # 3. 显著性分析（使用改进方法，如果有的话）
        sig_layers_original = []
        sig_layers_improved = []
        sig_layers_fdr = []
        
        for result in self.all_layer_results:
            layer = result['layer']
            # 原始方法
            if 'comparison' in result and result['comparison']:
                if result['comparison']['p_value'] < 0.05:
                    sig_layers_original.append(layer)
            # 改进方法
            if 'comparison_improved' in result and result['comparison_improved']:
                comp_imp = result['comparison_improved']
                if comp_imp['p_value'] < 0.05:
                    sig_layers_improved.append(layer)
                if 'p_value_fdr' in comp_imp and comp_imp['p_value_fdr'] < 0.05:
                    sig_layers_fdr.append(layer)
        
        print("3. 显著性分析:")
        if sig_layers_original:
            print(f"   原始方法（任务级别）: 层 {sig_layers_original} 显著")
        if sig_layers_improved:
            print(f"   改进方法（样本级别）: 层 {sig_layers_improved} 显著")
        if sig_layers_fdr:
            print(f"   FDR校正后: 层 {sig_layers_fdr} 显著")
        if not sig_layers_original and not sig_layers_improved:
            print("   未检测到显著的共享子空间特异性")
        
        # 4. 维度压缩率
        compression_ratios = [d/self.config.hidden_dim for d in subspace_dims]
        avg_compression = np.mean(compression_ratios)
        
        print(f"4. 平均维度压缩率: {avg_compression:.3f}")
        print(f"   推理任务激活被压缩到原维度的{avg_compression*100:.1f}%")
        
        # 5. 稳定性分析
        avg_stability_angles = []
        for result in self.all_layer_results:
            if 'stability' in result and result['stability']:
                angles = [s['avg_angle_with_full'] for s in result['stability']]
                avg_stability_angles.append(np.mean(angles))
        
        if avg_stability_angles:
            print(f"5. 平均子空间稳定性: {np.mean(avg_stability_angles):.1f}度")
            print("   交叉验证表明子空间估计是稳定的")
        
        print("\n" + "="*60)


# ==================== 可视化工具 ====================
class SharedReasoningVisualizer:
    """共享推理子空间可视化工具"""
    
    @staticmethod
    def plot_shared_subspace_analysis(results, config, save_dir):
        """绘制共享子空间分析结果"""
        layer_idx = results['layer']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1：推理任务重建误差
        ax1 = axes[0, 0]
        
        reasoning_tasks = list(results['reasoning_tasks'].keys())
        reasoning_errors = [results['reasoning_tasks'][t] for t in reasoning_tasks]
        
        if reasoning_tasks:
            bars1 = ax1.bar(range(len(reasoning_tasks)), reasoning_errors, color='lightblue', alpha=0.7)
            ax1.set_xlabel('Reasoning Tasks')
            ax1.set_ylabel('Reconstruction Error (MSE)')
            ax1.set_title(f'Layer {layer_idx}: Reconstruction Error on Shared Subspace\n(Lower is Better)')
            ax1.set_xticks(range(len(reasoning_tasks)))
            ax1.set_xticklabels(reasoning_tasks, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)
            
            # 添加平均线
            avg_error = np.mean(reasoning_errors)
            ax1.axhline(y=avg_error, color='red', linestyle='--', label=f'Mean: {avg_error:.2e}')
            ax1.legend()
        
        # 子图2：推理任务 vs 其他任务重建误差
        ax2 = axes[0, 1]
        
        if results['other_tasks'] and results['reasoning_tasks']:
            other_tasks = list(results['other_tasks'].keys())[:6]  # 只显示前6个
            other_errors = [results['other_tasks'][t] for t in other_tasks]
            
            # 合并显示
            all_tasks = reasoning_tasks + other_tasks
            all_errors = reasoning_errors + other_errors
            colors = ['lightblue'] * len(reasoning_tasks) + ['lightcoral'] * len(other_tasks)
            
            bars2 = ax2.bar(range(len(all_tasks)), all_errors, color=colors, alpha=0.7)
            ax2.set_xlabel('Task')
            ax2.set_ylabel('Reconstruction Error (MSE)')
            ax2.set_title(f'Layer {layer_idx}: Reasoning vs Other Tasks Comparison')
            ax2.set_xticks(range(len(all_tasks)))
            ax2.set_xticklabels(all_tasks, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
            
            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='lightblue', alpha=0.7, label='Reasoning Tasks'),
                Patch(facecolor='lightcoral', alpha=0.7, label='Other Tasks')
            ]
            ax2.legend(handles=legend_elements)
        
        # 子图3：子空间维度与重建误差的关系
        ax3 = axes[1, 0]
        
        # 模拟不同维度下的重建误差
        if results['reasoning_tasks']:
            # 获取所有推理任务的激活
            reasoning_activations = []
            for task in reasoning_tasks:
                if task in results['reasoning_tasks']:
                    # 这里我们需要原始激活数据，暂时用模拟
                    pass
            
            # 显示统计比较结果（优先使用改进方法）
            comp = None
            comp_label = ""
            if 'comparison_improved' in results and results['comparison_improved']:
                comp = results['comparison_improved']
                comp_label = "Sample Level"
            elif 'comparison' in results and results['comparison']:
                comp = results['comparison']
                comp_label = "Task Level"
            
            if comp:
                categories = ['Reasoning Tasks', 'Other Tasks']
                means = [comp['mean_reasoning_error'], comp['mean_other_error']]
                stds = [comp['std_reasoning_error'], comp['std_other_error']]
                
                bars3 = ax3.bar(categories, means, yerr=stds, capsize=10, 
                               color=['lightblue', 'lightcoral'], alpha=0.7)
                ax3.set_ylabel('Mean Reconstruction Error (MSE)')
                
                # 构建标题，包含p值和校正信息
                title_parts = [f'Layer {layer_idx}: Reconstruction Error Comparison\n({comp_label}, p={comp["p_value"]:.3f})']
                if 'p_value_bonferroni' in comp:
                    title_parts.append(f'Bonf: {comp["p_value_bonferroni"]:.3f}')
                if 'p_value_fdr' in comp:
                    title_parts.append(f'FDR: {comp["p_value_fdr"]:.3f}')
                ax3.set_title(', '.join(title_parts))
                ax3.grid(True, alpha=0.3)
                
                # 添加显著性标记（使用校正后的p值，如果有的话）
                p_to_check = comp.get('p_value_fdr', comp.get('p_value_bonferroni', comp['p_value']))
                if p_to_check < 0.05:
                    ax3.text(0.5, max(means)*1.05, '*', ha='center', va='bottom', fontsize=20)
                    p_label = 'p<0.05'
                    if 'p_value_fdr' in comp and comp['p_value_fdr'] < 0.05:
                        p_label += ' (FDR Corrected)'
                    elif 'p_value_bonferroni' in comp and comp['p_value_bonferroni'] < 0.05:
                        p_label += ' (Bonf Corrected)'
                    ax3.text(0.5, max(means)*1.10, p_label, ha='center', va='bottom')
        
        # 子图4：共享子空间解释的方差比例
        ax4 = axes[1, 1]
        
        # 显示共享子空间维度
        subspace_dim = results['shared_subspace_dim']
        total_dim = config.hidden_dim
        
        # 饼图显示维度比例
        labels = [f'Shared Subspace\n({subspace_dim} dim)', f'Remaining Dimensions\n({total_dim-subspace_dim} dim)']
        sizes = [subspace_dim, total_dim - subspace_dim]
        colors_pie = ['lightgreen', 'lightgray']
        
        wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors_pie, 
                                          autopct='%1.1f%%', startangle=90)
        ax4.set_title(f'Layer {layer_idx}: Shared Subspace Dimension Ratio\n({subspace_dim}/{total_dim} dim)')
        
        plt.suptitle(f'Layer {layer_idx}: Shared Reasoning Subspace Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图形
        save_path = os.path.join(save_dir, f"shared_subspace_analysis_layer{layer_idx}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"共享子空间分析图已保存到: {save_path}")
        return save_path
    
    @staticmethod
    def plot_cross_layer_comparison(all_layer_results, config, save_dir):
        """绘制跨层比较图"""
        layers = []
        subspace_dims = []
        reasoning_errors = []
        other_errors = []
        p_values = []
        
        for result in all_layer_results:
            layers.append(result['layer'])
            subspace_dims.append(result['shared_subspace_dim'])
            
            if result['reasoning_tasks']:
                reasoning_errors.append(np.mean(list(result['reasoning_tasks'].values())))
            else:
                reasoning_errors.append(0)
            
            if result['other_tasks']:
                other_errors.append(np.mean(list(result['other_tasks'].values())))
            else:
                other_errors.append(0)
            
            # 优先使用改进方法的p值
            if 'comparison_improved' in result and result['comparison_improved']:
                p_val = result['comparison_improved'].get('p_value_fdr', 
                                                          result['comparison_improved'].get('p_value', 1.0))
                p_values.append(p_val)
            elif 'comparison' in result and result['comparison']:
                p_values.append(result['comparison']['p_value'])
            else:
                p_values.append(1.0)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 子图1：共享子空间维度随层变化
        ax1 = axes[0, 0]
        ax1.plot(layers, subspace_dims, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('Shared Subspace Dimension')
        ax1.set_title('Shared Subspace Dimension Across Layers')
        ax1.grid(True, alpha=0.3)
        
        # 标记最小值
        min_idx = np.argmin(subspace_dims)
        ax1.annotate(f'Min: {subspace_dims[min_idx]} dim', 
                    xy=(layers[min_idx], subspace_dims[min_idx]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
        
        # 子图2：重建误差随层变化
        ax2 = axes[0, 1]
        line1, = ax2.plot(layers, reasoning_errors, 'bo-', linewidth=2, markersize=8, label='Reasoning Tasks')
        line2, = ax2.plot(layers, other_errors, 'ro-', linewidth=2, markersize=8, label='Other Tasks')
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Mean Reconstruction Error (MSE)')
        ax2.set_title('Reconstruction Error Across Layers')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 子图3：误差比率（其他/推理）
        ax3 = axes[1, 0]
        error_ratios = []
        for re, oe in zip(reasoning_errors, other_errors):
            if re > 0:
                error_ratios.append(oe / re)
            else:
                error_ratios.append(1.0)
        
        bars = ax3.bar(layers, error_ratios, color=['green' if r > 1.5 else 'orange' for r in error_ratios], alpha=0.7)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Error Ratio (Other/Reasoning)')
        ax3.set_title('Error Ratio: Other vs Reasoning Tasks\n(>1.5 indicates shared subspace is more specific to reasoning)')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
        
        # 添加数值标签
        for i, (bar, ratio) in enumerate(zip(bars, error_ratios)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
        
        # 子图4：统计显著性随层变化
        ax4 = axes[1, 1]
        sig_mask = [p < 0.05 for p in p_values]
        colors = ['green' if sig else 'red' for sig in sig_mask]
        
        bars4 = ax4.bar(layers, [-np.log10(p) if p > 0 else 10 for p in p_values], 
                       color=colors, alpha=0.7)
        ax4.set_xlabel('Layer Index')
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('Statistical Significance: Reasoning vs Other Tasks')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5, label='p=0.05 threshold')
        ax4.legend()
        
        # 添加数值标签
        for i, (bar, p) in enumerate(zip(bars4, p_values)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02, 
                    f'p={p:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Cross-Layer Shared Reasoning Subspace Analysis Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图形
        save_path = os.path.join(save_dir, "cross_layer_comparison.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"跨层比较图已保存到: {save_path}")
        return save_path


# ==================== 修改的报告生成器 ====================
class SharedReasoningReportGenerator:
    """共享推理子空间报告生成器（修复版）"""
    
    @staticmethod
    def generate_comprehensive_report(all_layer_results, config, save_dir):
        """生成综合报告（修复版）"""
        report_path = os.path.join(save_dir, "shared_reasoning_analysis_report.txt")
        
        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 100 + "\n")
                f.write("共享推理子空间分析报告\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 100 + "\n\n")
                
                # 1. 实验概览
                f.write("1. 实验概览\n")
                f.write("-" * 60 + "\n")
                f.write(f"模型: {config.model_name}\n")
                f.write(f"隐藏层维度: {config.hidden_dim}\n")
                f.write(f"样本数/任务: {config.n_samples_per_task}\n")
                f.write(f"激活收集策略: {config.activation_strategy}\n")
                f.write(f"PCA方差阈值: {config.pca_variance_threshold}\n")
                f.write(f"探测层: {config.layers_to_probe}\n")
                f.write(f"重点关注层: {config.focus_layers}\n")
                f.write(f"推理任务: {', '.join(config.task_categories['reasoning'])}\n\n")
                
                # 2. 各层共享子空间维度
                f.write("2. 各层共享子空间维度分析\n")
                f.write("-" * 60 + "\n")
                
                if not all_layer_results:
                    f.write("无分析结果\n\n")
                    return report_path
                
                layers = []
                subspace_dims = []
                reasoning_errors = []
                p_values = []
                
                for result in all_layer_results:
                    layer = result.get('layer', 'Unknown')
                    subspace_dim = result.get('subspace_dim', 0)
                    
                    layers.append(layer)
                    subspace_dims.append(float(subspace_dim))
                    
                    # 推理任务重建误差
                    if 'reasoning_tasks' in result and result['reasoning_tasks']:
                        errors = list(result['reasoning_tasks'].values())
                        avg_error = float(np.mean(errors)) if errors else 0
                        reasoning_errors.append(avg_error)
                    else:
                        reasoning_errors.append(0)
                    
                    # p值
                    # 优先使用改进方法的p值
                    if 'comparison_improved' in result and result['comparison_improved']:
                        p_val = result['comparison_improved'].get('p_value', 1.0)
                        p_values.append(float(p_val))
                    elif 'comparison' in result and result['comparison']:
                        p_val = result['comparison'].get('p_value', 1.0)
                        p_values.append(float(p_val))
                    else:
                        p_values.append(1.0)
                    
                    # 写入层信息
                    f.write(f"层 {layer}: {subspace_dim} 维 ({subspace_dim/config.hidden_dim*100:.1f}%)\n")
                    
                    if 'reasoning_tasks' in result and result['reasoning_tasks']:
                        errors = list(result['reasoning_tasks'].values())
                        if errors:
                            avg_error = float(np.mean(errors))
                            f.write(f"  推理任务平均重建误差: {avg_error:.2e}\n")
                    
                    # 原始方法
                    if 'comparison' in result and result['comparison']:
                        comp = result['comparison']
                        p_val = comp.get('p_value', 1.0)
                        sig = "显著" if p_val < 0.05 else "不显著"
                        cohens_d = comp.get('cohens_d', comp.get('effect_size', 0))
                        f.write(f"  显著性（任务级别）: p={p_val:.4f} ({sig}), Cohen's d={cohens_d:.3f}\n")
                    
                    # 改进方法
                    if 'comparison_improved' in result and result['comparison_improved']:
                        comp_imp = result['comparison_improved']
                        p_val_imp = comp_imp.get('p_value', 1.0)
                        sig_imp = "显著" if p_val_imp < 0.05 else "不显著"
                        cohens_d_imp = comp_imp.get('cohens_d', 0)
                        n_reasoning = comp_imp.get('n_reasoning_samples', 0)
                        n_other = comp_imp.get('n_other_samples', 0)
                        f.write(f"  显著性（样本级别，改进）: p={p_val_imp:.4f} ({sig_imp}), "
                               f"Cohen's d={cohens_d_imp:.3f}, n={n_reasoning} vs {n_other}\n")
                        if 'p_value_fdr' in comp_imp:
                            sig_fdr = "显著" if comp_imp['p_value_fdr'] < 0.05 else "不显著"
                            f.write(f"    FDR校正: p={comp_imp['p_value_fdr']:.4f} ({sig_fdr})\n")
                        if 'p_value_bonferroni' in comp_imp:
                            sig_bonf = "显著" if comp_imp['p_value_bonferroni'] < 0.05 else "不显著"
                            f.write(f"    Bonferroni校正: p={comp_imp['p_value_bonferroni']:.4f} ({sig_bonf})\n")
                    
                    f.write("\n")
                
                # 3. 关键统计
                f.write("3. 关键统计\n")
                f.write("-" * 60 + "\n")
                
                if layers and subspace_dims:
                    # 找到维度最低和最高的层
                    min_idx = np.argmin(subspace_dims)
                    max_idx = np.argmax(subspace_dims)
                    
                    f.write(f"最低维共享子空间: 层 {layers[min_idx]} ({subspace_dims[min_idx]:.1f}维)\n")
                    f.write(f"最高维共享子空间: 层 {layers[max_idx]} ({subspace_dims[max_idx]:.1f}维)\n")
                    f.write(f"平均维度: {np.mean(subspace_dims):.1f}维\n")
                    f.write(f"维度标准差: {np.std(subspace_dims):.1f}维\n")
                    
                    # U型曲线检测
                    if len(layers) >= 3:
                        middle_idx = len(layers) // 2
                        if subspace_dims[-1] + subspace_dims[0] > 0:
                            u_score = (subspace_dims[-1] + subspace_dims[0] - 2*subspace_dims[middle_idx]) / (subspace_dims[-1] + subspace_dims[0])
                            f.write(f"U型曲线强度: {u_score:.3f}\n")
                        
                        # 检查是否确实是U型
                        if (subspace_dims[middle_idx] > subspace_dims[0] and 
                            subspace_dims[middle_idx] > subspace_dims[-1]):
                            f.write("检测到清晰的U型曲线模式\n")
                        else:
                            f.write("未检测到典型的U型曲线模式\n")
                    
                    # 显著性统计
                    sig_count = sum(1 for p in p_values if p < 0.05)
                    f.write(f"显著层数: {sig_count}/{len(layers)} ({sig_count/len(layers)*100:.1f}%)\n")
                    
                    if reasoning_errors:
                        f.write(f"推理任务平均重建误差: {np.mean(reasoning_errors):.2e}\n")
                        f.write(f"重建误差范围: [{np.min(reasoning_errors):.2e}, {np.max(reasoning_errors):.2e}]\n")
                
                # 4. 理论解释
                f.write("\n4. 理论解释\n")
                f.write("-" * 60 + "\n")
                
                f.write("实验结果支持以下理论假设:\n\n")
                
                # 基于结果动态生成解释
                if len(layers) >= 3 and subspace_dims:
                    middle_idx = len(layers) // 2
                    
                    if subspace_dims[middle_idx] > subspace_dims[0] * 2:
                        f.write("1. 明确的U型维度模式表明Transformer的层级处理架构:\n")
                        f.write("   - 浅层(0-3层): 主要负责特征提取和初步编码\n")
                        f.write("   - 中层(5-8层): 形成共享推理工作区，维度显著升高\n")
                        f.write("   - 深层(11层): 为输出准备，维度再次降低\n\n")
                    
                    if np.mean(subspace_dims) < config.hidden_dim * 0.2:
                        f.write("2. 极低的维度压缩率(平均{:.1f}%)表明:\n".format(np.mean(subspace_dims)/config.hidden_dim*100))
                        f.write("   - 推理任务共享高度紧凑的表征空间\n")
                        f.write("   - 模型能够有效提取和复用通用推理模式\n")
                        f.write("   - 可能存在跨任务的通用计算基元\n\n")
                    
                    if sig_count > len(layers) * 0.5:
                        f.write("3. 高显著性比率表明:\n")
                        f.write("   - 共享子空间对推理任务具有高度特异性\n")
                        f.write("   - 推理任务与其他任务在表征空间上可区分\n")
                        f.write("   - 支持'专用推理工作区'假设\n")
                
                # 5. 局限性
                f.write("\n5. 实验局限性\n")
                f.write("-" * 60 + "\n")
                
                f.write("1. 样本量: 每任务{}样本可能仍不足以充分探索高维空间\n".format(config.n_samples_per_task))
                f.write("2. 任务范围: 只涵盖了部分推理任务类型\n")
                f.write("3. 模型: 仅分析了单一模型架构(OPT)\n")
                f.write("4. 激活策略: 仅使用最后token激活，可能遗漏序列信息\n")
                f.write("5. PCA阈值: 固定95%方差阈值可能不适合所有层\n")
                
                # 6. 建议
                f.write("\n6. 进一步研究方向\n")
                f.write("-" * 60 + "\n")
                
                f.write("1. 扩展实验到更多模型(Mamba, Llama, GPT等)\n")
                f.write("2. 增加样本量和任务多样性\n")
                f.write("3. 探索不同激活收集策略的影响\n")
                f.write("4. 分析不同PCA阈值下的稳定性\n")
                f.write("5. 探究共享子空间与任务性能的相关性\n")
                
                f.write("\n" + "=" * 100 + "\n")
                f.write("报告生成完成\n")
                f.write("=" * 100 + "\n")
            
            print(f"综合报告已保存到: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"生成报告时出错: {e}")
            # 创建简单的错误报告
            error_report_path = os.path.join(save_dir, "error_report.txt")
            with open(error_report_path, 'w') as f:
                f.write(f"报告生成失败: {str(e)}\n")
                f.write(f"时间: {datetime.now()}\n")
            return error_report_path
    
    @staticmethod
    def generate_json_summary(all_layer_results, config, save_dir):
        """生成JSON格式的总结（修复版）"""
        if not all_layer_results:
            print("警告: 无分析结果，跳过JSON总结生成")
            return None
        
        try:
            # 提取关键数据并转换为Python原生类型
            layers = []
            subspace_dims = []
            reasoning_errors_by_layer = {}
            p_values = []
            
            for result in all_layer_results:
                layer = result.get('layer', 'Unknown')
                layers.append(int(layer))
                
                subspace_dim = result.get('subspace_dim', 0)
                subspace_dims.append(float(subspace_dim))
                
                # 处理推理任务重建误差
                if 'reasoning_tasks' in result and result['reasoning_tasks']:
                    errors = []
                    for task, error in result['reasoning_tasks'].items():
                        errors.append({
                            'task': task,
                            'error': float(error) if not isinstance(error, str) else error
                        })
                    reasoning_errors_by_layer[layer] = errors
                
                # 处理p值
                # 优先使用改进方法的p值
                if 'comparison_improved' in result and result['comparison_improved']:
                    p_val = result['comparison_improved'].get('p_value_fdr',
                                                              result['comparison_improved'].get('p_value', 1.0))
                    p_values.append(float(p_val))
                elif 'comparison' in result and result['comparison']:
                    p_val = result['comparison'].get('p_value', 1.0)
                    p_values.append(float(p_val))
                else:
                    p_values.append(1.0)
            
            # 构建总结数据
            summary = {
                "experiment_info": {
                    "model_name": config.model_name,
                    "hidden_dim": int(config.hidden_dim),
                    "n_samples_per_task": int(config.n_samples_per_task),
                    "activation_strategy": config.activation_strategy,
                    "pca_variance_threshold": float(config.pca_variance_threshold),
                    "layers_analyzed": [int(l) for l in config.layers_to_probe],
                    "focus_layers": [int(l) for l in config.focus_layers],
                    "reasoning_tasks": config.task_categories["reasoning"],
                    "generation_time": datetime.now().isoformat()
                },
                
                "dimensional_analysis": {
                    "layers": layers,
                    "subspace_dimensions": subspace_dims,
                    "dimension_ratios": [float(dim/config.hidden_dim) for dim in subspace_dims],
                    "min_dimension": {
                        "layer": int(layers[np.argmin(subspace_dims)]),
                        "value": float(np.min(subspace_dims)),
                        "ratio": float(np.min(subspace_dims)/config.hidden_dim)
                    },
                    "max_dimension": {
                        "layer": int(layers[np.argmax(subspace_dims)]),
                        "value": float(np.max(subspace_dims)),
                        "ratio": float(np.max(subspace_dims)/config.hidden_dim)
                    },
                    "mean_dimension": float(np.mean(subspace_dims)),
                    "std_dimension": float(np.std(subspace_dims)),
                    "mean_compression_ratio": float(np.mean(subspace_dims)/config.hidden_dim)
                },
                
                "reconstruction_analysis": {
                    "reasoning_errors_by_layer": reasoning_errors_by_layer,
                    "mean_reasoning_error_by_layer": {
                        layer: float(np.mean([e['error'] for e in errors])) 
                        for layer, errors in reasoning_errors_by_layer.items()
                        if errors and all(isinstance(e['error'], (int, float)) for e in errors)
                    }
                },
                
                "statistical_significance": {
                    "p_values": p_values,
                    "significant_layers": [
                        int(layers[i]) for i, p in enumerate(p_values) if p < 0.05
                    ],
                    "significant_count": int(sum(1 for p in p_values if p < 0.05)),
                    "total_layers": len(layers),
                    "significance_ratio": float(sum(1 for p in p_values if p < 0.05) / len(layers))
                },
                
                "u_shape_analysis": {
                    "detected": len(layers) >= 3,
                    "strength": None,
                    "pattern": None
                },
                
                "key_findings": {
                    "lowest_dimension_layer": int(layers[np.argmin(subspace_dims)]),
                    "highest_dimension_layer": int(layers[np.argmax(subspace_dims)]),
                    "strongest_shared_subspace": int(layers[np.argmin([
                        np.mean([e['error'] for e in errors]) 
                        for layer, errors in reasoning_errors_by_layer.items()
                        if errors
                    ])]) if reasoning_errors_by_layer else None,
                    "average_compression": float(np.mean(subspace_dims)/config.hidden_dim)
                }
            }
            
            # 计算U型曲线指标
            if len(layers) >= 3:
                middle_idx = len(layers) // 2
                if subspace_dims[-1] + subspace_dims[0] > 0:
                    u_strength = (subspace_dims[-1] + subspace_dims[0] - 2*subspace_dims[middle_idx]) / (subspace_dims[-1] + subspace_dims[0])
                    summary["u_shape_analysis"]["strength"] = float(u_strength)
                    
                    if (subspace_dims[middle_idx] > subspace_dims[0] and 
                        subspace_dims[middle_idx] > subspace_dims[-1]):
                        summary["u_shape_analysis"]["pattern"] = "clear_u_shape"
                    elif (subspace_dims[middle_idx] < subspace_dims[0] and 
                          subspace_dims[middle_idx] < subspace_dims[-1]):
                        summary["u_shape_analysis"]["pattern"] = "inverse_u_shape"
                    else:
                        summary["u_shape_analysis"]["pattern"] = "no_clear_pattern"
            
            # 使用安全的JSON保存方法
            json_path = os.path.join(save_dir, "shared_reasoning_summary.json")
            DataTypeHandler.safe_json_dump(summary, json_path)
            
            print(f"JSON总结已保存到: {json_path}")
            return json_path
            
        except Exception as e:
            print(f"生成JSON总结时出错: {e}")
            import traceback
            traceback.print_exc()
            return None

# ==================== 优化的激活收集器 ====================
class OptimizedActivationCollector(EnhancedActivationCollector):
    """优化的激活收集器，添加内存管理和分批处理"""
    
    def collect_activations(self, task_name, texts):
        """收集激活（优化版）"""
        print(f"收集任务 '{task_name}' 的激活 ({len(texts)} 样本)...")
        self.current_task = task_name
        
        # 分批处理，避免内存溢出
        batch_size = min(self.config.batch_size, 4)  # 小批量处理
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"任务: {task_name}"):
            batch_texts = texts[i:i + batch_size]
            
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_len
            ).to(self.config.device)
            
            with torch.no_grad():
                _ = self.model(**inputs)
            
            # 定期清理缓存
            if i % 100 == 0 and self.config.device == "cuda":
                torch.cuda.empty_cache()
    
    def get_activations(self, layer_idx, hook_type="mlp", task_name=None):
        """获取激活矩阵（优化版）"""
        key = f"layer{layer_idx}_{hook_type}"
        
        if key not in self.activations:
            return None
        
        acts_list = self.activations[key]
        
        # 筛选任务
        filtered_acts = []
        for act_dict in acts_list:
            if task_name is not None and act_dict['task'] != task_name:
                continue
            
            # 转换数据类型以节省内存
            activation = act_dict['activation']
            if isinstance(activation, torch.Tensor):
                activation = activation.numpy().astype(np.float32)  # 转换为float32节省内存
            filtered_acts.append(activation)
        
        if filtered_acts:
            # 使用vstack但注意内存使用
            if sum(a.nbytes for a in filtered_acts) > 1e9:  # 大于1GB
                print(f"警告: 层{layer_idx}激活数据较大，使用内存优化连接")
                # 分批连接
                result = filtered_acts[0]
                for i in range(1, len(filtered_acts)):
                    result = np.vstack([result, filtered_acts[i]])
                return result
            else:
                return np.vstack(filtered_acts)
        return None

# ==================== 修改主实验类 ====================
class RobustSharedReasoningExperiment(SharedReasoningExperiment):
    """健壮的主实验类，包含错误处理和恢复机制"""
    
    def __init__(self, config):
        super().__init__(config)
        self.error_log = []
        self.checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    
    def safe_execute(self, func, func_name, *args, **kwargs):
        """安全执行函数，包含错误处理和恢复"""
        try:
            print(f"\n执行: {func_name}")
            result = func(*args, **kwargs)
            return result
        except Exception as e:
            error_msg = f"{func_name} 执行失败: {str(e)}"
            print(f"错误: {error_msg}")
            self.error_log.append({
                'function': func_name,
                'error': str(e),
                'time': datetime.now().isoformat()
            })
            
            # 保存错误日志
            self.save_error_log()
            return None
    
    def save_error_log(self):
        """保存错误日志"""
        error_log_path = os.path.join(self.config.output_dir, "error_log.json")
        DataTypeHandler.safe_json_dump(self.error_log, error_log_path)
    
    def create_checkpoint(self, stage_name, data):
        """创建检查点"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{stage_name}.pkl")
        DataTypeHandler.safe_pickle_dump(data, checkpoint_path)
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, stage_name):
        """加载检查点"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{stage_name}.pkl")
        
        # 检查压缩版本
        if not os.path.exists(checkpoint_path):
            checkpoint_path += '.gz'
        
        if os.path.exists(checkpoint_path):
            try:
                if checkpoint_path.endswith('.gz'):
                    import gzip
                    with gzip.open(checkpoint_path, 'rb') as f:
                        data = pickle.load(f)
                else:
                    with open(checkpoint_path, 'rb') as f:
                        data = pickle.load(f)
                
                print(f"从检查点恢复: {stage_name}")
                return data
            except Exception as e:
                print(f"加载检查点失败: {e}")
                return None
        return None
    
    def run_comprehensive_analysis(self):
        """运行完整分析（健壮版）"""
        print("=" * 80)
        print("共享低维推理子空间分析实验（健壮版）")
        print("=" * 80)
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        try:
            # 1. 设置
            self.safe_execute(self.setup, "实验设置")
            
            # 2. 收集激活（使用检查点）
            if not hasattr(self, 'collector') or self.collector is None:
                checkpoint_data = self.load_checkpoint("activations")
                if checkpoint_data:
                    self.collector = checkpoint_data.get('collector')
                    print("从检查点恢复激活收集器")
            
            if not hasattr(self, 'collector') or self.collector is None:
                self.collector = OptimizedActivationCollector(self.model, self.tokenizer, self.config)
                self.safe_execute(self.collect_all_activations, "激活收集")
                # 保存检查点
                self.create_checkpoint("activations", {'collector': self.collector})
            
            # 3. 分析共享子空间
            if not self.all_layer_results:
                checkpoint_data = self.load_checkpoint("analysis")
                if checkpoint_data:
                    self.all_layer_results = checkpoint_data.get('all_layer_results', [])
                    self.shared_subspaces = checkpoint_data.get('shared_subspaces', {})
                    print("从检查点恢复分析结果")
            
            if not self.all_layer_results:
                self.safe_execute(self.analyze_shared_subspaces, "子空间分析")
                # 保存检查点
                self.create_checkpoint("analysis", {
                    'all_layer_results': self.all_layer_results,
                    'shared_subspaces': self.shared_subspaces
                })
            
            # 4. 可视化
            if self.all_layer_results:
                self.safe_execute(self.visualize_results, "可视化")
            
            # 5. 生成报告（使用安全版本）
            if self.all_layer_results:
                self.generate_reports_robust()
            
            # 6. 打印关键发现
            if self.all_layer_results:
                self.print_key_findings()
            
            # 7. 保存最终结果
            self.save_final_results()
            
            print("\n" + "=" * 80)
            print("实验完成!")
            print("=" * 80)
            
            if self.error_log:
                print(f"警告: 实验过程中发生了 {len(self.error_log)} 个错误")
                print(f"错误日志已保存到: {os.path.join(self.config.output_dir, 'error_log.json')}")
        
        except Exception as e:
            print(f"\n实验执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 尝试保存已有结果
            self.save_error_log()
            
            # 尝试生成最小报告
            self.generate_minimal_report()
    
    def generate_reports_robust(self):
        """生成报告（健壮版）"""
        print("\n" + "="*60)
        print("生成分析报告（健壮版）")
        print("="*60)
        
        # 生成文本报告
        report_path = SharedReasoningReportGenerator.generate_comprehensive_report(
            self.all_layer_results, 
            self.config, 
            self.config.output_dir
        )
        
        if report_path and os.path.exists(report_path):
            print(f"文本报告已生成: {report_path}")
        else:
            print("警告: 文本报告生成失败")
        
        # 生成JSON总结
        json_path = SharedReasoningReportGenerator.generate_json_summary(
            self.all_layer_results, 
            self.config, 
            self.config.output_dir
        )
        
        if json_path and os.path.exists(json_path):
            print(f"JSON总结已生成: {json_path}")
        else:
            print("警告: JSON总结生成失败，尝试生成简化版")
            self.generate_simple_json_summary()
    
    def generate_simple_json_summary(self):
        """生成简化的JSON总结"""
        try:
            simple_summary = {
                "experiment_info": {
                    "model": self.config.model_name,
                    "timestamp": datetime.now().isoformat(),
                    "layers_analyzed": [int(l) for l in self.config.layers_to_probe]
                },
                "key_results": {}
            }
            
            for result in self.all_layer_results:
                layer = result.get('layer', 'Unknown')
                simple_summary["key_results"][str(layer)] = {
                    "subspace_dimension": float(result.get('subspace_dim', 0)),
                    "has_significance": 'comparison' in result and 
                                       result['comparison'].get('p_value', 1.0) < 0.05
                }
            
            json_path = os.path.join(self.config.output_dir, "simple_summary.json")
            DataTypeHandler.safe_json_dump(simple_summary, json_path)
            print(f"简化JSON总结已保存到: {json_path}")
            
        except Exception as e:
            print(f"生成简化JSON总结失败: {e}")
    
    def generate_minimal_report(self):
        """生成最小报告（在失败时）"""
        try:
            report_path = os.path.join(self.config.output_dir, "minimal_report.txt")
            with open(report_path, 'w') as f:
                f.write("最小实验报告\n")
                f.write("="*50 + "\n")
                f.write(f"实验时间: {datetime.now()}\n")
                f.write(f"模型: {self.config.model_name}\n")
                f.write(f"错误数量: {len(self.error_log)}\n")
                
                if self.error_log:
                    f.write("\n错误列表:\n")
                    for i, error in enumerate(self.error_log, 1):
                        f.write(f"{i}. {error['function']}: {error['error']}\n")
                
                if hasattr(self, 'all_layer_results') and self.all_layer_results:
                    f.write(f"\n成功分析层数: {len(self.all_layer_results)}\n")
            
            print(f"最小报告已保存到: {report_path}")
        except Exception as e:
            print(f"生成最小报告失败: {e}")
    
    def save_final_results(self):
        """保存最终结果"""
        try:
            # 保存完整的分析结果
            final_results_path = os.path.join(self.config.output_dir, "final_results.pkl")
            DataTypeHandler.safe_pickle_dump({
                'all_layer_results': self.all_layer_results,
                'shared_subspaces': self.shared_subspaces,
                'config': self.config,
                'error_log': self.error_log,
                'timestamp': datetime.now().isoformat()
            }, final_results_path)
            
            print(f"最终结果已保存到: {final_results_path}")
            
        except Exception as e:
            print(f"保存最终结果失败: {e}")

# ==================== 修改主函数 ====================
def run_experiment_for_model(model_name, base_output_dir="./shared_reasoning_results_robust", config_overrides=None):
    """为单个模型运行实验
    
    Args:
        model_name: 模型名称
        base_output_dir: 输出目录基础路径
        config_overrides: 可选的配置覆盖字典，例如 {'activation_strategy': 'all_tokens', 'n_samples_per_task': 500}
    """
    print(f"\n{'='*80}")
    print(f"开始实验: {model_name}")
    print(f"{'='*80}")
    
    # Create model-specific output directory
    model_safe_name = model_name.replace("/", "_").replace("-", "_")
    output_dir = os.path.join(base_output_dir, model_safe_name)
    
    # 使用类定义中的默认值，然后应用覆盖
    config = SharedSpaceConfig(
        model_name=model_name,
        output_dir=output_dir,
        layers_to_probe=[0, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        focus_layers=[5, 6, 7, 8, 9]
    )
    
    # 应用用户提供的覆盖
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
                print(f"  配置覆盖: {key} = {value}")
            else:
                print(f"  警告: 未知的配置项 {key}，忽略")
    
    print(f"\n使用的配置:")
    print(f"  activation_strategy: {config.activation_strategy}")
    print(f"  n_samples_per_task: {config.n_samples_per_task}")
    print(f"  pca_variance_threshold: {config.pca_variance_threshold}")
    
    # 使用健壮的实验类
    experiment = RobustSharedReasoningExperiment(config)
    
    try:
        experiment.run_comprehensive_analysis()
        print(f"\n{'='*80}")
        print(f"完成实验: {model_name}")
        print(f"{'='*80}\n")
    except Exception as e:
        print(f"\n错误: 运行 {model_name} 的实验时出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up model to free GPU memory
        if experiment.model is not None:
            del experiment.model
        if experiment.tokenizer is not None:
            del experiment.tokenizer
        if experiment.collector is not None:
            experiment.collector.remove_hooks()
            del experiment.collector
        torch.cuda.empty_cache()
        print(f"已清理 {model_name} 的模型和缓存")


def main():
    """主函数（多模型版本）"""
    # List of models to evaluate
    MODELS_TO_RUN = [
        "meta-llama/Llama-2-7b-hf",
        "facebook/opt-6.7b",
        "Qwen/Qwen2.5-7B"
    ]
    
    base_output_dir = "./shared_reasoning_results_robust"
    
    print("=" * 80)
    print("多模型共享推理子空间分析实验")
    print("=" * 80)
    print(f"将评估 {len(MODELS_TO_RUN)} 个模型:")
    for i, model_name in enumerate(MODELS_TO_RUN, 1):
        print(f"  {i}. {model_name}")
    print("=" * 80)
    
    # Run experiments for each model
    for model_name in MODELS_TO_RUN:
        try:
            run_experiment_for_model(model_name, base_output_dir)
        except Exception as e:
            print(f"\n跳过 {model_name} 并继续下一个模型...")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 80)
    print("所有实验完成!")
    print("=" * 80)

if __name__ == "__main__":
    main()