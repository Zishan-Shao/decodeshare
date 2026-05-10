"""
Joint subspace activation perturbation experiment:
- Load models and process datasets
- Collect activations from specified layers
- Compute joint subspace using PCA/SVD with improved numerical stability
- Disturb the joint subspace and merge back with original factors
- Compare loss with and without perturbation
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import random
import time

# Configuration
MODELS_TO_RUN = [
    "meta-llama/Llama-2-7b-hf",
    # "facebook/opt-6.7b",
    # "Qwen/Qwen2.5-7B"
]  # List of models to evaluate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_SEQ_LEN = 128
N_SAMPLES = 64  # Number of samples to use for both subspace computation and evaluation
NOISE_SCALE = 1.0  # Scale of noise to add to joint subspace
MAX_NOISE_RATIO = 0.1  # 最大噪声比例
CLIP_VALUE = 10.0  # 新增：裁剪值
LAYER_INDICES = [21]  # Layers to collect activations from
PCA_VARIANCE_THRESHOLD = 0.95  # Variance threshold for PCA
MIN_SUBSPACE_DIM = 1  # Minimum subspace dimension
MAX_SUBSPACE_DIM = 4096
ACTIVATION_STRATEGY = "all_tokens"  # "last_token", "mean", "max", "all_tokens"
EPS = 1e-8  # 更小的epsilon用于数值稳定性

# Datasets to evaluate (set to None to use all available)
DATASETS_TO_RUN = None  # None = all, or specify: ["wikitext", "gsm8k", "commonsenseqa", ...]



def load_wikitext_data(n_samples=100, max_retries=3):
    """Load wikitext dataset with retry logic"""
    print("Loading wikitext dataset...")

    for attempt in range(max_retries):
        try:
            try:
                dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            except Exception as e1:
                print(f"  Attempt {attempt + 1}: Trying alternative loading method...")
                try:
                    dataset = load_dataset("wikitext", split="train")
                except Exception as e2:
                    try:
                        dataset = load_dataset("wikitext", "wikitext-2", split="train")
                    except Exception as e3:
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 2
                            print(f"  Retrying in {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        raise e3

            # 修改：确保获取足够长的文本
            texts = []
            for item in dataset:
                text = item["text"].strip()
                if len(text) > 500:  # 只选择长度大于500字符的文本
                    texts.append(text)
                if len(texts) >= n_samples * 2:  # 获取更多样本，稍后筛选
                    break

            if not texts:
                raise ValueError("No valid texts found in wikitext dataset")

            # 进一步筛选：选择单词数较多的文本
            texts_with_word_count = [(t, len(t.split())) for t in texts]
            texts_with_word_count.sort(key=lambda x: x[1], reverse=True)

            # 选择前n_samples个最长的文本
            selected_texts = [t for t, _ in texts_with_word_count[:n_samples]]

            # 打印统计信息
            avg_words = sum(len(t.split()) for t in selected_texts) / len(selected_texts)
            avg_chars = sum(len(t) for t in selected_texts) / len(selected_texts)
            print(f"  ✓ Successfully loaded {len(selected_texts)} samples")
            print(f"  Average words per sample: {avg_words:.1f}")
            print(f"  Average chars per sample: {avg_chars:.1f}")

            return selected_texts

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ All {max_retries} attempts failed: {e}")
                raise


def load_gsm8k_data(n_samples=100, max_retries=3):
    """Load gsm8k dataset with retry logic"""
    print("Loading gsm8k dataset...")

    for attempt in range(max_retries):
        try:
            dataset = load_dataset("gsm8k", "main", split="train")
            texts = []
            for item in dataset:
                question = item.get("question", "")
                answer = item.get("answer", "")
                if question:
                    text = f"Question: {question}\nAnswer: {answer}" if answer else question
                    texts.append(text)
                if len(texts) >= n_samples:
                    break

            if not texts:
                raise ValueError("No valid texts found in gsm8k dataset")

            print(f"  ✓ Successfully loaded {len(texts)} samples")
            return texts

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ All {max_retries} attempts failed: {e}")
                raise


def load_commonsenseqa_data(n_samples=100, max_retries=3):
    """Load CommonsenseQA dataset with retry logic"""
    print("Loading commonsenseqa dataset...")

    for attempt in range(max_retries):
        try:
            dataset = load_dataset("commonsense_qa", "default", split="train")
            texts = []
            for item in dataset:
                question = item.get("question", "")
                choices = item.get("choices", {})
                if isinstance(choices, dict) and "text" in choices:
                    choice_text = " ".join([str(c) for c in choices["text"]])
                elif isinstance(choices, list):
                    choice_text = " ".join([str(c) for c in choices])
                else:
                    choice_text = str(choices) if choices else ""
                answer = item.get("answerKey", "")
                if question:
                    text = f"Question: {question}\nChoices: {choice_text}\nAnswer: {answer}"
                    texts.append(text)
                if len(texts) >= n_samples:
                    break

            if not texts:
                raise ValueError("No valid texts found in commonsenseqa dataset")

            print(f"  ✓ Successfully loaded {len(texts)} samples")
            return texts

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ All {max_retries} attempts failed: {e}")
                raise


def load_strategyqa_data(n_samples=100, max_retries=3):
    """Load StrategyQA dataset with retry logic"""
    print("Loading strategyqa dataset...")

    for attempt in range(max_retries):
        try:
            try:
                dataset = load_dataset("metaeval/strategyqa", split="train")
            except:
                dataset = load_dataset("strategyqa", split="train")

            texts = []
            for item in dataset:
                question = item.get("question", "")
                answer = item.get("answer", "")
                facts = item.get("facts", [])
                facts_text = " ".join([str(f) for f in facts]) if isinstance(facts, list) else str(facts)
                if question:
                    text = f"Question: {question}\nFacts: {facts_text}\nAnswer: {answer}"
                    texts.append(text)
                if len(texts) >= n_samples:
                    break

            if not texts:
                raise ValueError("No valid texts found in strategyqa dataset")

            print(f"  ✓ Successfully loaded {len(texts)} samples")
            return texts

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ All {max_retries} attempts failed: {e}")
                raise


def load_aqua_data(n_samples=100, max_retries=3):
    """Load AQuA dataset with retry logic"""
    print("Loading aqua dataset...")

    for attempt in range(max_retries):
        try:
            dataset = load_dataset("aqua_rat", split="train")
            texts = []
            for item in dataset:
                question = item.get("question", "")
                options = item.get("options", "")
                correct = item.get("correct", "")
                if question:
                    text = f"Question: {question}\nOptions: {options}\nCorrect: {correct}"
                    texts.append(text)
                if len(texts) >= n_samples:
                    break

            if not texts:
                raise ValueError("No valid texts found in aqua dataset")

            print(f"  ✓ Successfully loaded {len(texts)} samples")
            return texts

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Attempt {attempt + 1} failed: {e}")
                print(f"  Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"  ✗ All {max_retries} attempts failed: {e}")
                raise


# Dataset loader registry
DATASET_LOADERS = {
    "wikitext": load_wikitext_data,
    "gsm8k": load_gsm8k_data,
    "commonsenseqa": load_commonsenseqa_data,
    "strategyqa": load_strategyqa_data,
    "aqua": load_aqua_data,
}


def get_model_layers(model):
    """Get the layers from a model, handling different architectures"""
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        return model.model.decoder.layers, 'opt'
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers, 'llama'
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, 'gpt2'
    else:
        raise ValueError(f"Could not find layers in model. Model type: {type(model)}")


class ActivationCollector:
    """Collect activations from specified layers"""

    def __init__(self, layer_indices, activation_strategy="all_tokens"):
        self.layer_indices = layer_indices
        self.activations = {idx: [] for idx in layer_indices}
        self.hooks = []
        self.activation_strategy = activation_strategy

    def create_hook(self, layer_idx):
        """Create a hook function for a specific layer"""
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply activation strategy
            if self.activation_strategy == "last_token":
                if len(hidden_states.shape) == 3:
                    act = hidden_states[:, -1, :].detach().cpu()
                elif len(hidden_states.shape) == 2:
                    act = hidden_states.detach().cpu()
                else:
                    return

            elif self.activation_strategy == "mean":
                if len(hidden_states.shape) == 3:
                    act = hidden_states.mean(dim=1).detach().cpu()
                elif len(hidden_states.shape) == 2:
                    act = hidden_states.detach().cpu()
                else:
                    return

            elif self.activation_strategy == "max":
                if len(hidden_states.shape) == 3:
                    act, _ = hidden_states.max(dim=1)
                    act = act.detach().cpu()
                elif len(hidden_states.shape) == 2:
                    act = hidden_states.detach().cpu()
                else:
                    return

            elif self.activation_strategy == "all_tokens":
                if len(hidden_states.shape) == 3:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    act = hidden_states.view(-1, hidden_dim).detach().cpu()
                elif len(hidden_states.shape) == 2:
                    act = hidden_states.detach().cpu()
                else:
                    return

            else:
                raise ValueError(f"Unknown activation strategy: {self.activation_strategy}")

            self.activations[layer_idx].append(act)

        return hook

    def register_hooks(self, model):
        """Register hooks to specified layers"""
        layers, arch_type = get_model_layers(model)

        for layer_idx in self.layer_indices:
            if layer_idx >= len(layers):
                print(f"Warning: Layer {layer_idx} not available. Skipping.")
                continue
            layer = layers[layer_idx]
            hook = self.create_hook(layer_idx)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def get_activations(self, layer_idx):
        """Get collected activations for a layer as numpy array"""
        if layer_idx not in self.activations:
            return None
        if not self.activations[layer_idx]:
            return None
        acts = torch.cat(self.activations[layer_idx], dim=0)
        return acts.numpy()

    def clear(self):
        """Clear collected activations"""
        for idx in self.activations:
            self.activations[idx] = []



def compute_joint_subspace(activations_dict, variance_threshold=0.95, min_dim=1, max_dim=2000):
    """
    Compute joint subspace from activations across multiple layers using PCA

    Args:
        activations_dict: dict mapping layer_idx -> numpy array of activations [n_samples, hidden_dim]
        variance_threshold: variance threshold for PCA
        min_dim: minimum subspace dimension
        max_dim: maximum subspace dimension

    Returns:
        joint_subspace: numpy array [hidden_dim, k] where k is the subspace dimension
        subspace_dim: int, the dimension of the joint subspace
    """
    # Stack activations from all layers
    all_activations = []
    for layer_idx, acts in activations_dict.items():
        if acts is not None and acts.shape[0] > 0:
            all_activations.append(acts)

    if not all_activations:
        return None, 0

    # Combine activations: [n_samples_total, hidden_dim]
    X_combined = np.vstack(all_activations)

    n_samples, hidden_dim = X_combined.shape

    # Check if we have enough samples for PCA
    if n_samples < 2:
        print(f"Warning: Only {n_samples} sample(s) available. Need at least 2 samples for PCA.")
        return None, 0

    # IMPORTANT: Convert to float64 to avoid overflow and improve numerical stability
    X_combined = X_combined.astype(np.float64)

    # Check for constant or near-constant features
    feature_stds = np.std(X_combined, axis=0, dtype=np.float64)
    constant_features = np.sum(feature_stds < 1e-12)
    if constant_features > 0:
        print(f"Warning: {constant_features}/{hidden_dim} features have near-zero variance (std < 1e-12)")
        print(f"  Removing constant features...")
        valid_features = feature_stds >= 1e-12
        X_combined = X_combined[:, valid_features]
        hidden_dim = X_combined.shape[1]
        if hidden_dim == 0:
            print(f"Error: All features are constant!")
            return None, 0

    # Check for NaN or Inf values
    if np.any(np.isnan(X_combined)) or np.any(np.isinf(X_combined)):
        print(f"Warning: Data contains NaN or Inf values. Replacing with 0...")
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

    # IMPORTANT: Center the data to avoid large mean values causing overflow
    X_mean = np.mean(X_combined, axis=0, dtype=np.float64)
    X_centered = X_combined - X_mean

    # Scale the data to prevent overflow in covariance computation
    # Use feature-wise scaling to maintain relative importance
    feature_scales = np.std(X_centered, axis=0, dtype=np.float64)
    # Avoid division by zero for features with zero std (though we removed them)
    feature_scales = np.where(feature_scales < 1e-12, 1.0, feature_scales)
    X_scaled = X_centered / feature_scales

    # Limit the number of components
    max_components = min(n_samples - 1, hidden_dim, max_dim)

    if max_components < 1:
        print(f"Warning: Cannot compute PCA with max_components={max_components}")
        return None, 0

    # Compute PCA with float64 precision
    try:
        pca = PCA(n_components=max_components, copy=True, whiten=False,
                  svd_solver='auto', tol=1e-12, iterated_power='auto',
                  random_state=None)
        pca.fit(X_scaled)
    except Exception as e:
        print(f"PCA fitting failed: {e}")
        print(f"  Trying with reduced components...")
        # Try with fewer components
        try:
            max_components = min(max_components, 100)
            pca = PCA(n_components=max_components)
            pca.fit(X_scaled)
        except Exception as e2:
            print(f"  PCA still failed: {e2}")
            return None, 0

    # Determine subspace dimension based on variance threshold
    if hasattr(pca, 'explained_variance_ratio_'):
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(cumsum >= variance_threshold) + 1

        if k == 0 or cumsum[-1] < variance_threshold:
            k = len(cumsum)
    else:
        # Fallback: use all components
        k = max_components

    # Apply dimension constraints
    k = max(min(k, max_dim, max_components), min_dim)

    # Ensure k doesn't exceed available components
    max_available_components = len(pca.components_) if hasattr(pca, 'components_') else 0
    if k > max_available_components:
        k = max_available_components

    if k == 0:
        print(f"Warning: No components selected!")
        return None, 0

    # Extract joint subspace basis from scaled space
    if hasattr(pca, 'components_'):
        # components_ are in scaled space: [k, hidden_dim]
        joint_subspace_scaled = pca.components_[:k].T  # [hidden_dim, k]

        # Transform back to original (centered) space
        joint_subspace_centered = joint_subspace_scaled * feature_scales.reshape(-1, 1)
    else:
        print(f"Warning: PCA has no components_ attribute")
        return None, 0

    # If we removed constant features, pad with zeros to restore original dimension
    if constant_features > 0:
        full_subspace = np.zeros((len(valid_features), k), dtype=np.float64)
        full_subspace[valid_features, :] = joint_subspace_centered
        joint_subspace = full_subspace
    else:
        joint_subspace = joint_subspace_centered

    # Compute explained variance for reporting
    explained_var = 0.0
    if hasattr(pca, 'explained_variance_ratio_') and k > 0 and k <= len(cumsum):
        explained_var = cumsum[k-1] * 100

    print(f"Joint subspace: {k}/{hidden_dim} dim ({k/hidden_dim*100:.1f}%), explains {explained_var:.1f}% variance")

    # IMPORTANT: Convert back to float32 for compatibility with PyTorch
    joint_subspace = joint_subspace.astype(np.float32)

    return joint_subspace, k



class JointSubspaceRemovalHook:
    """Hook to remove the joint subspace component from activations"""

    def __init__(self, layer_idx, joint_subspace, enabled=True, track_stats=False, eps=1e-6,
                 preserve_statistics=True):
        self.layer_idx = layer_idx
        self.joint_subspace_np = joint_subspace
        self.enabled = enabled
        self.track_stats = track_stats
        self.eps = eps
        self.preserve_statistics = preserve_statistics
        self.stats = {
            'original_variances': [],
            'removed_variances': [],
            'variance_ratios': []
        }

    def __call__(self, module, input, output):
        """Hook function that removes the joint subspace component"""
        if not self.enabled:
            return output

        # Get device and dtype from output
        if isinstance(output, tuple):
            device = output[0].device
            dtype = output[0].dtype
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            device = output.device
            dtype = output.dtype
            hidden_states = output
            other_outputs = ()

        # Convert joint subspace to match output dtype and device
        joint_subspace = torch.tensor(self.joint_subspace_np, dtype=dtype, device=device)

        # Store original shape
        original_shape = hidden_states.shape

        # Reshape to 2D: [batch*seq, hidden_dim]
        if len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_dim)
        elif len(hidden_states.shape) == 2:
            hidden_flat = hidden_states
        else:
            return output

        # 保存原始数据用于回退
        hidden_flat_original = hidden_flat.clone()

        # 记录原始统计信息
        with torch.no_grad():
            if self.preserve_statistics:
                original_mean = hidden_flat.mean(dim=0, keepdim=True)
                original_std = hidden_flat.std(dim=0, keepdim=True)

                # 避免除以0
                std_mask = original_std < self.eps
                safe_std = original_std.clone()
                safe_std[std_mask] = 1.0

                # 标准化输入
                hidden_normalized = (hidden_flat - original_mean) / safe_std

        # 确保联合子空间是正交的
        U = joint_subspace  # [hidden_dim, k]
        k = U.shape[1]

        # 使用SVD确保正交性（比QR更稳定）
        try:
            # 转换为float32进行SVD
            if U.dtype != torch.float32:
                U_fp32 = U.float()
            else:
                U_fp32 = U

            # 使用SVD确保正交性
            U_svd, S, Vh = torch.linalg.svd(U_fp32, full_matrices=False)
            # SVD已经确保U_svd是正交的
            if U.dtype != torch.float32:
                U_orth = U_svd.to(dtype=dtype)
            else:
                U_orth = U_svd
        except Exception as e:
            print(f"Warning: SVD failed: {e}")
            # 如果失败，使用原始U并添加小的随机噪声来避免奇异性
            U_orth = U + torch.randn_like(U) * self.eps

        # 检查U_orth是否正交
        with torch.no_grad():
            # 转换为float32检查正交性
            if U_orth.dtype != torch.float32:
                U_check = U_orth.float()
            else:
                U_check = U_orth

            ortho_check = U_check.T @ U_check
            expected_identity = torch.eye(ortho_check.shape[0], dtype=ortho_check.dtype, device=ortho_check.device)
            max_deviation = (ortho_check - expected_identity).abs().max().item()
            if max_deviation > 1e-3:
                print(f"  Warning: Joint subspace columns are not properly orthonormal! Max deviation: {max_deviation:.6f}")

        # 计算投影到联合子空间
        if self.preserve_statistics:
            projection_coeffs = hidden_normalized @ U_orth  # [batch*seq, k]
            # 计算联合子空间成分
            joint_subspace_component = projection_coeffs @ U_orth.T
            # 移除联合子空间成分
            orthogonal_complement = hidden_normalized - joint_subspace_component

            # with torch.no_grad():
            #     # 关键检查：残余信号在子空间上的投影是否接近零？
            #     residual_projection = orthogonal_complement @ U_orth  # [batch*seq, k]
            #     projection_norm = torch.norm(residual_projection, p=2, dim=1).mean()  # 平均L2范数
            #     projection_max = torch.norm(residual_projection, p=2, dim=1).max()    # 最大L2范数

            #     # 原始信号在子空间上的投影范数，作为对比基线
            #     original_projection_norm = torch.norm(projection_coeffs, p=2, dim=1).mean()

            #     print(f"[DEBUG-移除检查] 层 {self.layer_idx}:")
            #     print(f"  原始信号投影平均范数: {original_projection_norm.item():.6e}")
            #     print(f"  残余信号投影平均范数: {projection_norm.item():.6e}")
            #     print(f"  残余信号投影最大范数: {projection_max.item():.6e}")
            #     print(f"  移除比率(平均): {projection_norm.item()/(original_projection_norm.item()+1e-12):.6e}")

        else:
            projection_coeffs = hidden_flat @ U_orth  # [batch*seq, k]
            # 计算联合子空间成分
            joint_subspace_component = projection_coeffs @ U_orth.T
            # 移除联合子空间成分
            orthogonal_complement = hidden_flat - joint_subspace_component

        # 如果保持统计特性，需要恢复原始的统计信息
        if self.preserve_statistics:
            # 恢复原始的均值和方差
            orthogonal_complement = orthogonal_complement * safe_std + original_mean

            # 对于标准差为0的维度，恢复为原始值
            if std_mask.any():
                orthogonal_complement[:, std_mask.squeeze()] = hidden_flat_original[:, std_mask.squeeze()]

        # 检查数值问题
        if torch.isnan(orthogonal_complement).any() or torch.isinf(orthogonal_complement).any():
            print(f"Warning: NaN or Inf in orthogonal_complement for layer {self.layer_idx}")
            print(f"  Shape: {orthogonal_complement.shape}")
            print(f"  NaN count: {torch.isnan(orthogonal_complement).sum().item()}")
            print(f"  Inf count: {torch.isinf(orthogonal_complement).sum().item()}")

            # 尝试修复
            orthogonal_complement = torch.nan_to_num(orthogonal_complement, nan=0.0, posinf=0.0, neginf=0.0)

            # 如果修复失败，使用原始值
            if torch.isnan(orthogonal_complement).any() or torch.isinf(orthogonal_complement).any():
                print(f"  Critical: Could not fix NaN/Inf, using original activations")
                orthogonal_complement = hidden_flat_original

        # 恢复原始形状
        perturbed_states = orthogonal_complement.reshape(original_shape)

        # 返回扰动后的输出
        if isinstance(output, tuple):
            return (perturbed_states,) + other_outputs
        else:
            return perturbed_states


class RandomDimensionsRemovalHook:
    """Hook to remove random dimensions (fair comparison)"""

    def __init__(self, layer_idx, hidden_dim, subspace_dim, joint_subspace=None, seed=None,
                 enabled=True, preserve_statistics=True, eps=1e-6):
        self.layer_idx = layer_idx
        self.hidden_dim = hidden_dim
        self.subspace_dim = subspace_dim
        self.enabled = enabled
        self.preserve_statistics = preserve_statistics
        self.eps = eps

        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Find dimensions to remove
        all_dims = np.arange(hidden_dim)

        if joint_subspace is not None:
            dim_importance = np.sum(np.abs(joint_subspace), axis=1)
            sorted_indices = np.argsort(dim_importance)
            candidate_dims = sorted_indices[:hidden_dim - subspace_dim]

            if len(candidate_dims) >= subspace_dim:
                self.dims_to_remove = np.random.choice(candidate_dims, size=subspace_dim, replace=False)
            else:
                self.dims_to_remove = candidate_dims[:subspace_dim]

            joint_subspace_dims = np.where(dim_importance > np.percentile(dim_importance, 100 - subspace_dim/hidden_dim*100))[0]
            overlap = len(np.intersect1d(self.dims_to_remove, joint_subspace_dims))
            print(f"  Random removal: {subspace_dim} dims selected, {overlap} overlap with joint subspace")
        else:
            self.dims_to_remove = np.random.choice(all_dims, size=min(subspace_dim, hidden_dim), replace=False)
            print(f"  Random removal: {len(self.dims_to_remove)} dims selected (no joint subspace info)")

        self.dims_to_remove = torch.tensor(self.dims_to_remove, dtype=torch.long)

    def __call__(self, module, input, output):
        """Hook function that removes random dimensions (sets them to zero)"""
        if not self.enabled:
            return output

        # Handle tuple outputs
        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = ()

        # Store original shape
        original_shape = hidden_states.shape

        # Reshape to 2D: [batch*seq, hidden_dim]
        if len(hidden_states.shape) == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
            hidden_flat = hidden_states.reshape(-1, hidden_dim)
        elif len(hidden_states.shape) == 2:
            hidden_flat = hidden_states
        else:
            return output

        # 保存原始数据用于回退
        hidden_flat_original = hidden_flat.clone()

        # 记录原始统计信息
        with torch.no_grad():
            if self.preserve_statistics:
                original_mean = hidden_flat.mean(dim=0, keepdim=True)
                original_std = hidden_flat.std(dim=0, keepdim=True)

                # 避免除以0
                std_mask = original_std < self.eps
                safe_std = original_std.clone()
                safe_std[std_mask] = 1.0

                # 标准化输入
                hidden_normalized = (hidden_flat - original_mean) / safe_std

        # Move dims_to_remove to same device
        device = hidden_flat.device
        dims_to_remove = self.dims_to_remove.to(device)

        # Remove dimensions by setting them to zero
        if self.preserve_statistics:
            perturbed_flat = hidden_normalized.clone()
        else:
            perturbed_flat = hidden_flat.clone()

        perturbed_flat[:, dims_to_remove] = 0.0

        # 如果保持统计特性，需要恢复原始的统计信息
        if self.preserve_statistics:
            # 恢复原始的均值和方差
            perturbed_flat = perturbed_flat * safe_std + original_mean

            # 对于标准差为0的维度，恢复为原始值
            if std_mask.any():
                perturbed_flat[:, std_mask.squeeze()] = hidden_flat_original[:, std_mask.squeeze()]

        # 检查数值稳定性
        if torch.isnan(perturbed_flat).any() or torch.isinf(perturbed_flat).any():
            print(f"Warning: NaN or Inf in RandomDimensionsRemoval perturbed_flat")
            print(f"  Shape: {perturbed_flat.shape}")
            print(f"  NaN count: {torch.isnan(perturbed_flat).sum().item()}")
            print(f"  Inf count: {torch.isinf(perturbed_flat).sum().item()}")

            # 尝试修复
            perturbed_flat = torch.nan_to_num(perturbed_flat, nan=0.0, posinf=0.0, neginf=0.0)

            # 如果修复失败，使用原始值
            if torch.isnan(perturbed_flat).any() or torch.isinf(perturbed_flat).any():
                print(f"  Critical: Could not fix NaN/Inf, using original activations")
                perturbed_flat = hidden_flat_original

        # Restore original shape
        perturbed_states = perturbed_flat.reshape(original_shape)

        # Return perturbed output
        if isinstance(output, tuple):
            return (perturbed_states,) + other_outputs
        else:
            return perturbed_states


class SimpleNoisePerturbationHook:
    """Hook to add simple random noise directly to activations (baseline comparison)"""

    def __init__(self, noise_scale=0.1, enabled=True, preserve_statistics=True,
                 eps=1e-8, max_noise_ratio=0.5, clip_value=10.0):
        self.noise_scale = noise_scale
        self.enabled = enabled
        self.preserve_statistics = preserve_statistics
        self.eps = eps
        self.max_noise_ratio = max_noise_ratio
        self.clip_value = clip_value

    def __call__(self, module, input, output):
        if not self.enabled:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            hidden_states = output
            other_outputs = ()

        # 保存原始数据用于回退
        hidden_states_original = hidden_states.clone()

        try:
            # 存储原始形状
            original_shape = hidden_states.shape

            # 重塑为2D: [batch*seq, hidden_dim]
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_flat = hidden_states.reshape(-1, hidden_dim)
            elif len(hidden_states.shape) == 2:
                hidden_flat = hidden_states
            else:
                return output

            # 如果需要保持统计特性
            if self.preserve_statistics:
                with torch.no_grad():
                    # 计算均值和标准差，确保数值稳定性
                    original_mean = hidden_flat.mean(dim=0, keepdim=True)
                    original_std = hidden_flat.std(dim=0, keepdim=True)

                    # 避免除以0
                    std_mask = original_std < self.eps
                    safe_std = original_std.clone()
                    safe_std[std_mask] = 1.0

                    # 标准化输入
                    hidden_normalized = (hidden_flat - original_mean) / safe_std

                    # 确保没有NaN/Inf
                    if torch.isnan(hidden_normalized).any() or torch.isinf(hidden_normalized).any():
                        hidden_normalized = torch.nan_to_num(hidden_normalized, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)

                # 生成噪声 - 使用标准正态分布，限制幅度
                noise = torch.randn_like(hidden_normalized) * self.noise_scale

                # 控制噪声幅度，避免过大
                if self.max_noise_ratio > 0:
                    noise_norm = torch.norm(noise, dim=1, keepdim=True)
                    hidden_norm = torch.norm(hidden_normalized, dim=1, keepdim=True)
                    # 避免除以0
                    safe_hidden_norm = torch.where(hidden_norm < self.eps, torch.ones_like(hidden_norm), hidden_norm)
                    safe_noise_ratio = torch.clamp(noise_norm / (safe_hidden_norm + self.eps), max=self.max_noise_ratio)
                    noise = noise / (noise_norm + self.eps) * (safe_hidden_norm * safe_noise_ratio)

                perturbed_flat = hidden_normalized + noise

                # 恢复原始统计特性
                perturbed_flat = perturbed_flat * safe_std + original_mean

                # 对于标准差为0的维度，恢复为原始值
                if std_mask.any():
                    perturbed_flat[:, std_mask.squeeze()] = hidden_flat[:, std_mask.squeeze()]
            else:
                # 直接添加噪声
                noise = torch.randn_like(hidden_flat) * self.noise_scale

                # 控制噪声幅度
                if self.max_noise_ratio > 0:
                    noise_norm = torch.norm(noise, dim=1, keepdim=True)
                    hidden_norm = torch.norm(hidden_flat, dim=1, keepdim=True)
                    safe_hidden_norm = torch.where(hidden_norm < self.eps, torch.ones_like(hidden_norm), hidden_norm)
                    safe_noise_ratio = torch.clamp(noise_norm / (safe_hidden_norm + self.eps), max=self.max_noise_ratio)
                    noise = noise / (noise_norm + self.eps) * (safe_hidden_norm * safe_noise_ratio)

                perturbed_flat = hidden_flat + noise

            # 确保没有NaN/Inf
            if torch.isnan(perturbed_flat).any() or torch.isinf(perturbed_flat).any():
                perturbed_flat = torch.nan_to_num(perturbed_flat, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)

            # 恢复原始形状
            perturbed_states = perturbed_flat.reshape(original_shape)

        except Exception as e:
            print(f"Error in SimpleNoisePerturbationHook: {e}")
            print("  Falling back to original activations")
            perturbed_states = hidden_states_original

        if isinstance(output, tuple):
            return (perturbed_states,) + other_outputs
        else:
            return perturbed_states


class JointSubspacePerturbationHook:
    """Hook to perturb activations using joint subspace - 与JointSubspaceRemovalHook保持一致的实现"""

    def __init__(self, layer_idx, joint_subspace, noise_scale=0.1, enabled=True, eps=1e-8,
                 preserve_statistics=True, max_noise_ratio=0.3, clip_value=10.0):
        self.layer_idx = layer_idx
        self.joint_subspace_np = joint_subspace
        self.noise_scale = noise_scale
        self.enabled = enabled
        self.eps = eps
        self.preserve_statistics = preserve_statistics
        self.max_noise_ratio = max_noise_ratio
        self.clip_value = clip_value

    def __call__(self, module, input, output):
        """Hook function that perturbs activations using joint subspace - 与JointSubspaceRemovalHook保持一致"""
        if not self.enabled:
            return output

        # 获取设备、数据类型和激活值 - 与JointSubspaceRemovalHook完全相同的逻辑
        if isinstance(output, tuple):
            device = output[0].device
            dtype = output[0].dtype
            hidden_states = output[0]
            other_outputs = output[1:]
        else:
            device = output.device
            dtype = output.dtype
            hidden_states = output
            other_outputs = ()

        # 保存原始数据用于回退
        hidden_states_original = hidden_states.clone()

        try:
            # 转换联合子空间到合适的设备/数据类型 - 与JointSubspaceRemovalHook相同
            joint_subspace = torch.tensor(self.joint_subspace_np, dtype=dtype, device=device)

            # 存储原始形状 - 与JointSubspaceRemovalHook相同
            original_shape = hidden_states.shape

            # 重塑为2D: [batch*seq, hidden_dim] - 与JointSubspaceRemovalHook相同
            if len(hidden_states.shape) == 3:
                batch_size, seq_len, hidden_dim = hidden_states.shape
                hidden_flat = hidden_states.reshape(-1, hidden_dim)
            elif len(hidden_states.shape) == 2:
                hidden_flat = hidden_states
            else:
                return output

            # 保存原始数据用于回退 - 与JointSubspaceRemovalHook相同
            hidden_flat_original = hidden_flat.clone()

            # 记录原始统计信息 - 与JointSubspaceRemovalHook完全相同
            with torch.no_grad():
                if self.preserve_statistics:
                    original_mean = hidden_flat.mean(dim=0, keepdim=True)
                    original_std = hidden_flat.std(dim=0, keepdim=True)

                    # 避免除以0 - 与JointSubspaceRemovalHook相同
                    std_mask = original_std < self.eps
                    safe_std = original_std.clone()
                    safe_std[std_mask] = 1.0

                    # 标准化输入 - 与JointSubspaceRemovalHook相同
                    hidden_normalized = (hidden_flat - original_mean) / safe_std

            # 确保联合子空间是正交的 - 使用与JointSubspaceRemovalHook完全相同的SVD方法
            U = joint_subspace  # [hidden_dim, k]
            k = U.shape[1]

            # 使用SVD确保正交性（与JointSubspaceRemovalHook完全相同的实现）
            try:
                # 转换为float32进行SVD（与JointSubspaceRemovalHook相同）
                if U.dtype != torch.float32:
                    U_fp32 = U.float()
                else:
                    U_fp32 = U

                # 使用SVD确保正交性（与JointSubspaceRemovalHook相同）
                U_svd, S, Vh = torch.linalg.svd(U_fp32, full_matrices=False)
                # SVD已经确保U_svd是正交的
                if U.dtype != torch.float32:
                    U_orth = U_svd.to(dtype=dtype)
                else:
                    U_orth = U_svd
            except Exception as e:
                print(f"Warning: SVD failed: {e}")
                # 如果失败，使用原始U并添加小的随机噪声来避免奇异性（与JointSubspaceRemovalHook相同）
                U_orth = U + torch.randn_like(U) * self.eps

            # 检查U_orth是否正交（与JointSubspaceRemovalHook相同）
            with torch.no_grad():
                # 转换为float32检查正交性
                if U_orth.dtype != torch.float32:
                    U_check = U_orth.float()
                else:
                    U_check = U_orth

                ortho_check = U_check.T @ U_check
                expected_identity = torch.eye(ortho_check.shape[0], dtype=ortho_check.dtype, device=ortho_check.device)
                max_deviation = (ortho_check - expected_identity).abs().max().item()
                if max_deviation > 1e-3:
                    print(f"  Warning: Joint subspace columns are not properly orthonormal! Max deviation: {max_deviation:.6f}")

            # 计算投影到联合子空间 - 与JointSubspaceRemovalHook完全相同的逻辑
            if self.preserve_statistics:
                projection_coeffs = hidden_normalized @ U_orth  # [batch*seq, k]
                # 计算联合子空间成分
                joint_subspace_component = projection_coeffs @ U_orth.T
                # 移除联合子空间成分（得到正交补）
                orthogonal_complement = hidden_normalized - joint_subspace_component
            else:
                projection_coeffs = hidden_flat @ U_orth  # [batch*seq, k]
                # 计算联合子空间成分
                joint_subspace_component = projection_coeffs @ U_orth.T
                # 移除联合子空间成分（得到正交补）
                orthogonal_complement = hidden_flat - joint_subspace_component

            # 关键修改点：添加噪声到投影系数，然后重建扰动后的激活
            # 这是与JointSubspaceRemovalHook的唯一不同之处

            # 在投影系数上添加噪声
            noise = torch.randn_like(projection_coeffs) * self.noise_scale

            # 控制噪声幅度，确保不会太大
            if self.max_noise_ratio > 0:
                noise_norm = torch.norm(noise, dim=1, keepdim=True)
                projection_norm = torch.norm(projection_coeffs, dim=1, keepdim=True)
                # 避免除以0
                safe_projection_norm = torch.where(projection_norm < self.eps, torch.ones_like(projection_norm), projection_norm)
                safe_noise_ratio = torch.clamp(noise_norm / (safe_projection_norm + self.eps), max=self.max_noise_ratio)
                noise = noise / (noise_norm + self.eps) * (safe_projection_norm * safe_noise_ratio)

            # 扰动后的投影系数
            perturbed_projection_coeffs = projection_coeffs + noise

            # 重建扰动后的激活（正交补 + 扰动后的子空间成分）
            if self.preserve_statistics:
                # 使用扰动后的投影系数重建子空间成分
                perturbed_joint_component = perturbed_projection_coeffs @ U_orth.T
                # 重建扰动后的激活
                perturbed_reconstruction = orthogonal_complement + perturbed_joint_component
            else:
                perturbed_joint_component = perturbed_projection_coeffs @ U_orth.T
                perturbed_reconstruction = orthogonal_complement + perturbed_joint_component

            # 如果保持统计特性，需要恢复原始的统计信息 - 与JointSubspaceRemovalHook相同
            if self.preserve_statistics:
                # 恢复原始的均值和方差
                perturbed_reconstruction = perturbed_reconstruction * safe_std + original_mean

                # 对于标准差为0的维度，恢复为原始值
                if std_mask.any():
                    perturbed_reconstruction[:, std_mask.squeeze()] = hidden_flat[:, std_mask.squeeze()]

            # 检查数值问题 - 与JointSubspaceRemovalHook相同
            if torch.isnan(perturbed_reconstruction).any() or torch.isinf(perturbed_reconstruction).any():
                print(f"Warning: NaN or Inf in perturbed_reconstruction for layer {self.layer_idx}")
                print(f"  Shape: {perturbed_reconstruction.shape}")
                print(f"  NaN count: {torch.isnan(perturbed_reconstruction).sum().item()}")
                print(f"  Inf count: {torch.isinf(perturbed_reconstruction).sum().item()}")

                # 尝试修复
                perturbed_reconstruction = torch.nan_to_num(perturbed_reconstruction, nan=0.0, posinf=0.0, neginf=0.0)

                # 如果修复失败，使用原始值
                if torch.isnan(perturbed_reconstruction).any() or torch.isinf(perturbed_reconstruction).any():
                    print(f"  Critical: Could not fix NaN/Inf, using original activations")
                    perturbed_reconstruction = hidden_flat_original

            # 恢复原始形状 - 与JointSubspaceRemovalHook相同
            perturbed_states = perturbed_reconstruction.reshape(original_shape)

            # 返回扰动后的输出 - 与JointSubspaceRemovalHook相同
            if isinstance(output, tuple):
                return (perturbed_states,) + other_outputs
            else:
                return perturbed_states

        except Exception as e:
            print(f"Error in JointSubspacePerturbationHook for layer {self.layer_idx}: {e}")
            import traceback
            traceback.print_exc()
            print("  Falling back to original activations")
            perturbed_states = hidden_states_original

            # 返回扰动后的输出
            if isinstance(output, tuple):
                return (perturbed_states,) + other_outputs
            else:
                return perturbed_states



def collect_activations(model, tokenizer, texts, layer_indices, device=DEVICE, activation_strategy=ACTIVATION_STRATEGY):
    """Collect activations from specified layers"""
    collector = ActivationCollector(layer_indices, activation_strategy=activation_strategy)
    collector.register_hooks(model)

    model.eval()
    total_tokens = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Collecting activations"):
            batch_texts = texts[i:i + BATCH_SIZE]

            # Tokenize with debugging
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_attention_mask=True,
                return_length=True
            )

            # 打印调试信息
            lengths = inputs["length"]
            print(f"  Batch {i//BATCH_SIZE}: lengths = {lengths.tolist()}, total tokens = {sum(lengths.tolist())}")
            total_tokens += sum(lengths.tolist())

            inputs = {k: v.to(device) for k, v in inputs.items() if k != "length"}

            model(**inputs)

    collector.remove_hooks()

    # 打印总token数
    print(f"  Total tokens collected: {total_tokens}")

    return collector


def compute_loss(model, tokenizer, texts, hooks=None, device=DEVICE):
    """Compute average loss on texts with enhanced error handling"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    failed_batches = 0
    max_failed_batches = 3  # 最多允许失败的批次

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Computing loss"):
            batch_texts = texts[i:i + BATCH_SIZE]

            try:
                inputs = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_SEQ_LEN
                ).to(device)

                # 前向传播，确保使用稳定的数值
                with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度以确保稳定性
                    outputs = model(**inputs, labels=inputs["input_ids"])

                loss = outputs.loss

                # 检查loss是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Batch {i//BATCH_SIZE} has NaN/Inf loss: {loss.item()}")
                    failed_batches += 1
                    if failed_batches >= max_failed_batches:
                        print(f"Too many failed batches ({failed_batches}), skipping...")
                        break
                    continue

                num_tokens = inputs["attention_mask"].sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE}: {e}")
                failed_batches += 1
                if failed_batches >= max_failed_batches:
                    print(f"Too many failed batches ({failed_batches}), skipping...")
                    break
                continue

    if total_tokens == 0:
        print(f"Warning: No valid tokens processed, returning NaN")
        return float('nan')

    avg_loss = total_loss / total_tokens

    # 检查最终loss
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        print(f"Warning: Final loss is NaN/Inf: {avg_loss}")

    return avg_loss


def run_experiment_for_model(model_name, datasets):
    """Run the full experiment for a single model"""
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"{'='*80}")

    print(f"Loading model: {model_name}")

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return {}

    model = model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        layers, arch_type = get_model_layers(model)
    except Exception as e:
        print(f"Failed to get layers for {model_name}: {e}")
        return {}

    print(f"Architecture type: {arch_type}")
    print(f"Model has {len(layers)} layers")

    if len(layers) < max(LAYER_INDICES):
        target_layers = [int(idx * len(layers) / 12) for idx in LAYER_INDICES]
        target_layers = [min(l, len(layers) - 1) for l in target_layers]
        print(f"Adjusted target layers from {LAYER_INDICES} to {target_layers} (model has {len(layers)} layers)")
    else:
        target_layers = LAYER_INDICES

    valid_layers = [idx for idx in target_layers if idx < len(layers)]
    if not valid_layers:
        print(f"Warning: None of the specified layers {target_layers} are valid for {model_name}")
        return {}

    print(f"Target layers: {valid_layers}")

    results = {}

    for dataset_name, texts in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*50}")

        eval_texts = texts[:N_SAMPLES]
        print(f"Using {len(eval_texts)} samples for both subspace computation and evaluation")

        print(f"\nStep 1: Collecting activations from layers {valid_layers}...")
        print(f"  Using activation strategy: {ACTIVATION_STRATEGY}")
        collector = collect_activations(model, tokenizer, eval_texts, valid_layers, device=DEVICE, activation_strategy=ACTIVATION_STRATEGY)

        activations_dict = {}
        for layer_idx in valid_layers:
            acts = collector.get_activations(layer_idx)
            if acts is not None:
                activations_dict[layer_idx] = acts
                print(f"  Layer {layer_idx}: {acts.shape}")

        if not activations_dict:
            print(f"  No activations collected for {dataset_name}, skipping...")
            continue

        print(f"\nStep 2: Computing joint subspace...")
        joint_subspace, subspace_dim = compute_joint_subspace(
            activations_dict,
            variance_threshold=PCA_VARIANCE_THRESHOLD,
            min_dim=MIN_SUBSPACE_DIM,
            max_dim=MAX_SUBSPACE_DIM
        )

        if joint_subspace is None:
            print(f"  Failed to compute joint subspace for {dataset_name}, skipping...")
            continue

        print(f"\nStep 3: Computing baseline loss (no perturbation)...")
        baseline_loss = compute_loss(model, tokenizer, eval_texts, hooks=None, device=DEVICE)
        print(f"Baseline loss: {baseline_loss:.4f}")

        print(f"\nStep 4: Computing loss with simple noise perturbation (baseline comparison)...")

        simple_noise_hooks = []
        for layer_idx in valid_layers:
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            hook = SimpleNoisePerturbationHook(noise_scale=NOISE_SCALE, enabled=True, preserve_statistics=True)
            handle = layer.register_forward_hook(hook)
            simple_noise_hooks.append(handle)

        try:
            simple_noise_loss = compute_loss(model, tokenizer, eval_texts, hooks=simple_noise_hooks, device=DEVICE)
            print(f"Simple noise loss: {simple_noise_loss:.4f}")
        finally:
            for handle in simple_noise_hooks:
                handle.remove()

        print(f"\nStep 5: Computing loss with joint subspace REMOVAL...")

        removal_hooks = []
        for layer_idx in valid_layers:
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            hook = JointSubspaceRemovalHook(
                layer_idx, joint_subspace, enabled=True, track_stats=False, preserve_statistics=True
            )
            handle = layer.register_forward_hook(hook)
            removal_hooks.append(handle)

        try:
            removal_loss = compute_loss(model, tokenizer, eval_texts, hooks=removal_hooks, device=DEVICE)
            print(f"Joint subspace removal loss: {removal_loss:.4f}")
        finally:
            for handle in removal_hooks:
                handle.remove()

        print(f"\nStep 5b: Computing loss with RANDOM dimensions removal...")

        if hasattr(model.config, 'hidden_size'):
            hidden_dim = model.config.hidden_size
        else:
            hidden_dim = activations_dict[valid_layers[0]].shape[1]

        random_removal_hooks = []
        for layer_idx in valid_layers:
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            hook = RandomDimensionsRemovalHook(
                layer_idx, hidden_dim, subspace_dim,
                joint_subspace=joint_subspace, seed=layer_idx, enabled=True, preserve_statistics=True
            )
            handle = layer.register_forward_hook(hook)
            random_removal_hooks.append(handle)

        try:
            random_removal_loss = compute_loss(model, tokenizer, eval_texts, hooks=random_removal_hooks, device=DEVICE)
            print(f"Random subspace removal loss: {random_removal_loss:.4f}")
        finally:
            for handle in random_removal_hooks:
                handle.remove()

        print(f"\nStep 6: Computing loss with joint subspace perturbation (noise)...")

        joint_subspace_hooks = []
        for layer_idx in valid_layers:
            if layer_idx >= len(layers):
                continue
            layer = layers[layer_idx]
            hook = JointSubspacePerturbationHook(
                layer_idx, joint_subspace, noise_scale=NOISE_SCALE, enabled=True, preserve_statistics=True, max_noise_ratio=MAX_NOISE_RATIO
            )
            handle = layer.register_forward_hook(hook)
            joint_subspace_hooks.append(handle)

        try:
            joint_subspace_loss = compute_loss(model, tokenizer, eval_texts, hooks=joint_subspace_hooks, device=DEVICE)
            print(f"Joint subspace perturbation loss: {joint_subspace_loss:.4f}")

            simple_noise_diff = simple_noise_loss - baseline_loss
            removal_diff = removal_loss - baseline_loss
            random_removal_diff = random_removal_loss - baseline_loss
            joint_subspace_diff = joint_subspace_loss - baseline_loss

            simple_noise_ratio = simple_noise_loss / baseline_loss if baseline_loss > 0 else float('inf')
            removal_ratio = removal_loss / baseline_loss if baseline_loss > 0 else float('inf')
            random_removal_ratio = random_removal_loss / baseline_loss if baseline_loss > 0 else float('inf')
            joint_subspace_ratio = joint_subspace_loss / baseline_loss if baseline_loss > 0 else float('inf')

            print(f"\nResults for {dataset_name}:")
            print(f"  Baseline loss:              {baseline_loss:.4f}")
            print(f"  Simple noise loss:          {simple_noise_loss:.4f} "
                  f"(+{simple_noise_diff:.4f}, {simple_noise_diff/baseline_loss*100:.2f}% increase)")
            print(f"  Joint subspace REMOVAL:     {removal_loss:.4f} "
                  f"(+{removal_diff:.4f}, {removal_diff/baseline_loss*100:.2f}% increase)")
            print(f"  Random dimensions REMOVAL:  {random_removal_loss:.4f} "
                  f"(+{random_removal_diff:.4f}, {random_removal_diff/baseline_loss*100:.2f}% increase)")
            print(f"  Joint subspace perturbation: {joint_subspace_loss:.4f} "
                  f"(+{joint_subspace_diff:.4f}, {joint_subspace_diff/baseline_loss*100:.2f}% increase)")
            print(f"  Joint subspace dim:         {subspace_dim}")

            if simple_noise_loss > 0:
                relative_perturbation = (joint_subspace_loss - simple_noise_loss) / simple_noise_loss * 100
                print(f"  Perturbation vs noise:     Joint subspace perturbation is {relative_perturbation:.2f}% "
                      f"{'worse' if relative_perturbation > 0 else 'better'} than simple noise")

            if random_removal_loss > 0:
                relative_joint_vs_random = (removal_loss - random_removal_loss) / random_removal_loss * 100
                print(f"  Joint vs Random removal:   Joint subspace removal is {relative_joint_vs_random:.2f}% "
                      f"{'worse' if relative_joint_vs_random > 0 else 'better'} than random dimensions removal")

                if abs(relative_joint_vs_random) > 5.0:
                    print(f"  → Joint subspace removal is {'significantly worse' if relative_joint_vs_random > 0 else 'significantly better'} than random dimensions removal")
                else:
                    print(f"  → Joint subspace removal has similar impact as random dimensions removal")

            results[dataset_name] = {
                "baseline_loss": baseline_loss,
                "simple_noise_loss": simple_noise_loss,
                "removal_loss": removal_loss,
                "random_removal_loss": random_removal_loss,
                "joint_subspace_loss": joint_subspace_loss,
                "simple_noise_diff": simple_noise_diff,
                "removal_diff": removal_diff,
                "random_removal_diff": random_removal_diff,
                "joint_subspace_diff": joint_subspace_diff,
                "simple_noise_ratio": simple_noise_ratio,
                "removal_ratio": removal_ratio,
                "random_removal_ratio": random_removal_ratio,
                "joint_subspace_ratio": joint_subspace_ratio,
                "subspace_dim": subspace_dim
            }
        finally:
            for handle in joint_subspace_hooks:
                handle.remove()

    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Baseline:                    {result['baseline_loss']:.4f}")
        print(f"  Simple noise:                {result['simple_noise_loss']:.4f} "
              f"(+{result['simple_noise_diff']/result['baseline_loss']*100:.2f}%)")
        print(f"  Joint subspace REMOVAL:      {result['removal_loss']:.4f} "
              f"(+{result['removal_diff']/result['baseline_loss']*100:.2f}%)")
        print(f"  Random dimensions REMOVAL:  {result['random_removal_loss']:.4f} "
              f"(+{result['random_removal_diff']/result['baseline_loss']*100:.2f}%)")
        print(f"  Joint subspace perturbation:  {result['joint_subspace_loss']:.4f} "
              f"(+{result['joint_subspace_diff']/result['baseline_loss']*100:.2f}%)")
        print(f"  Joint subspace dim:           {result['subspace_dim']}")

    del model
    del tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    """Main function that runs experiments for all models"""
    print(f"Using device: {DEVICE}")

    print("\n" + "="*50)
    print("Loading datasets...")

    if DATASETS_TO_RUN is None:
        datasets_to_load = list(DATASET_LOADERS.keys())
    else:
        datasets_to_load = [d.lower() for d in DATASETS_TO_RUN]
        invalid = [d for d in datasets_to_load if d not in DATASET_LOADERS]
        if invalid:
            print(f"Warning: Invalid dataset names: {invalid}")
            print(f"Available datasets: {list(DATASET_LOADERS.keys())}")
            datasets_to_load = [d for d in datasets_to_load if d in DATASET_LOADERS]

    if not datasets_to_load:
        raise ValueError("No valid datasets to load!")

    print(f"Loading {len(datasets_to_load)} dataset(s): {datasets_to_load}")

    datasets = {}
    for dataset_name in datasets_to_load:
        try:
            loader = DATASET_LOADERS[dataset_name]
            texts = loader(N_SAMPLES)
            datasets[dataset_name] = texts
            print(f"  ✓ {dataset_name}: {len(texts)} samples")
        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {e}")
            print(f"    Skipping {dataset_name}...")

    if not datasets:
        raise ValueError("No datasets were successfully loaded!")

    print(f"\nSuccessfully loaded {len(datasets)} dataset(s)")

    for model_name in MODELS_TO_RUN:
        try:
            print(f"\n{'='*80}")
            print(f"Starting experiment for: {model_name}")
            print(f"{'='*80}")
            run_experiment_for_model(model_name, datasets)
            print(f"\n{'='*80}")
            print(f"Completed experiment for: {model_name}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"\nError running experiment for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping {model_name} and continuing with next model...")
            continue

    print("\nAll experiments completed!")


if __name__ == "__main__":
    results = main()