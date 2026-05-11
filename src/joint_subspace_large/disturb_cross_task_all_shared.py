"""
disturb_cross_task_all_shared.py (更完整，更focused，只专注于fully shared basis):
Focuses on finding truly shared basis vectors across ALL tasks
Implements Step 5.5 analysis: identifying basis vectors that are consistently important across multiple tasks
Uses statistical testing to determine if shared subspaces are significantly more important than random subspaces

- Statistical rigor: Multiple trials (N_STATISTICAL_TRIALS=10) with hypothesis testing
- Domain analysis: Tests same-domain (reasoning tasks) vs cross-domain (math + language)
- Comprehensive evaluation: 32+32+32+32+32 = 160 samples across 5 tasks

Question Answered:
1. Are there truly shared representations across reasoning tasks?
2. Do shared subspaces hurt performance on untrained tasks (generalization)?
3. Statistical significance of shared vs random subspaces?

NOTE (FIX):
- Added robust get_model_layers() that supports Gemma3ForConditionalGeneration (and many other HF model wrappers)
  by (1) checking known attribute paths and (2) falling back to an automatic ModuleList search.
- Also made hidden_dim inference robust (using model.config) to avoid hard-coding 4096 for non-LLaMA models.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer
# from scipy import stats  # Not available, using numpy-based implementation
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
MAX_SEQ_LEN = 128 * 2
N_SAMPLES = 128 * 4  # Number of samples to use for both subspace computation and evaluation
MAX_SAMPLES_PER_TASK = 512  # Maximum samples per task (will be adjusted to satisfy samples * seq_len > hidden_dim)

# NOTE: Keep as a fallback default. Actual hidden_dim will be inferred from model.config when model is loaded.
HIDDEN_DIM = 4096  # Default hidden dimension (4096 for Llama-2-7b, adjust for other models)

N_STATISTICAL_TRIALS = 30  # Number of trials for statistical hypothesis testing
NOISE_SCALE = 1.0  # Scale of noise to add to joint subspace
MAX_NOISE_RATIO = 0.1  # 最大噪声比例
CLIP_VALUE = 10.0  # 新增：裁剪值
LAYER_INDICES = [8]  # Layers to collect activations from
PCA_VARIANCE_THRESHOLD = 0.95  # Variance threshold for PCA
MIN_SUBSPACE_DIM = 1  # Minimum subspace dimension
MAX_SUBSPACE_DIM = 4096
ACTIVATION_STRATEGY = "all_tokens"  # "last_token", "mean", "max", "all_tokens"
EPS = 1e-8  # 更小的epsilon用于数值稳定性

# Datasets to evaluate (set to None to use all available)
DATASETS_TO_RUN = None  # ["strategyqa", "aqua"]  # Test on these two reasoning benchmarks


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
            except Exception:
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


def find_fully_shared_basis_improved(
    task_contributions,
    all_tasks,
    cross_subspace_dim,
    min_tasks_shared=None,
    relative_threshold=0.001,
    top_k_components=None,
):
    """
    改进版：寻找共享basis，支持不同阈值和最少任务数要求
    """
    if min_tasks_shared is None:
        min_tasks_shared = len(all_tasks)  # 默认为所有任务

    if top_k_components is None:
        top_k_components = cross_subspace_dim  # 默认检查所有成分

    print(f"寻找被至少{min_tasks_shared}个任务共享的基向量...")
    print(f"  相对阈值: 方差贡献 > 任务总方差的{relative_threshold*100:.1f}%")
    print(f"  考虑前{min(top_k_components, cross_subspace_dim)}个最重要的成分")

    # 计算每个任务的总方差
    task_total_variances = {}
    for task_name in all_tasks:
        if task_name in task_contributions and "total_variance" in task_contributions[task_name]:
            task_total_variances[task_name] = task_contributions[task_name]["total_variance"]
        elif task_name in task_contributions and "raw_variances" in task_contributions[task_name]:
            task_total_variances[task_name] = np.sum(task_contributions[task_name]["raw_variances"])
        else:
            print(f"  警告: 任务 {task_name} 没有方差数据")
            return []

    # 为每个成分记录哪些任务有显著贡献
    component_task_significance = {}

    # 只考虑最重要的top_k_components个成分
    num_components_to_check = min(cross_subspace_dim, top_k_components)

    for comp_idx in range(num_components_to_check):
        significant_tasks = []

        for task_name in all_tasks:
            if (
                task_name in task_contributions
                and "raw_variances" in task_contributions[task_name]
                and len(task_contributions[task_name]["raw_variances"]) > comp_idx
            ):
                variance = task_contributions[task_name]["raw_variances"][comp_idx]
                total_variance = task_total_variances.get(task_name, 1.0)

                # 使用相对阈值
                if total_variance > 0 and variance > total_variance * relative_threshold:
                    significant_tasks.append(task_name)

        if significant_tasks:
            component_task_significance[comp_idx] = {"tasks": significant_tasks, "count": len(significant_tasks)}

    # 找出满足最少任务数要求的成分
    shared_indices = []
    for comp_idx, info in component_task_significance.items():
        if info["count"] >= min_tasks_shared:
            shared_indices.append(comp_idx)

    print(f"  发现 {len(shared_indices)}/{num_components_to_check} 个被至少{min_tasks_shared}个任务共享的成分")

    if shared_indices:
        # 分析共享basis的特征
        print(f"  共享basis分析:")

        # 计算每个共享basis的平均方差贡献
        basis_info = []
        for idx in shared_indices:
            avg_var = 0
            avg_rel = 0
            task_counts = 0

            for task_name in all_tasks:
                if (
                    task_name in task_contributions
                    and "raw_variances" in task_contributions[task_name]
                    and len(task_contributions[task_name]["raw_variances"]) > idx
                ):
                    var = task_contributions[task_name]["raw_variances"][idx]
                    total_var = task_total_variances.get(task_name, 1.0)

                    if total_var > 0:
                        avg_var += var
                        avg_rel += var / total_var
                        task_counts += 1

            if task_counts > 0:
                avg_var /= task_counts
                avg_rel /= task_counts

                # 统计这个basis被多少个任务共享
                shared_tasks = component_task_significance[idx]["tasks"]
                missing_tasks = [t for t in all_tasks if t not in shared_tasks]

                basis_info.append(
                    {
                        "idx": idx,
                        "avg_variance": avg_var,
                        "avg_relative": avg_rel,
                        "shared_task_count": len(shared_tasks),
                        "missing_tasks": missing_tasks,
                    }
                )

        # 按平均方差排序
        basis_info.sort(key=lambda x: x["avg_variance"], reverse=True)

        # 打印前10个共享basis
        if len(basis_info) > 0:
            print(f"  前{min(10, len(basis_info))}个共享basis:")
            for info in basis_info[:10]:
                print(f"    成分#{info['idx']}: 被{info['shared_task_count']}个任务共享，缺少{info['missing_tasks']}")
                print(f"      平均方差贡献: {info['avg_variance']:.4e}")
                print(f"      平均相对贡献: {info['avg_relative']:.4f}")
        else:
            print("  没有找到共享basis")

    return shared_indices


# Dataset loader registry
DATASET_LOADERS = {
    "wikitext": load_wikitext_data,
    "gsm8k": load_gsm8k_data,
    "commonsenseqa": load_commonsenseqa_data,
    "strategyqa": load_strategyqa_data,
    "aqua": load_aqua_data,
}


def calculate_optimal_samples(max_samples, seq_len, hidden_dim, min_samples=10):
    """
    Calculate optimal number of samples ensuring samples * seq_len > hidden_dim
    for sufficient data for PCA computation.

    Args:
        max_samples: Maximum samples available or desired
        seq_len: Sequence length (MAX_SEQ_LEN)
        hidden_dim: Hidden dimension of the model
        min_samples: Minimum samples required

    Returns:
        Optimal number of samples satisfying the constraint
    """
    # Minimum samples needed to satisfy samples * seq_len > hidden_dim
    min_required = (hidden_dim // seq_len) + 1

    # Ensure we have at least the minimum required
    optimal_samples = max(min_required, min_samples)

    # But don't exceed the maximum available/desired
    optimal_samples = min(optimal_samples, max_samples)

    return optimal_samples


def perform_hypothesis_test(baseline_losses, treatment_losses, test_name="test", alpha=0.05):
    """
    Perform statistical hypothesis test to determine if treatment significantly affects performance.
    Uses numpy-based implementation instead of scipy.

    Args:
        baseline_losses: List of baseline loss measurements
        treatment_losses: List of treatment loss measurements
        test_name: Name of the test for reporting
        alpha: Significance level

    Returns:
        Dict with test results including p-value approximation, significance, effect size
    """
    baseline_losses = np.array(baseline_losses)
    treatment_losses = np.array(treatment_losses)

    n1, n2 = len(baseline_losses), len(treatment_losses)

    if n1 < 2 or n2 < 2:
        # Not enough samples for meaningful statistical test
        mean_diff = np.mean(treatment_losses) - np.mean(baseline_losses)
        return {
            "test_type": "insufficient_samples",
            "is_significant": False,
            "mean_difference": mean_diff,
            "p_value": 1.0,
            "alpha": alpha,
            "n_baseline": n1,
            "n_treatment": n2,
        }

    # Calculate basic statistics
    mean1, std1 = np.mean(baseline_losses), np.std(baseline_losses, ddof=1)
    mean2, std2 = np.mean(treatment_losses), np.std(treatment_losses, ddof=1)
    mean_diff = mean2 - mean1

    # Calculate pooled standard deviation for effect size
    if n1 + n2 - 2 > 0:
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
    else:
        effect_size = 0

    # Handle cases with no variation (all samples identical)
    if std1 == 0 and std2 == 0:
        # No variation in either group
        if abs(mean_diff) < 1e-10:
            # Means are essentially identical - no difference
            effect_size = 0
            t_stat = 0
            p_value = 1.0
            test_type = "no_variation_identical_means"
        else:
            # Means differ but no variation - perfect separation
            effect_size = float("inf") if mean_diff > 0 else float("-inf")
            t_stat = float("inf") if mean_diff > 0 else float("-inf")
            p_value = 0.0
            test_type = "no_variation_different_means"
    elif std1 == 0 or std2 == 0:
        # One group has no variation
        effect_size = float("inf") if mean_diff > 0 else float("-inf")
        t_stat = float("inf") if mean_diff > 0 else float("-inf")
        p_value = 0.0
        test_type = "one_group_no_variation"
    else:
        # Normal case with variation in both groups
        # Simple t-test approximation (Welch's t-test for unequal variances)
        se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
        t_stat = mean_diff / se_diff if se_diff > 0 else 0

        # Approximate degrees of freedom (Welch-Satterthwaite equation)
        df_numerator = (std1**2 / n1 + std2**2 / n2) ** 2
        df_denominator = (std1**2 / n1) ** 2 / (n1 - 1) + (std2**2 / n2) ** 2 / (n2 - 1)
        df = df_numerator / df_denominator if df_denominator > 0 else min(n1, n2) - 1

        # Better p-value approximation for t-distribution
        if abs(t_stat) > 10:  # For very large t-statistics, p-value is essentially 0
            p_value = 0.0
        elif df > 30:
            # Use normal approximation for large df
            p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - np.exp(-2 * t_stat**2 / np.pi))))
        else:
            # For small df, use a more conservative approach
            if abs(t_stat) < 2:
                p_value = 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - np.exp(-2 * t_stat**2 / np.pi))))
            else:
                p_value = min(
                    0.1, 2 * (1 - 0.5 * (1 + np.sign(t_stat) * np.sqrt(1 - np.exp(-2 * t_stat**2 / np.pi))))
                )

        test_type = "approximate_t_test"

    # Clamp p-value to valid range
    p_value = np.clip(p_value, 0.0, 1.0)
    is_significant = p_value < alpha

    # Confidence interval approximation
    if n1 >= 2 and n2 >= 2:
        se_diff = np.sqrt(std1**2 / n1 + std2**2 / n2)
        ci_margin = 1.96 * se_diff  # 95% CI
        ci_lower = mean_diff - ci_margin
        ci_upper = mean_diff + ci_margin
    else:
        ci_lower = ci_upper = mean_diff

    result = {
        "test_type": "approximate_t_test",
        "t_statistic": t_stat,
        "p_value": p_value,
        "is_significant": is_significant,
        "alpha": alpha,
        "mean_difference": mean_diff,
        "effect_size": effect_size,
        "confidence_interval": (ci_lower, ci_upper),
        "baseline_mean": mean1,
        "baseline_std": std1,
        "treatment_mean": mean2,
        "treatment_std": std2,
        "n_baseline": n1,
        "n_treatment": n2,
    }

    # Print results
    print(f"\n📊 Hypothesis Test: {test_name}")
    print(f"  Test type: {test_type}")
    print(f"  Sample sizes: baseline={n1}, treatment={n2}")
    print(f"  Means: baseline={mean1:.4f}±{std1:.4f}, treatment={mean2:.4f}±{std2:.4f}")
    print(f"  Mean difference: {mean_diff:+.4f} (95% CI: {ci_lower:+.4f}, {ci_upper:+.4f})")
    print(f"  Effect size (Cohen's d): {effect_size:+.3f}")
    print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    print(f"  Significant at α={alpha}: {'YES' if is_significant else 'NO'}")
    if is_significant:
        direction = "worse" if mean_diff > 0 else "better"
        print(f"  → Treatment significantly {direction} than baseline (p < {alpha})")
    else:
        print(f"  → No significant difference between treatment and baseline (p ≥ {alpha})")

    return result


# =========================
# NEW: robust layer finding
# =========================
def _get_attr_chain(obj, chain):
    """Safely resolve obj.a.b.c ... returning None if any missing."""
    cur = obj
    for attr in chain:
        if cur is None or not hasattr(cur, attr):
            return None
        cur = getattr(cur, attr)
    return cur


def _auto_find_transformer_layers(model):
    """
    Fallback: automatically locate the most likely transformer block ModuleList inside a HF model.
    Returns (module_list, name) or (None, None).
    """
    candidates = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList):
            continue
        if len(module) == 0:
            continue

        lname = name.lower()

        # Skip obvious non-block lists
        if any(k in lname for k in ["embed", "embedding", "token", "position", "rotary", "vision", "image", "patch"]):
            continue

        elem = module[0]
        elem_name = elem.__class__.__name__.lower()

        # Heuristic: transformer blocks usually contain attention and/or mlp
        has_attn = any(hasattr(elem, k) for k in ["self_attn", "attn", "attention", "self_attention"])
        has_mlp = any(hasattr(elem, k) for k in ["mlp", "ffn", "feed_forward", "feedforward", "dense_h_to_4h"])

        # Accept if it looks like a block/layer or has attn/mlp
        if not (has_attn or has_mlp or ("block" in elem_name) or ("layer" in elem_name) or ("decoder" in elem_name)):
            continue

        # Scoring to prefer decoder/layers over encoder/others
        score = 0
        score += len(module) * 10

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

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0]
    return best[3], best[2]


def infer_hidden_dim_from_model(model, fallback=HIDDEN_DIM):
    """
    Infer hidden dimension robustly from model.config (works across many HF architectures).
    """
    cfg = getattr(model, "config", None)
    if cfg is not None:
        # Common config fields
        for attr in ["hidden_size", "n_embd", "d_model", "model_dim", "dim", "hidden_dim"]:
            if hasattr(cfg, attr):
                val = getattr(cfg, attr)
                if isinstance(val, int) and val > 0:
                    return val

        # Nested text configs (some multi-modal wrappers)
        for nested in ["text_config", "language_config", "decoder_config"]:
            if hasattr(cfg, nested):
                sub = getattr(cfg, nested)
                for attr in ["hidden_size", "n_embd", "d_model", "dim", "hidden_dim"]:
                    if hasattr(sub, attr):
                        val = getattr(sub, attr)
                        if isinstance(val, int) and val > 0:
                            return val

    return fallback


def get_model_layers(model):
    """
    Get the transformer block layers from a model, handling different architectures.

    FIX:
    - Adds support for Gemma3ForConditionalGeneration and other wrapped models by checking
      common wrapper paths (language_model/text_model) and by auto-searching ModuleLists.
    """
    # 1) Known explicit paths (fast + reliable)
    known_paths = [
        (("model", "decoder", "layers"), "opt"),
        (("model", "layers"), "llama_like"),  # llama/mistral/gemma/gemma2 often land here
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

    # 2) Wrapped language model paths (Gemma3ForConditionalGeneration-like, multi-modal, etc.)
    wrapped_paths = [
        (("language_model", "model", "layers"), "wrapped_language_model"),
        (("language_model", "layers"), "wrapped_language_model"),
        (("model", "language_model", "model", "layers"), "model_language_model"),
        (("model", "language_model", "layers"), "model_language_model"),
        (("text_model", "model", "layers"), "wrapped_text_model"),
        (("text_model", "layers"), "wrapped_text_model"),
        (("model", "text_model", "model", "layers"), "model_text_model"),
        (("model", "text_model", "layers"), "model_text_model"),
        # Some wrappers nest another "language_model"
        (("model", "language_model", "language_model", "model", "layers"), "double_wrapped_language_model"),
        (("language_model", "language_model", "model", "layers"), "double_wrapped_language_model"),
    ]

    for chain, arch in wrapped_paths:
        layers = _get_attr_chain(model, chain)
        if layers is not None:
            return layers, arch

    # 3) Generic fallback: scan for plausible ModuleList of transformer blocks
    layers, name = _auto_find_transformer_layers(model)
    if layers is not None:
        print(f"[get_model_layers] Auto-detected transformer layers at: {name} (len={len(layers)})")
        return layers, f"auto:{name}"

    # 4) Give a helpful error with model type
    raise ValueError(f"Could not find layers in model. Model type: {type(model)}")


class CrossTaskActivationCollector:
    """Collect activations from multiple tasks for cross-task subspace computation"""

    def __init__(self, layer_indices, activation_strategy="all_tokens"):
        self.layer_indices = layer_indices
        self.activations = {idx: [] for idx in layer_indices}
        self.task_activations = {}  # 存储每个任务的激活 {task_name: {layer_idx: np.array}}
        self.hooks = []
        self.activation_strategy = activation_strategy
        self.current_task = None

    def set_current_task(self, task_name):
        """Set current task name for activation tagging"""
        self.current_task = task_name
        if task_name not in self.task_activations:
            self.task_activations[task_name] = {idx: [] for idx in self.layer_indices}

    def create_hook(self, layer_idx):
        """Create a hook function for a specific layer"""

        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Apply activation strategy
            if self.activation_strategy == "all_tokens":
                if len(hidden_states.shape) == 3:
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    #act = hidden_states.view(-1, hidden_dim).detach().cpu()
                    act = hidden_states.reshape(-1, hidden_dim).detach().cpu()
                    # 或者 hidden_states.contiguous().view(-1, hidden_dim)
                elif len(hidden_states.shape) == 2:
                    act = hidden_states.detach().cpu()
                else:
                    return
            else:
                # 简化处理，跨任务实验主要用all_tokens
                if len(hidden_states.shape) == 3:
                    act = hidden_states[:, -1, :].detach().cpu()
                else:
                    act = hidden_states.detach().cpu()

            # Store in global activations
            self.activations[layer_idx].append(act)

            # Store in task-specific activations
            if self.current_task is not None:
                self.task_activations[self.current_task][layer_idx].append(act)

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

    def get_global_activations(self, layer_idx):
        """Get combined activations from all tasks as numpy array"""
        if layer_idx not in self.activations:
            return None
        if not self.activations[layer_idx]:
            return None
        acts = torch.cat(self.activations[layer_idx], dim=0)
        return acts.numpy()

    def get_task_activations(self, task_name, layer_idx):
        """Get activations for a specific task"""
        if task_name not in self.task_activations:
            return None
        if layer_idx not in self.task_activations[task_name]:
            return None
        if not self.task_activations[task_name][layer_idx]:
            return None
        acts = torch.cat(self.task_activations[task_name][layer_idx], dim=0)
        return acts.numpy()

    def clear(self):
        """Clear all collected activations"""
        for idx in self.activations:
            self.activations[idx] = []
        for task in self.task_activations:
            for idx in self.task_activations[task]:
                self.task_activations[task][idx] = []


def compute_cross_task_subspace(
    task_activations_dict, variance_threshold=0.95, min_dim=1, max_dim=2000, return_full_pca=False
):
    """
    Compute joint subspace from activations across MULTIPLE TASKS using PCA

    Args:
        task_activations_dict: dict mapping task_name -> dict mapping layer_idx -> np.array
        variance_threshold: variance threshold for PCA
        min_dim: minimum subspace dimension
        max_dim: maximum subspace dimension

    Returns:
        cross_task_subspace: numpy array [hidden_dim, k] where k is the subspace dimension
        subspace_dim: int, the dimension of the cross-task subspace
        task_variance_contributions: dict mapping task_name -> list of variance contributions
    """
    # Stack activations from all tasks and all layers
    all_activations = []
    task_sample_counts = {}
    task_start_indices = {}

    print(f"\nCombining activations from {len(task_activations_dict)} tasks...")

    current_idx = 0
    for task_name, layer_activations in task_activations_dict.items():
        task_activations = []
        for layer_idx, acts in layer_activations.items():
            if acts is not None and acts.shape[0] > 0:
                task_activations.append(acts)

        if task_activations:
            # Combine activations from all layers within this task
            X_task = np.vstack(task_activations)
            all_activations.append(X_task)
            task_sample_counts[task_name] = X_task.shape[0]
            task_start_indices[task_name] = current_idx
            current_idx += X_task.shape[0]
            print(f"  {task_name}: {X_task.shape[0]} samples")

    if not all_activations:
        return None, 0, {} if not return_full_pca else (None, 0, {}, {})

    # Combine all task activations: [n_samples_total, hidden_dim]
    X_combined = np.vstack(all_activations)
    n_samples_total, hidden_dim = X_combined.shape

    print(f"\nTotal samples for cross-task PCA: {n_samples_total}")

    # Convert to float64 for numerical stability
    X_combined = X_combined.astype(np.float64)

    # Center and scale the data
    X_mean = np.mean(X_combined, axis=0, dtype=np.float64)
    X_centered = X_combined - X_mean
    feature_scales = np.std(X_centered, axis=0, dtype=np.float64)
    feature_scales = np.where(feature_scales < 1e-12, 1.0, feature_scales)
    X_scaled = X_centered / feature_scales

    # Limit the number of components
    max_components = min(n_samples_total - 1, hidden_dim, max_dim)

    # Compute PCA
    print("Computing cross-task PCA...")
    try:
        pca = PCA(n_components=max_components)
        pca.fit(X_scaled)
    except Exception as e:
        print(f"Cross-task PCA failed: {e}")
        return None, 0, {} if not return_full_pca else (None, 0, {}, {})

    # Determine subspace dimension based on variance threshold
    if hasattr(pca, "explained_variance_ratio_"):
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        k = np.argmax(cumsum >= variance_threshold) + 1
        if k == 0 or cumsum[-1] < variance_threshold:
            k = len(cumsum)
    else:
        k = max_components
        cumsum = np.zeros(max_components, dtype=np.float64)

    # Apply dimension constraints
    k = max(min(k, max_dim, max_components), min_dim)

    # Extract subspace basis
    if hasattr(pca, "components_"):
        joint_subspace_scaled = pca.components_[:k].T
        joint_subspace_centered = joint_subspace_scaled * feature_scales.reshape(-1, 1)
        cross_task_subspace = joint_subspace_centered.astype(np.float32)
    else:
        print("Warning: PCA has no components_ attribute")
        return None, 0, {} if not return_full_pca else (None, 0, {}, {})

    # Compute task-specific variance contributions
    task_variance_contributions = {}
    if hasattr(pca, "components_") and k > 0:
        print("\nAnalyzing task contributions to cross-task subspace...")
        for task_name, start_idx in task_start_indices.items():
            task_count = task_sample_counts[task_name]
            end_idx = start_idx + task_count

            # Get task data in scaled space
            X_task_scaled = X_scaled[start_idx:end_idx, :]

            # Project task data onto cross-task subspace
            projection = X_task_scaled @ joint_subspace_scaled  # [task_samples, k]

            # Compute variance explained by each component for this task
            task_variances = np.var(projection, axis=0)
            total_task_variance = np.sum(task_variances)

            # Normalize by total variance in the subspace
            if total_task_variance > 0:
                normalized_contributions = task_variances / total_task_variance
            else:
                normalized_contributions = np.zeros(k)

            task_variance_contributions[task_name] = {
                "raw_variances": task_variances,
                "normalized": normalized_contributions,
                "total_variance": total_task_variance,
                "sample_count": task_count,
            }

            # Print summary for this task
            top_5_idx = np.argsort(task_variances)[-5:][::-1]
            print(f"  {task_name}:")
            print(f"    Samples: {task_count}")
            print(f"    Total variance in subspace: {total_task_variance:.2e}")
            print(f"    Top 5 components by variance: {top_5_idx.tolist()}")
            print(f"    Top 5 variances: {task_variances[top_5_idx].round(6)}")

    explained_var = cumsum[k - 1] * 100 if k > 0 and k <= len(cumsum) else 0.0
    print(
        f"\nCross-task subspace: {k}/{hidden_dim} dim ({k/hidden_dim*100:.1f}%), "
        f"explains {explained_var:.1f}% variance"
    )

    # 在 compute_cross_task_subspace 函数末尾，返回前添加：
    if cross_task_subspace is not None:
        # 归一化子空间基
        norms = np.linalg.norm(cross_task_subspace, axis=0, keepdims=True)
        cross_task_subspace = cross_task_subspace / (norms + 1e-12)

        # 验证质量
        cross_task_subspace = validate_subspace_basis(cross_task_subspace, "跨任务子空间")

    if return_full_pca:
        # Return full PCA information for proper random control generation
        full_pca_info = {
            "components": pca.components_.T if hasattr(pca, "components_") else None,  # [hidden_dim, n_components]
            "feature_scales": feature_scales,  # [hidden_dim]
            "explained_variance_ratio": pca.explained_variance_ratio_
            if hasattr(pca, "explained_variance_ratio_")
            else None,
            "max_components": max_components,
        }
        return cross_task_subspace, k, task_variance_contributions, full_pca_info
    else:
        return cross_task_subspace, k, task_variance_contributions


def validate_subspace_basis(subspace, name="子空间"):
    """验证子空间基的质量"""
    k = subspace.shape[1]

    print(f"\n[VALIDATE] {name}验证:")
    print(f"  形状: {subspace.shape}")

    # 计算列向量范数
    norms = np.linalg.norm(subspace, axis=0)
    print(f"  列向量范数范围: [{np.min(norms):.6f}, {np.max(norms):.6f}]")

    # 检查正交性
    if k > 1:
        ortho_matrix = subspace.T @ subspace
        np.fill_diagonal(ortho_matrix, 0)  # 移除对角线
        max_off_diag = np.max(np.abs(ortho_matrix))
        print(f"  最大非对角线元素: {max_off_diag:.6e}")

    # 检查归一化程度
    expected_norm = 1.0
    norm_errors = np.abs(norms - expected_norm)
    avg_norm_error = np.mean(norm_errors)
    print(f"  平均范数误差: {avg_norm_error:.6e}")

    # 如果范数误差太大，建议重新归一化
    if avg_norm_error > 0.1:
        print(f"  [建议] 对{name}进行归一化")
        normalized = subspace / norms
        return normalized
    else:
        return subspace


class JointSubspaceRemovalHook:
    """Hook to remove the joint subspace component from activations

    The subspace is computed from globally scaled activations (centered + scaled by feature std).
    During removal, we have two options:
    1. preserve_statistics=False: Remove subspace directly from raw activations
    2. preserve_statistics=True: Remove subspace from normalized activations, then restore stats

    Note: The normalization during removal uses local (per-batch) statistics, which differs
    from the global normalization used during subspace computation. This may cause slight
    inconsistencies in the geometric space where removal occurs.
    """

    def __init__(
        self,
        layer_idx,
        joint_subspace,
        enabled=True,
        track_stats=False,
        eps=1e-6,
        preserve_statistics=True,
        strength=1.0,
    ):
        self.layer_idx = layer_idx
        self.joint_subspace_np = joint_subspace
        self.enabled = enabled
        self.track_stats = track_stats
        self.eps = eps
        self.preserve_statistics = preserve_statistics
        self.strength = strength  # Intervention strength (0.0 = no intervention, 1.0 = full removal)
        self.stats = {"original_variances": [], "removed_variances": [], "variance_ratios": []}

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

                # 标准化输入 - 使用与训练时相同的方式
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
                print(
                    f"  Warning: Joint subspace columns are not properly orthonormal! Max deviation: {max_deviation:.6f}"
                )

        # 计算投影到联合子空间
        if self.preserve_statistics:
            projection_coeffs = hidden_normalized @ U_orth  # [batch*seq, k]
            # 计算联合子空间成分
            joint_subspace_component = projection_coeffs @ U_orth.T
            # 按强度移除联合子空间成分
            orthogonal_complement = hidden_normalized - self.strength * joint_subspace_component
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


def compute_loss(model, tokenizer, texts, hooks=None, device=DEVICE):
    """Compute average loss, perplexity, and accuracy on texts with enhanced error handling"""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_predictions = 0
    total_predictions = 0
    failed_batches = 0
    max_failed_batches = 3  # 最多允许失败的批次

    # Filter out empty or too-short texts
    valid_texts = [text for text in texts if text and len(text.strip()) > 10]
    if len(valid_texts) < len(texts):
        print(f"Filtered out {len(texts) - len(valid_texts)} empty/short texts, {len(valid_texts)} remaining")

    if not valid_texts:
        print(f"Warning: No valid texts remaining after filtering, returning NaN")
        return {"loss": float("nan"), "perplexity": float("nan"), "accuracy": float("nan")}

    with torch.no_grad():
        for i in tqdm(range(0, len(valid_texts), BATCH_SIZE), desc="Computing metrics"):
            batch_texts = valid_texts[i : i + BATCH_SIZE]

            try:
                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN
                ).to(device)

                # Check for empty attention masks
                attention_mask = inputs["attention_mask"]
                valid_tokens = attention_mask.sum(dim=1)  # Sum over sequence dimension

                # Skip batches with no valid tokens
                if valid_tokens.sum().item() == 0:
                    print(f"Warning: Batch {i//BATCH_SIZE} has no valid tokens, skipping")
                    failed_batches += 1
                    if failed_batches >= max_failed_batches:
                        print(f"Too many failed batches ({failed_batches}), skipping...")
                        break
                    continue

                # 前向传播，确保使用稳定的数值
                with torch.cuda.amp.autocast(enabled=False):  # 禁用混合精度以确保稳定性
                    #outputs = model(**inputs, labels=inputs["input_ids"])
                    labels = inputs["input_ids"].clone()
                    labels[inputs["attention_mask"] == 0] = -100
                    outputs = model(**inputs, labels=labels)

                loss = outputs.loss

                # 检查loss是否为NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Batch {i//BATCH_SIZE} has NaN/Inf loss: {loss.item()}")
                    failed_batches += 1
                    if failed_batches >= max_failed_batches:
                        print(f"Too many failed batches ({failed_batches}), skipping...")
                        break
                    continue

                num_tokens = valid_tokens.sum().item()

                # Additional safety check
                if num_tokens == 0:
                    print(f"Warning: Batch {i//BATCH_SIZE} computed 0 tokens after processing, skipping")
                    failed_batches += 1
                    if failed_batches >= max_failed_batches:
                        print(f"Too many failed batches ({failed_batches}), skipping...")
                        break
                    continue

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

                # Compute accuracy
                if hasattr(outputs, "logits"):
                    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                    labels = inputs["input_ids"]  # [batch_size, seq_len]

                    # Shift logits and labels for causal LM prediction
                    shift_logits = logits[..., :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
                    shift_labels = labels[..., 1:].contiguous()  # [batch_size, seq_len-1]

                    # Get predictions
                    predictions = shift_logits.argmax(dim=-1)  # [batch_size, seq_len-1]

                    # Create mask for valid positions
                    attention_mask_shifted = attention_mask[..., 1:].contiguous()  # [batch_size, seq_len-1]
                    valid_mask = attention_mask_shifted.bool()

                    # Also exclude pad tokens from accuracy calculation
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    non_pad_mask = shift_labels != pad_token_id

                    # Combine masks
                    # final_mask = valid_mask & non_pad_mask
                    final_mask = attention_mask_shifted.bool()

                    # Count correct predictions
                    correct = (predictions == shift_labels) & final_mask
                    batch_correct = correct.sum().item()
                    batch_total = final_mask.sum().item()

                    correct_predictions += batch_correct
                    total_predictions += batch_total

            except Exception as e:
                print(f"Error processing batch {i//BATCH_SIZE}: {e}")
                failed_batches += 1
                if failed_batches >= max_failed_batches:
                    print(f"Too many failed batches ({failed_batches}), skipping...")
                    break
                continue

    if total_tokens == 0:
        print(f"Warning: No valid tokens processed, returning NaN")
        return {"loss": float("nan"), "perplexity": float("nan"), "accuracy": float("nan")}

    avg_loss = total_loss / total_tokens

    # Compute perplexity
    try:
        perplexity = math.exp(avg_loss)
    except (ValueError, OverflowError):
        perplexity = float("inf")

    # Compute accuracy
    accuracy = 0.0 if total_predictions == 0 else correct_predictions / total_predictions

    # 检查最终loss
    if np.isnan(avg_loss) or np.isinf(avg_loss):
        print(f"Warning: Final loss is NaN/Inf: {avg_loss}")

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": accuracy,
        "total_tokens": total_tokens,
        "correct_predictions": correct_predictions,
        "total_predictions": total_predictions,
    }


def run_cross_task_experiment_fast_55(model_name, datasets):
    """Fast version: Run ONLY the essential steps for Step 5.5 analysis"""
    print(f"\n{'='*80}")
    print(f"FAST CROSS-TASK EXPERIMENT (Step 5.5 only): {model_name}")
    print(f"{'='*80}")

    # Load model and tokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        return {}

    model = model.to(DEVICE)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Infer hidden dim (FIX: do not assume 4096 for Gemma3 etc.)
    hidden_dim = infer_hidden_dim_from_model(model, fallback=HIDDEN_DIM)
    print(f"[Info] Inferred hidden_dim = {hidden_dim}")

    # Get model layers
    try:
        layers, arch_type = get_model_layers(model)
    except Exception as e:
        print(f"Failed to get layers for {model_name}: {e}")
        return {}

    print(f"Architecture type: {arch_type}")
    print(f"Model has {len(layers)} layers")

    # Use specified layers
    target_layers = LAYER_INDICES
    valid_layers = [idx for idx in target_layers if idx < len(layers)]
    if not valid_layers:
        print(f"Warning: None of the specified layers {target_layers} are valid")
        return {}

    print(f"Target layers: {valid_layers}")

    # Step 1: Collect activations from ALL tasks
    print(f"\n{'='*50}")
    print("Step 1: Collecting activations from ALL tasks...")
    print(f"{'='*50}")

    # Initialize cross-task collector
    cross_collector = CrossTaskActivationCollector(valid_layers, activation_strategy=ACTIVATION_STRATEGY)
    cross_collector.register_hooks(model)

    # Collect activations for each task (增加样本数量)
    all_task_texts = {}
    for dataset_name, texts in datasets.items():
        print(f"Collecting activations for task: {dataset_name}")

        # Calculate optimal samples for activation collection (use inferred hidden_dim)
        max_available = min(len(texts), MAX_SAMPLES_PER_TASK)
        optimal_samples = calculate_optimal_samples(max_available, MAX_SEQ_LEN, hidden_dim)

        eval_texts = texts[:optimal_samples]
        all_task_texts[dataset_name] = eval_texts

        constraint_satisfied = optimal_samples * MAX_SEQ_LEN > hidden_dim
        print(
            f"  Using {len(eval_texts)} samples (constraint satisfied: {constraint_satisfied}, "
            f"{len(eval_texts)}×{MAX_SEQ_LEN}={len(eval_texts)*MAX_SEQ_LEN} > {hidden_dim})"
        )

        cross_collector.set_current_task(dataset_name)

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(eval_texts), BATCH_SIZE), desc=f"Processing {dataset_name}"):
                batch_texts = eval_texts[i : i + BATCH_SIZE]

                inputs = tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN
                ).to(DEVICE)

                model(**inputs)

    cross_collector.remove_hooks()

    # Step 2: Compute cross-task subspace only
    print(f"\n{'='*50}")
    print("Step 2: Computing CROSS-TASK subspace (fast)...")
    print(f"{'='*50}")

    # Prepare task activations dict for cross-task PCA
    task_activations_dict = {}
    for dataset_name in datasets.keys():
        layer_activations = {}
        for layer_idx in valid_layers:
            acts = cross_collector.get_task_activations(dataset_name, layer_idx)
            if acts is not None and acts.shape[0] > 0:
                # Calculate optimal samples for PCA (ensure sufficient data for PCA computation)
                max_available = acts.shape[0]
                #optimal_samples = calculate_optimal_samples(max_available, MAX_SEQ_LEN, hidden_dim)
                n_samples = min(max_available, 20000)
                indices = np.random.choice(max_available, size=n_samples, replace=False)

                # n_samples = min(max_available, optimal_samples)

                # if n_samples < optimal_samples:
                #     print(
                #         f"  Warning: {dataset_name} only has {acts.shape[0]} activations (need {optimal_samples} for optimal PCA)"
                #     )

                # # Use all available samples or subsample if too many
                # if max_available <= optimal_samples:
                #     indices = np.arange(max_available)
                # else:
                #     indices = np.random.choice(max_available, size=n_samples, replace=False)

                layer_activations[layer_idx] = acts[indices]

        if layer_activations:
            task_activations_dict[dataset_name] = layer_activations
            total_samples = sum(a.shape[0] for a in layer_activations.values())
            print(f"  {dataset_name}: Using {total_samples} total samples for PCA")

    if not task_activations_dict:
        print("No activations collected for any task")
        return {}

    # 检查每个任务的样本数量
    print("\n样本数量统计:")
    for task_name, layer_acts in task_activations_dict.items():
        for layer_idx, acts in layer_acts.items():
            print(f"  {task_name} (layer {layer_idx}): {acts.shape[0]} samples")

    # Compute cross-task subspace
    print("\nComputing cross-task subspace...")
    cross_task_subspace, cross_subspace_dim, task_contributions, full_pca_info = compute_cross_task_subspace(
        task_activations_dict,
        variance_threshold=PCA_VARIANCE_THRESHOLD,
        min_dim=MIN_SUBSPACE_DIM,
        max_dim=MAX_SUBSPACE_DIM,
        return_full_pca=True,
    )

    if cross_task_subspace is None:
        print("Failed to compute cross-task subspace")
        return {}

    # Update hidden_dim from subspace (more reliable than config in some wrappers)
    hidden_dim = int(cross_task_subspace.shape[0])
    print(f"\nCross-task subspace computed: {cross_subspace_dim} dimensions (hidden_dim={hidden_dim})")

    # ================================================================
    # STEP 5.5 ONLY: Identify and test TRULY SHARED basis
    # ================================================================

    print(f"\n{'='*50}")
    print("MAIN STEP: Identifying TRULY SHARED BASIS across ALL tasks")
    print(f"{'='*50}")

    def select_random_non_shared_basis(
        task_contributions, shared_indices, num_basis, cross_subspace_dim, task_name, trial_seed=None
    ):
        """
        Select random basis vectors that are not shared (low contribution to this task)
        trial_seed allows for different selections across trials
        """
        if trial_seed is not None:
            np.random.seed(trial_seed)

        if task_name not in task_contributions or "raw_variances" not in task_contributions[task_name]:
            # Fallback: select random indices
            all_indices = list(range(cross_subspace_dim))
            non_shared = [idx for idx in all_indices if idx not in shared_indices]
            if len(non_shared) >= num_basis:
                return np.random.choice(non_shared, size=num_basis, replace=False).tolist()
            else:
                return np.random.choice(all_indices, size=num_basis, replace=False).tolist()

        variances = task_contributions[task_name]["raw_variances"]

        # Get indices sorted by variance (low to high for this task)
        all_indices = np.argsort(variances)  # Sort by increasing variance

        # Exclude shared indices
        non_shared_indices = [idx for idx in all_indices if idx not in shared_indices]

        # For statistical testing, we want variation between trials
        if len(non_shared_indices) >= num_basis:
            # Select randomly from the lower half of non-shared indices (lower variance = less important)
            candidates = non_shared_indices[: max(num_basis * 2, len(non_shared_indices) // 2)]
            selected = np.random.choice(candidates, size=num_basis, replace=False).tolist()
        else:
            # If not enough non-shared, pad with random from remaining
            selected = non_shared_indices[:]
            remaining = [idx for idx in all_indices if idx not in selected and idx not in shared_indices]
            if len(remaining) > 0:
                additional = np.random.choice(remaining, size=num_basis - len(selected), replace=False)
                selected.extend(additional.tolist())

        return selected

    def test_shared_vs_random_basis_importance(
        model,
        tokenizer,
        eval_texts,
        cross_task_subspace,
        shared_indices,
        task_contributions,
        dataset_name,
        cross_subspace_dim,
        full_pca_info=None,
        sample_size=16,
        n_trials=3,
    ):
        """
        Test importance of shared basis vs random non-shared basis vectors with statistical testing
        """
        if not shared_indices:
            print(f"  No shared basis vectors found")
            return None

        num_basis = len(shared_indices)
        print(f"  Testing {dataset_name}: {num_basis} shared vs {num_basis} random non-shared basis vectors")

        # Run multiple trials for statistical testing
        trial_results = []
        print(f"\n🔄 Running {n_trials} trials for statistical testing...")

        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...")
            try:
                trial_seed = 42 + trial * 100  # Use spaced seeds to ensure different randomization
                if trial_seed is not None:
                    np.random.seed(trial_seed)
                    torch.manual_seed(trial_seed)

                # Use different random subset of eval_texts for each trial
                if len(eval_texts) > sample_size:
                    eval_subset = np.random.choice(eval_texts, size=sample_size, replace=False).tolist()
                else:
                    eval_subset = eval_texts[:sample_size]

                # Select random non-shared basis vectors for this task
                random_indices = select_random_non_shared_basis(
                    task_contributions, shared_indices, num_basis, cross_subspace_dim, dataset_name, trial_seed
                )

                # Compute baseline loss on random subset
                baseline_metrics = compute_loss(model, tokenizer, eval_subset, hooks=None, device=DEVICE)
                baseline_loss = baseline_metrics["loss"]

                result = {
                    "baseline_loss": baseline_loss,
                    "baseline_perplexity": baseline_metrics["perplexity"],
                    "baseline_accuracy": baseline_metrics["accuracy"],
                    "num_basis": num_basis,
                    "shared_indices": shared_indices,
                    "trial_seed": trial_seed,
                }

                # Test shared basis removal
                try:
                    shared_subspace = cross_task_subspace[:, shared_indices].copy()
                    norms = np.linalg.norm(shared_subspace, axis=0, keepdims=True)
                    shared_subspace = shared_subspace / (norms + 1e-12)

                    shared_hooks = []
                    for layer_idx in valid_layers:
                        if layer_idx >= len(layers):
                            continue
                        layer = layers[layer_idx]
                        hook = JointSubspaceRemovalHook(
                            layer_idx, shared_subspace, enabled=True, track_stats=False, preserve_statistics=True
                        )
                        handle = layer.register_forward_hook(hook)
                        shared_hooks.append(handle)

                    shared_metrics = compute_loss(model, tokenizer, eval_subset, hooks=shared_hooks, device=DEVICE)
                    shared_removal_loss = shared_metrics["loss"]

                    result.update(
                        {
                            "shared_removal_loss": shared_removal_loss,
                            "shared_perplexity": shared_metrics["perplexity"],
                            "shared_accuracy": shared_metrics["accuracy"],
                            "shared_success": True,
                        }
                    )

                    # Clean up hooks
                    for handle in shared_hooks:
                        handle.remove()

                except Exception as e:
                    result.update({"shared_removal_loss": None, "shared_success": False})

                # Test random basis removal (from SAME PCA space, using non-shared components)
                try:
                    if full_pca_info and "components" in full_pca_info and full_pca_info["components"] is not None:
                        # PRINCIPLED APPROACH: Use components from the SAME PCA decomposition
                        np.random.seed(trial_seed + 1000)  # Different seed from shared test

                        # Get all available PCA components
                        all_components = full_pca_info["components"]  # [hidden_dim, n_components]
                        feature_scales = full_pca_info["feature_scales"]  # [hidden_dim]
                        max_components = full_pca_info["max_components"]

                        # Available components: exclude the shared ones (0 to cross_subspace_dim-1)
                        available_component_indices = [i for i in range(max_components) if i not in shared_indices]

                        # Select same number of components as shared subspace size, not necessarily full cross_subspace_dim
                        control_k = num_basis

                        if len(available_component_indices) < control_k:
                            print(
                                f"  Warning: Only {len(available_component_indices)} non-shared components available, using all of them"
                            )
                            selected_indices = available_component_indices
                        else:
                            selected_indices = np.random.choice(
                                available_component_indices, size=control_k, replace=False
                            ).tolist()

                        # Extract selected components and rescale to original space
                        selected_components_scaled = all_components[:, selected_indices]  # [hidden_dim, selected_k]
                        selected_components_centered = selected_components_scaled * feature_scales.reshape(-1, 1)

                        random_subspace = selected_components_centered.astype(np.float32)
                        print(
                            f"  Random control: Selected {len(selected_indices)} components from PCA space (excluding shared components)"
                        )

                        # Verify orthogonality with shared subspace (approximate)
                        if control_k > 0:
                            shared_subspace_for_check = cross_task_subspace[:, shared_indices].copy()
                            ortho_check = np.abs(shared_subspace_for_check.T @ random_subspace).max()
                            print(f"  Orthogonality check (shared vs random): max overlap = {ortho_check:.6f}")
                    else:
                        # Fallback: Create random orthogonal subspace
                        print(f"  Fallback: Creating random subspace (full PCA info not available)")
                        random_matrix = np.random.randn(hidden_dim, num_basis)
                        Q, R = np.linalg.qr(random_matrix)
                        random_subspace = Q[:, :num_basis].copy()

                    random_hooks = []
                    for layer_idx in valid_layers:
                        if layer_idx >= len(layers):
                            continue
                        layer = layers[layer_idx]
                        hook = JointSubspaceRemovalHook(
                            layer_idx, random_subspace, enabled=True, track_stats=False, preserve_statistics=True
                        )
                        handle = layer.register_forward_hook(hook)
                        random_hooks.append(handle)

                    random_metrics = compute_loss(model, tokenizer, eval_subset, hooks=random_hooks, device=DEVICE)
                    random_removal_loss = random_metrics["loss"]

                    result.update(
                        {
                            "random_removal_loss": random_removal_loss,
                            "random_perplexity": random_metrics["perplexity"],
                            "random_accuracy": random_metrics["accuracy"],
                            "random_success": True,
                            "random_note": "matched_basis_count_control",
                        }
                    )

                    # Clean up hooks
                    for handle in random_hooks:
                        handle.remove()

                except Exception as e:
                    result.update({"random_removal_loss": None, "random_success": False})

                if result:
                    trial_results.append(result)
            except Exception as e:
                print(f"    Trial {trial + 1} failed: {e}")
                continue

        if not trial_results:
            print("  No successful trials completed")
            return None

        print(f"  Completed {len(trial_results)}/{n_trials} trials successfully")

        # Aggregate results across trials
        baseline_losses = [r["baseline_loss"] for r in trial_results if r is not None]
        baseline_perplexities = [
            r["baseline_perplexity"] for r in trial_results if r is not None and not np.isnan(r["baseline_perplexity"])
        ]
        baseline_accuracies = [
            r["baseline_accuracy"] for r in trial_results if r is not None and not np.isnan(r["baseline_accuracy"])
        ]

        shared_losses = [r["shared_removal_loss"] for r in trial_results if r and r.get("shared_success")]
        shared_perplexities = [
            r["shared_perplexity"]
            for r in trial_results
            if r and r.get("shared_success") and not np.isnan(r["shared_perplexity"])
        ]
        shared_accuracies = [
            r["shared_accuracy"] for r in trial_results if r and r.get("shared_success") and not np.isnan(r["shared_accuracy"])
        ]

        random_losses = [r["random_removal_loss"] for r in trial_results if r and r.get("random_success")]
        random_perplexities = [
            r["random_perplexity"]
            for r in trial_results
            if r and r.get("random_success") and not np.isnan(r["random_perplexity"])
        ]
        random_accuracies = [
            r["random_accuracy"] for r in trial_results if r and r.get("random_success") and not np.isnan(r["random_accuracy"])
        ]

        # Determine appropriate metrics for this dataset
        is_language_modeling = dataset_name.lower() in ["wikitext"]
        is_reasoning_task = dataset_name.lower() in ["gsm8k", "commonsenseqa", "strategyqa", "aqua"]

        print(f"\n📈 Results across {len(trial_results)} trials:")

        # Calculate statistics
        if baseline_losses:
            baseline_std = np.std(baseline_losses)
            print(f"  Baseline - Loss: {np.mean(baseline_losses):.4f} ± {baseline_std:.4f}", end="")
            if is_language_modeling and baseline_perplexities:
                print(f", PPL: {np.mean(baseline_perplexities):.2f} ± {np.std(baseline_perplexities):.2f}", end="")
            elif is_reasoning_task and baseline_accuracies:
                print(
                    f", Acc: {np.mean(baseline_accuracies)*100:.2f}% ± {np.std(baseline_accuracies)*100:.2f}%",
                    end="",
                )
            print()
            if baseline_std == 0 and len(baseline_losses) > 1:
                print("  ⚠️  Warning: No variation in baseline losses across trials")

        if shared_losses:
            shared_mean = np.mean(shared_losses)
            shared_std = np.std(shared_losses)
            shared_impact = (shared_mean - np.mean(baseline_losses)) / np.mean(baseline_losses) * 100
            print(
                f"  Shared removal - Loss: {shared_mean:.4f} ± {shared_std:.4f} (impact: +{shared_impact:.2f}%)",
                end="",
            )
            if is_language_modeling and shared_perplexities and baseline_perplexities:
                shared_ppl_mean = np.mean(shared_perplexities)
                shared_ppl_std = np.std(shared_perplexities)
                baseline_ppl_mean = np.mean(baseline_perplexities)
                shared_ppl_impact = (shared_ppl_mean - baseline_ppl_mean) / baseline_ppl_mean * 100
                print(
                    f", PPL: {shared_ppl_mean:.2f} ± {shared_ppl_std:.2f} (impact: +{shared_ppl_impact:.2f}%)",
                    end="",
                )
            elif is_reasoning_task and shared_accuracies and baseline_accuracies:
                shared_acc_mean = np.mean(shared_accuracies)
                shared_acc_std = np.std(shared_accuracies)
                baseline_acc_mean = np.mean(baseline_accuracies)
                shared_acc_impact = (shared_acc_mean - baseline_acc_mean) / baseline_acc_mean * 100
                print(
                    f", Acc: {shared_acc_mean*100:.2f}% ± {shared_acc_std*100:.2f}% (impact: {shared_acc_impact:+.2f}%)",
                    end="",
                )
            print()
            if shared_std == 0 and len(shared_losses) > 1:
                print("  ⚠️  Warning: No variation in shared removal losses across trials")

        if random_losses:
            random_mean = np.mean(random_losses)
            random_std = np.std(random_losses)
            random_impact = (random_mean - np.mean(baseline_losses)) / np.mean(baseline_losses) * 100
            print(
                f"  Random removal - Loss: {random_mean:.4f} ± {random_std:.4f} (impact: +{random_impact:.2f}%)",
                end="",
            )
            if is_language_modeling and random_perplexities and baseline_perplexities:
                random_ppl_mean = np.mean(random_perplexities)
                random_ppl_std = np.std(random_perplexities)
                baseline_ppl_mean = np.mean(baseline_perplexities)
                random_ppl_impact = (random_ppl_mean - baseline_ppl_mean) / baseline_ppl_mean * 100
                print(
                    f", PPL: {random_ppl_mean:.2f} ± {random_ppl_std:.2f} (impact: +{random_ppl_impact:.2f}%)",
                    end="",
                )
            elif is_reasoning_task and random_accuracies and baseline_accuracies:
                random_acc_mean = np.mean(random_accuracies)
                random_acc_std = np.std(random_accuracies)
                baseline_acc_mean = np.mean(baseline_accuracies)
                random_acc_impact = (random_acc_mean - baseline_acc_mean) / baseline_acc_mean * 100
                print(
                    f", Acc: {random_acc_mean*100:.2f}% ± {random_acc_std*100:.2f}% (impact: {random_acc_impact:+.2f}%)",
                    end="",
                )
            print()
            if random_std == 0 and len(random_losses) > 1:
                print("  ⚠️  Warning: No variation in random removal losses across trials")

        # Perform hypothesis tests
        statistical_results = {}

        if len(baseline_losses) > 1 and len(shared_losses) > 1:
            statistical_results["baseline_vs_shared"] = perform_hypothesis_test(
                baseline_losses, shared_losses, f"{dataset_name}: Baseline vs Shared Removal"
            )

        if len(baseline_losses) > 1 and len(random_losses) > 1:
            statistical_results["baseline_vs_random"] = perform_hypothesis_test(
                baseline_losses, random_losses, f"{dataset_name}: Baseline vs Random Removal"
            )

        if len(shared_losses) > 1 and len(random_losses) > 1:
            statistical_results["shared_vs_random"] = perform_hypothesis_test(
                shared_losses, random_losses, f"{dataset_name}: Shared vs Random Removal"
            )

        # Aggregate results
        results = {
            "dataset_name": dataset_name,
            "num_basis": num_basis,
            "n_trials": len(trial_results),
            "baseline_losses": baseline_losses,
            "baseline_perplexities": baseline_perplexities,
            "baseline_accuracies": baseline_accuracies,
            "shared_losses": shared_losses,
            "shared_perplexities": shared_perplexities,
            "shared_accuracies": shared_accuracies,
            "random_losses": random_losses,
            "random_perplexities": random_perplexities,
            "random_accuracies": random_accuracies,
            "statistical_tests": statistical_results,
        }

        # Summary of significance
        print(f"\n🎯 Statistical Significance Summary for {dataset_name}:")
        if "baseline_vs_shared" in statistical_results:
            test_result = statistical_results["baseline_vs_shared"]
            sig = "SIGNIFICANT" if test_result["is_significant"] else "not significant"
            effect = "worse" if test_result["mean_difference"] > 0 else "better"
            print(f"  Shared removal vs baseline: {sig} ({effect}, p={test_result['p_value']:.4f})")

        if "baseline_vs_random" in statistical_results:
            test_result = statistical_results["baseline_vs_random"]
            sig = "SIGNIFICANT" if test_result["is_significant"] else "not significant"
            effect = "worse" if test_result["mean_difference"] > 0 else "better"
            print(f"  Random removal vs baseline: {sig} ({effect}, p={test_result['p_value']:.4f})")

        if "shared_vs_random" in statistical_results:
            test_result = statistical_results["shared_vs_random"]
            sig = "SIGNIFICANT" if test_result["is_significant"] else "not significant"
            random_vs_shared = "better" if test_result["mean_difference"] < 0 else "worse"
            print(
                f"  Shared vs random removal: {sig} (random {random_vs_shared} than shared, p={test_result['p_value']:.4f})"
            )

        return results

    # 执行主要分析
    print(f"\nA: 识别被所有任务共享的基向量...")
    all_tasks = list(datasets.keys())

    # 方法1：改进的阈值方法
    print(f"\n方法1: 改进的阈值方法")
    threshold_shared_indices = find_fully_shared_basis_improved(
        task_contributions,
        all_tasks,
        cross_subspace_dim,
        min_tasks_shared=2,  # 至少2个任务共享（你可改回你想要的阈值）
        relative_threshold=0.001,  # 0.1%的阈值
        top_k_components=cross_subspace_dim,  # 检查所有成分
    )

    # Use only threshold method for finding shared basis
    all_shared_indices = threshold_shared_indices
    all_shared_indices.sort()

    # 如果没有找到任何共享basis，尝试放宽条件
    if len(all_shared_indices) == 0:
        print(f"\n⚠️ 警告：没有找到任何共享basis，尝试进一步放宽条件...")

        # 进一步放宽阈值
        very_relaxed_indices = find_fully_shared_basis_improved(
            task_contributions,
            all_tasks,
            cross_subspace_dim,
            min_tasks_shared=3,  # 至少3个任务共享
            relative_threshold=0.0005,  # 0.05%的阈值
            top_k_components=cross_subspace_dim,  # 检查所有成分
        )

        if very_relaxed_indices:
            all_shared_indices = very_relaxed_indices
            print(f"  放宽条件后找到 {len(all_shared_indices)} 个共享basis")
        else:
            print(f"  ❌ 即使放宽条件也没有找到共享basis")

    print(f"\nB: 最终共享basis分析")
    print(f"  找到 {len(all_shared_indices)} 个共享basis")

    print(f"\nC: 快速测试共享基向量的重要性...")
    shared_impacts = {}

    if all_shared_indices:
        # 打印共享basis的详细信息
        print(f"\n  共享basis详细信息:")
        print(f"  总数: {len(all_shared_indices)} 个")
        print(
            f"  占跨任务子空间比例: {len(all_shared_indices)}/{cross_subspace_dim} ({len(all_shared_indices)/cross_subspace_dim*100:.1f}%)"
        )
        print(f"  前20个共享basis索引: {all_shared_indices[:20]}")

        # 测试所有任务
        for dataset_name in all_tasks:
            print(f"\n  测试 {dataset_name}:")
            eval_texts = all_task_texts[dataset_name]

            result = test_shared_vs_random_basis_importance(
                model,
                tokenizer,
                eval_texts,
                cross_task_subspace,
                all_shared_indices,
                task_contributions,
                dataset_name,
                cross_subspace_dim,
                full_pca_info=full_pca_info,
                sample_size=16,
                n_trials=N_STATISTICAL_TRIALS,
            )

            if result:
                shared_impacts[dataset_name] = result
    else:
        print("  未找到被所有任务共享的基向量")

    # ================================================================
    # 简化版结果总结
    # ================================================================
    print(f"\n{'='*50}")
    print("FAST RESULTS SUMMARY")
    print(f"{'='*50}")

    print(f"\n🔬 跨任务子空间:")
    print(f"  - 维度: {cross_subspace_dim}/{hidden_dim} ({cross_subspace_dim/hidden_dim*100:.1f}%)")

    print(f"\n📊 任务对跨任务子空间的贡献:")
    for task_name, contrib in task_contributions.items():
        if "total_variance" in contrib:
            print(f"  - {task_name}: 总方差贡献 = {contrib['total_variance']:.2e}")

    if all_shared_indices:
        print(f"\n🌟 共享basis分析:")
        print(f"  - 发现 {len(all_shared_indices)} 个被至少若干任务共享的basis")
        print(
            f"  - 占总跨任务子空间: {len(all_shared_indices)}/{cross_subspace_dim} ({len(all_shared_indices)/cross_subspace_dim*100:.1f}%)"
        )
        print(f"  - 前10个共享basis索引: {all_shared_indices[:10]}")

        # 分析共享basis的重要性
        if shared_impacts:
            print(f"\n🎯 Basis重要性测试 (共享vs随机) - {N_STATISTICAL_TRIALS} trials:")
            for dataset_name, result in shared_impacts.items():
                print(f"  {dataset_name}:")

                # Determine appropriate metrics for this dataset
                is_language_modeling = dataset_name.lower() in ["wikitext"]
                is_reasoning_task = dataset_name.lower() in ["gsm8k", "commonsenseqa", "strategyqa", "aqua"]

                # Show aggregated results
                if result.get("baseline_losses"):
                    baseline_mean = np.mean(result["baseline_losses"])
                    baseline_std = np.std(result["baseline_losses"])
                    print(f"    - Baseline: Loss={baseline_mean:.4f}±{baseline_std:.4f}", end="")
                    if is_language_modeling and result.get("baseline_perplexities"):
                        baseline_ppl = np.mean(result["baseline_perplexities"])
                        print(f", PPL={baseline_ppl:.2f}", end="")
                    elif is_reasoning_task and result.get("baseline_accuracies"):
                        baseline_acc = np.mean(result["baseline_accuracies"]) * 100
                        print(f", Acc={baseline_acc:.2f}%", end="")
                    print(f" ({len(result['baseline_losses'])} trials)")

                if result.get("shared_losses"):
                    shared_mean = np.mean(result["shared_losses"])
                    shared_std = np.std(result["shared_losses"])
                    baseline_mean = np.mean(result["baseline_losses"])
                    shared_impact = (shared_mean - baseline_mean) / baseline_mean * 100
                    print(f"    - 共享basis移除: Loss={shared_mean:.4f}±{shared_std:.4f} ({shared_impact:+.2f}%)", end="")
                    if is_language_modeling and result.get("shared_perplexities") and result.get("baseline_perplexities"):
                        shared_ppl = np.mean(result["shared_perplexities"])
                        baseline_ppl = np.mean(result["baseline_perplexities"])
                        shared_ppl_impact = (shared_ppl - baseline_ppl) / baseline_ppl * 100
                        print(f", PPL={shared_ppl:.2f} ({shared_ppl_impact:+.2f}%)", end="")
                    elif is_reasoning_task and result.get("shared_accuracies") and result.get("baseline_accuracies"):
                        shared_acc = np.mean(result["shared_accuracies"]) * 100
                        baseline_acc = np.mean(result["baseline_accuracies"]) * 100
                        shared_acc_impact = (shared_acc - baseline_acc) / baseline_acc * 100
                        print(f", Acc={shared_acc:.2f}% ({shared_acc_impact:+.2f}%)", end="")
                    print()

                if result.get("random_losses"):
                    random_mean = np.mean(result["random_losses"])
                    random_std = np.std(result["random_losses"])
                    baseline_mean = np.mean(result["baseline_losses"])
                    random_impact = (random_mean - baseline_mean) / baseline_mean * 100
                    print(f"    - 随机basis移除: Loss={random_mean:.4f}±{random_std:.4f} ({random_impact:+.2f}%)", end="")
                    if is_language_modeling and result.get("random_perplexities") and result.get("baseline_perplexities"):
                        random_ppl = np.mean(result["random_perplexities"])
                        baseline_ppl = np.mean(result["baseline_perplexities"])
                        random_ppl_impact = (random_ppl - baseline_ppl) / baseline_ppl * 100
                        print(f", PPL={random_ppl:.2f} ({random_ppl_impact:+.2f}%)", end="")
                    elif is_reasoning_task and result.get("random_accuracies") and result.get("baseline_accuracies"):
                        random_acc = np.mean(result["random_accuracies"]) * 100
                        baseline_acc = np.mean(result["baseline_accuracies"]) * 100
                        random_acc_impact = (random_acc - baseline_acc) / baseline_acc * 100
                        print(f", Acc={random_acc:.2f}% ({random_acc_impact:+.2f}%)", end="")
                    print()

                # Show statistical test results
                if result.get("statistical_tests"):
                    tests = result["statistical_tests"]
                    print(f"    - 统计检验:")

                    if "baseline_vs_shared" in tests:
                        test_result = tests["baseline_vs_shared"]
                        sig_symbol = "✓" if test_result["is_significant"] else "✗"
                        print(f"      {sig_symbol} Shared vs baseline: p={test_result['p_value']:.4f}, d={test_result['effect_size']:+.3f}")

                    if "baseline_vs_random" in tests:
                        test_result = tests["baseline_vs_random"]
                        sig_symbol = "✓" if test_result["is_significant"] else "✗"
                        print(f"      {sig_symbol} Random vs baseline: p={test_result['p_value']:.4f}, d={test_result['effect_size']:+.3f}")

                    if "shared_vs_random" in tests:
                        test_result = tests["shared_vs_random"]
                        sig_symbol = "✓" if test_result["is_significant"] else "✗"
                        random_vs_shared = "better" if test_result["mean_difference"] < 0 else "worse"
                        print(f"      {sig_symbol} Shared vs random: p={test_result['p_value']:.4f} (random {random_vs_shared} than shared)")
    else:
        print(f"\n❌ 未找到任何共享的basis")

    # 清理
    del model
    del tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return {
        "cross_task_subspace": cross_task_subspace,
        "cross_subspace_dim": cross_subspace_dim,
        "task_contributions": task_contributions,
        "fully_shared_indices": all_shared_indices,
        "shared_impacts": shared_impacts,
    }


def main():
    """Main function that runs CROSS-TASK experiments only"""
    print(f"Using device: {DEVICE}")

    print("\n" + "=" * 80)
    print("FAST CROSS-TASK SUBSPACE ANALYSIS (Step 5.5 only)")
    print("=" * 80)

    # Load datasets
    print("\n" + "=" * 50)
    print("Loading datasets for cross-task analysis...")

    print(f"DATASETS_TO_RUN configuration: {DATASETS_TO_RUN}")

    if DATASETS_TO_RUN is None:
        datasets_to_load = list(DATASET_LOADERS.keys())
        print("Using ALL available datasets (DATASETS_TO_RUN = None)")
    else:
        datasets_to_load = [d.lower() for d in DATASETS_TO_RUN]
        print(f"Using SPECIFIED datasets: {datasets_to_load}")
        invalid = [d for d in datasets_to_load if d not in DATASET_LOADERS]
        if invalid:
            print(f"Warning: Invalid dataset names: {invalid}")
            print(f"Available datasets: {list(DATASET_LOADERS.keys())}")
            datasets_to_load = [d for d in datasets_to_load if d in DATASET_LOADERS]

    if not datasets_to_load:
        raise ValueError("No valid datasets to load!")

    print(f"Final datasets to load ({len(datasets_to_load)}): {datasets_to_load}")

    datasets = {}
    for dataset_name in datasets_to_load:
        try:
            loader = DATASET_LOADERS[dataset_name]
            # Load enough samples to satisfy the constraint: samples * seq_len > hidden_dim
            # NOTE: we use fallback HIDDEN_DIM here since model not loaded yet.
            min_required_for_pca = calculate_optimal_samples(MAX_SAMPLES_PER_TASK, MAX_SEQ_LEN, HIDDEN_DIM)
            load_samples = min(MAX_SAMPLES_PER_TASK, max(N_SAMPLES * 2, min_required_for_pca))
            texts = loader(load_samples)
            datasets[dataset_name] = texts
            print(f"  ✓ {dataset_name}: {len(texts)} samples")
        except Exception as e:
            print(f"  ✗ Failed to load {dataset_name}: {e}")
            print(f"    Skipping {dataset_name}...")

    if not datasets:
        raise ValueError("No datasets were successfully loaded!")

    print(f"\nSuccessfully loaded {len(datasets)} dataset(s)")

    # 运行快速版本的跨任务实验（只对指定的模型）
    for model_name in MODELS_TO_RUN:
        try:
            print(f"\n{'='*80}")
            print(f"Starting FAST CROSS-TASK ANALYSIS (Step 5.5 only) for: {model_name}")
            print(f"{'='*80}")

            # 只运行快速版本（Step 5.5）
            cross_task_results = run_cross_task_experiment_fast_55(model_name, datasets)

            if cross_task_results:
                print(f"\n{'='*80}")
                print(f"FAST ANALYSIS COMPLETED for: {model_name}")
                print(f"{'='*80}")

                # 打印关键结果
                if "cross_subspace_dim" in cross_task_results:
                    cross_dim = cross_task_results["cross_subspace_dim"]
                    inferred_hidden_dim = int(cross_task_results["cross_task_subspace"].shape[0])
                    print(f"\n关键发现:")
                    print(f"  1. 跨任务子空间维度: {cross_dim}/{inferred_hidden_dim} ({cross_dim/inferred_hidden_dim*100:.1f}%)")

                    # 共享basis分析
                    if "fully_shared_indices" in cross_task_results:
                        shared_indices = cross_task_results["fully_shared_indices"]
                        print(f"  2. 完全共享basis数量: {len(shared_indices)} (占{cross_dim}的{len(shared_indices)/cross_dim*100:.1f}%)")

                        if len(shared_indices) > 0:
                            print(f"  3. 前10个共享basis索引: {shared_indices[:10]}")

                    # 共享basis的重要性
                    if "shared_impacts" in cross_task_results and cross_task_results["shared_impacts"]:
                        print(f"\n  4. Basis重要性 (共享vs随机):")
                        for task_name, impact in cross_task_results["shared_impacts"].items():
                            if impact.get("statistical_tests"):
                                tests = impact["statistical_tests"]
                                if "shared_vs_random" in tests:
                                    test_result = tests["shared_vs_random"]
                                    sig_symbol = "✓" if test_result["is_significant"] else "✗"
                                    random_vs_shared = "better" if test_result["mean_difference"] < 0 else "worse"
                                    print(
                                        f"     {task_name}: {sig_symbol} Random significantly {random_vs_shared} than shared (p={test_result['p_value']:.4f})"
                                    )
                                else:
                                    print(f"     {task_name}: Insufficient data for statistical testing")
                            else:
                                print(f"     {task_name}: Insufficient data for statistical testing")

                print(f"\n{'='*80}")
                print(f"Fast cross-task experiment completed successfully!")
                print(f"{'='*80}\n")
            else:
                print(f"\n✗ Fast cross-task experiment failed for {model_name}")

        except Exception as e:
            print(f"\nError running fast cross-task experiment for {model_name}: {e}")
            import traceback

            traceback.print_exc()
            print(f"Skipping {model_name}...")
            continue

    print("\n" + "=" * 80)
    print("ALL FAST CROSS-TASK EXPERIMENTS COMPLETED!")
    print("=" * 80)


if __name__ == "__main__":
    results = main()
