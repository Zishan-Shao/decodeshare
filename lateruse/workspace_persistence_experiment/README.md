
# Workspace / Shared-Subspace Persistence Experiments

这套脚本对应你描述的两类检验：

- **Q1 Structural persistence**：共享子空间在不同层是否重叠？
  - 输出：layer×layer 相似度 heatmap + 相似度随层距离变化的 1D 曲线
- **Q2 Functional persistence**：即使基旋转，“共享坐标”是否能跨层线性预测？
  - 输出：不同层对 (ℓ→m) 的线性回归 R²，以及 R² vs layer distance 曲线
- （可选）**Cross-layer patching**：在目标层做 shared ablation，再用源层预测的 shared component patch 回去

## 依赖

建议：

```bash
pip install -U torch transformers datasets numpy pandas matplotlib tqdm
```

## 文件结构

- `benchmark_dataloaders_aqua_prefix_default.py`（你提供的 data loader）
- `data_utils.py`：封装 data loader
- `model_utils.py`：模型/层定位/激活收集
- `subspace_utils.py`：PCA、共享子空间估计、子空间相似度、Ridge+R²
- `01_estimate_shared_subspaces.py`：估计 Q_S(ℓ)
- `02_structural_persistence.py`：Q1 热图 + 距离曲线
- `03_collect_shared_coords.py`：收集 decode 轨迹的共享坐标 z_t(ℓ)
- `04_coordinate_persistence.py`：Q2 线性可预测性（R²）
- `05_cross_layer_patching_optional.py`：可选 patching 强检验（仅 MC 字母答案）

## 约定

- **层索引是 0-based**，指第 ell 个 transformer block。
  - LLaMA 2 7B (32 layers) 的合法范围是 [0,31]
- 子空间相似度用 principal angles：`mean(sigma^2)`（sigma 是 Qa^T Qb 的奇异值）

## 推荐最小运行流程

### Step 1：估计多个层的共享子空间

（下面 layers 示例是 24 层模型；32 层请自己换成类似分布）

```bash
python 01_estimate_shared_subspaces.py \
  --model YOUR_MODEL \
  --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
  --layers 2,5,8,11,14,17,20,23 \
  --n_subspace 256 --pca_dim 128 \
  --answer_prefix "Final answer:" \
  --out_dir results_run1
```

会生成：`results_run1/shared_subspaces.pt`

### Step 2：Q1 结构持久性

```bash
python 02_structural_persistence.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --out_dir results_run1
```

会生成：
- `results_run1/figs/q1_heatmap_similarity.png`
- `results_run1/figs/q1_similarity_vs_distance.png`

### Step 3：收集共享坐标轨迹

```bash
python 03_collect_shared_coords.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
  --n_train_per_task 40 --n_test_per_task 40 \
  --max_new_tokens 16 \
  --out_dir results_run1
```

会生成：
- `results_run1/coords_train.pt`
- `results_run1/coords_test.pt`

### Step 4：Q2 功能持久性（线性可预测性）

```bash
python 04_coordinate_persistence.py \
  --coords_train results_run1/coords_train.pt \
  --coords_test  results_run1/coords_test.pt \
  --pair_mode adjacent --max_hop 2
```

会生成：
- `results_run1/results_coord_persistence.csv`
- `results_run1/figs/q2_r2_vs_distance.png`

## 可选：cross-layer patching（强因果）

```bash
python 05_cross_layer_patching_optional.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --model YOUR_MODEL \
  --task aqua \
  --src_layer 11 --tgt_layer 14 \
  --n_calib 200 --n_test 200 \
  --answer_prefix "Final answer: " \
  --out_dir results_run1
```

注意这里 answer_prefix 末尾建议加空格，让“下一个 token 更像是字母”。

## 常见坑

1) **padding side**
`model_utils.load_model_and_tokenizer()` 已强制 `tokenizer.padding_side="left"`，否则 hook 里 `hs[:, -1, :]` 可能落在 pad 上。

2) **答案前缀一致性**
如果你希望“对齐到答案 token”，建议统一用 `--add_answer_prefix 1` 并设置 `--answer_prefix`。

3) **k_shared 的自动选择**
默认用 eigvals 阈值 `tau_shared≈0.2`。如果你发现 k 过大/过小，建议：
- 固定 `--k_shared 32`（最稳）
- 或调整 `--tau_shared`（更严格/更宽松）

4) **计算量**
最慢的是 `03_collect_shared_coords.py`（逐步 decode）。可以先把 `--n_*_per_task` 和 `--max_new_tokens` 调小做 pilot。

## 新增：Q3 shared ↔ residual 跨层耦合（相关 + 因果）

这里的 **residual** 指你在 01 输出里保存的 `Q_control_pool_by_layer`（matched nonshared/control pool）。
它不是完整的 d-k 正交补，但作为“非共享方向探针”非常实用：
- 维数小（默认 64），跨层/跨条件分析稳定
- 与该层的 Q_shared 正交（同一层内），便于解释

### Step 6：相关性分析（结构 + 功能）

```bash
python 06_shared_residual_flow_analysis.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --model YOUR_MODEL \
  --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
  --n_train_per_task 40 --n_test_per_task 40 \
  --max_new_tokens 16 \
  --out_dir results_run1
```

会生成：
- `figs/q3_struct_shared_vs_resid_heatmap.png`
- `figs/q3_r2_shared_to_resid_state_heatmap.png`
- `figs/q3_r2_shared_to_resid_update_heatmap.png`
- `figs/q3_r2_vs_distance.png`
- `shared_residual_train.pt` / `shared_residual_test.pt`
- `shared_residual_r2_results.csv`

### Step 7：因果证据（在 src_layer 消融 shared 或 residual）

```bash
python 07_shared_residual_causal_evidence.py \
  --shared_pt results_run1/shared_subspaces.pt \
  --model YOUR_MODEL \
  --tasks aqua,commonsenseqa,strategyqa,arc_challenge,openbookqa \
  --src_layer 11 \
  --ablate shared \
  --n_prompts 80 \
  --max_new_tokens 16 \
  --out_dir results_run1
```

输出：
- `shared_residual_causal.csv`
- `figs/q3_causal_effect_vs_layer.png`
- 终端会打印平均 token logprob 变化（证明干预对输出有影响）

如果你也想看“消融 residual 对 shared 的影响”，把 `--ablate residual` 即可。
