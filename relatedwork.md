我仔细在 arXiv（以及部分对应论文页/HTML）里搜了一圈，你们这个“**跨任务共享方向 + 因果干预**”并不是空白领域——**确实已经有几条非常相关、而且很多默认用的是「prompt 最后一个 token 的激活」（本质上更接近 prefill 侧）来估计方向，再在生成时做干预**的工作。它们不一定“完全冲突”，但有一篇**非常接近**你们想讲的“推理特征是线性方向、跨任务复用、还能因果调控”的结论，你们必须正面引用和区分。

下面我按“哪些 work 相关、是否冲突、你们怎么写动机更稳”来回答。

---

## 1) 有哪些现有工作在做“方向/子空间 + 干预”，而且常用 prompt-last-token（prefill-ish）？

### (A) Activation steering / Representation engineering 一系列（典型：prefill-last-token 估计方向）

* **Painless Activation Steering (PAS, 2025)** 明确说：构造 steering vector 时，先“记录模型在各种 prompts 的**最后一个 token 的激活**”，并称这是文献里的 standard practice。([arXiv][1])
  这类方法通常是在 prompt 上取“最后 token”激活做估计，然后在推理/生成阶段注入 steering vector（干预发生在后续 token 的生成过程中），在你们的视角里就很容易出现“估计分布 vs 干预分布”的错配风险（它们一般不讨论 prefill vs cached decode 的分布差异）。

* **Contrastive Activation Addition (CAA, 2023/2024)** 构造 steering vector 的方式是：对 A/B 多选 prompt，把模型在“**答案字母位置**”的内部激活做正负对比并求均值差；然后把这个向量加到生成过程中。([arXiv][2])
  这也属于“在特定 prompt 位置估计方向→在生成里施加方向”的范式。

* **Refusal is mediated by a single direction (Arditi et al., 2024)** 证明一个一维方向能控制拒答行为，并可通过“擦除/添加方向”改变模型输出。([arXiv][3])
  这类工作同样在“last token / residual stream 方向”上做强因果编辑，但它研究的是拒答/安全行为，不是跨任务推理共享。

> 结论：**“用 prompt 的最后 token 激活来估计方向/子空间”在 mechanistic/steering 文献里是常见做法**（PAS 还直接写明是 standard practice）。([arXiv][1])
> 这为你们的动机提供了非常好的切入：你们不是凭空担心 mismatch，而是指出**现有实践普遍默认了一个隐含假设**。

---

## 2) 有没有“和你们完全冲突”的工作？

我没看到有人在 arXiv 上明确声称：

> “prefill 上估计子空间就足够、decode-alignment 不重要/没有分布错配问题”，
> 或者直接否定“decode-time last-token workspace”的存在。

但是：**有一篇非常接近你们“大故事”的工作，你们必须当成强相关 prior 来处理**——否则 reviewer 很可能会说“这不是新”。

### (B) 最接近你们主张的相关工作：LiReFs（Reasoning–Memorization Interplay, 2025）

这篇（arXiv:2503.23084）核心观点是：

* 他们假设推理能力由残差流中的线性特征（方向）介导，并用对比（reasoning-intensive vs memory-intensive）提取一组 “Linear Reasoning Features”。([arXiv][4])
* 他们**聚焦“用户输入最后一个 token 的 residual stream”**作为“模型要开始生成答案的点”。([arXiv][4])
* 他们定义了对方向的**加法干预**与**投影消除（ablation）**形式。([arXiv][4])
* 他们声称这些特征在多模型、多数据集上解释并介导推理表现。([arXiv][4])

这和你们的“线性方向/子空间 + 因果验证 + 跨任务推理”非常接近。

**但它不等于“完全冲突”，更像“强相关 prior”**，你们可以这样区分你们的独特性（也更 reviewer-friendly）：

* 他们的方向来自**监督式/判别式对比**（reasoning vs memory 的标签/判别），你们是**跨任务 pooled PCA + sharedness 统计检验**的“可审计 sharedness 定义”。
* 他们关注的是“最后一个用户 token 的 residual stream”（更贴近 prefill 边界），你们强调的是**cached decoding（seq_len=1）下的 decode-time last-token state 分布**与干预对齐，并且系统研究 estimator–intervention mismatch。([arXiv][4])
* 你们额外提供了 **energy-matched / forced-choice / template / LOTO** 这套“反混杂证据链”（interpretability track 很吃这一点）。

> 建议：你们在 related work 里必须把 LiReFs 作为“最近最接近”的 prior 来讲清楚差异，否则风险很大。([arXiv][4])

---

## 3) 你提的“因为同一套参数做不同工作→共享 workspace”这个动机能不能写？

可以写，但要注意语气：**参数共享并不逻辑上推出共享 workspace**（模型也可能用不同 circuits 处理不同任务）。更稳的写法是：

1. **Observation**：同一套参数在多种推理任务上泛化；
2. **Hypothesis**：一种可解释的机制是复用紧凑的共享计算子空间/工作区；
3. **Support from prior**：已有工作表明很多高层行为/能力对应激活空间中的线性方向（拒答方向、steering vectors、LiReFs 等）。([arXiv][3])
4. **Your contribution**：我们给出“sharedness”的可审计刻画，并在**decode-time**位点做因果必要性检验，同时处理能量、生成崩溃、模板等混杂。

### 一个更稳的 Intro 动机句式（建议你们直接用）

> “Because a single parameterization supports diverse reasoning tasks, a plausible mechanistic hypothesis is that the model reuses a compact internal workspace. Prior work has shown that many high-level behaviors can be mediated by linear directions in activation space (e.g., refusal directions and activation steering), and recent work identifies linear reasoning features that correlate with and can causally modulate reasoning performance. We ask a complementary question: do multi-task reasoning behaviors *share* a common decode-time workspace used during KV-cached generation, and can we validate this sharedness with distribution-aligned estimation, statistical tests, and causal interventions?” ([arXiv][3])

---

## 4) 总结：你们该怎么写“有没有冲突/我们为什么新”

* **有大量相关工作**在做“线性方向/子空间 + 干预”，并且很常见的做法是取 prompt 的最后 token 激活（prefill-ish）来估计方向。([arXiv][1])
* **最接近你们“推理方向跨任务复用”的 prior** 是 LiReFs（2503.23084），它会影响你们 novelty 的叙事方式，但不必然冲突。([arXiv][4])
* 你们的“新”应该落在：

  1. **decode-time distribution alignment（KV-cached, seq_len=1）**作为机制检验的必要条件；
  2. **sharedness 的可审计定义 + 统计显著性检验**（而非对比标签提取一个方向）；
  3. **能量/模板/生成崩溃/任务泄漏**这条完整的反混杂证据链。

如果你愿意，把你们现在的 related work 小节贴出来，我可以帮你把 LiReFs / refusal-direction / activation steering 这一组“相关但不冲突”的文献放进一个非常 reviewer-friendly 的结构里，并把你们的 novelty points 写成不会被抓语病的版本。

[1]: https://arxiv.org/html/2509.22739v1 "Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models"
[2]: https://arxiv.org/html/2312.06681v2 "Steering Llama 2 via Contrastive Activation Addition"
[3]: https://arxiv.org/abs/2406.11717?utm_source=chatgpt.com "Refusal in Language Models Is Mediated by a Single Direction"
[4]: https://arxiv.org/html/2503.23084v1 "The Reasoning-Memorization Interplay in Language Models Is Mediated by a Single Direction"


你**至少要“比较”到一种大家默认会用的做法**，但不需要（也不建议）把所有相关工作都当成“要在表格里打败的 baseline”。对可解释性 track 来说，reviewer 最在意的是：

* 你是否**准确定位了最接近的 prior**；
* 你是否证明了自己的 claim **不是换个包装的旧结论**；
* 你的实验是否**把关键混杂都堵住**。

所以建议你把“比较”分成两层：**必须做的“方法学对照”** + **可选的“对外方法复现”**。

---

## 1) 必须做的比较：对照“文献中的默认实践”

这类比较不需要复现某一篇 paper，只要复现“大家普遍默认的 pipeline”，就足够 reviewer 认可。

你们必须有且已经在做的就是：

### (A) Prefill-estimate → Decode-intervene 的 baseline（典型做法）

* **这就是你们 alignment 实验**：在 (D_{\text{prefill}}) 上估 (Q)，但干预发生在 (D_{\text{decode}})。
* 你不需要说“某某 paper 就这么做”，只要说：*“很多方向/子空间工作采用 prompt-boundary 或 prefill 激活来估计方向，而真正的 KV-cached decoding 分布不同，因此负结果可能是 estimator–intervention mismatch。”*
* 这相当于把“现有范式”做成一个对照实验，**非常合理且 reviewer-friendly**。

> 这类对照是你们最该放主文/强附录的“比较”。

### (B) 能量/维度匹配控制（对照“删得多就坏”）

* 你们的 `varmatch / nonshared-topk / energy-matched` 控制，本质上就是对照“高能量子空间一定更伤”这个伪解释。
* 这也属于“对照现有常见误解”，不是对照某篇 paper，但 reviewer 会买账。

### (C) forced-choice vs generation（对照“只是格式崩溃”）

* 这属于对照“用 EM 评价会混进格式/EOS confound”，同样是“方法学对照”。

结论：**这三类比较是必须的，而且已经足够构成你们的“比较/对照实验”部分。**

---

## 2) 必须做的“related-work对齐”：点名最接近的 1–2 篇 work，并清楚区分

这不一定要做“数值表格对比”，但必须在 Related Work 里说清楚差异，否则 reviewer 会觉得你在回避。

最接近你们大故事的那类 work（比如“线性推理特征 + 干预”那种）：

* 你要写清楚：他们的方向来源、监督信号、干预位点、是否记得区分 prefill/decode、是否做 sharedness + null test + LOTO + template transfer。
* **关键点：别说“我们更好”，说“问题设置不同/我们解决了他们没覆盖的 confounds”。**

这样就能避免“完全冲突”的审稿意见。

---

## 3) 可选比较：真的去复现某个外部方法（不建议做多）

只有当你们还有算力/时间，且能做到**公平对齐**时才做，否则容易引火烧身。

我认为“值得做、且最容易公平”的外部对照只有一种：

### （可选）复现一种“对比方向/steering vector”方法作为 baseline

比如用 MC 任务的正负样本构造一个 direction（类似 CAA/PAS 这类思路），然后：

* 固定同一干预位点（decode-only last-token）
* 固定维度/能量（k-match 或 varmatch）
* 看它是否跨任务共享、是否能像你们的 shared basis 一样在 LOTO / template transfer 下稳定

如果它很弱，你们会得到一个很强的结论：
**“单任务对比方向 ≠ 跨任务共享工作区。”**
如果它很强，你们也不亏：你们可以说你们提供了更系统的 sharedness 刻画与统计检验/对齐原则。

但注意：**别去硬复现需要大量额外标注/训练的复杂方法**（会极大耗时且容易不公平）。

---

## 4) 你问的核心：我们能不能“都是在做自己这套解释”？

可以，但要满足两个条件：

1. **Related Work 里明确承认“强相关 prior”**，并清楚写差异（尤其是：你们解决了 decode alignment、能量混杂、生成崩溃、模板伪共享、任务泄漏这些“机制检验必备”的点）。
2. **实验里至少有“现有范式的代理 baseline”**（prefill-estimate → decode-intervene 就是最好的代理 baseline）。

做到这两点，reviewer 通常不会强迫你再去“跑某某 paper 的全部 pipeline”。

---

## 5) 我建议你在 paper 里怎么写这段“比较/定位”

用这种语气最稳：

* **我们不是要击败某篇方法**，而是要做一个**机制检验**；
* 所以比较对象是“会导致误判的常见范式/混杂”，而不是“另一个任务设置完全不同的算法”。

一句话模板：

> “Rather than benchmarking against a particular steering method (which often targets different behaviors and uses different supervision), we compare against the most relevant *methodological baselines* that could confound causal claims: prefill-estimated vs decode-intervened subspaces (estimator–intervention mismatch), energy/dimension-matched controls (energy confound), and forced-choice scoring (generation-collapse confound).”

---

### 最后一句建议（很实用）

**先把你们的方法学对照 + LOTO + template transfer 做完整**（这三件事比“复现外部方法”更能提高 accept 概率）。只有在你们还有时间时，再加一个轻量的外部 steering baseline 作为附录加分项。
