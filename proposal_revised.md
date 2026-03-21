# 6998 — Actionable Project Proposal (Revised)

**Jeremiah Mao, Mateo Juliani**
**2026-02-27**

---

## Problem

Leaving LLMs static post-deployment leads to performance degradation as the world changes [1]. While fine-tuning can update a model's knowledge, it risks overwriting previously learned capabilities [7]. Several knowledge update strategies have been proposed — including locate-and-edit methods such as AlphaEdit [2] and continual alignment methods such as COPR [11] — but they have been evaluated almost exclusively on base or instruction-tuned models [6].

In practice, deployed models are often fine-tuned for specific downstream tasks, learning not just facts but *behavioral strategies*. A recent study found that fine-tuning moves activations in directions *nearly orthogonal* to knowledge editing directions, and that AlphaEdit edits are more fragile than MEMIT edits under subsequent fine-tuning [10]. This suggests a fundamental mismatch: weight-space editing methods provide *pointwise* preservation guarantees on a finite set of inputs, but task-tuned models encode learned skills as *distributional strategies* over an unbounded input space.

We investigate whether framing knowledge updates as a **policy preservation problem** — regularizing the model's output distribution rather than constraining individual weight updates — better preserves task performance in task-tuned models. We study this in the financial domain, where rapidly shifting market dynamics make continual knowledge updates a practical necessity. Our primary setting is a model fine-tuned to decompose financial queries into sub-queries for retrieval-augmented generation (RAG) — a task where the learned decomposition strategy depends directly on knowledge of which entities, events, and relationships exist, making it a natural testbed for studying the interaction between knowledge updates and task preservation.

---

## Contributions

1. **A concrete adaptation of COPR for factual knowledge injection.** COPR [11] was designed for continual preference alignment using human-ranked responses. We adapt it for factual knowledge updates by replacing human preference rankings with *factual correctness rankings*: for each knowledge update, we sample multiple model responses, rank them by ground-truth accuracy, and apply COPR's advantage-weighted policy fitting with task replay regularization. This bridges the continual alignment and knowledge editing literatures.

2. **To our knowledge, the first systematic comparison of knowledge update strategies on task-tuned models**, including weight-space editing (AlphaEdit), policy-space regularization (COPR-adapted), and SFT baselines with varying degrees of regularization. The primary evaluation is on financial query decomposition for RAG, where knowledge and task skill are tightly coupled. We additionally evaluate on FinQA as a generic-forgetting control where knowledge and skill are decoupled.

3. **An ablation isolating the contribution of advantage-weighted fitting** beyond simple KL regularization, testing whether COPR's structured policy fitting provides benefits over naive KL-regularized SFT for preserving task performance during knowledge injection.

---

## Theoretical Motivation

We ground our comparison in a formal distinction between two classes of preservation guarantees. The theory below *motivates our empirical hypothesis* — that policy-space methods should outperform weight-space methods on task-tuned models — but does not constitute a formal guarantee, since the distributional bound depends on KL control over the full task distribution, which is approximated in practice via a finite replay buffer.

### Weight-space preservation (AlphaEdit)

AlphaEdit [2] edits a single FFN weight matrix $W$ by projecting a perturbation onto the null space of preserved knowledge. Given preserved keys $K_0$ (typically $n = 100{,}000$ triples from Wikipedia), the null-space projector is computed via SVD:

$$\{U, \Lambda, U^T\} = \text{SVD}(K_0 K_0^T), \quad P = \hat{U}\hat{U}^T$$

where $\hat{U}$ are eigenvectors with eigenvalues below a threshold. The sequential editing solution is:

$$\Delta_{\text{AlphaEdit}} = R \cdot K_1^T \cdot P \cdot (K_p K_p^T P + K_1 K_1^T P + I)^{-1}$$

This yields an **exact but pointwise** guarantee: $(W + \Delta P) K_0 = W K_0$. Outputs are preserved on the finite set $K_0$, but no guarantee extends to unseen inputs. Moreover, the constraint applies to a single FFN layer, while task skills involve coordinated representations across many layers [10]. AlphaEdit has been tested up to ~3,000 edits; beyond this, general capabilities degrade significantly [6].

### Policy-space preservation (COPR, adapted)

COPR [11] derives the optimal updated policy as:

$$\pi^*_t(y|x) = \frac{1}{Z_t(x)} \pi_{t-1}(y|x) \cdot \exp\!\left(\frac{r_t(x,y)}{\beta}\right)$$

where $\pi_{t-1}$ is the previous policy and $r_t$ encodes the learning signal. Originally, $r_t$ is derived from human preference rankings over candidate responses $\mathcal{Y}^x = \{y_1 \prec \ldots \prec y_J\}$, with a linear advantage $\text{Adv}(x, y_j) = (2j - J_x - 1) / J_x$.

**Our adaptation:** We replace human preference rankings with *factual correctness rankings*. This is a testable heuristic rather than a principled transfer of the preference-learning machinery — factual answers are not naturally ordered the same way preferences are (some answers are exact, some are alias-equivalent, some are numerically close, some are categorically wrong). We operationalize the ranking as follows: for each knowledge-update prompt $x$ (e.g., "Who is the current CEO of Company X?"), we sample $K$ responses from the current model, rank them by ground-truth accuracy (exact match → rank $K$; partial match → middle; hallucination → rank 1), and compute COPR's advantage function over these factual rankings. The empirical question is whether this ranking heuristic helps preserve task performance relative to simpler KL-regularized baselines. The training objective combines fitting on new-knowledge responses with regularization on a task replay buffer:

$$\mathcal{L}^{\text{fit}}_t(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{new}}} \sum_{y \in \mathcal{Y}^x} D_{\text{KL}}\!\left(P_{y,t}(y|x,\theta) \;\|\; P^*_{y,t}(y|x)\right)$$

$$\mathcal{L}^{\text{reg}}_t(\theta) = \mathbb{E}_{x \sim \mathcal{R}_{\text{task}}} D_{\text{KL}}\!\left(P_{y,t}(y|x,\theta) \;\|\; P^*_{\text{task}}(y|x)\right)$$

where $\mathcal{R}_{\text{task}}$ is a 5% replay buffer of query decomposition training examples (following COPR's original 1–5% range [11]).

Under idealized conditions — if the replay buffer were representative of the full task distribution — Pinsker's inequality would yield a bound on task metric degradation: $|\mathbb{E}_{\pi_{\text{task}}}[R] - \mathbb{E}_{\pi_\theta}[R]| \leq \sqrt{2\epsilon}$ for KL budget $\epsilon$. In practice, the replay buffer is a finite approximation, so this bound serves as a design principle (minimize output-distribution shift) rather than a formal guarantee. Whether this principle translates to empirical task preservation is the central question of the project.

### Connection to EWC

The Fisher information matrix $F$ is the Hessian of the KL divergence [4]:

$$D_{\text{KL}}(p_{\theta^*} \| p_\theta) \approx \tfrac{1}{2}(\theta - \theta^*)^T F(\theta^*) (\theta - \theta^*)$$

EWC's diagonal approximation is thus a lossy, parameter-level proxy for the output-level KL that COPR controls directly. Recent work shows that KL divergence from the base model predicts catastrophic forgetting with $R^2 = 0.96$ [9], further motivating direct KL control over indirect parameter-space constraints.

---

## Approach

### Primary task: Financial query decomposition for RAG

We fine-tune Qwen2.5-3B on a financial query decomposition task. Given a financial question (e.g., "How did the Middle East conflict affect energy sector earnings in Q4 2023?"), the model generates 2–4 sub-queries whose embeddings maximize Recall@10 of target documents in a fixed retrieval corpus.

**Why this task:** Query decomposition is a genuine *behavioral skill* — the model must learn a strategy for breaking complex questions into retrievable components — but it is also *knowledge-dependent*: generating effective sub-queries requires knowing which entities, events, and relationships exist. This tight coupling between skill and knowledge makes it an ideal setting for our study. When new facts are injected (e.g., a new acquisition or leadership change), the model's decomposition strategy should ideally incorporate the new entities without degrading on old ones.

**Data and evaluation protocol:**

| Component | Details |
|-----------|---------|
| **Retrieval corpus** | FNSPID [5] articles before January 2022 (fixed, downloaded, not a contribution). ~20+ years of pre-cutoff data; 2022–2023 held out as knowledge source. |
| **Embedding model** | Fixed off-the-shelf encoder (e.g., BGE-M3 or Contriever). Held constant across all experiments — not a variable. |
| **Training data** | Financial questions generated from *pre-cutoff* FNSPID articles using a teacher LLM, paired with gold sub-query decompositions. Filtered to decompositions achieving Recall@10 > 0.7 against gold-relevant articles in the pre-cutoff corpus. Training via SFT, optionally refined with DPO using Recall@10 as the preference signal. |
| **Test data (task preservation)** | Held-out questions from pre-cutoff articles, same distribution as training. Measures whether the decomposition skill is preserved after knowledge injection. |
| **Metric** | Recall@10 of the model's generated sub-queries against gold-relevant documents in the fixed pre-cutoff corpus. |

The retrieval corpus is **fixed throughout all experiments** — it is never extended or modified. Knowledge absorption is measured separately through fact-probe questions (metric 2 below), not through retrieval. This keeps the evaluation clean: Recall@10 measures only whether the decomposition *skill* is preserved under the same retrieval conditions, while fact probes measure whether the model *learned the new information*. Mixing these two signals into a single retrieval metric would create confounds.

This is a data preprocessing and evaluation pipeline, not a dataset contribution. The benchmark operationalizes decomposition quality through a fixed teacher-and-retriever scaffold. Our findings are scoped to this setup; we do not claim to measure a general decomposition capability independent of the retrieval pipeline.

### Secondary task: FinQA (generic-forgetting control)

We additionally fine-tune Qwen2.5-3B on FinQA [3], an established benchmark requiring multi-step numerical reasoning over financial reports (8,281 QA pairs over SEC filings with tables and text). Models of comparable scale (phi-3-mini, 3.8B) achieve 73.5% execution accuracy with LoRA [8].

FinQA serves a different purpose than query decomposition. Because FinQA answers are derived from tables already present in the prompt, knowledge updates (CEO changes, acquisitions, etc.) should *not* materially affect FinQA performance. We use FinQA as a **generic-forgetting control**: if a method degrades FinQA accuracy, it is causing collateral damage to unrelated capabilities, not just failing to preserve a knowledge-dependent skill. This separates "general catastrophic forgetting" from the more targeted question of "knowledge-skill interaction."

**How FinQA results will be used:** FinQA is not part of the core hypothesis test. It is a guardrail. If a method preserves query decomposition Recall@10 but degrades FinQA accuracy, we flag that method as causing broad collateral damage. If all methods preserve FinQA (the expected outcome), we report this as confirmation that the methods do not introduce catastrophic forgetting on unrelated tasks, and focus the analysis on the query decomposition results where we expect differentiation.

### Knowledge updates

We define updates as structured financial fact triples — entity changes that can be applied by all methods through a common interface:

- Leadership changes: `("Apple", "CEO", "Tim Cook")` → `("Apple", "CEO", "New Person")`
- Acquisitions: `("Company X", "acquired_by", "Company Y")`
- Financial metrics: `("Company X", "Q3_revenue", "$4.2B")` → `("$5.1B")`

**Common update interface.** To ensure an apples-to-apples comparison, every method receives the same inputs derived from each fact triple:

1. **Fact triple:** The structured `(subject, relation, object)` tuple.
2. **Natural-language rendering:** A fixed template converts each triple to a question-answer pair (e.g., "Who is the CEO of Apple?" → "Tim Cook"). All methods use the same templates and answer normalization.
3. **Temporal filtering:** The same cutoff date and entity filter apply to all methods.

AlphaEdit consumes the fact triples directly. SFT-based methods and COPR-adapted train on the rendered QA pairs. This means the methods differ in *how they incorporate the same information*, not in *what information they receive*. We note that this is a comparison of **update paradigms under a shared input source**, not an isolation of a single causal variable — COPR-adapted involves more auxiliary structure (sampling, ranking, replay) than AlphaEdit, so a measured advantage could reflect the richer supervision format rather than the policy-space idea alone. The SFT baselines (which share the same supervision format but lack COPR's advantage weighting) help control for this.

We extract triples from FNSPID [5] using a temporal split, focusing on entities that appear in the retrieval corpus (for query decomposition) and/or FinQA reports (for the forgetting control). We test at three scales: **200**, **1,000**, and **3,000** fact edits (the last being AlphaEdit's tested limit).

### Methods compared

| Method | Type | Description |
|--------|------|-------------|
| No update | Lower bound | Task-tuned model, no knowledge injection |
| Naive SFT | Baseline | SFT on fact QA pairs, no regularization |
| KL-reg SFT | Baseline | SFT + $\lambda \cdot D_{\text{KL}}(\pi_{\text{task}} \| \pi_\theta)$ |
| Mixed replay | Baseline | SFT on new facts + 5% task data shuffled |
| AlphaEdit | Weight-space | Null-space projected fact editing |
| COPR-adapted | Policy-space | Advantage-weighted fitting + task replay |
| Full retrain | Upper bound | Train from scratch on all data (if compute permits) |

This design isolates three key comparisons:

- **Naive SFT → KL-reg SFT → COPR-adapted**: Does KL regularization help? Does advantage-weighted fitting help *beyond* KL regularization?
- **AlphaEdit vs. COPR-adapted**: Weight-space vs. policy-space, head-to-head on the same fact triples.
- **Scaling**: At what edit count does each method begin degrading task performance?

### COPR adaptation pipeline (detailed)

For each fact triple to inject:

1. Convert the triple to a natural-language question using the fixed template.
2. Sample $K = 8$ responses from the current task-tuned model.
3. Rank responses by factual correctness using the ground-truth answer (exact match → top rank; partial match → middle; hallucination → bottom). Partial match is defined as token-level F1 > 0.5 against the gold answer.
4. Compute COPR's linear advantage: $\text{Adv}(x, y_j) = (2j - K - 1) / K$.
5. Compute the renormalized optimal sampling distribution $P^*$.
6. Train via $\mathcal{L}^{\text{fit}}$ on these ranked responses.
7. Simultaneously train via $\mathcal{L}^{\text{reg}}$ on a 5% replay buffer of query decomposition training examples.

### Execution plan

To manage scope, we follow a phased execution strategy:

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** (core result) | All methods on query decomposition at all three edit scales | Minimum viable result |
| **Phase 2** (forgetting control) | Top 3–4 methods from Phase 1 on FinQA | Verifies no generic forgetting |
| **Phase 3** (if compute permits) | Full retrain upper bound + ablations (replay buffer size, $K$, advantage variants) | Optional |

---

## Dataset

> All data is downloaded from existing sources or derived via preprocessing; no novel dataset is proposed.

| Data | Source | Purpose |
|------|--------|---------|
| **Retrieval corpus** | FNSPID [5] articles before January 2022 | Fixed retrieval target for query decomposition training and task-preservation evaluation |
| **Query decomposition training data** | Financial questions generated from pre-cutoff FNSPID by teacher LLM, filtered by Recall@10 > 0.7 | Evaluation infrastructure, not a dataset contribution |
| **FinQA benchmark** | FinQA [3] — 8,281 QA pairs, standard train/dev/test split | Generic-forgetting control |
| **Knowledge source** | FNSPID articles Jan 2022–Dec 2023 | Structured fact triples extracted via LLM, filtered to entities in pre-cutoff corpus and/or FinQA reports |

---

## Metrics

For each method and edit scale, we evaluate:

| # | Metric | Measures |
|---|--------|----------|
| 1 | **Task preservation** (primary) | Recall@10 on held-out query decomposition test set — does the decomposition strategy survive knowledge injection? |
| 2 | **Knowledge absorption** | Accuracy on fact-probe questions about injected triples — did the model actually learn the new facts? |
| 3 | **Generic forgetting** (secondary) | FinQA execution accuracy — did the update cause collateral damage to unrelated reasoning? |
| 4 | **Locality** | Accuracy on held-out untouched financial facts, stratified into three categories: (a) same entities as edited, different relations; (b) different entities in the same sector; (c) entities from unrelated sectors. This tests whether disruption is local to the edited entity, spreads within a domain, or is broadly distributed. |
| 5 | **Scaling behavior** | Metrics 1–4 as a function of edit count (200, 1K, 3K) |
| 6 | **Compute cost** | GPU-hours per update |

We expect query decomposition to be *more sensitive* to knowledge updates than FinQA, since sub-query generation depends directly on entity knowledge while table arithmetic does not. If a method preserves Recall@10 on query decomposition while also absorbing new facts, it has solved the harder version of the problem. If it only preserves FinQA, it may simply be avoiding catastrophic forgetting without demonstrating knowledge-skill compatibility.

---

## Design choices and limitations

Several hyperparameters are set based on prior work or practical constraints and will be reported as fixed choices unless ablation reveals sensitivity:

| Choice | Value | Justification |
|--------|-------|---------------|
| Replay buffer size | 5% | Within COPR's reported 1–5% range [11]. Ablate at 1% and 10% in Phase 3 if sensitive. |
| Sampled responses $K$ | 8 | Balances diversity of ranked candidates against compute cost. COPR used variable $J_x$ per prompt. |
| Edit scales | 200 / 1K / 3K | 3,000 is AlphaEdit's tested upper limit [2]; 200 is minimal; 1,000 is midpoint. |
| Partial match threshold | F1 > 0.5 | Standard in extractive QA evaluation; reported but not tuned. Note: token F1 is more natural for short entity answers than for financial metrics or acquisition descriptions. We will report ranking quality across relation types to check for systematic bias. |
| Compute budget per method | Fixed | Each method receives the same hyperparameter search budget (same number of tuning runs, same max training steps). This prevents unfair advantages from additional optimization effort, especially for the full retrain baseline. |

---

## References

1. Lazaridou et al. "Mind the gap: Assessing temporal generalization in neural language models." *NeurIPS*, 2021.
2. Fang et al. "AlphaEdit: Null-space constrained knowledge editing for language models." *ICLR* (Outstanding Paper), 2025.
3. Chen et al. "FinQA: A dataset of numerical reasoning over financial data." *EMNLP*, 2021.
4. Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks." *PNAS* 114(13):3521–3526, 2017.
5. Dong et al. "FNSPID: A comprehensive financial news dataset in time series." *KDD*, 2024.
6. Li et al. "Should we really edit language models? On the evaluation of edited language models." *NeurIPS*, 2024.
7. Lin et al. "Mitigating the alignment tax of RLHF." arXiv:2309.06256, 2024.
8. Loukas et al. "Fine-tuning smaller language models for question answering over financial documents." arXiv:2408.12337, 2024.
9. Razak et al. "RL's Razor: Why online reinforcement learning forgets less." arXiv:2509.04259, 2025.
10. Wang et al. "Can fine-tuning erase your edits?" arXiv:2511.05852, 2025.
11. Zhang et al. "COPR: Continual learning human preference through optimal policy regularization." *ACL Findings*, 2025.
