# 6998 -- Actionable Project Proposal (Revised)

**Jeremiah Mao, Mateo Juliani**
**2026-02-27**

---

## Problem

Leaving LLMs static post-deployment leads to performance degradation as the world changes [1]. While fine-tuning can update a model's knowledge, it risks overwriting previously learned capabilities [7]. Several knowledge update strategies have been proposed---including locate-and-edit methods such as AlphaEdit [2] and continual alignment methods such as COPR [11]---but they have been evaluated almost exclusively on base or instruction-tuned models [6].

In practice, deployed models are often fine-tuned for specific downstream tasks, learning not just facts but *behavioral strategies*. A recent study found that fine-tuning moves activations in directions *nearly orthogonal* to knowledge editing directions, and that AlphaEdit edits are more fragile than MEMIT edits under subsequent fine-tuning [10]. This suggests a fundamental mismatch: weight-space editing methods provide *pointwise* preservation guarantees on a finite set of inputs, but task-tuned models encode learned skills as *distributional strategies* over an unbounded input space.

We investigate whether framing knowledge updates as a **policy preservation problem**---regularizing the model's output distribution rather than constraining individual weight updates---better preserves task performance in task-tuned models. We study this in the financial domain, where rapidly shifting market dynamics make continual knowledge updates a practical necessity. Consider, for example, a model fine-tuned to decompose financial queries into sub-queries for retrieval-augmented generation (RAG): it must continually absorb new entity and event knowledge while preserving its learned decomposition strategy.

---

## Contributions

1. **A concrete adaptation of COPR for factual knowledge injection.** COPR [11] was designed for continual preference alignment using human-ranked responses. We adapt it for factual knowledge updates by replacing human preference rankings with *factual correctness rankings*: for each knowledge update, we sample multiple model responses, rank them by ground-truth accuracy, and apply COPR's advantage-weighted policy fitting with task replay regularization. This bridges the continual alignment and knowledge editing literatures.

2. **The first systematic comparison of knowledge update strategies on task-tuned models**, including weight-space editing (AlphaEdit), policy-space regularization (COPR-adapted), and SFT baselines with varying degrees of regularization, evaluated across two distinct financial tasks---numerical reasoning (FinQA) and query decomposition for RAG---to test generality.

3. **An ablation isolating the contribution of advantage-weighted fitting** beyond simple KL regularization, testing whether COPR's structured policy fitting provides benefits over naive KL-regularized SFT for preserving task performance during knowledge injection.

---

## Theoretical Motivation

We ground our comparison in a formal distinction between two classes of preservation guarantees.

### Weight-space preservation (AlphaEdit)

AlphaEdit [2] edits a single FFN weight matrix W by projecting a perturbation onto the null space of preserved knowledge. Given preserved keys K_0 (typically n = 100,000 triples from Wikipedia), the null-space projector is computed via SVD:

> {U, Lambda, U^T} = SVD(K_0 K_0^T),  P = U_hat U_hat^T

where U_hat are eigenvectors with eigenvalues below a threshold. The sequential editing solution is:

> Delta_AlphaEdit = R * K_1^T * P * (K_p K_p^T P + K_1 K_1^T P + I)^(-1)

This yields an **exact but pointwise** guarantee: (W + Delta P) K_0 = W K_0. Outputs are preserved on the finite set K_0, but no guarantee extends to unseen inputs. Moreover, the constraint applies to a single FFN layer, while task skills involve coordinated representations across many layers [10]. AlphaEdit has been tested up to ~3,000 edits; beyond this, general capabilities degrade significantly [6].

### Policy-space preservation (COPR, adapted)

COPR [11] derives the optimal updated policy as:

> pi*_t(y|x) = (1/Z_t(x)) * pi_{t-1}(y|x) * exp(r_t(x,y) / beta)

where pi_{t-1} is the previous policy and r_t encodes the learning signal. Originally, r_t is derived from human preference rankings over candidate responses Y^x = {y_1 < ... < y_J}, with a linear advantage Adv(x, y_j) = (2j - J_x - 1) / J_x.

**Our adaptation:** We replace human preference rankings with *factual correctness rankings*. For each knowledge-update prompt x (e.g., "Who is the current CEO of Company X?"), we sample K responses from the current model, rank them by ground-truth accuracy (correct answer -> rank K; hallucinations -> rank 1), and compute COPR's advantage function over these factual rankings. The training objective combines fitting on new-knowledge responses with regularization on a task replay buffer:

> L_fit(theta) = E_{x ~ D_new} sum_{y in Y^x} D_KL(P_{y,t}(y|x,theta) || P*_{y,t}(y|x))
>
> L_reg(theta) = E_{x ~ R_task} D_KL(P_{y,t}(y|x,theta) || P*_task(y|x))

where R_task is a 5% replay buffer of task training data (FinQA or query decomposition examples, depending on the task model).

This yields a **distributional** guarantee via Pinsker's inequality. If E_{x ~ D_task}[D_KL(pi_task || pi_theta)] <= epsilon, then for any bounded task metric R with ||R||_inf <= 1:

> |E_{pi_task}[R] - E_{pi_theta}[R]| <= sqrt(2 epsilon)

### Connection to EWC

The Fisher information matrix F is the Hessian of the KL divergence [4]: D_KL(p_{theta*} || p_theta) ~ (1/2)(theta - theta*)^T F(theta*) (theta - theta*). EWC's diagonal approximation is thus a lossy, parameter-level proxy for the output-level KL that COPR controls directly. Recent work shows that KL divergence from the base model predicts catastrophic forgetting with R^2 = 0.96 [9], further motivating direct KL control.

### Proposition

Let pi_task be a task-tuned policy and D_task the task input distribution.

**(a) Pointwise (AlphaEdit):** For finite S = {k_1, ..., k_n}: (W + Delta P)k_i = Wk_i for all k_i in S. No guarantee for x not in S.

**(b) Distributional (COPR):** If E_x[D_KL(pi_task || pi_theta)] <= epsilon, then |E_{pi_task}[R] - E_{pi_theta}[R]| <= sqrt(2 epsilon) for any bounded R.

For task-tuned models---where the learned skill is a distributional strategy---guarantee (b) is the appropriate abstraction.

---

## Approach

### Task models

We fine-tune Qwen2.5-3B on two financial tasks that represent distinct behavioral skills:

1. **FinQA** [3]: Multi-step numerical reasoning over financial reports (8,281 QA pairs over SEC filings with tables and text). The model must parse tables, identify quantities, and compose arithmetic programs. Models of comparable scale (phi-3-mini, 3.8B) achieve 73.5% execution accuracy with LoRA [8]. This is an established benchmark that ensures reproducibility.

2. **Financial query decomposition for RAG**: Given a financial question (e.g., "How did the Middle East conflict affect energy sector earnings in Q4 2023?"), the model generates 2--4 sub-queries whose embeddings maximize Recall@10 of target documents in a retrieval corpus. We train this via SFT on decomposition examples, optionally refined with DPO using Recall@10 as the preference signal. This task is more directly sensitive to entity knowledge---generating effective sub-queries requires knowing *which* entities, events, and relationships are relevant---making it a stronger test of whether knowledge updates interact with task performance.

Using two tasks lets us test whether our findings generalize across task types: FinQA tests preservation of a *procedural reasoning* skill, while query decomposition tests preservation of a *strategic generation* skill that depends more directly on entity knowledge.

### Knowledge updates

We define updates as structured financial fact triples---entity changes that can be applied by all methods on equal footing:

- Leadership changes: `("Apple", "CEO", "Tim Cook")` -> `("Apple", "CEO", "New Person")`
- Acquisitions: `("Company X", "acquired_by", "Company Y")`
- Financial metrics: `("Company X", "Q3_revenue", "$4.2B")` -> `("$5.1B")`

We extract these triples from FNSPID [5] (15.7M timestamped financial news records, 1999--2023) using a temporal split, focusing on entities that also appear in FinQA reports so that representational changes from knowledge injection can plausibly interact with the task. We test at three scales: 200, 1,000, and 3,000 fact edits (the last being AlphaEdit's tested limit).

### Methods compared

| Method | Type | Description |
|--------|------|-------------|
| No update | Lower bound | Task-tuned model, no knowledge injection |
| Naive SFT | Baseline | SFT on fact QA pairs, no regularization |
| KL-reg SFT | Baseline | SFT + lambda * D_KL(pi_task \|\| pi_theta) |
| Mixed replay | Baseline | SFT on new facts + 5% task data shuffled |
| AlphaEdit | Weight-space | Null-space projected fact editing |
| COPR-adapted | Policy-space | Advantage-weighted fitting + task replay |
| Full retrain | Upper bound | Train from scratch on all data |

This design isolates three key comparisons:

- **Naive SFT -> KL-reg SFT -> COPR-adapted**: Does KL regularization help? Does advantage-weighted fitting help *beyond* KL regularization?
- **AlphaEdit vs. COPR-adapted**: Weight-space vs. policy-space, head-to-head on the same fact triples.
- **Scaling**: At what edit count does each method begin degrading task performance?

### COPR adaptation pipeline (detailed)

For each fact triple to inject:

1. Convert the triple to a natural-language question (e.g., "Who is the CEO of Apple?").
2. Sample K = 8 responses from the current task-tuned model.
3. Rank responses by factual correctness using the ground-truth answer (exact match -> top rank; partial match -> middle; hallucination -> bottom).
4. Compute COPR's linear advantage: Adv(x, y_j) = (2j - K - 1) / K.
5. Compute the renormalized optimal sampling distribution P*.
6. Train via L_fit on these ranked responses.
7. Simultaneously train via L_reg on a 5% replay buffer of task training examples (FinQA or query decomposition).

---

## Dataset

- **Task 1 benchmark:** FinQA [3]---8,281 QA pairs requiring multi-step numerical reasoning over financial reports. Train/dev/test split per the original paper.
- **Task 2 data:** We construct query decomposition training data from FNSPID [5] by pairing financial questions with gold sub-query decompositions, using the FNSPID news corpus as the retrieval target. We evaluate via Recall@10: given a question, the model's generated sub-queries are embedded and used to retrieve from the corpus, and we measure whether the gold-relevant documents appear in the top 10. Training data is generated using a teacher LLM and filtered by retrieval performance.
- **Knowledge source:** FNSPID---15.7M timestamped financial news records (1999--2023) across 4,775 S&P 500 companies. We extract structured fact triples from articles after a temporal cutoff, filtered to entities appearing in FinQA reports and/or the retrieval corpus.
- **Fact triple extraction:** We use an LLM to extract (subject, relation, object) triples from post-cutoff FNSPID articles, then verify against ground truth. This is *not* a dataset contribution---it is a preprocessing step to create fair inputs for all methods.

---

## Metrics

For each method, task, and edit scale, we evaluate:

1. **Task preservation:** FinQA execution accuracy (Task 1) and Recall@10 on held-out queries (Task 2).
2. **Knowledge absorption:** Accuracy on fact-probe questions about the injected triples (e.g., "Who is the CEO of Apple?").
3. **Scaling behavior:** Task preservation and knowledge absorption as a function of edit count (200, 1K, 3K).
4. **Compute cost:** GPU-hours per update.

We expect Task 2 (query decomposition) to be *more sensitive* to knowledge updates than Task 1 (FinQA), since generating effective sub-queries depends directly on entity knowledge, while numerical reasoning over tables is more procedural. If this differential sensitivity is observed, it strengthens the argument that the choice of update method matters most when the task and knowledge are tightly coupled.

---

## References

[1] Lazaridou et al. "Mind the gap: Assessing temporal generalization in neural language models." NeurIPS, 2021.

[2] Fang et al. "AlphaEdit: Null-space constrained knowledge editing for language models." ICLR (Outstanding Paper), 2025.

[3] Chen et al. "FinQA: A dataset of numerical reasoning over financial data." EMNLP, 2021.

[4] Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks." PNAS 114(13):3521--3526, 2017.

[5] Dong et al. "FNSPID: A comprehensive financial news dataset in time series." KDD, 2024.

[6] Li et al. "Should we really edit language models? On the evaluation of edited language models." NeurIPS, 2024.

[7] Lin et al. "Mitigating the alignment tax of RLHF." arXiv:2309.06256, 2024.

[8] Loukas et al. "Fine-tuning smaller language models for question answering over financial documents." arXiv:2408.12337, 2024.

[9] Razak et al. "RL's Razor: Why online reinforcement learning forgets less." arXiv:2509.04259, 2025.

[10] Wang et al. "Can fine-tuning erase your edits?" arXiv:2511.05852, 2025.

[11] Zhang et al. "COPR: Continual learning human preference through optimal policy regularization." ACL Findings, 2025.

[12] Zhang et al. "CPPO: Continual learning for reinforcement learning with human feedback." ICLR, 2024.
