# Regime Matters: Evaluating COPR for Knowledge Injection Under Batch and Sequential Update Schedules

## Abstract

Knowledge injection methods for large language models are typically benchmarked under a single update regime, which obscures how their inductive biases interact with deployment schedules. This paper evaluates six LoRA-based update procedures, including three novel variants of Continual-Optimization Policy Ranking (COPR, Zhang et al. 2025), on a single backbone (Qwen3-4B-Instruct-2507, task-tuned for query decomposition over financial news) across a batch regime (1K and 3K edits at once) and a sequential regime of fifteen rounds of 200 disjoint edits. The headline finding is a negative transfer: a simple KL-regularized SFT baseline dominates the COPR family on absorption in *both* regimes --- at 3K batch edits by a 36-40% relative gap at 10-20x less compute, and across 15 sequential rounds by a 9-3 win tally (2 ties) on per-round absorption, with an 0.114 versus 0.107-0.110 lead on trajectory-averaged absorption. An earlier draft reported a +21% absorption advantage for a gold-injection COPR variant at round 10; we extended the stream to round 15 and the advantage did not replicate (three scattered wins, not a trend). What survives is locality: `copr_gold_injection_anchored` sustains a +47% overall locality gap over `kl_reg_sft` to round 15 and +17% on the 15-round average, at 10-12x per-round compute. `copr_anchored` (anchoring without gold injection) is reported as a clean negative result (absorption F1 0.066, compositional F1 0.025, below the no-update baseline). Two mechanistic analyses ground these findings. A LoRA-delta analysis shows that all four COPR variants share a near-identical top input subspace (first-principal-angle cosine 0.997 pairwise) and differ mainly in stable rank and in total Frobenius norm (`copr_gold_injection_anchored` highest at 1.472). A hidden-state geometry probe shows that every method shifts the QA-direction representation substantially more than the QD-direction representation for the same subject: plain `copr` and `copr_anchored` actively anti-align (cosine drops by -0.14 and -0.10), while gold injection is the only mechanism that partially couples the two formats (shift ratio 1.44 at best, versus 2.1-3.2 elsewhere). The paper argues that gold injection collapses COPR toward KL-regularized SFT --- which explains why it matches but does not beat KL-SFT on absorption --- and that the format-coupling signature is a property of gradient-based editing under a QA-formatted loss, not a LoRA artifact, citing the Reversal-Curse and knowledge-storage-vs-extraction literature in support. Numbers are quoted verbatim from Phase 1-7 CSV artifacts; standard errors and probe counts are reported with the means.

## 1. Introduction

Injecting new factual knowledge into the parameters of an already-tuned language model is an operational need in any deployment where the world changes after training. Existing methods fall into two broad families: constrained parameter edits that localize changes to a small set of MLP weights (ROME, Meng et al. 2022; MEMIT, Meng et al. 2023), and fine-tuning-style updates that use gradient descent on edit-specific objectives (SFT, KL-regularized SFT, direct-preference-style objectives such as KDPO, Rafailov-style preference contrast extended to editing; COPR, Zhang et al. 2025). A third family --- retrieval-augmented generation --- side-steps parametric editing entirely and is therefore out of scope here.

Most of the editing literature reports results in one of two default postures: a single batch of edits applied once, or a stream of single edits applied one at a time with or without checkpoint chaining. These two postures correspond to meaningfully different operational settings. A batch update is a push: a data-engineering pipeline produces a list of new facts and a training loop internalizes them in a single pass. A sequential update is a drumbeat: disjoint batches of facts arrive on a schedule, each round inherits the previous round's checkpoint, and the model must absorb the new batch without catastrophic interference from prior rounds. The operational costs and the failure modes differ, and so does the appropriate inductive bias of the update method.

This paper tests the hypothesis that the choice of knowledge-injection method depends critically on the update regime, and reports a negative transfer result: at 4B-LoRA scale, the simple KL-regularized SFT baseline dominates the COPR family in *both* regimes on absorption, and the only metric on which a COPR variant holds a durable advantage is overall locality under the novel `copr_gold_injection_anchored` variant. The claim is supported by a controlled experiment on a single backbone (Qwen3-4B-Instruct-2507, LoRA-task-tuned for query decomposition on FNSPID pre-2022 news) with disjoint triple batches drawn from the same distribution, holding data, evaluation, compute budget (on a per-update basis), and seed fixed across methods. Three contributions follow.

First, the paper introduces **gold-answer injection** for COPR as a novel modification to the candidate set, motivated by a failure mode we call the *K-sample-all-wrong pathology*. The original COPR design (Zhang et al. 2025) samples K candidate completions from the current policy and fits a reference-anchored preference ranking among them. If all K self-samples miss the new gold fact --- a near-universal failure mode when the fact is genuinely unknown to the base model --- the ranking contains no gradient signal toward the correct answer. Injecting the gold completion as an additional candidate guarantees a correct anchor in the ranking at every step. Mechanistically, gold injection collapses COPR toward KL-regularized SFT: once the gold candidate dominates the rank signal, the MSE fit approximates cross-entropy on the gold answer with the reference term as a side constraint. Phase 6 and Phase 7 evidence (largest Frobenius norms, smallest hidden-state shift ratios) is consistent with this collapse.

Second, the paper introduces **anchored COPR** --- an attempt to improve the reference policy by mixing a task-replay-normalized reference into the per-candidate log pi_ref. This is reported as a clean negative result on its own: `copr_anchored` absorption F1 collapses to 0.066 and the compositional F1 drops to 0.025, below the no-update baseline of 0.052. Anchoring *alone* is harmful because the anchor is unopposed when no gold signal is in the candidate set. The recovery variant, `copr_gold_injection_anchored` (gold injection + anchoring), is the paper's best locality method at a +47% gap over `kl_reg_sft` at round 15 and is the only COPR variant with a durable advantage over KL-regularized SFT on any metric.

Third, the paper provides a **regime characterization** grounded in two mechanistic analyses --- a LoRA-delta subspace analysis (Phase 6) and a hidden-state geometry probe (Phase 7) --- alongside a 15-round sequential sweep that explicitly disconfirms a round-10 "inversion" earlier drafts reported. All four COPR variants write into nearly the same top input subspace (first-principal-angle cosine 0.997 pairwise) and differ in stable rank (`copr_gold_injection` narrowest at 2.54) and total Frobenius norm (`copr_gold_injection_anchored` highest at 1.472). The hidden-state probe shows every method --- including the gold-injection variants --- moves the QA-direction representation substantially more than the QD-direction representation for the same subject: plain `copr` and `copr_anchored` actively anti-align (cosine drops -0.14 and -0.10), while gold injection partially couples the two formats (shift ratio 1.44 at best versus 2.1-3.2 elsewhere). The paper argues this format-coupling signature is a property of gradient-based editing under QA-only loss rather than a LoRA artifact, placing the observation in the Reversal-Curse and knowledge-storage-vs-extraction literature.

The central finding is: **KL-regularized SFT is the strongest absorption method at 4B/LoRA across both regimes tested**; its absorption dominance is 36-40% at 3K batch and 9-3 in a 14-round sequential head-to-head with `copr_gold_injection`; the round-10 inversion earlier drafts reported was an outlier that did not replicate in rounds 11-15. **The sequential regime's one durable COPR advantage is locality under `copr_gold_injection_anchored`**: +47% at round 15, +17% on 15-round average, at 10-12x per-round compute. All compositional (two-hop) edits degrade bridging-entity recall below the no-update baseline --- a shared limitation the paper reports prominently rather than obscuring.

## 2. Related Work

**Parametric knowledge editing.** ROME (Meng et al. 2022) formulates single-fact editing as a rank-one update to a single MLP layer identified via causal tracing. MEMIT (Meng et al. 2023) extends this to mass-editing thousands of facts at once via a layer-wise least-squares formulation. MEND (Mitchell et al. 2022a) trains a hypernetwork that maps an edit-example gradient into a low-rank weight update, while SERAC (Mitchell et al. 2022b) keeps edits in an external memory with a scope classifier and counterfactual model. Knowledge-Neurons (Dai et al. 2022) identifies individual MLP neurons whose activations are causally tied to factual predictions. The Yao et al. (2023) survey taxonomizes these memory-based, meta-learning, and locate-then-edit families and explicitly flags continual and compositional evaluation as open gaps --- the two gaps the present paper's Phases 3-4 target. All five methods above are batch or single-shot editors; none is evaluated under a stream of disjoint edit rounds with checkpoint chaining, which is the sequential regime studied here.

**Preference-style editing.** KDPO (Rozner et al. 2024) is the closest prior work: it casts a single fact edit as a DPO (Rafailov et al. 2023) problem --- the new fact is the preferred completion and the old fact the rejected one --- with the pre-edit model as the KL reference. Two design points distinguish it from the COPR variants studied here. First, KDPO uses the pre-edit base as the KL anchor, which drifts as edits accumulate over rounds; COPR's KL-replay anchor (Section 3.2) regularizes instead against a task-replay mixture, which does not drift with the policy. Second, KDPO does not inject gold answers into a candidate pool --- it is a two-way DPO contrast, not a K-way ranking. Gold injection (this paper) addresses the K-sample-all-wrong failure mode that KDPO side-steps by construction but cannot exploit. IPO (Azar et al. 2024) and related work on DPO overfitting motivate the KL regularization terms in both `kl_reg_sft` and the COPR family.

**COPR and continual preference alignment.** COPR (Zhang et al. 2025, ACL Findings) was proposed for continual *preference* alignment: maintaining helpfulness/harmlessness across a stream of preference-labeled tasks via KL regularization anchored on a task-replay buffer, avoiding catastrophic alignment drift. CPPO (Zhang et al. 2024) is a parallel continual-RLHF line with replay. The present paper ports COPR's inductive bias from preference tasks to knowledge injection, treating each new fact as a mini preference task with the gold completion as the preferred candidate. The novel gold-injection modification (Section 3.3) is specifically required because, unlike in preference alignment where the preferred and rejected completions are both reasonable behaviors, in knowledge injection the pre-edit policy puts near-zero probability on the gold answer and self-samples will therefore all miss it.

**LoRA, PEFT, and LoRA-based editing.** Low-rank adaptation (Hu et al. 2022) is the training substrate used throughout; its rank-r structure implies a strong inductive bias toward a low-dimensional update subspace, which grounds the Phase 6 mechanistic analysis. Aghajanyan et al. (2021) give the empirical evidence that fine-tuning occupies an intrinsically low-dimensional subspace and justifies why rank 16-32 suffices for absorbing a fact. MELO (Yu et al. 2024) and LoRAHub (Huang et al. 2023) validate LoRA as a medium for editing and for cross-task composition respectively; the present paper stays within a single LoRA slot and lets the training objective arbitrate conflicts. Zhu et al. (2024) analyze asymmetries in LoRA B-vs-A subspaces, which motivates the first-principal-angle (input-subspace) metric used in Phase 6.

**Continual learning for LLMs.** InsCL (Wang et al. 2024) demonstrates that a similarity-selected task-replay buffer beats EWC-style regularization (Kirkpatrick et al. 2017) on instruction-tuning continual learning; Luo et al. (2023) empirically documents catastrophic forgetting under continual SFT, directly motivating the task-preservation axis (Recall@10 on qd_temporal_test) in Section 5. The Wu et al. (2024) continual-learning-for-LLMs survey places the methods here in the replay-plus-regularization quadrant of the design space. O-LoRA (Wang et al. 2023) proposes orthogonal subspace learning for continual tuning; the Phase 6 subspace-overlap analysis is partially a response to that line, and finds that the COPR variants do *not* occupy orthogonal subspaces --- they share a near-identical first principal direction.

**Compositional and multi-hop evaluation.** Cohen et al. (2024) shows that editing a single fact rarely propagates to its logical consequences (e.g., editing a spouse does not update "mother-in-law"); the RippleEdits benchmark is the direct motivation for Phase 4. MQuAKE (Zhong et al. 2023) is methodologically closest to the compositional probe set used here; the 500 two-hop probes (Section 4) are constructed on the same "edited-fact-as-bridge" pattern, adapted to financial triples, with second-hop templates inspired by HotpotQA (Yang et al. 2018), MuSiQue (Trivedi et al. 2022), and 2WikiMultihopQA (Ho et al. 2020). The degradation of bridging-entity recall across all updated methods (Section 5.3) is consistent with Cohen et al.'s ripple-effect finding.

**Knowledge-editing benchmarks.** Counterfact (Meng et al. 2022) and zsRE (Levy et al. 2017, re-used by De Cao et al. 2021) provide ground-truth edit tuples. This paper does not use these directly; it constructs its own fact-triple dataset from FNSPID financial news (Shah et al. 2023), which allows a fair temporal cutoff and a task-domain (query-decomposition) test set that the baseline is skilled at.

## 3. Methods

### 3.1 Problem formulation

Let f_theta be a task-tuned LM and let T = {(x_i, y_i)} be a set of new fact triples, each rendered as a question-answer pair (x_i, y_i). The knowledge-injection problem is to produce an updated model f_theta' that (i) **absorbs** the new facts --- i.e., generates y_i given x_i --- while (ii) **preserving** task capability on an in-domain test set, (iii) **localizing** the change so that unrelated facts are not disturbed, and (iv) **composing** with pre-existing knowledge for multi-hop reasoning. This paper instantiates the four objectives with four metric families:

- **Preservation**: Recall@10 of the task-tuned query-decomposition model on the held-out qd_temporal test split (n=104).
- **Absorption**: For each injected fact, exact-match, token-F1, and a paraphrase-worst-case F1 across k paraphrases of (x_i). Reported as fact-level means and worst-case F1.
- **Locality**: Token-F1 on three strata of unrelated facts --- same entity (n=182 probes), other sector (n=1817), same sector (n=1). The n=1 stratum is acknowledged and discussed in Limitations.
- **Compositional**: Two-hop QA probes (n=500, generated with Gemini-3.1-Flash-Lite) measuring exact-match, final-answer containment, bridging-entity recall, and token-F1.

### 3.2 Base objective and methods

Let pi_theta denote the update policy (task-tuned backbone + update LoRA). Let pi_ref denote the pre-update reference. Let D_T be a mini-batch of fact QA pairs and D_R a task-replay mini-batch from the qd_temporal train split.

**naive_sft.** Standard cross-entropy on D_T only, no replay, no reference regularizer.

**kl_reg_sft.** Cross-entropy on D_T plus lambda * KL(pi_theta || pi_ref) on D_R, with lambda = 0.1. This is a strong generic baseline that pays for an extra replay forward pass but no candidate sampling.

**copr (paper-faithful).** For each (x_i, y_i) in D_T, sample K=8 candidate completions from the current pi_theta, score each against y_i with token-F1, rank, and fit pi_theta's per-candidate log-prob to the reference-anchored rank signal with an MSE loss. Add a gated gold-NLL SFT anchor (weight alpha=0.25, gate threshold 0.5, `always_apply=False`) and a KL replay loss on D_R with `replay_pct=0.05`. beta=0.1, partial_match_threshold=0.5, max_new_tokens=128.

**copr_gold_injection (novel).** As above, except the gold completion y_i is injected as one of the K candidates (replacing the lowest-ranked self-sample). SFT anchor is OFF (alpha=0.0). This is the core novel contribution: it guarantees a correct anchor in the ranking even when self-samples are uniformly wrong, which is typical when the fact is genuinely unknown to pi_theta.

**copr_gold_injection_anchored (novel).** Gold injection ON plus the gated gold-NLL SFT anchor (alpha=0.25). This tests whether the two ideas compose or interfere.

**copr_anchored (novel, negative result).** Paper-faithful COPR, plus a task-replay-normalized reference policy: log pi_ref'(y|x) = log pi_ref(y|x) + task_anchor_alpha * logmeanexp_{y_j ~ D_R}(log pi_ref(y_j|x_j)), with task_anchor_alpha=0.3 and task_anchor_n_samples=4. Motivation: the reference should be anchored to the task manifold so that candidates that drift off-task are penalized. Phase 3 reports that this harms rather than helps.

### 3.3 Why gold injection

In pilot runs, the K=8 self-samples for a novel fact were almost always all incorrect by F1 (worst-F1 near zero), so the COPR ranking was a ranking over wrong answers. The MSE fit then reinforces whichever wrong answer is most plausible, rather than pulling the policy toward the correct one. Injecting gold into the candidate set guarantees that the top-ranked candidate is correct and that the MSE fit has a non-degenerate target. Phase 7 confirms a corresponding representational effect: `copr_gold_injection` and `copr_gold_injection_anchored` are the only methods whose hidden-state shift ratio `direct/related` drops toward 1.0 (1.60 and 1.44 respectively, versus 2.1-3.2 for all other methods), i.e., they are the only methods that move the QD-direction representation substantially at all. Phase 6 shows they also carry the two largest Frobenius norms (1.43, 1.47), consistent with a larger effective update driven by a correct target in the ranking.

### 3.4 Training setup

Update-time LoRA: r=16, alpha=32, on attention Q/K/V/O and MLP up/down/gate projections of the merged task-tuned backbone (itself trained at r=32/alpha=64). Optimizer AdamW, lr=2e-5 for SFT variants, lr=1e-5 for COPR variants, 3 epochs per update, bf16, batch size 8 for SFT (grad_accum=2) and 4 for COPR (grad_accum=4), max_seq_length=512. Seed=42 across all runs. Batch regime (Phases 1-2) trains a single update on the full batch; sequential regime (Phase 3) chains checkpoints: round k starts from round k-1's `model/` directory.

## 4. Experimental Setup

**Backbone.** Qwen3-4B-Instruct-2507, bf16. Task tuning: LoRA r=32/alpha=64 on query-decomposition SFT (FNSPID financial news, pre-2022 cutoff split, n=414 train / 104 test / 518 post_test). The task-tuned LoRA is merged before update-time LoRA is applied.

**Data.** 96,897 filtered fact triples from FNSPID (pre-cutoff only). Batch scales: 200, 1000, 3000 triples. Sequential: 10 rounds of 200 disjoint triples each (sampled once with seed=42; metadata in `data/fnspid/triples/sequential/metadata.json`). Compositional probes: 500 two-hop probes generated with Gemini-3.1-Flash-Lite from the same triple pool. Locality facts: 96,697 unrelated FNSPID facts sampled into the three strata.

**Metrics.** See Section 3.1. All absorption metrics are computed on a paraphrase-augmented probe set (n_probes ranges from 395 to 6000 depending on scale). Preservation uses Recall@10 on the qd_temporal test split (n=104). Compositional uses n=500 two-hop probes.

**Compute.** All runs executed on a single NVIDIA A100-80GB (or equivalent) with peak memory 6-20 GB depending on method. Per-run wall-clock is in `final_results/run_metadata.csv`; aggregated gpu_hours is reproduced in the results tables below. Compute is reported prominently because it is a first-class factor in the paper's conclusion.

**Reporting.** Every number in the results tables is copied verbatim from the corresponding row of the CSV artifacts in `final_results/`. No number is re-derived from raw logs. Standard errors are reported where meaningful; where they are not (compositional with n=500, no bootstraps computed) they are omitted and the paper does not make significance claims.

## 5. Results

Results are organized by regime. Each subsection has a table, a short qualitative narrative, and an honest accounting of where differences are small or noisy. Preservation here is Recall@10; reported as mean +/- std, n=104 unless noted.

### 5.1 Batch regime

**Scale 1000 (Phase 1).**

| Method | Preservation (mean +/- std, n=104) | Absorption F1 | Absorption worst-F1 | Locality overall F1 (n=2000) | GPU-h |
|---|---|---|---|---|---|
| naive_sft | 0.204 +/- 0.330 | 0.121 | 0.091 | 0.056 | 0.057 |
| kl_reg_sft | 0.236 +/- 0.336 | 0.141 | 0.120 | 0.062 | 0.190 |
| copr | 0.213 +/- 0.318 | 0.084 | 0.052 | 0.028 | 3.08 |
| copr_gold_injection | 0.234 +/- 0.342 | 0.096 | 0.061 | 0.029 | 3.01 |
| copr_gold_injection_anchored | 0.215 +/- 0.329 | 0.094 | 0.060 | 0.030 | 3.18 |
| copr_anchored | 0.215 +/- 0.323 | 0.104 | 0.066 | 0.037 | 4.04 |

**Scale 3000 (Phase 2).**

| Method | Preservation | Absorption F1 | Absorption worst-F1 | Locality overall F1 | GPU-h |
|---|---|---|---|---|---|
| naive_sft | 0.123 +/- 0.273 | 0.142 | 0.112 | 0.059 | 0.184 |
| kl_reg_sft | 0.253 +/- 0.354 | 0.173 | 0.142 | 0.071 | 0.569 |
| copr | 0.237 +/- 0.350 | 0.104 | 0.061 | 0.026 | 9.05 |
| copr_gold_injection | 0.264 +/- 0.367 | 0.108 | 0.063 | 0.030 | 8.90 |
| copr_gold_injection_anchored | 0.245 +/- 0.347 | 0.111 | 0.064 | 0.029 | 9.68 |
| copr_anchored | not run at 3K | | | | |

`kl_reg_sft` is the best method at batch scale 3000 on absorption F1 (0.173) and locality overall F1 (0.071), and preserves task capability at 0.253 +/- 0.354 (within one standard error of the best). The COPR family runs at 9-10 GPU-hours for absorption F1 in the 0.104-0.111 band --- a 36-40% relative gap below `kl_reg_sft` absorption (gaps of 39.9%, 37.6%, 35.8% for `copr`, `copr_gi`, `copr_gi_anchored` respectively) --- and locality F1 roughly 0.03 versus `kl_reg_sft` at 0.071 (a 58% relative gap). `naive_sft` degrades preservation sharply at 3K (0.123, below all other methods) while improving absorption less than `kl_reg_sft`, illustrating why the KL regularizer is worth its cost. The COPR premium is substantial (10-20x `kl_reg_sft`) and does not pay off in batch at the scales tested.

The standard deviation on preservation is 0.27-0.37 across methods at n=104, meaning the standard error on each mean is roughly 0.027-0.036. Many of the preservation differences between methods are within one SE of each other, and the paper does not claim significance on preservation alone.

### 5.2 Sequential regime

Phase 3 runs 15 rounds of 200 disjoint fact triples per method, chaining checkpoints. The extended stream (rounds 11-15) was run after an earlier draft claimed a round-10 absorption advantage for `copr_gold_injection` that we wanted to verify rather than report opportunistically. The round-10 numbers are preserved below for continuity with the earlier draft and for comparison to the round-15 endpoint.

**Round-15 endpoint (primary).**

| Method | Preservation | Absorption F1 | Absorption worst-F1 | Locality overall F1 | GPU-h (per round, avg) |
|---|---|---|---|---|---|
| naive_sft | 0.256 +/- 0.345 | 0.088 | 0.041 | 0.046 | 0.013 |
| **kl_reg_sft** | 0.240 +/- 0.344 | **0.118** | 0.090 | 0.048 | 0.051 |
| copr | 0.228 +/- 0.338 | 0.062 | 0.016 | 0.029 | 0.60 |
| copr_gold_injection | 0.213 +/- 0.331 | 0.115 | 0.086 | 0.064 | 0.58 |
| **copr_gold_injection_anchored** | 0.191 +/- 0.321 | 0.119 | 0.084 | **0.071** | 0.65 |
| copr_anchored (negative) | 0.258 +/- 0.355 | 0.076 | 0.023 | 0.037 | 0.62 |

At the true endpoint (round 15): **absorption is essentially tied between `kl_reg_sft` (0.118), `copr_gold_injection_anchored` (0.119), and `copr_gold_injection` (0.115)**. The +21% absorption "win" that `copr_gold_injection` held at round 10 (0.162 vs 0.134) does not survive to round 15. Locality, however, does sustain: `copr_gold_injection_anchored` at 0.071 versus `kl_reg_sft` at 0.048 is a +47% gap that persists to the endpoint. Preservation is 0.10-0.15 SE apart across methods at round 15 except for `copr_gold_injection_anchored` (0.191, roughly 1.5 SE below `kl_reg_sft`), so preservation differences are mostly within noise; the locality win again comes paired with a preservation cost.

**Round-10 endpoint (for comparison with earlier drafts).** `copr_gold_injection` absorption 0.162 vs `kl_reg_sft` 0.134 (+21%); `copr_gold_injection_anchored` locality 0.077 vs `kl_reg_sft` 0.051 (+50%). These numbers were the basis of a "sequential inversion" claim in earlier drafts. The extended stream (below) shows the absorption gap was not replicated. The locality gap was.

**Head-to-head tally across all 14 rounds where both methods have data.** We count each round as a "win" for the method with higher absorption F1 when the gap is > 0.005; "tie" otherwise.

| Round | kl_reg_sft | copr_gold_injection | Δ | Winner |
|---|---|---|---|---|
| 1 | 0.082 | 0.069 | -0.013 | kl_reg |
| 2 | 0.112 | 0.086 | -0.026 | kl_reg |
| 3 | 0.086 | 0.081 | -0.005 | kl_reg |
| 4 | 0.121 | 0.115 | -0.007 | kl_reg |
| 5 | 0.110 | 0.094 | -0.016 | kl_reg |
| 6 | 0.141 | 0.118 | -0.023 | kl_reg |
| 7 | 0.135 | 0.127 | -0.008 | kl_reg |
| 8 | 0.112 | 0.109 | -0.003 | tie |
| 9 | 0.100 | (missing) | --- | --- |
| 10 | 0.134 | 0.162 | +0.028 | copr_gi |
| 11 | 0.132 | 0.111 | -0.021 | kl_reg |
| 12 | 0.098 | 0.115 | +0.017 | copr_gi |
| 13 | 0.109 | 0.129 | +0.020 | copr_gi |
| 14 | 0.119 | 0.108 | -0.010 | kl_reg |
| 15 | 0.118 | 0.115 | -0.003 | tie |

Across 14 rounds with data: `kl_reg_sft` wins 9, `copr_gold_injection` wins 3, tied 2. **The round-10 result was one of three scattered wins, not the start of a trend.** No sustained absorption advantage exists at 4B/LoRA in our data.

**15-round averages (over rounds with data).**

| Method | Rounds | Preservation mean | Absorption mean | Locality mean |
|---|---|---|---|---|
| naive_sft | 15 | 0.214 | 0.081 | 0.036 |
| **kl_reg_sft** | 15 | 0.227 | **0.114** | 0.050 |
| copr | 14 | 0.231 | 0.076 | 0.026 |
| copr_gold_injection | 14 | 0.224 | 0.110 | 0.052 |
| **copr_gold_injection_anchored** | 14 | 0.199 | 0.107 | **0.059** |
| copr_anchored | 13 | 0.237 | 0.064 | 0.029 |

**`kl_reg_sft` holds the highest trajectory-averaged absorption across all 15 rounds** (0.114), narrowly beating the gold-injection variants (0.107-0.110). `copr_gold_injection_anchored` holds the highest trajectory-averaged locality (0.059 vs 0.050 for `kl_reg_sft`, +17%). Preservation is a wash (all within roughly 0.01 except naive_sft's low).

The negative result on `copr_anchored` stands: absorption stays below 0.089 throughout (never exceeds that), compositional token-F1 at round 10 is 0.025 (below no-update baseline 0.052). The mechanism is direct: with no gold in the candidate set and a task-replay-normalized reference pushing the policy toward the task manifold, there is no correct signal to offset the anchor pressure, and absorption does not happen.

**Trajectory gaps.** Rounds 6, 7 are missing for `copr_anchored` (faiss crash in the initial sweep, not re-run); round 9 is missing for `copr`, `copr_gold_injection`, `copr_gold_injection_anchored` (a single-batch failure that did not propagate the chain). The paper does not impute and reports means only over rounds with data.

**Figure 1 (placeholder).** Per-round absorption F1 (y-axis, 0-0.20) versus round index (1-15) with one line per method. Caption: "Absorption trajectories over 15 sequential rounds. `kl_reg_sft` leads or ties in 11/14 contested rounds against `copr_gold_injection`; the round-10 spike for `copr_gold_injection` does not persist. Absorption at round 15 is tied among `kl_reg_sft`, `copr_gold_injection`, and `copr_gold_injection_anchored` within 0.004 F1." Data source: `final_results/phase3_sequential_trajectory.csv`.

**Figure 2 (placeholder).** Per-round locality overall F1 over 15 rounds. Caption: "`copr_gold_injection_anchored` locality climbs to and sustains roughly 0.07 at the stream endpoint, surpassing `kl_reg_sft` (0.048 at round 15) by +47%." Data source: same.

### 5.3 Compositional (Phase 4)

The compositional metric is a two-hop question constructed from the injected triples and a second hop through the base model's prior knowledge (n=500 probes, generated by Gemini-3.1-Flash-Lite). Round-10 checkpoints are evaluated:

| Method | Exact match | Contains final answer | Contains bridging entity | Token F1 |
|---|---|---|---|---|
| no_update_baseline | 0.000 | 0.014 | **0.102** | 0.052 |
| naive_sft | 0.002 | 0.018 | 0.058 | 0.069 |
| kl_reg_sft | 0.014 | 0.016 | 0.024 | 0.083 |
| copr | 0.000 | 0.014 | 0.072 | 0.066 |
| copr_gold_injection | 0.012 | 0.018 | 0.042 | 0.088 |
| copr_gold_injection_anchored | 0.010 | 0.016 | 0.052 | 0.088 |
| copr_anchored | 0.004 | 0.010 | 0.014 | 0.025 |

Two observations. First, token-F1 on compositional improves over no_update for all methods except `copr_anchored` (0.025 vs 0.052), and the gold-injection variants tie at 0.088 --- the best on this metric. Second, and importantly, **every update method degrades bridging-entity recall** below the no-update baseline: no_update 0.102 versus the best updated method (`copr`) at 0.072 and the worst (`copr_anchored`) at 0.014. The updates absorb the new fact but interfere with the model's ability to retrieve the bridging entity from its prior knowledge. This is a shared limitation of all parametric edits at this scale and is not solved by any of the six methods.

Exact-match on compositional is near zero across the board (0-0.014); two-hop generation is a hard task at 4B parameters and the paper does not claim it as a strength.

### 5.4 Mechanistic (Phase 6)

Phase 6 analyzes the per-site (per-LoRA-injection-site) delta W = B A * scaling for the round-10 checkpoints of the four COPR variants. 252 sites per adapter (all attention Q/K/V/O and MLP up/down/gate across 32 layers).

**Overall magnitude and rank.**

| Method | Total Frobenius norm | Mean stable rank | Mean effective rank |
|---|---|---|---|
| copr | 1.325 | 3.19 | 9.36 |
| copr_gi | 1.433 | 2.54 | 7.75 |
| copr_gi_anchored | 1.472 | 2.72 | 8.30 |
| copr_anchored | 1.282 | 2.62 | 7.89 |

`copr_gi_anchored` has the highest total Frobenius norm at 1.472, followed by `copr_gi` at 1.433, `copr` at 1.325, and `copr_anchored` at 1.282. `copr_gi` has the narrowest stable rank (2.54) --- it writes a middling amount, concentrated on fewer directions than any other variant. Gold injection (with or without anchoring) produces the two highest Frobenius norms, consistent with the picture that the correct-answer signal in the ranking drives a larger effective update than a ranking over self-samples alone.

**Subspace overlap.** Pairwise principal-angle cosines between the *input* subspaces (spanned by the A matrices, averaged over 252 sites):

| Pair | Mean cos (first angle) | Mean cos (all angles) |
|---|---|---|
| copr vs copr_gi | 0.997 | 0.833 |
| copr vs copr_gi_anchored | 0.997 | 0.835 |
| copr vs copr_anchored | 0.997 | 0.819 |
| copr_gi vs copr_gi_anchored | 0.997 | 0.868 |
| copr_gi vs copr_anchored | 0.997 | 0.831 |
| copr_gi_anchored vs copr_anchored | 0.997 | 0.823 |

All four variants share an essentially identical first-principal-direction input subspace (cos = 0.997). The differences among them live in the lower-rank tail (mean cos over all angles drops to 0.82-0.87). This is consistent with a picture in which the LoRA adapter has a strong top direction dictated by the task and the fact distribution, with method-specific refinements in the remaining rank-15 subspace.

**Per-module distribution.** From the per-module Frobenius norms (`phase6_lora_deltas.csv`, aggregated across all 32 layers), all four variants write most heavily to `down_proj`, `up_proj`, `gate_proj`, and `q_proj`, with `k_proj` and `v_proj` receiving materially less mass. The published aggregate CSV does not resolve per-layer distribution; a per-layer breakdown of the saved adapters is future work and would allow a direct test of whether `copr_gi`'s absorption lead corresponds to earlier-layer lexical rewriting, as we conjectured in a prior draft.

**Figure 3 (placeholder).** Per-module Frobenius norm across the four COPR variants. Caption: "All four variants concentrate mass in `down_proj`, `up_proj`, `gate_proj`, and `q_proj`; `k_proj` and `v_proj` receive materially less." Data source: `final_results/phase6_lora_deltas.csv`.

### 5.5 Hidden-state geometry (Phase 7)

Phase 6 characterizes *what the LoRA deltas look like*. Phase 7 asks a different question: *do the updates move the model's internal representations in a way that couples the injected fact to the downstream task format?* For each of 50 injected facts, we construct three prompts:

- **Q_direct**: the QA-style prompt used during the update (e.g., `Who is X's CEO?`).
- **Q_task_related**: a QD-style prompt whose input references the same subject X (the task format the preservation metric targets).
- **Q_task_unrelated**: a QD-style prompt with an unrelated post-cutoff subject (control).

We extract the final-layer hidden state at the last prompt token (Qwen3-4B, `output_hidden_states=True`) and report three quantities per method: mean cosine similarity `cos(h_direct, h_related)`, `cos(h_direct, h_unrelated)`, and the shift ratio `||h_direct(updated) - h_direct(baseline)|| / ||h_related(updated) - h_related(baseline)||`. Baseline is the no-update checkpoint. A ratio near 1 with rising cosine indicates *subject-level integration*; a ratio substantially above 1 with flat or dropping cosine indicates *format-coupling* --- the update moves the QA direction but leaves the QD direction approximately fixed. Full numbers are in `final_results/phase7_manifold_analysis.csv` with per-fact pairs in `phase7_manifold_pairs.json` (n=50 facts).

| Method | cos(h_direct, h_related) | Δcos vs no_update | shift_direct | shift_related | shift ratio (direct / related) |
|---|---|---|---|---|---|
| no_update | 0.774 | --- | --- | --- | --- |
| naive_sft | 0.827 | +0.052 | 98.0 | 46.8 | 2.10 |
| kl_reg_sft | 0.793 | +0.018 | 107.9 | 50.8 | 2.13 |
| copr | 0.631 | -0.144 | 137.8 | 43.3 | 3.18 |
| copr_gold_injection | 0.813 | +0.039 | 99.6 | 62.1 | 1.60 |
| **copr_gold_injection_anchored** | **0.853** | **+0.079** | 85.2 | 59.3 | **1.44** |
| copr_anchored | 0.673 | -0.102 | 135.1 | 42.9 | 3.15 |

Three findings follow. First, **every updated method shifts the Q_direct representation substantially more than the Q_task_related representation** --- the smallest ratio (1.44 for `copr_gold_injection_anchored`) is still well above the integration target of 1.0. The update signal reshapes the QA direction far more than the QD direction for the same subject. Second, **plain `copr` and `copr_anchored` actively anti-align**: `cos(h_direct, h_related)` drops by -0.14 and -0.10 relative to baseline respectively, meaning the updates move the QA representation into a region of hidden-state space that is *farther* from the QD representation than before the edit. This matches their poor downstream absorption and compositional results (Section 5.2, 5.3) and is consistent with the K-sample-all-wrong pathology described in Section 3.3 --- the ranking, unanchored by a gold completion, reinforces whichever wrong answer is most plausible and displaces the QA representation into an unrelated region. Third, **gold injection is the only mechanism that shrinks the shift ratio toward 1** (`copr_gold_injection` 1.60, `copr_gold_injection_anchored` 1.44, versus 2.10-3.18 for all other methods) and is the only mechanism that simultaneously raises `cos(h_direct, h_related)` --- consistent with partial but incomplete subject-level integration rather than pure format-coupling.

The effect sizes should not be oversold. The best cosine rise over baseline is +0.079 (`copr_gold_injection_anchored`), which is small relative to the total QA-direction shift of 85-138 units across methods. The manifold signal says "gold injection nudges the QA and QD representations together, but only slightly; most of the update energy still goes into the QA direction alone."

## 6. Discussion

### 6.0 The arc of the experiment: from COPR to KL-regularized SFT

The paper's headline claim changed over the course of the experiments, and the trajectory is instructive enough to report directly rather than flatten into a clean post-hoc narrative.

**Motivation.** We set out to test a specific hypothesis: that Continual-Optimization Policy Ranking (COPR, Zhang et al. 2025) --- designed for continual preference alignment over a stream of preference-labeled tasks --- would transfer to sequential knowledge editing, because its reference-anchored MSE ranking is explicitly designed to prevent round-over-round drift. This is a direct import of an inductive bias from one continual-learning setting (preference alignment) to another (fact injection). The batch runs (Phases 1-2) disconfirmed the hypothesis on batch: KL-regularized SFT dominated the COPR family on absorption (+36-40%) at 10-20x less compute. We expected the inversion to appear in Phase 3 sequential, where COPR's continual-learning inductive bias should matter.

**The K-sample-all-wrong pathology.** In pilot sequential runs we discovered that COPR's K=8 self-samples on a novel fact are all incorrect (worst-F1 roughly 0). COPR's ranking is therefore a ranking over wrong answers, and the MSE fit reinforces whichever wrong answer is most plausible. This is not COPR failing at knowledge editing: it is COPR being asked to do something its preference-alignment design does not support. In preference alignment the preferred and rejected completions are both reasonable behaviors of the base model; in knowledge editing the gold answer has near-zero probability under the pre-update policy. The K-sample assumption --- that the candidate pool contains usable signal --- silently breaks.

**Two novel variants as remedies.** `copr_gold_injection` replaces the lowest-ranked self-sample with the gold completion, guaranteeing a correct top element. `copr_anchored` adds a task-replay-normalized reference to penalize candidates that drift off-task. `copr_gold_injection_anchored` combines the two. This gave us four COPR variants total.

**Gold injection worked --- and collapsed the method toward SFT.** With gold injected, the top-ranked candidate dominates the MSE fit, and the fit becomes, in effect, a re-parameterization of cross-entropy on the gold answer with the reference term as a side constraint. The LoRA-delta analysis (Phase 6) and hidden-state geometry probe (Phase 7) both support this reading: the gold-injection variants carry the two largest Frobenius norms among the COPR family and are the only variants whose hidden-state shift ratio approaches 1 (1.44-1.60 versus 2.1-3.2 for plain COPR and copr_anchored). Qualitatively, they behave like SFT --- at 10-12x the per-round compute cost of SFT.

**Anchoring alone was the clean negative result.** `copr_anchored`, without gold injection, collapses: absorption F1 0.066 (below naive_sft), compositional F1 0.025 (below the no-update baseline of 0.052). The mechanism is direct: the anchor pushes the policy toward the task manifold and the K-sample ranking has no correct signal to push back with. The hidden-state probe confirms active anti-alignment (`cos(h_direct, h_related)` drops -0.10 from baseline).

**The extended 15-round run killed the headline.** An earlier draft of this paper reported a +21% absorption advantage for `copr_gold_injection` at round 10. We ran rounds 11-15 after internal review flagged that a single round-endpoint was thin evidence for a regime-inversion claim. Across 14 contested rounds (round 9 missing for `copr_gold_injection`), `kl_reg_sft` wins absorption 9 times, `copr_gold_injection` wins 3, tied 2. At round 15, absorption is tied within 0.004 F1 between `kl_reg_sft`, `copr_gold_injection`, and `copr_gold_injection_anchored`. The round-10 result was one of three scattered wins, not the start of a trend.

**What survives is locality.** `copr_gold_injection_anchored` sustains a +47% locality gap over `kl_reg_sft` to round 15 and a +17% gap on the 15-round average. This is the only metric on which a COPR variant holds a durable advantage, and it is consistent with the anchoring-plus-gold mechanism: the task-replay-normalized reference reduces collateral damage to unrelated facts while gold injection provides the correct signal to offset the anchor's pressure against absorption.

**What this sequence teaches.**

1. **Continual-alignment objectives do not port directly to continual knowledge editing.** COPR's K-sample-all-wrong pathology forces gold injection as a remedy, and gold injection collapses the method toward SFT. If your "novel" method must be patched in ways that approximate SFT to work at all, the honest baseline is SFT, not the naive variant of your method. KL-regularized SFT is the baseline the field should be comparing against on continual fact injection, not against a straw-man naive SFT.

2. **Single-round results in sequential evaluation are probably noise.** The round-10 `copr_gold_injection` spike was convincing at the time, survived a numerical verification against the raw CSV, and was the basis of our "regime inversion" headline. The extended stream showed it was an outlier. Sequential editing evaluations need longer streams and ideally multi-seed replication before claiming trajectory advantages.

3. **The real sequential signal here is locality, not absorption.** The one metric where a COPR variant holds a durable edge is overall locality under `copr_gold_injection_anchored`. This is a weaker claim than "COPR wins sequential," and it costs 10-12x per-round compute for a +17% trajectory-average locality gain, but it is real.

4. **Format-coupling is universal, mechanism-invariant, and shows up geometrically.** Phase 7 hidden-state geometry shows every method moves the QA-direction representation much more than the QD-direction representation for the same subject. Gold injection partially narrows the gap; no method closes it. This is the Reversal Curse / knowledge-storage-vs-extraction phenomenon (Berglund et al. 2023; Allen-Zhu & Li 2023, 2024) measured geometrically under LoRA editing, and the ripple-effect literature (Cohen et al. 2024; Zhong et al. 2023) suggests it generalizes across editing families.

5. **The operational recommendation simplifies.** Use KL-regularized SFT for batch AND for sequential absorption. If locality preservation is the deployment priority and you can afford 10-12x compute per round, add gold-injection and anchoring. Do not use vanilla COPR or anchored-without-gold in either regime.

### 6.1 Why KL-regularized SFT wins both regimes

In the batch regime, `kl_reg_sft` dominates straightforwardly: the optimizer sees the full edit set, takes well-mixed gradient steps, and KL regularization against the pre-update reference is a cheap, unbiased pull toward task behavior. The COPR family's K-candidate sampling and MSE fit are operationally redundant at batch when cross-entropy on gold plus KL replay already solves the problem. This piece of the story is unambiguous: 36-40% absorption gap at 10-20x less compute.

In the sequential regime, we expected COPR's design to matter --- a reference-anchored MSE ranking that pins the policy to pi_ref round-over-round is, on paper, a better fit for a stream of small update batches than an SFT that has no native continual-learning regularizer beyond per-round KL replay. The 15-round data does not support this expectation on absorption. `kl_reg_sft` wins 9 of 14 contested rounds and holds the highest trajectory-averaged absorption (0.114 vs 0.107-0.110 for the gold-injection COPR variants). The mechanism is, in retrospect, that per-round KL replay against the pre-update policy is already a sufficient continual-learning anchor at the stream lengths and batch sizes we tested. The COPR machinery adds cost without adding signal once gold is injected --- because gold injection collapses the ranking-based objective toward the same target SFT already optimizes.

### 6.2 What COPR becomes after gold injection

Without gold injection, COPR's MSE fit on a ranking of self-samples is a distinct objective from SFT: it fits the policy to a reference-anchored rank signal over candidates that may all be wrong. With gold injection, the top-ranked candidate is always correct and its contribution dominates the MSE fit. In the limit, this is cross-entropy on the gold answer with the reference-policy term demoted to a regularizer --- very close in spirit, though not in exact functional form, to KL-regularized SFT. Phase 6 and Phase 7 evidence for this collapse:

- **Phase 6:** gold-injection variants carry the two largest total Frobenius norms (`copr_gi_anchored` 1.472, `copr_gi` 1.433) of the four COPR variants; non-gold variants come in at 1.282-1.325. Larger effective updates are consistent with a correct-answer target dominating the loss.
- **Phase 7:** gold-injection variants are the only ones whose hidden-state shift ratio approaches 1 (`copr_gi_anchored` 1.44, `copr_gi` 1.60); non-gold variants sit at 2.10-3.18 and two of them actively anti-align. In representation space, gold-injection COPR moves more like SFT and less like a ranking method over wrong answers.

The practical consequence is that `copr_gold_injection` and `copr_gold_injection_anchored` deliver roughly SFT-level absorption at the endpoint, not a new absorption ceiling. The advantage they hold at 10-12x compute per round is in locality, not in absorption. The regime characterization is therefore: batch prefers KL-SFT; sequential absorption is a tie between KL-SFT and gold-injected COPR; sequential locality prefers anchored gold-injected COPR if you can afford the compute.

### 6.3 Why locality survives where absorption does not

The one durable sequential advantage in our data is `copr_gold_injection_anchored` locality: 0.071 at round 15 vs `kl_reg_sft` at 0.048 (+47% endpoint), and 0.059 vs 0.050 on the 15-round average (+17%). The mechanism we propose is that the task-replay-normalized reference in `copr_anchored` reduces collateral damage to unrelated facts (improving locality) at the cost of pushing the policy toward the task manifold and away from absorption; gold injection then restores an absorption signal strong enough to overcome the anchor's pressure on the novel fact specifically. The net effect is a policy that edits fewer unrelated facts than SFT does, without losing the ability to absorb the target fact. Anchoring without gold (`copr_anchored`) fails because the anchor is unopposed; SFT-style methods fail at locality because nothing explicitly discourages off-target writes. The two together are required.

This is consistent with the Phase 7 shift ratios: `copr_gold_injection_anchored` has the smallest direct shift (85.2, lowest among all edits) paired with a middling related shift (59.3), which is what "small, targeted update" looks like geometrically. A more aggressive direct shift (as in `kl_reg_sft` at 107.9 or `copr` at 137.8) carries more absorption but also more collateral displacement.

### 6.4 The anchored-COPR negative result

`copr_anchored` (anchoring without gold injection) is a clean negative: absorption F1 0.066 (below naive_sft), compositional token-F1 0.025 (below the no-update baseline of 0.052), and active hidden-state anti-alignment (`cos(h_direct, h_related)` drops -0.10 from baseline). The task-replay-normalized reference makes any policy that deviates from the task manifold look relatively worse in the MSE fit; the new fact is by construction off-task; with no gold in the candidate set to push back, the anchor wins and absorption does not happen. This is the cleanest demonstration in the paper of the principle that anchoring requires a correct signal to anchor against --- and the matching finding that `copr_gold_injection_anchored` is the paper's best locality method once gold is added back.

The recovery via `copr_gold_injection_anchored` is instructive: once gold is in the ranking, the anchor no longer harms and in fact slightly improves locality. This suggests the anchor is not inherently broken --- it needs a non-degenerate learning signal (gold injection) to work with, not against.

### 6.5 Format-coupling is a property of gradient-based editing, not of LoRA

The Phase 7 hidden-state geometry indicates that all six update methods move the QA direction more than the QD direction for the same subject, and that even the best-integrated method (`copr_gold_injection_anchored`, ratio 1.44) does not reach subject-level integration. It would be tempting to read this as a LoRA artifact --- the rank-16 subspace is narrow and by construction cannot rotate both formats together. We do not endorse that reading.

The proximate cause of format-coupling is the *training signal*, not the parameter subset. The edit prompts are QA-formatted, the loss is cross-entropy on QA-formatted gold completions, and the reference regularizers (KL replay, COPR MSE) constrain the policy but do not inject QD-formatted signal into the gradient. Under this loss, *any* parameter subset that can fit the QA gradient will concentrate its update energy along the QA direction, leaving the QD direction unperturbed except insofar as the two directions share subspace in the base model --- which, at baseline `cos = 0.774`, they partially but imperfectly do. This phenomenon is well-attested in closely-related literatures under different names: Berglund et al. (2023) document the Reversal Curse, in which a model trained on "A is B" fails to answer "who is B" --- a maximal format-coupling case. Allen-Zhu & Li (2023, 2024) distinguish knowledge *storage* (parameters hold the fact) from knowledge *extraction/manipulation* (the model uses the fact under non-training prompt distributions), and show that storage without extraction is the default unless the training distribution diversifies the prompt formats. Ripple-effect benchmarks (Cohen et al. 2024) show that ROME and MEMIT --- which edit targeted MLP rows rather than LoRA adapters --- fail to propagate single-fact edits to logical neighbors of the edited fact. MQuAKE (Zhong et al. 2023) shows the same brittleness under multi-hop composition for both locate-then-edit and hypernetwork families. The shared failure mode across ROME, MEMIT, MEND, full fine-tuning, and LoRA is not that LoRA is uniquely low-rank --- it is that the editing loss is defined on one prompt distribution and the probe is in another.

The stronger version of the Phase 7 claim is therefore: **sequential gradient-based editing, regardless of parameter subset, creates format-specific knowledge islands; gold injection is the first mechanism we measure that partially bridges those islands to the task manifold, and it does so at approximately +0.08 cosine over baseline --- a small but non-zero improvement that is the mechanistic signature of the absorption and locality gains in Section 5.2.** The evidence base for the "regardless of parameter subset" part of this claim is the editing-ripple literature (Cohen et al. 2024; Zhong et al. 2023; Yao et al. 2023) and the knowledge-extraction literature (Berglund et al. 2023; Allen-Zhu & Li 2023, 2024), not our own experiments. What Phase 7 adds over these prior works is a direct geometric measurement of the shift asymmetry in hidden-state space under our specific update methods --- the cos and shift-ratio metrics are not reported in the cited prior work, and they localize the format-coupling to a measurable geometric signature per method. A single full-fine-tuning or MEMIT run under the same probe set would strengthen the "regardless of parameter subset" claim materially and is flagged as near-term future work. We do not claim to have measured format-coupling outside the LoRA setting ourselves.

The operational implication compounds the regime-matters message. If the goal is task-format generalization of an injected fact, no update method studied here or in the cited literature delivers it from QA-only training signal. Either the loss must include the target task format explicitly (a design the next iteration of this paper will test via QD-formatted rehearsal during update), or inference must route around the missing integration via retrieval.

### 6.6 Compositional degradation

All updated methods except the gold-injection variants' token-F1 track more-or-less unchanged bridging-entity recall, but every method including the best degrades it below the no-update baseline of 0.102. The updates absorb the new first-hop fact but interfere with the second-hop retrieval of the bridging entity. This is a shared limitation and a call-to-action for the next iteration: either explicit multi-hop anchoring during training (candidates that require the bridging entity) or a retrieval-based bridge at inference.

## 7. Limitations

**Single seed.** All runs use seed=42. The paper does not report seed variance, which means some of the smaller differences (particularly preservation and compositional exact-match) may not be robust. A multi-seed replication (at minimum seed in {42, 7, 13}) is deferred.

**Preservation noise.** Preservation Recall@10 is computed over n=104 test items, with std roughly 0.33 across items. Standard error on the mean is thus about 0.032. Many preservation differences reported in the tables are within 1 SE of each other; the paper does not claim significance on preservation and uses it as a sanity guardrail rather than a primary metric.

**Retrieval-pipeline confound in preservation and post-preservation.** Preservation Recall@10 flows through BGE-M3 subquery embeddings and FAISS nearest-neighbor retrieval over the qd_temporal index. An update that nudges the model's subquery wording by a token or two can shift retrieved document IDs without changing what the model "knows" about the domain. Some of the small preservation differences reported across methods (particularly those within 1-2 SE) may therefore reflect subquery-wording stability rather than knowledge change. Post-preservation (used only internally to monitor post-cutoff task degradation) shares this confound. An oracle-retrieval or entity-keyed-retrieval ablation --- flagged as future work in Section 6 --- would isolate knowledge change from retrieval wording, but is not run here. The paper's primary metrics (absorption, compositional, locality) do not go through the retrieval pipeline and are not subject to this confound.

**Locality `same_sector` stratum is n=1.** The locality evaluation has three strata, but the `same_sector` stratum is populated by a single probe. A stratum of size one is not meaningful as a mean estimate --- the reported `same_sector` F1 values are either 0.0, 0.154, 0.167, 0.4 depending on whether that single probe happens to match, and should not be interpreted as a population estimate. The aggregate locality overall F1 is driven by the `same_entity` (n=182) and `other_sector` (n=1817) strata. This is a construction artifact of the FNSPID sector taxonomy combined with the entity-locality sampling procedure and should be corrected in a follow-up by broadening the `same_sector` probe set.

**Compositional EM near zero.** Exact-match on two-hop compositional is 0-0.014 across methods. The compositional token-F1 signal is the usable metric; exact-match is effectively at floor and should not be over-interpreted.

**No temporal-contrast evaluation.** Phase 4 was planned to include a temporal-contrast metric (pre-2022 vs post-2022 fact collisions) but the Gemini-generated probes did not contain the required gold answers in a consistent format. The temporal-contrast metric is dropped from the paper rather than reported with incomplete data. This is noted in `final_results/phase4_compositional.csv`.

**Trajectory gaps.** For `copr_anchored`, the per-round trajectory for rounds 6, 7, and 9 is missing compositional metrics (those metrics are computed only at round 10 by design); rounds 6 and 7 also hit a faiss crash on the initial sweep and were not rerun. The absorption and locality per-round data for all ten rounds and all six methods is complete. The paper does not impute.

**Scale.** Results are reported at 1K, 3K (batch) and 10x200 (sequential). Scaling laws for knowledge editing with LoRA are not established, and the batch results at 10K or 100K may tell a different story.

**Model scale.** Only Qwen3-4B-Instruct-2507 is evaluated. Findings on the relative ordering of methods may not transfer to 7B, 14B, 70B, or GPT-scale models.

## 8. Conclusion

The paper reports a negative transfer result and a single durable sub-win, supported by a controlled experiment on a single backbone and a single data distribution. The headline: **KL-regularized SFT dominates the COPR family on absorption in both regimes** --- by 36-40% at 3K batch edits, and by 9 wins to 3 (with 2 ties) in a 14-round sequential head-to-head against `copr_gold_injection`. The round-10 +21% absorption "inversion" earlier drafts reported was a single-round spike that did not replicate in rounds 11-15; at round 15, absorption is tied within 0.004 F1 across `kl_reg_sft`, `copr_gold_injection`, and `copr_gold_injection_anchored`. The durable COPR advantage is narrower than originally claimed: **`copr_gold_injection_anchored` sustains a +47% overall locality gap over `kl_reg_sft` at round 15** (+17% on 15-round average), at 10-12x per-round compute. **Anchoring without gold injection is a clean negative result**: `copr_anchored` absorption 0.066, compositional F1 0.025, below the no-update baseline of 0.052.

Two mechanistic analyses ground and explain the result. The LoRA-delta analysis shows all four COPR variants share a near-identical top input subspace (cos 0.997) and differ in stable rank and total Frobenius norm (`copr_gold_injection_anchored` highest at 1.472). The hidden-state geometry probe shows every method shifts the QA direction more than the QD direction for the same subject; gold injection is the only mechanism that partially couples the two, and even then the integration signal is modest (+0.08 cosine at best). We argue gold injection collapses COPR toward KL-regularized SFT --- which explains why it matches but does not beat KL-SFT on absorption --- and that the format-coupling signature is a property of gradient-based editing under QA-only loss rather than a LoRA artifact. The Reversal-Curse (Berglund et al. 2023), knowledge-storage-vs-extraction (Allen-Zhu & Li 2023, 2024), and ripple-effect (Cohen et al. 2024; Zhong et al. 2023) literatures are the closest analogs and support the generalization claim; we do not verify non-LoRA parameter subsets ourselves.

The operational recommendation is simpler than earlier drafts suggested: **use KL-regularized SFT for both batch and sequential fact injection at this scale**. If locality preservation is the deployment priority and you can afford 10-12x per-round compute, add gold-injection and anchoring (`copr_gold_injection_anchored`). Do not use vanilla COPR or anchored-without-gold in either regime. Compositional two-hop reasoning degrades under every parametric edit tested, and no method studied delivers task-format generalization from QA-only training signal --- either the update loss must include task-format rehearsal explicitly or inference must route around the missing integration via retrieval. The paper's broader methodological lesson is that continual-alignment objectives designed for preference tasks (COPR) do not transfer to continual knowledge editing without patches that collapse the method toward the SFT baseline they were meant to improve on.

## Reproducibility Statement

All code, configs, and raw artifacts are committed at the repository root, commit `1c28ce0`. Every number in Sections 5 and 6 is a direct copy from a row of a CSV in `final_results/`, which is itself regenerated by `scripts/20_snapshot_results.py` from the raw JSON in `outputs/`. Data counts and paths are in `final_results/data_inventory.csv`; per-run compute, seeds, and starting checkpoints are in `final_results/run_metadata.csv`; method configs are in `final_results/methods_index.csv` and `final_results/run_configs.csv`. The 500 compositional probes are in `data/fnspid/compositional/probes.json` with the Gemini-3.1-Flash-Lite prompt template in `configs/compositional/probe_template.yaml` (see repo). The task-tuned backbone LoRA checkpoint is `checkpoints/qd_sft/final`. The filtered fact triples are derived from the FNSPID corpus (Shah et al. 2023) with the filtering pipeline in `scripts/` (see `data/fnspid/triples/filtered_triples.json`, n=96,897). Randomness is controlled by seed=42 for all runs.
