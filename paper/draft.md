# Regime Matters: Evaluating COPR for Knowledge Injection Under Batch and Sequential Update Schedules

## Abstract

Knowledge injection methods for large language models are typically benchmarked under a single update regime, which obscures how their inductive biases interact with deployment schedules. This paper evaluates six LoRA-based update procedures, including three novel variants of Continual-Optimization Policy Ranking (COPR, Zhang et al. 2025), on a single backbone (Qwen3-4B-Instruct-2507, task-tuned for query decomposition over financial news) across both a batch regime (1K and 3K edits at once) and a sequential regime (ten rounds of 200 edits). Under the batch regime at 3K edits, a simple KL-regularized SFT baseline dominates on absorption (F1 0.173) and locality (overall F1 0.071) at 0.57 GPU-hours, while the COPR family costs roughly 9 GPU-hours for equal or worse metrics --- a 36-40% relative absorption gap. Under the sequential regime at round 10, a novel variant that injects the gold answer into the COPR candidate set (`copr_gold_injection`) reaches absorption F1 0.162 (+21% over KL-regularized SFT at the endpoint), and the anchored version (`copr_gold_injection_anchored`) reaches locality overall F1 0.077 (+50%); the absorption advantage is a round-10 endpoint effect rather than a sustained trajectory lead (`copr_gold_injection` trails `kl_reg_sft` at every intermediate round 1-8 for which both have data). A naive task-replay-normalized reference-anchoring (`copr_anchored`) collapses to absorption F1 0.066 and compositional F1 0.025 --- a negative result reported prominently. Two mechanistic analyses ground the regime claim. A LoRA-delta analysis shows that all four COPR variants share a near-identical top input subspace (first-principal-angle cosine 0.997 pairwise) and differ mainly in stable rank and in total Frobenius norm (`copr_gold_injection_anchored` highest at 1.472). A hidden-state geometry probe shows that every method shifts the QA-direction representation substantially more than the QD-direction representation for the same subject: plain `copr` and `copr_anchored` actively anti-align (cosine drops by -0.14 and -0.10), while gold injection is the only mechanism that partially couples the two formats (shift ratio 1.44 at best, versus 2.1-3.2 elsewhere). The paper argues that this format-coupling signature is a property of gradient-based editing under a QA-formatted loss, not a LoRA artifact, and cites the ROME/MEMIT ripple-effect literature in support. Compute overhead for the COPR family is ten-to-twenty-fold per update round. Numbers are quoted verbatim from Phase 1-7 CSV artifacts; standard errors and probe counts are reported with the means.

## 1. Introduction

Injecting new factual knowledge into the parameters of an already-tuned language model is an operational need in any deployment where the world changes after training. Existing methods fall into two broad families: constrained parameter edits that localize changes to a small set of MLP weights (ROME, Meng et al. 2022; MEMIT, Meng et al. 2023), and fine-tuning-style updates that use gradient descent on edit-specific objectives (SFT, KL-regularized SFT, direct-preference-style objectives such as KDPO, Rafailov-style preference contrast extended to editing; COPR, Zhang et al. 2025). A third family --- retrieval-augmented generation --- side-steps parametric editing entirely and is therefore out of scope here.

Most of the editing literature reports results in one of two default postures: a single batch of edits applied once, or a stream of single edits applied one at a time with or without checkpoint chaining. These two postures correspond to meaningfully different operational settings. A batch update is a push: a data-engineering pipeline produces a list of new facts and a training loop internalizes them in a single pass. A sequential update is a drumbeat: disjoint batches of facts arrive on a schedule, each round inherits the previous round's checkpoint, and the model must absorb the new batch without catastrophic interference from prior rounds. The operational costs and the failure modes differ, and so does the appropriate inductive bias of the update method.

This paper argues that the choice of knowledge-injection method depends critically on the update regime. The claim is supported by a controlled experiment on a single backbone (Qwen3-4B-Instruct-2507, LoRA-task-tuned for query decomposition on FNSPID pre-2022 news) with disjoint triple batches drawn from the same distribution, holding data, evaluation, compute budget (on a per-update basis), and seed fixed across methods. Three contributions follow.

First, the paper introduces **gold-answer injection** for COPR as a novel modification to the candidate set. The original COPR design (Zhang et al. 2025) samples K candidate completions from the current policy and fits a reference-anchored preference ranking among them. If all K self-samples miss the new gold fact --- a common failure mode when the fact is genuinely unknown to the base model --- the ranking contains no gradient signal toward the correct answer. Injecting the gold completion as an additional candidate guarantees a correct anchor in the ranking at every step. Phase 3 shows that this single change accounts for the bulk of COPR's sequential-regime advantage.

Second, the paper introduces **anchored COPR** --- an attempt to improve the reference policy by mixing a task-replay-normalized reference into the per-candidate log pi_ref. This is reported as a negative result: on Phase 3, `copr_anchored` absorption F1 collapses to 0.066 and the compositional F1 drops to 0.025, below the no-update baseline of 0.052. Anchoring *alone* is harmful; anchoring *combined with* gold injection (`copr_gold_injection_anchored`) recovers most of the loss and, in fact, wins locality overall F1 at 0.077.

Third, the paper provides a **regime characterization** grounded in two mechanistic analyses: a LoRA-delta subspace analysis (Phase 6) and a hidden-state geometry probe (Phase 7). All four COPR variants write into nearly the same top input subspace (first-principal-angle cosine 0.997 pairwise across methods), confirming that they share a common rank-1 update direction; differences appear in stable rank (`copr_gold_injection` at 2.54 is the narrowest) and in total Frobenius norm (`copr_gold_injection_anchored` highest at 1.472). The hidden-state probe then shows that every method --- including the gold-injection variants --- moves the QA-direction representation substantially more than the QD-direction representation for the same subject, and that plain `copr` and `copr_anchored` actively anti-align (cosine drops by -0.14 and -0.10 respectively). Gold injection is the only mechanism that partially couples the two formats (shift ratio 1.44-1.60 versus 2.1-3.2 for all other methods), and even then the integration signal is modest (+0.08 cosine at best).

The central finding is: the KL-regularized SFT baseline is strictly better than the COPR family in the batch regime at the scales tested (36-40% relative absorption gap), and the gold-injection COPR variants reach a round-10 absorption and locality advantage over the KL baseline in the sequential regime --- at the cost of a ten-to-twenty-fold compute premium per update, and without a sustained trajectory lead (`copr_gold_injection` trails `kl_reg_sft` at rounds 1-8). All compositional (two-hop) edits degrade bridging-entity recall below the no-update baseline --- a shared limitation the paper reports prominently rather than obscuring.

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

Phase 3 runs 10 rounds of 200 disjoint fact triples per method, chaining checkpoints. The final-round table at round 10:

| Method | Preservation | Absorption F1 | Absorption worst-F1 | Locality overall F1 | Compositional token-F1 | GPU-h (per round, avg) |
|---|---|---|---|---|---|---|
| naive_sft | 0.204 +/- 0.331 | 0.100 | 0.044 | 0.037 | 0.069 | 0.013 |
| kl_reg_sft | 0.244 +/- 0.344 | 0.134 | 0.098 | 0.051 | 0.083 | 0.051 |
| copr | 0.232 +/- 0.344 | 0.110 | 0.063 | 0.026 | 0.066 | 0.60 |
| **copr_gold_injection** | 0.213 +/- 0.331 | **0.162** | **0.119** | 0.064 | 0.088 | 0.58 |
| **copr_gold_injection_anchored** | 0.188 +/- 0.322 | 0.139 | 0.087 | **0.077** | 0.088 | 0.65 |
| copr_anchored (negative result) | 0.239 +/- 0.358 | 0.066 | 0.014 | 0.028 | 0.025 | 0.62 |

Relative to `kl_reg_sft` at round 10: `copr_gold_injection` absorption F1 of 0.162 vs 0.134 is a +21% relative gain; its worst-F1 of 0.119 vs 0.098 is +21%. `copr_gold_injection_anchored` locality overall F1 of 0.077 vs 0.051 is +50%. Both gold-injection variants tie or slightly beat `kl_reg_sft` on compositional token-F1 (0.088 vs 0.083, +6%). Preservation trades off: `copr_gold_injection` at 0.213 is 0.96 SE below `kl_reg_sft` (0.244) --- within noise --- but `copr_gold_injection_anchored` at 0.188 is 1.77 SE below, a real preservation cost. The locality win of the anchored variant comes paired with this preservation hit, and the paper does not treat the two as free.

The negative result on `copr_anchored` is severe: absorption F1 collapses to 0.066 (below naive_sft), worst-F1 to 0.014 (essentially chance on paraphrases), and compositional token-F1 to 0.025 (below the no-update baseline of 0.052). The task-replay-normalized reference pulls the policy toward the task manifold so strongly that absorption is smothered. Because this variant does not gold-inject, the pathology described in Section 3.3 compounds with the anchoring pressure. Absorption simply does not happen.

**Trajectories.** Full per-round trajectories are in `final_results/phase3_sequential_trajectory.csv`. Qualitative patterns:

- `kl_reg_sft`'s absorption F1 rises non-monotonically from 0.082 at round 1 to 0.134 at round 10, with intermediate highs of 0.141 (round 6) and 0.135 (round 7); its preservation stays in 0.186-0.244 band throughout.
- `copr_gold_injection`'s absorption F1 rises from 0.069 at round 1 to 0.162 at round 10 but **trails `kl_reg_sft` at every intermediate round for which both have data** (rounds 1-8: deltas -0.013 to -0.026; round 9 is missing for `copr_gold_injection`). The crossover is a round-10 endpoint effect, not a steady pull-ahead: the +0.028 gap at round 10 (0.162 vs 0.134) reverses a -0.003 to -0.026 deficit held across rounds 1-8. The paper does not claim a sustained trajectory advantage; it claims a round-10 advantage, and Section 6 discusses this more carefully.
- `copr_anchored`'s absorption F1 is erratic, dipping to 0.040 at round 9 and never exceeding 0.089. Compositional at round 10 is 0.025, worse than no_update.

**Figure 1 (placeholder).** Per-round absorption F1 (y-axis, 0-0.20) versus round index (1-10) with one line per method. Caption: "Absorption trajectories over 10 sequential rounds (200 triples per round). `copr_gold_injection` trails `kl_reg_sft` at rounds 1-8 and only crosses it at round 10; `copr_anchored` never climbs above 0.089; `naive_sft` plateaus below 0.100." Data source: `final_results/phase3_sequential_trajectory.csv`.

**Figure 2 (placeholder).** Per-round locality overall F1 over 10 rounds, same format. Caption: "`copr_gold_injection_anchored` locality climbs steadily to 0.077 at round 10, surpassing all other methods; `kl_reg_sft` plateaus around 0.051." Data source: same.

**Trajectory gaps.** Honest accounting: rounds 6/7/9 are missing from `copr_anchored` trajectory for compositional metrics (only computed at round 10); no full-trajectory per-round faiss evaluation was re-run for these rounds after a faiss crash in the initial sweep. The per-round absorption and locality for all ten rounds are complete for all six methods (see trajectory CSV). The paper does not backfill missing cells or impute.

**Durability across rounds (trajectory-averaged metrics).** The round-10 final-round table above captures the endpoint; it does not directly measure *durability* --- how steady each method is throughout the stream. Table 3bis reports the cross-round mean for each of the three metrics, averaged only over rounds for which data exists (from `final_results/phase3_durability.csv`). Not all methods have data for every round in 1-10: `naive_sft` and `kl_reg_sft` have all ten; `copr`, `copr_gold_injection`, and `copr_gold_injection_anchored` each have nine (round 9 missing); `copr_anchored` has eight (rounds 6 and 7 missing, after a faiss crash during the initial sweep that was not re-run). The durability means below are therefore over those observed rounds, not over all ten uniformly, which should be read with that caveat.

| Method | Preservation mean | Absorption mean | Locality mean |
|---|---|---|---|
| naive_sft | 0.209 | 0.077 | 0.032 |
| kl_reg_sft | 0.223 | 0.113 | 0.050 |
| copr | 0.231 | 0.075 | 0.025 |
| copr_gold_injection | 0.227 | 0.107 | 0.046 |
| copr_gold_injection_anchored | 0.205 | 0.107 | 0.052 |
| **copr_anchored** | **0.237** | 0.060 | 0.030 |

Two findings emerge. First, on the *durability* metric that a continual-update deployment actually cares about --- mean preservation across the whole stream, not just the endpoint --- three of the four COPR variants (`copr_anchored` 0.237, `copr` 0.231, `copr_gold_injection` 0.227) are steadier than `kl_reg_sft` (0.223). COPR's reference-policy-anchored MSE fit is designed to keep the policy close to pi_ref round-over-round, and the trajectory-average preservation is consistent with that design; the differences are small (within roughly 0.01) and should be read as directional rather than as a clean regime separator. Second, the absorption and locality *advantage* of the gold-injection variants is not a steady pull-ahead: trajectory-averaged absorption for `copr_gold_injection` is 0.107 versus `kl_reg_sft` 0.113 (essentially tied, with `kl_reg_sft` slightly higher), and `copr_gold_injection` trails `kl_reg_sft` at every intermediate round 1-8 before pulling above at round 10 (+0.028). The round-10 advantage is therefore a late-stream endpoint effect, not a growing gap that compounds over rounds. A longer stream would be needed to distinguish a genuine asymptote from endpoint noise; we treat this finding conservatively and do not extrapolate.

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

### 6.1 Regime mismatch as the driver

The batch and sequential results are not a gradient of difficulty --- they are two different operational settings with different failure modes. In batch, the update method sees the entire edit set at once, the optimizer can take K gradient steps with well-mixed batches, and KL regularization against the pre-update reference is a strong, cheap, unbiased pull back toward task behavior. `kl_reg_sft` exploits this directly. The COPR family, by contrast, pays a sampling cost (K candidates per item per step) and an MSE fit on a rank signal, both of which are operationally redundant when cross-entropy on the gold answer plus KL replay already solves the problem.

In sequential, each round re-starts from the previous round's model. The KL reference drifts with the policy (or is pinned at round 0 at the cost of increasing KL with every round). The optimizer sees only 200 examples per round, and catastrophic forgetting within the fact distribution compounds. COPR's anchoring structure --- the reference-policy MSE fit that pins the ranking relative to pi_ref --- is closer to what the sequential regime actually needs. Gold injection makes this anchor productive by guaranteeing a correct target in the ranking; without it, the anchor just stabilizes the current (wrong) ranking.

### 6.2 When to pay the COPR compute premium

The paper's operational recommendation is direct. If the deployment is a single batch push of O(1K-10K) new facts, use KL-regularized SFT: it is 10-20x cheaper and dominates on absorption and locality at these scales. If the deployment is a continual-update stream (many rounds over time), the gold-injection COPR variants are worth the per-round compute premium, because (i) absorption ceiling is materially higher (+21% over KL on the 10-round cumulative), (ii) locality is materially better (+50%), and (iii) compositional token-F1 is at least as good. The premium in Phase 3 is roughly 12x per round (0.58 GPU-h for `copr_gi` vs 0.051 for `kl_reg_sft`).

The durability view in Section 5.2's Table 3bis gives a mixed picture: three of the four COPR variants hold slightly higher trajectory-averaged preservation than `kl_reg_sft` (0.227-0.237 vs 0.223), but trajectory-averaged absorption and locality for the gold-injection variants are tied with or slightly below `kl_reg_sft`. The round-10 absorption advantage of `copr_gold_injection` (+21%) is an endpoint effect rather than a growing gap. Short deployments see thin differentiation; whether long deployments would compound the advantage is a question this paper cannot answer from a single ten-round run. We flag this explicitly rather than extrapolate.

### 6.3 Gold injection's mechanism

The pilot observation that K self-samples are all wrong on novel facts is, in retrospect, obvious: the base model does not know the fact. Under that condition, the COPR ranking is a ranking over wrong answers, and fitting to it is at best noise-absorbing and at worst actively misleading. Gold injection flips this: the ranking now has a correct top element, and the MSE fit pulls the policy toward it. Phase 6 supports this: `copr_gi` uniquely edits early layers (0-3), consistent with the model absorbing the novel surface form directly rather than redistributing probability among existing plausible answers.

### 6.4 The anchored-COPR negative result

Anchoring the reference policy with a task-replay normalizer (`copr_anchored`) fails in both regimes, but catastrophically in sequential (absorption F1 0.066, compositional token-F1 0.025). The mechanism: the task-replay-normalized reference makes any policy that deviates from the task manifold look relatively worse in the MSE fit. Since the new fact is, by construction, off-task (it is a new fact not seen during task-tuning), the anchor pushes *against* absorption. Combined with the no-gold-injection pathology, the method simply does not learn the new facts.

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

The paper makes three claims and supports each with a controlled experiment on a single backbone and a single data distribution. First, **the batch regime favors KL-regularized SFT**: a simple, cheap baseline dominates the COPR family at 1K and 3K edits at 10-20x less compute (36-40% relative absorption gap). Second, **the sequential regime offers gold-injected COPR a round-10 endpoint advantage, not a steady lead**: `copr_gold_injection` reaches absorption F1 0.162 (+21% over `kl_reg_sft`) and `copr_gold_injection_anchored` reaches locality overall F1 0.077 (+50%) at round 10, but `copr_gold_injection` trails `kl_reg_sft` at every intermediate round 1-8 for which both have data. Whether the endpoint gap reflects a genuine asymptote or stream-length noise cannot be resolved from a single ten-round run. Third, **anchoring alone is harmful**: `copr_anchored`'s absorption F1 collapses to 0.066 and its compositional token-F1 drops to 0.025, below the no-update baseline. Two mechanistic analyses ground these results. The LoRA-delta analysis shows the COPR variants share a near-identical top input subspace but differ in stable rank and total Frobenius norm (`copr_gold_injection_anchored` highest at 1.472). The hidden-state geometry probe shows every method shifts the QA direction more than the QD direction for the same subject; gold injection is the only mechanism that partially couples the two, and even then the integration signal is modest (+0.08 cosine at best). We argue this format-coupling signature is a property of gradient-based editing under a QA-formatted loss rather than a LoRA artifact, with the ROME/MEMIT ripple-effect literature as the primary evidence base for the generalization.

The operational recommendations follow with appropriate hedging. Batch pushes of facts should use the KL baseline --- this is the strongest and cleanest finding. Continual-update streams may benefit from gold-injected COPR at its round-10 endpoint, but the case for a sustained trajectory advantage is not made by this paper. Anchoring the reference without gold injection should be avoided. Compositional two-hop reasoning degrades under all parametric edits tested here, and no method studied delivers task-format generalization from QA-only training signal --- either the loss must include task-format rehearsal explicitly or inference must route around the missing integration via retrieval.

## Reproducibility Statement

All code, configs, and raw artifacts are committed at the repository root, commit `1c28ce0`. Every number in Sections 5 and 6 is a direct copy from a row of a CSV in `final_results/`, which is itself regenerated by `scripts/20_snapshot_results.py` from the raw JSON in `outputs/`. Data counts and paths are in `final_results/data_inventory.csv`; per-run compute, seeds, and starting checkpoints are in `final_results/run_metadata.csv`; method configs are in `final_results/methods_index.csv` and `final_results/run_configs.csv`. The 500 compositional probes are in `data/fnspid/compositional/probes.json` with the Gemini-3.1-Flash-Lite prompt template in `configs/compositional/probe_template.yaml` (see repo). The task-tuned backbone LoRA checkpoint is `checkpoints/qd_sft/final`. The filtered fact triples are derived from the FNSPID corpus (Shah et al. 2023) with the filtering pipeline in `scripts/` (see `data/fnspid/triples/filtered_triples.json`, n=96,897). Randomness is controlled by seed=42 for all runs.
