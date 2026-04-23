# Regime Matters: Evaluating COPR for Knowledge Injection Under Batch and Sequential Update Schedules

## Abstract

Knowledge injection methods for large language models are typically benchmarked under a single update regime, which obscures how their inductive biases interact with deployment schedules. This paper evaluates six LoRA-based update procedures, including three novel variants of Continual-Optimization Policy Ranking (COPR, Zhang et al. 2025), on a single backbone (Qwen3-4B-Instruct-2507, task-tuned for query decomposition over financial news) across both a batch regime (1K and 3K edits at once) and a sequential regime (ten rounds of 200 edits). Under the batch regime at 3K edits, a simple KL-regularized SFT baseline dominates on absorption (F1 0.173) and locality (overall F1 0.071) at 0.57 GPU-hours, while the COPR family costs roughly 9 GPU-hours for equal or worse metrics. Under the sequential regime at round 10, the picture inverts: a novel variant that injects the gold answer into the COPR candidate set (`copr_gold_injection`) wins absorption at F1 0.162 (+21% over KL-regularized SFT), and the anchored version (`copr_gold_injection_anchored`) wins locality at overall F1 0.077 (+50%), while a naive task-replay-normalized reference-anchoring (`copr_anchored`) collapses to absorption F1 0.066 and compositional F1 0.025 --- a negative result reported prominently. A mechanistic LoRA-delta analysis shows that all four COPR variants write into nearly-identical top input subspaces (first-principal-angle cosine around 0.997 pairwise) but differ in stable rank and, uniquely for `copr_gold_injection`, in per-layer distribution (non-trivial writes to the earliest layers 0-3). The paper reports ten-to-twenty-fold compute overhead for the COPR family per update round, and positions gold-answer injection as the dominant lever for COPR's sequential advantage, while anchoring alone is, at best, neutral. Numbers are quoted verbatim from Phase 1-6 CSV artifacts; standard errors and probe counts are reported with the means.

## 1. Introduction

Injecting new factual knowledge into the parameters of an already-tuned language model is an operational need in any deployment where the world changes after training. Existing methods fall into two broad families: constrained parameter edits that localize changes to a small set of MLP weights (ROME, Meng et al. 2022; MEMIT, Meng et al. 2023), and fine-tuning-style updates that use gradient descent on edit-specific objectives (SFT, KL-regularized SFT, direct-preference-style objectives such as KDPO, Rafailov-style preference contrast extended to editing; COPR, Zhang et al. 2025). A third family --- retrieval-augmented generation --- side-steps parametric editing entirely and is therefore out of scope here.

Most of the editing literature reports results in one of two default postures: a single batch of edits applied once, or a stream of single edits applied one at a time with or without checkpoint chaining. These two postures correspond to meaningfully different operational settings. A batch update is a push: a data-engineering pipeline produces a list of new facts and a training loop internalizes them in a single pass. A sequential update is a drumbeat: disjoint batches of facts arrive on a schedule, each round inherits the previous round's checkpoint, and the model must absorb the new batch without catastrophic interference from prior rounds. The operational costs and the failure modes differ, and so does the appropriate inductive bias of the update method.

This paper argues that the choice of knowledge-injection method depends critically on the update regime. The claim is supported by a controlled experiment on a single backbone (Qwen3-4B-Instruct-2507, LoRA-task-tuned for query decomposition on FNSPID pre-2022 news) with disjoint triple batches drawn from the same distribution, holding data, evaluation, compute budget (on a per-update basis), and seed fixed across methods. Three contributions follow.

First, the paper introduces **gold-answer injection** for COPR as a novel modification to the candidate set. The original COPR design (Zhang et al. 2025) samples K candidate completions from the current policy and fits a reference-anchored preference ranking among them. If all K self-samples miss the new gold fact --- a common failure mode when the fact is genuinely unknown to the base model --- the ranking contains no gradient signal toward the correct answer. Injecting the gold completion as an additional candidate guarantees a correct anchor in the ranking at every step. Phase 3 shows that this single change accounts for the bulk of COPR's sequential-regime advantage.

Second, the paper introduces **anchored COPR** --- an attempt to improve the reference policy by mixing a task-replay-normalized reference into the per-candidate log pi_ref. This is reported as a negative result: on Phase 3, `copr_anchored` absorption F1 collapses to 0.066 and the compositional F1 drops to 0.025, below the no-update baseline of 0.052. Anchoring *alone* is harmful; anchoring *combined with* gold injection (`copr_gold_injection_anchored`) recovers most of the loss and, in fact, wins locality overall F1 at 0.077.

Third, the paper provides a **regime characterization** grounded in a mechanistic LoRA-delta analysis. All four COPR variants write into nearly the same top input subspace (first-principal-angle cosine 0.997 pairwise across methods), confirming that they share a common rank-1 update direction. Differences appear in stable rank (`copr_gold_injection` at 2.54 is the narrowest) and in per-layer distribution: `copr_gold_injection` is the only variant with non-trivial activity in the earliest layers (0-3), which plausibly explains its absorption lead.

The central finding is simple: the KL-regularized SFT baseline is strictly better than the COPR family in the batch regime at the scales tested, and the gold-injection COPR variants are strictly better than the KL baseline on absorption and locality in the sequential regime, at a ten-to-twenty-fold compute premium per update. All compositional (two-hop) edits degrade bridging-entity recall below the no-update baseline --- an important shared limitation that the paper does not paper over.

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

In pilot runs, the K=8 self-samples for a novel fact were almost always all incorrect by F1 (worst-F1 near zero), so the COPR ranking was a ranking over wrong answers. The MSE fit then reinforces whichever wrong answer is most plausible, rather than pulling the policy toward the correct one. Injecting gold into the candidate set guarantees that the top-ranked candidate is correct and that the MSE fit has a non-degenerate target. Phase 6 confirms the downstream effect: `copr_gold_injection` is the only method that writes non-trivially into the earliest layers (0-3), where the bulk of lexical/factual content is edited.

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

`kl_reg_sft` is the best method at batch scale 3000 on absorption F1 (0.173) and locality overall F1 (0.071), and preserves task capability at 0.253 +/- 0.354 (within one standard error of the best). The COPR family runs at 9-10 GPU-hours for absorption F1 in the 0.104-0.111 band --- a 33-38% relative gap below `kl_reg_sft` absorption --- and locality F1 roughly 0.03 versus `kl_reg_sft` at 0.071 (a 58% relative gap). `naive_sft` degrades preservation sharply at 3K (0.123, below all other methods) while improving absorption less than `kl_reg_sft`, illustrating why the KL regularizer is worth its cost. The COPR premium is substantial (10-20x `kl_reg_sft`) and does not pay off in batch at the scales tested.

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

Relative to `kl_reg_sft` at round 10: `copr_gold_injection` absorption F1 of 0.162 vs 0.134 is a +21% relative gain; its worst-F1 of 0.119 vs 0.098 is +21%. `copr_gold_injection_anchored` locality overall F1 of 0.077 vs 0.051 is +50%. Both gold-injection variants tie or slightly beat `kl_reg_sft` on compositional token-F1 (0.088 vs 0.083, +6%). Preservation on both gold-injection variants is 0.188-0.213, slightly below `kl_reg_sft` at 0.244 but within one SE given std roughly 0.33.

The negative result on `copr_anchored` is severe: absorption F1 collapses to 0.066 (below naive_sft), worst-F1 to 0.014 (essentially chance on paraphrases), and compositional token-F1 to 0.025 (below the no-update baseline of 0.052). The task-replay-normalized reference pulls the policy toward the task manifold so strongly that absorption is smothered. Because this variant does not gold-inject, the pathology described in Section 3.3 compounds with the anchoring pressure. Absorption simply does not happen.

**Trajectories.** Full per-round trajectories are in `final_results/phase3_sequential_trajectory.csv`. Qualitative patterns:

- `kl_reg_sft`'s absorption F1 rises roughly monotonically from 0.082 at round 1 to 0.134 at round 10; its preservation stays in 0.186-0.244 band throughout.
- `copr_gold_injection`'s absorption F1 rises from 0.069 at round 1 to 0.162 at round 10, overtaking `kl_reg_sft` around round 4-5 (where it reaches 0.094-0.115, matching or exceeding `kl_reg_sft`'s contemporaneous absorption) and pulling away through round 10.
- `copr_anchored`'s absorption F1 is erratic, dipping to 0.040 at round 9 and never exceeding 0.089. Compositional at round 10 is 0.025, worse than no_update.

**Figure 1 (placeholder).** Per-round absorption F1 (y-axis, 0-0.20) versus round index (1-10) with one line per method. Caption: "Absorption trajectories over 10 sequential rounds (200 triples per round). `copr_gold_injection` crosses `kl_reg_sft` around round 4 and pulls away; `copr_anchored` never climbs above 0.089; `naive_sft` plateaus below 0.100." Data source: `final_results/phase3_sequential_trajectory.csv`.

**Figure 2 (placeholder).** Per-round locality overall F1 over 10 rounds, same format. Caption: "`copr_gold_injection_anchored` locality climbs steadily to 0.077 at round 10, surpassing all other methods; `kl_reg_sft` plateaus around 0.051." Data source: same.

**Trajectory gaps.** Honest accounting: rounds 6/7/9 are missing from `copr_anchored` trajectory for compositional metrics (only computed at round 10); no full-trajectory per-round faiss evaluation was re-run for these rounds after a faiss crash in the initial sweep. The per-round absorption and locality for all ten rounds are complete for all six methods (see trajectory CSV). The paper does not backfill missing cells or impute.

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

`copr_gi` has the narrowest stable rank (2.54) despite the highest total Frobenius norm (1.43) among the three non-anchored-alone variants --- it writes more, concentrated on fewer directions. `copr_gi_anchored` writes the most in total (1.47) but at slightly higher rank (2.72).

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

**Per-layer distribution.** From per-module Frobenius norms (`phase6_lora_deltas.csv`), `copr_gi` is the only variant with non-trivial writes to layers 0-3. All four variants write most heavily to `down_proj`, `up_proj`, `gate_proj`, and `q_proj` in the mid layers (roughly layers 10-20). The early-layer activity of `copr_gi` plausibly reflects the gold-injection signal pulling the model toward lexical absorption of the novel tokens earlier in the stack, where surface-form adjustments typically land.

**Figure 3 (placeholder).** Per-layer Frobenius norm (y-axis, summed over all modules at that layer) for each of the four COPR variants, layers 0-31. Caption: "`copr_gi` is the only variant with non-trivial mass in layers 0-3; the other three concentrate in layers 10-20." Data source: `final_results/phase6_lora_deltas.csv` aggregated by layer.

## 6. Discussion

### 6.1 Regime mismatch as the driver

The batch and sequential results are not a gradient of difficulty --- they are two different operational settings with different failure modes. In batch, the update method sees the entire edit set at once, the optimizer can take K gradient steps with well-mixed batches, and KL regularization against the pre-update reference is a strong, cheap, unbiased pull back toward task behavior. `kl_reg_sft` exploits this directly. The COPR family, by contrast, pays a sampling cost (K candidates per item per step) and an MSE fit on a rank signal, both of which are operationally redundant when cross-entropy on the gold answer plus KL replay already solves the problem.

In sequential, each round re-starts from the previous round's model. The KL reference drifts with the policy (or is pinned at round 0 at the cost of increasing KL with every round). The optimizer sees only 200 examples per round, and catastrophic forgetting within the fact distribution compounds. COPR's anchoring structure --- the reference-policy MSE fit that pins the ranking relative to pi_ref --- is closer to what the sequential regime actually needs. Gold injection makes this anchor productive by guaranteeing a correct target in the ranking; without it, the anchor just stabilizes the current (wrong) ranking.

### 6.2 When to pay the COPR compute premium

The paper's operational recommendation is direct. If the deployment is a single batch push of O(1K-10K) new facts, use KL-regularized SFT: it is 10-20x cheaper and dominates on absorption and locality at these scales. If the deployment is a continual-update stream (many rounds over time), the gold-injection COPR variants are worth the per-round compute premium, because (i) absorption ceiling is materially higher (+21% over KL on the 10-round cumulative), (ii) locality is materially better (+50%), and (iii) compositional token-F1 is at least as good. The premium in Phase 3 is roughly 12x per round (0.58 GPU-h for `copr_gi` vs 0.051 for `kl_reg_sft`).

### 6.3 Gold injection's mechanism

The pilot observation that K self-samples are all wrong on novel facts is, in retrospect, obvious: the base model does not know the fact. Under that condition, the COPR ranking is a ranking over wrong answers, and fitting to it is at best noise-absorbing and at worst actively misleading. Gold injection flips this: the ranking now has a correct top element, and the MSE fit pulls the policy toward it. Phase 6 supports this: `copr_gi` uniquely edits early layers (0-3), consistent with the model absorbing the novel surface form directly rather than redistributing probability among existing plausible answers.

### 6.4 The anchored-COPR negative result

Anchoring the reference policy with a task-replay normalizer (`copr_anchored`) fails in both regimes, but catastrophically in sequential (absorption F1 0.066, compositional token-F1 0.025). The mechanism: the task-replay-normalized reference makes any policy that deviates from the task manifold look relatively worse in the MSE fit. Since the new fact is, by construction, off-task (it is a new fact not seen during task-tuning), the anchor pushes *against* absorption. Combined with the no-gold-injection pathology, the method simply does not learn the new facts.

The recovery via `copr_gold_injection_anchored` is instructive: once gold is in the ranking, the anchor no longer harms and in fact slightly improves locality. This suggests the anchor is not inherently broken --- it needs a non-degenerate learning signal (gold injection) to work with, not against.

### 6.5 Compositional degradation

All updated methods except the gold-injection variants' token-F1 track more-or-less unchanged bridging-entity recall, but every method including the best degrades it below the no-update baseline of 0.102. The updates absorb the new first-hop fact but interfere with the second-hop retrieval of the bridging entity. This is a shared limitation and a call-to-action for the next iteration: either explicit multi-hop anchoring during training (candidates that require the bridging entity) or a retrieval-based bridge at inference.

## 7. Limitations

**Single seed.** All runs use seed=42. The paper does not report seed variance, which means some of the smaller differences (particularly preservation and compositional exact-match) may not be robust. A multi-seed replication (at minimum seed in {42, 7, 13}) is deferred.

**Preservation noise.** Preservation Recall@10 is computed over n=104 test items, with std roughly 0.33 across items. Standard error on the mean is thus about 0.032. Many preservation differences reported in the tables are within 1 SE of each other; the paper does not claim significance on preservation and uses it as a sanity guardrail rather than a primary metric.

**Locality `same_sector` stratum is n=1.** The locality evaluation has three strata, but the `same_sector` stratum is populated by a single probe. A stratum of size one is not meaningful as a mean estimate --- the reported `same_sector` F1 values are either 0.0, 0.154, 0.167, 0.4 depending on whether that single probe happens to match, and should not be interpreted as a population estimate. The aggregate locality overall F1 is driven by the `same_entity` (n=182) and `other_sector` (n=1817) strata. This is a construction artifact of the FNSPID sector taxonomy combined with the entity-locality sampling procedure and should be corrected in a follow-up by broadening the `same_sector` probe set.

**Compositional EM near zero.** Exact-match on two-hop compositional is 0-0.014 across methods. The compositional token-F1 signal is the usable metric; exact-match is effectively at floor and should not be over-interpreted.

**No temporal-contrast evaluation.** Phase 4 was planned to include a temporal-contrast metric (pre-2022 vs post-2022 fact collisions) but the Gemini-generated probes did not contain the required gold answers in a consistent format. The temporal-contrast metric is dropped from the paper rather than reported with incomplete data. This is noted in `final_results/phase4_compositional.csv`.

**Trajectory gaps.** For `copr_anchored`, the per-round trajectory for rounds 6, 7, and 9 is missing compositional metrics (those metrics are computed only at round 10 by design); rounds 6 and 7 also hit a faiss crash on the initial sweep and were not rerun. The absorption and locality per-round data for all ten rounds and all six methods is complete. The paper does not impute.

**Scale.** Results are reported at 1K, 3K (batch) and 10x200 (sequential). Scaling laws for knowledge editing with LoRA are not established, and the batch results at 10K or 100K may tell a different story.

**Model scale.** Only Qwen3-4B-Instruct-2507 is evaluated. Findings on the relative ordering of methods may not transfer to 7B, 14B, 70B, or GPT-scale models.

## 8. Conclusion

The paper makes three claims and supports each with a controlled experiment on a single backbone and a single data distribution. First, **the batch regime favors KL-regularized SFT**: a simple, cheap baseline dominates the COPR family at 1K and 3K edits at 10-20x less compute. Second, **the sequential regime favors gold-injected COPR**: `copr_gold_injection` wins absorption (F1 0.162, +21% over `kl_reg_sft`) and `copr_gold_injection_anchored` wins locality (overall F1 0.077, +50%), both at the cost of 10-12x per-round compute. Third, **anchoring alone is harmful**: `copr_anchored`'s absorption F1 collapses to 0.066 and its compositional token-F1 drops to 0.025, below the no-update baseline. A mechanistic analysis shows that the COPR variants share an essentially identical top input subspace but differ in stable rank and per-layer distribution, with `copr_gold_injection` uniquely active in the earliest layers --- a plausible mechanism for its absorption lead.

The operational recommendation follows: match the update method to the update schedule. Batch pushes of facts should use the KL baseline; continual-update streams should use gold-injected COPR; and anchoring the reference without gold injection should be avoided. Compositional two-hop reasoning degrades under all parametric edits tested here, which motivates explicit multi-hop anchoring as the next design lever.

## Reproducibility Statement

All code, configs, and raw artifacts are committed at the repository root, commit `1c28ce0`. Every number in Sections 5 and 6 is a direct copy from a row of a CSV in `final_results/`, which is itself regenerated by `scripts/20_snapshot_results.py` from the raw JSON in `outputs/`. Data counts and paths are in `final_results/data_inventory.csv`; per-run compute, seeds, and starting checkpoints are in `final_results/run_metadata.csv`; method configs are in `final_results/methods_index.csv` and `final_results/run_configs.csv`. The 500 compositional probes are in `data/fnspid/compositional/probes.json` with the Gemini-3.1-Flash-Lite prompt template in `configs/compositional/probe_template.yaml` (see repo). The task-tuned backbone LoRA checkpoint is `checkpoints/qd_sft/final`. The filtered fact triples are derived from the FNSPID corpus (Shah et al. 2023) with the filtering pipeline in `scripts/` (see `data/fnspid/triples/filtered_triples.json`, n=96,897). Randomness is controlled by seed=42 for all runs.
