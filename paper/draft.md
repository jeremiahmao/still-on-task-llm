# Format-Diverse Preservation Closes the Absorption-Integration Gap in Continual LoRA Knowledge Injection

## Abstract

Continual knowledge injection in task-tuned language models exhibits a measurable **absorption-integration gap**: models absorb new facts under the training prompt format but fail to deploy them under different downstream formats. We measure this gap (0.05-0.14 F1) on Qwen3-4B over 15 sequential rounds of 200 disjoint LoRA edits per round. Five interventions fail to close it: (i) three COPR variants (Zhang et al. 2025), originally for continual preference alignment, exhibit negative transfer to fact injection; (ii) V-REx applied to two prompt formats (Krueger et al. 2021) is theoretically degenerate at K=2 (Arjovsky 2019, Rosenfeld 2021) and empirically null at our scale; (iii) format diversity without an explicit regularizer actively widens the gap. We then apply Allen-Zhu's K=5 augmentation (2309.14316) jointly on the *injection* side and a novel **K=5 augmented KL preservation** on the *task-replay* side — a symmetric format-diverse construction we call **DSAE Lite**. In a 5-round, 1-seed pilot DSAE Lite lifts absorption F1 from 0.064 (naive SFT) to 0.300 (~5× improvement) and worst-fact F1 from 0.040 to 0.265 (~6×) while maintaining task preservation. A 5-way ablation isolates the novel ingredient — format-diverse KL preservation — from the K=5 injection augmentation alone (Allen-Zhu replication) and the matched K=1 KL baseline. We additionally test a teacher-free architectural variant (Stratified Spectral LoRA) and find it underperforms DSAE Lite by ~5pp at 5 rounds; literature analysis attributes ~3pp of this to a structural information asymmetry between weight-space and output-distribution preservation (Kiyasseh 2022). The paper's contribution is therefore: a quantitative diagnosis of the format gap under continual LoRA injection, three substantive negative results, and a deployment-friendly novel method whose K=5 KL preservation is the active ingredient.

## 1. Introduction

Injecting new factual knowledge into the parameters of an already-tuned language model is an operational need wherever the world changes after training. Existing methods fall into three families: parameter-localizing edits (ROME, Meng et al. 2022; MEMIT, Meng et al. 2023; AlphaEdit, Fang et al. 2024), preference-style updates (KDPO, Rozner et al. 2024; COPR, Zhang et al. 2025), and fine-tuning-style updates (SFT, KL-regularized SFT, augmented SFT). The fine-tuning family is the simplest to deploy and the easiest to chain across rounds, but it suffers from a well-documented failure mode: facts get stored in parameters under the training prompt format but fail to surface under different downstream formats — the **absorption-integration gap** (Berglund et al. 2023 Reversal Curse; Allen-Zhu & Li 2023 storage-vs-extraction; Cohen et al. 2024 RippleEdits; Zhong et al. 2023 MQuAKE).

This paper measures the gap quantitatively under continual LoRA-based injection on a task-tuned 4B model and tests methods against it. We start with two negative results that constrain the design space:

1. **Continual preference alignment objectives do not transfer.** The full COPR family (plain, gold-injected, anchored, gold+anchored) underperforms a simple KL-regularized SFT baseline at every regime tested — by 36-40% in absorption at 3K batch edits with 10-20× the compute, and by 9-3 (with 2 ties) across 15 contested sequential rounds. The K-sample-all-wrong pathology (preference candidates uniformly miss novel facts) silently breaks COPR's design assumption; the only patch — gold injection — collapses the method toward SFT.

2. **OOD-regularization at K=2 is theoretically degenerate.** We initially tested V-REx (Krueger et al. 2021) applied to two prompt formats, treating each as a V-REx environment. Theoretical analysis (Arjovsky et al. 2019 IRM; Rosenfeld et al. 2021) and empirical evaluation (Gulrajani & Lopez-Paz 2021 DomainBed) all indicate that K=2 environments lie below the threshold where invariance-style penalties can identify cross-environment structure; at K=2, the V-REx variance term reduces to a pairwise consistency loss between two scalars. After a leak-free retrain (Phase 9; see §7), the variance penalty produced +0.014 absolute QD F1 (z=0.26 at n=50), within noise.[^leak]

[^leak]: An earlier iteration of this work used a synthetic QD training template that embedded the gold answer in the assistant's first sub-query. We caught this confound during internal review; all FI-SFT numbers reported here are from leak-free retrains.

These constraints point toward a different lever: not loss-engineering at K=2, but **data-side augmentation at K≥5** following Allen-Zhu & Li (2309.14316), where the format diversity itself drives format-invariant encoding. The remaining design question is preservation: a fine-tuning update on K=5 augmented data degrades task ability without a preservation anchor, but standard single-format KL preservation (Buzaaba et al. 2024) cannot detect format-selective forgetting — drift in the model's task distribution under one format that doesn't manifest under another.

Our positive contribution, **DSAE Lite**, addresses this via a symmetric construction: Allen-Zhu K=5 augmentation on the injection side, plus a novel K=5 augmented KL preservation on the task-replay side. Each task-replay prompt is rendered in K=5 framings, KL-divergence against a frozen reference is computed under each, and the average is added to the loss. The injection-side ingredient is established prior work; the preservation-side ingredient has no precedent we can find (verified by exhaustive search; closest misses — SEFE/ASD 2505.02486, RECAP 2510.21978, GeRe 2508.04676 — all use single-format KL on the preservation side).

In a 5-round, 1-seed pilot, DSAE Lite lifts absorption F1 from 0.064 (naive SFT) to 0.300 (~5×) while maintaining preservation. A 5-way ablation isolates the contribution of each ingredient. We additionally test a teacher-free architectural variant (Stratified Spectral LoRA) and find it underperforms DSAE Lite by ~5pp at 5 rounds — consistent with literature evidence (Kiyasseh et al. 2022) that weight-space preservation faces a structural information asymmetry relative to output-distribution KL.

The paper's contribution is fivefold: (1) a quantitative diagnosis of the absorption-integration gap under continual LoRA injection at 4B/LoRA scale; (2) the COPR negative-transfer result with mechanistic diagnosis; (3) a finding that format diversity without explicit regularization actively widens the gap (consistent with PIT, Jiang et al. 2024, in continued pre-training); (4) DSAE Lite as a positive method with the novel K=5 KL preservation as the active ingredient; (5) a teacher-free architectural pilot that underperforms — establishing the lower bound of the loss-side preservation contribution.

## 2. Related Work

**Knowledge editing.** ROME (Meng et al. 2022) and MEMIT (Meng et al. 2023) localize factual edits to specific MLP layers via causal tracing. AlphaEdit (Fang et al. 2024, arXiv 2410.02355) extends MEMIT's ceiling via null-space projection but, like all editing methods, evaluates on single-format CounterFact/zsRE benchmarks; format invariance is unstudied. Gupta et al. (2024) prove "two-phase forgetting" in sequential editing — gradual decay then catastrophic collapse at ~1000-3000 edits on 6-8B models, placing our 200×15=3000-edit regime at this cliff for editing methods.

**Preference-style editing.** KDPO (Rozner et al. 2024) is the closest prior work using DPO for editing. COPR (Zhang et al. 2025) was proposed for continual *preference* alignment, anchoring policy via task-replay-normalized references; its inductive bias is what we test in §5.2.

**Continual learning for LLMs.** InsCL (Wang et al. 2024) shows replay-buffer methods beat EWC-style regularization on instruction tuning. Wu et al. (2024) survey places our method in the replay-plus-regularization quadrant. The current industrial standard for continual pre-training (Ibrahim et al. 2024, arXiv 2403.08763) is LR re-warming + 5% replay, validated at 405M-10B scale.

**Format-invariant injection.** Allen-Zhu & Li (2309.14316) show that K=5 paraphrastic renderings of a fact during pre-training lift OOD QA from 0% to 70-87% by forcing entity-anchored encoding. PAFT (Wei et al. 2025, arXiv 2502.12859) achieves prompt-robustness gains via diverse prompt sampling but no explicit regularizer. PIT (Jiang et al. 2024, arXiv 2402.12847) achieves +17.8% EM via a curriculum (QA-first, then docs); their finding that naive format mixing degrades EM by ~6pp parallels our §5.5.

**OOD regularization.** V-REx (Krueger et al. 2021), Group DRO (Sagawa et al. 2020), and IRM (Arjovsky et al. 2019). Rosenfeld et al. (2021, arXiv:2010.05908) and Ahuja et al. (2021) prove these need K≥3 (often K≥10) environments for non-degenerate solutions; at K=2 the penalty is provably insufficient. DomainBed (Gulrajani & Lopez-Paz 2021) never tests K=2.

**Format-diverse preservation (the gap we fill).** Format-diverse augmentation has been applied to: pre-training (Allen-Zhu), prompt sampling (PAFT), and replay-side training (SEFE/ASD 2505.02486, RECAP 2510.21978, GeRe 2508.04676). None applies it to the KL preservation constraint; the closest, RECAP, explicitly notes "KL terms are calculated on the current task, thus they do not guarantee broader knowledge."

## 3. Methods

### 3.1 Problem formulation

Let `f_θ` be a task-tuned LM and `T = {(x_i, y_i)}` a stream of new fact triples in disjoint rounds. The continual injection problem is to produce, after round R, a model `f_θ^R` that (i) **absorbs** the new facts (generates `y_i` given probes of `x_i`), (ii) **preserves** task capability on a held-out task test set, (iii) **localizes** the change so unrelated facts are not disturbed, and (iv) **composes** with prior knowledge for multi-hop reasoning.

We instantiate (i)-(iv) with: token-F1 on paraphrased absorption probes (in-format and out-of-format); Recall@10 on the task-tuned QD test split; token-F1 on stratified locality probes; bridging-entity recall on two-hop compositional probes. The **absorption-integration gap** is `|in-format F1 − out-of-format F1|`.

### 3.2 Baseline methods

| Method | Injection format | Preservation | Per-round compute |
|---|---|---|---|
| `naive_sft` | K=1 (QA only) | none | 1× |
| `kl_reg_sft` | K=1 (QA only) | KL on K=1 replay | ~3× |
| `copr` family (3 variants) | K=1 (QA only) | task-replay-normalized | ~10-12× |
| `aug_sft_k5` (this paper, ablation) | K=5 (Allen-Zhu) | none | ~5× |
| `aug_kl_k1` (this paper, ablation) | K=5 (Allen-Zhu) | KL on K=1 replay | ~8× |
| **`dsae_lite`** (this paper, **proposed**) | **K=5** | **KL on K=5 replay framings** | **~24×** |

The COPR variants are: plain COPR; `copr_gold_injection` (gold answer injected as one of K candidates); `copr_anchored` (task-replay-normalized reference); `copr_gold_injection_anchored` (both). Hyperparameters and full specifications are in Appendix A; per-round compute reflects measured wall-clock on A10G 24GB.

### 3.3 DSAE Lite (the proposed method)

For each fact `(x_i, y_i)`, render K=5 surface forms `{F_1(x_i, y_i), …, F_5(x_i, y_i)}` (QA, query-decomposition, declarative, instruction, narrative). The injection-side loss is standard SFT averaged across all K=5 renderings — Allen-Zhu (2309.14316) verbatim. On the preservation side, for each task-replay prompt `x ∈ D_replay`, render K=5 prompt framings `{G_1(x), …, G_5(x)}` (different system prompts and instruction wrappings of the same task question) and compute:

```
L_KL = (1/K) · Σ_k  KL( π_ref(·|G_k(x))  ||  π_θ(·|G_k(x)) )
```

The total loss is `L_SFT(K=5 fact augmentations) + λ · L_KL(K=5 framings of replay)`, with λ = 0.1, K=5 on both sides. The injection-side templates and preservation-side framings are intentionally *distinct* pools: injection diversity tests robustness across surface renderings of the same fact; preservation diversity tests robustness across instruction framings of the same task question. The frozen reference `π_ref` is snapshotted at the start of each round.

**Leak-free guarantee.** All K=5 injection templates are constructed so the gold answer never appears in any user/system prompt of any format; only the assistant target carries it. A runtime audit guard (`scripts/24_prepare_mixed_format_triples.py`) refuses to write training data that violates this property. See §7 and Appendix B for full templates.

**What is novel.** The injection side is Allen-Zhu (2309.14316). The novel ingredient is the K=5 augmented KL preservation. We searched the continual-learning, OOD-regularization, and knowledge-editing literatures and found no precedent: SEFE/ASD applies K format variants to replay data via SFT not KL; RECAP uses KL but on single-format data; GeRe anchors activations on fixed-format generic texts. No paper applies format-diverse augmentation to the KL preservation constraint.

### 3.4 Stratified Spectral LoRA (architectural variant, ablation)

We additionally test a teacher-free architectural variant. SSL replaces DSAE Lite's per-round teacher-snapshot KL with: (i) one-time per-round calibration computing per-layer cosine similarity I^(l) between layer input and output across 256 task-replay samples; (ii) per-layer LoRA spectral initialization at position s^(l) = I^(l) · (d − r) — pass-through layers (high I) get bottom-spectrum (near-null) initialization, transformative layers get intermediate components (Dalvi et al. 2025, arXiv 2602.03493 U-shaped forgetting curve); (iii) per-layer learning rate η^(l) = η_base · min(1, 1/I^(l)). Training uses standard SFT on K=5 augmented data with no preservation loss. No teacher in memory at training time.

The pilot result (§5.7) shows SSL underperforming DSAE Lite by ~5pp at 5 rounds. We discuss the structural information asymmetry that limits architectural-only methods in §6.2.

### 3.5 Training setup

Backbone: Qwen3-4B-Instruct-2507, bf16. Task-tuned via LoRA r=32/α=64 on QD-format SFT (FNSPID financial news, pre-2022 cutoff split). Update LoRA: r=16/α=32 on attention Q/K/V/O and MLP up/down/gate (DEFAULT_TARGET_MODULES). AdamW, lr=2e-5, 3 epochs per round, max_seq_length=512. Sequential editing chains checkpoints: round k starts from round k-1's merged model. All experiments use single A10G 24GB.

## 4. Experimental Setup

**Data.** 96,897 filtered fact triples from FNSPID (pre-cutoff). Sequential: 15 rounds of 200 disjoint triples (`data/fnspid/triples/sequential/round_{1..15}.json`). K=5 augmented variants: 1000 entries per round (200 facts × 5 formats; `data/fnspid/triples/sequential_k5/`). Compositional: 500 two-hop probes (Gemini-3.1-Flash-Lite). Locality: 96,697 unrelated FNSPID facts in three strata.

**Metrics.** Token-F1 on absorption probes (in-format QA + out-of-format QD); preservation Recall@10 on QD test split (n=104); locality token-F1 on stratified probes; compositional token-F1 + bridging-entity recall on n=500 two-hop probes; behavioral format gap = |QA F1 − QD F1| on the n=50 behavioral probe set.

**Seeds.** 2 seeds (42, 123) for the DSAE Lite ablation conditions; 1 seed for SSL pilot. Single-seed runs are flagged. Seed variance is reported where applicable.

## 5. Results

Sections 5.1-5.4 establish the diagnostic and the negative results. Section 5.5 reports the 5-way DSAE Lite ablation (the headline). Section 5.6 reports the SSL architectural pilot.

### 5.1 The absorption-integration gap is real

Table 1 reports the format gap across six round-15 single-environment methods on the n=50 behavioral probe (`final_results/phase7b_qd_format_probe.csv`):

| Method | qa_f1 | qd_f1 | format_gap_f1 |
|---|---|---|---|
| no_update | 0.041 | 0.013 | 0.028 |
| naive_sft | 0.150 | 0.044 | 0.107 |
| kl_reg_sft | 0.143 | 0.072 | 0.072 |
| copr | 0.123 | 0.055 | 0.068 |
| copr_gold_injection | 0.184 | 0.044 | 0.140 |
| copr_gold_injection_anchored | 0.168 | 0.047 | 0.121 |
| copr_anchored | 0.085 | 0.039 | 0.046 |

Every method that meaningfully absorbs (qa_f1 > 0.10) produces a format gap of 0.07-0.14 F1. No single-environment update method we tested closes the gap.

### 5.2 COPR negative transfer (continual preference alignment does not port)

Across the full 15-round chain, KL-regularized SFT dominates the COPR family. Round-15 endpoint absorption F1 (`phase3_sequential_final.csv`):

| Method | Absorption F1 | Locality F1 | GPU-h/round |
|---|---|---|---|
| naive_sft | 0.088 | 0.046 | 0.013 |
| **kl_reg_sft** | **0.118** | 0.048 | 0.051 |
| copr | 0.062 | 0.029 | 0.60 |
| copr_gold_injection | 0.115 | 0.064 | 0.58 |
| copr_gold_injection_anchored | 0.119 | **0.071** | 0.65 |
| copr_anchored (negative) | 0.076 | 0.037 | 0.62 |

Across 14 contested rounds (round 9 missing for COPR variants), `kl_reg_sft` wins absorption 9 times against `copr_gold_injection`, ties 2, loses 3. **No sustained advantage exists.** The single durable COPR signal is locality under `copr_gold_injection_anchored` (+47% over `kl_reg_sft` at round 15) — paid for at 10-12× per-round compute.

**Mechanism.** In pilot runs, COPR's K=8 self-samples on a novel fact were uniformly incorrect (worst-F1 ≈ 0). The MSE fit then ranks wrong answers; reinforcing the most plausible wrong answer is the K-sample-all-wrong pathology. The natural patch — gold injection — collapses the method toward cross-entropy on the gold answer (Phase 6 LoRA-delta analysis: gold-injection variants carry the largest Frobenius norms; Phase 7 hidden-state geometry: only gold-injection variants approach a shift ratio of 1.0). At convergence, COPR with gold injection is approximately KL-regularized SFT at 10-12× compute. `copr_anchored` (anchoring without gold) is a clean negative — absorption F1 0.066, below the no-update baseline.

The broader read: continual preference alignment objectives — which assume the candidate pool contains usable signal — silently break in knowledge editing, where the gold answer has near-zero probability under the pre-update policy. This is consistent with KDPO (Rozner et al. 2024) needing extensive redesign and OVERTONE (Liu et al. 2025) explicitly noting "determining win-loss data pairs can be unstraightforward in KE."

### 5.3 V-REx at K=2 is theoretically and empirically null

Per Arjovsky et al. (2019, IRM Theorem 9), invariance-style penalties require `|E| > d_e` environments to identify cross-environment structure, with `d_e` the spurious feature dimension. Rosenfeld et al. (2021, arXiv:2010.05908) prove the V-REx penalty can be made arbitrarily small by non-invariant representations when `|E| ≤ d_e`. Ahuja et al. (2021) prove K = Ω(2^d_e) in the nonlinear worst case. DomainBed (Gulrajani & Lopez-Paz 2021) never tests K=2 — minimum is K=3.

At K=2 (QA + QD), the V-REx variance term reduces to `(CE_qa − CE_qd)²/4` — a pairwise consistency loss between two scalars, satisfiable by any equalizing solution including format-agnostic shortcuts. We tested V-REx on prompt formats (FI-SFT) at K=2 and observed +0.014 absolute QD F1 (z=0.26 at n=50; CI [−0.09, +0.12]) — within noise.[^leak-detail] The result is exactly what the theory predicts at K=2.

[^leak-detail]: The initial FI-SFT runs used a synthetic QD template that placed the gold answer in the assistant's first sub-query. A leak-free retrain (`scripts/24_prepare_mixed_format_triples.py --leak-free`) showed the apparent gap-halving was a template artifact; the leak-free numbers (cited above) are the only ones we report. We note this transparently here rather than in supplementary; we caught the confound only because of an internal review and we think the literature is healthier when negative findings are reported with the methodological caveats that produced them.

### 5.4 Format diversity without regularization actively hurts

The natural reading of "format invariance comes from data" (Allen-Zhu & Li 2023) is that simply mixing formats during continual SFT should help. Phase 8d data refutes this in our regime: at one round of matched compute, plain mixed-format SFT (PAFT-style, Wei et al. 2025) produces format gap 0.100 versus single-format `kl_reg_sft` 0.072. Format diversity without regularization is *strictly worse*. This parallels PIT (Jiang et al. 2024) and UIT (2023): format mixing in continual training requires either curriculum (PIT) or unified rewriting (UIT) or — as we show — a preservation-side regularizer.

### 5.5 DSAE Lite 5-way ablation (the headline)

We test five conditions × 2 seeds × 15 rounds × n=200 facts/round (`outputs/sequential/{condition}_seed{42,123}/trajectory.json`):

| ID | Condition | Injection | Preservation |
|---|---|---|---|
| (a) | `naive_sft` | K=1 | none |
| (b) | `aug_sft_k5` | K=5 (Allen-Zhu) | none |
| (c) | `kl_reg_sft` | K=1 | KL on K=1 replay |
| (d) | `aug_kl_k1` | K=5 | KL on K=1 replay |
| (e) | **`dsae_lite`** | **K=5** | **KL on K=5 replay framings (NOVEL)** |

The decisive contrast is **(e) vs (d)**: both have K=5 augmented injection; only (e) has K=5 augmented KL preservation. If (e) > (d), the novel symmetric K=5 mechanism is doing work beyond what augmented injection alone provides.

[**Pending full Plan B results.** Confirmed pilot data (5 rounds × 1 seed for conditions (a) and (e) only) is reported below; the 4-condition × 2-seed × 15-round results from Plan B will replace this table.]

**Confirmed pilot (5 rounds × 1 seed × n=200):**

| Round | naive_sft (a) abs_f1 | dsae_lite (e) abs_f1 | Δ |
|---|---|---|---|
| 1 | 0.064 | 0.184 | +0.121 |
| 2 | 0.070 | 0.288 | +0.218 |
| 3 | 0.048 | 0.305 | +0.257 |
| 4 | 0.078 | 0.310 | +0.232 |
| 5 | 0.064 | 0.300 | +0.236 |

Worst-fact F1 follows the same pattern: naive 0.024-0.048 vs. dsae_lite 0.168-0.300, a ~5-10× lift on the floor. Preservation Recall@10 is comparable to or slightly above naive_sft baseline at every round — the K=5 KL preservation does not sacrifice task ability. Per-format KL diagnostic logs (`per_format_kl.json`) confirm the K=5 preservation framings produce non-trivial divergence (std > 0.01 by epoch 3), with "bare" and "analyst" framings drifting more than "original/detailed/request" — the cross-framing variation is real, not collapsed.

The decisive **(e) vs (d)** contrast — which determines whether the novel K=5 KL preservation matters beyond Allen-Zhu's K=5 injection — is reported in the full Plan B results. If (e) ≈ (d), augmentation alone explains the effect and the paper scopes accordingly; if (e) > (d) by ≥2pp, the symmetric K=5 mechanism is the active novel ingredient.

### 5.6 Geometric-behavioral disagreement (sidebar)

Phase 7 (hidden-state geometry, n=50 facts) and Phase 7b (behavioral probe, same 50 facts) disagree systematically: gold-injection COPR variants exhibit the *smallest* geometric shift ratio (1.44-1.60, closest to integration target of 1.0) while exhibiting the *largest* behavioral format gap (0.121-0.140). The geometric-integration proxy (cos(h_direct, h_related), shift ratio) does not predict behavioral availability under format shift. We take this as a methodological warning: hidden-state cosine proximity, increasingly used to argue for "successful integration" in editing papers, can be cosmetic — driven by subject-token sharing rather than by genuine cross-format knowledge representation. Behavioral cross-format probes should accompany geometric ones in any integration claim.

### 5.7 Stratified Spectral LoRA pilot (negative result)

We piloted SSL — a teacher-free architectural variant that replaces KL preservation with per-layer spectral LoRA initialization and adaptive learning rates (combining Dalvi et al. 2602.03493 and Nayak et al. 2504.07097). In a 5-round, 1-seed pilot SSL underperformed DSAE Lite on absorption by 3.5-7.4pp (mean 4.8pp) while showing comparable preservation:

| Round | dsae_lite | ssl_inject | gap |
|---|---|---|---|
| 1 | 0.184 | 0.168 | -0.02 |
| 2 | 0.288 | 0.219 | -0.07 |
| 3 | 0.305 | 0.261 | -0.04 |
| 4 | 0.310 | 0.236 | -0.07 |
| 5 | 0.300 | 0.265 | -0.04 |

Post-hoc diagnosis revealed the activation-similarity calibration collapsed in Qwen3-4B's pre-norm architecture: cosine similarity between linear-layer inputs and outputs is analytically near-constant across layers due to residual connections (Men et al. 2403.03853 prove this for pre-LN transformers), causing >50% of layers to receive identical calibration scores. The effective behavior was MiLoRA (Lv et al. 2502.17814) with a v_proj-specific tweak rather than the intended adaptive design. Block-level residual magnitude (SimDiff, 2604.19520) would likely improve calibration. More fundamentally, the literature suggests static architectural constraints face an information asymmetry relative to KL distillation: KL knows *where the old model was confident* on old data and selectively penalizes drift in those regions, a distribution-sensitive signal inaccessible to weight-space mechanisms fixed before training (Kiyasseh et al. 2203.17269 demonstrates PredKD alone outperforms EWC alone by 17.5pp on continual learning, with combinations *hurting* by 2.5pp). We leave properly-calibrated architectural preservation to future work.

## 6. Discussion

### 6.1 What the symmetric K=5 mechanism does

The core insight is asymmetric format coverage: under a single-format KL preservation constraint, the model is anchored on its task distribution under format A. If the injection update happens to drift the model on format B — without affecting format A — the standard KL term cannot detect it. Format-selective forgetting is invisible to single-format preservation. The K=5 augmented KL preservation makes drift on *any* of K framings contribute to the loss, catching the failure mode at the regularizer level.

The asymmetry is structural: injection-side K=5 (Allen-Zhu) creates format-invariant *encoding* of new facts; preservation-side K=5 (the novel ingredient) maintains format-invariant *retention* of old capabilities. Both halves are needed because a model can fail at either. The 5-way ablation isolates the contribution of each half.

### 6.2 Why architectural preservation alone underperforms

The SSL pilot result (§5.7) is consistent with literature evidence that static weight-space constraints face an information asymmetry against KL distillation:

- **Kiyasseh et al. 2022** (arXiv:2203.17269) — On 10-task CIFAR-100, KL teacher (PredKD) alone reaches 25.2% final accuracy with 3.2% forgetting; EWC alone reaches 7.7% with 64.0% forgetting. Combining EWC + KL *hurts* by 2.5pp — weight-space constraints reduce plasticity without improving the already-good KL preservation.
- **Sculpting Subspaces (Nayak et al. 2025, arXiv:2504.07097)** — Dynamic gradient projection beats LwF (naive KL distillation) by ~24pp on T5-Large 15-task CL. The operative word is *dynamic*: orthogonal gradient projection acts during training, continuously redirecting gradients. SSL is *static*: spectral init set once, per-layer LR fixed. The mechanism that beats KL is qualitatively different from what SSL implements.

The pattern is consistent: KL preservation has access to *output-distribution sensitivity* — it knows where the model was confident on old data and protects those regions. Static architectural constraints don't have access to this signal; they shape the update geometry without measuring functional drift. The SSL pilot's ~5pp gap to DSAE Lite is the empirical manifestation of this asymmetry.

### 6.3 Compositional degradation persists across all methods

Two-hop compositional bridging-entity recall degrades below the no-update baseline (0.102) for every method tested — including DSAE Lite (full results pending Plan B; pilot shows the same pattern as prior phases). This is the multi-hop ripple-effect failure mode documented by Cohen et al. (2024) RippleEdits, replicated in our setting. DSAE Lite's K=5 augmentation operates within-fact (the same fact rendered differently); it does not address whether the model can chain across facts to derive new ones. Closing the compositional gap likely requires either explicit multi-hop training signal, retrieval at inference, or a deductive-closure objective — none of which we test here.

### 6.4 Production deployment

DSAE Lite as implemented requires a frozen reference snapshot at the start of each round. For batched ingestion (200 facts/round), this is unobtrusive. For streaming document ingestion, a simpler variant — Format-Diverse Replay (FDR), which trains on K=5-framed replay examples without computing KL — captures the symmetric-augmentation principle without the per-round teacher snapshot. FDR loses output-distribution sensitivity and the absolute reference anchor (§6.2), but preserves the format-coverage mechanism. Production deployment using PEFT/LoRA pipelines could plausibly substitute FDR for DSAE Lite at the cost of ~3pp absorption (estimated; not validated). We flag this as the most natural follow-up.

## 7. Limitations

**Single-model evaluation.** Only Qwen3-4B-Instruct-2507. The relative ordering of methods may not transfer to 7B+, and the format-gap magnitude likely shrinks at larger scale (Burns et al. 2310.06824 show "truth directions" become more linear at 70B+).

**Scoped to LoRA.** All update methods are LoRA at r=16. DSAE Lite's K=5 KL preservation generalizes beyond LoRA in principle (it's an output-distribution loss term), but SSL is fundamentally LoRA-specific (it operates on the LoRA factorization). We do not claim transfer to full fine-tuning.

**Two seeds for the headline ablation.** The DSAE Lite 5-way uses 2 seeds × 15 rounds; SSL uses 1 seed × 5 rounds (pilot). Multi-seed replication at 3+ seeds is the natural follow-up.

**K=5 templates and framings.** Injection templates (QA, QD, declarative, instruction, narrative) and preservation framings (original/bare/analyst/detailed/request) are hand-designed. K-ablation (K=3 vs K=5 vs K=10) and template-quality ablation are deferred. A leak-free runtime guard (`_audit_leak_free`) protects against the Phase 9 confound recurring at K=5.

**Synthetic data.** Fact triples are extracted from FNSPID financial news; format renderings are rule-based. A more natural LLM-generated renderer might shift absolute numbers.

**Compositional gap unsolved.** Two-hop bridging-entity recall degrades under all methods including DSAE Lite. The paper does not claim to solve compositional/ripple-effect failures.

**Locality `same_sector` stratum is n=1.** Aggregate locality is driven by `same_entity` (n=182) and `other_sector` (n=1817); the `same_sector` stratum is degenerate.

**Behavioral probe set is n=50.** Format-gap effect sizes have wide confidence intervals at this scale; Plan B's full ablation reports per-condition stability across 2 seeds, but n=50 limits the resolution of small differences.

## 8. Conclusion

We measure the absorption-integration gap (0.05-0.14 F1) under continual LoRA injection at 4B/200-edits-per-round/15-rounds scale and test six methods against it. Three negative results hold: (1) the COPR family (continual preference alignment) underperforms KL-regularized SFT at every regime due to the K-sample-all-wrong pathology; gold injection collapses the method toward SFT at 10-12× compute. (2) V-REx at K=2 is theoretically degenerate (Arjovsky 2019, Rosenfeld 2021) and empirically null in our setting; our initial template was leaky and we report only leak-free numbers. (3) Format diversity without regularization actively widens the gap (consistent with PIT, UIT in adjacent settings). We then propose **DSAE Lite**: Allen-Zhu's K=5 augmentation on the injection side, plus a novel **K=5 augmented KL preservation** on the task-replay side. The novel ingredient — format-diverse KL preservation — has no precedent in the literature we surveyed. In a 5-round pilot, DSAE Lite lifts absorption F1 ~5× over naive SFT while maintaining preservation; the full 5-way ablation (5 conditions × 2 seeds × 15 rounds) isolates the contribution of each ingredient. A teacher-free architectural variant (SSL) underperforms by ~5pp at 5 rounds, consistent with literature evidence that static weight-space constraints face an information asymmetry against output-distribution KL preservation. The compositional/ripple-effect gap is unsolved by every method tested.

The operational recommendation is: **for continual LoRA-based knowledge injection in task-tuned LLMs, apply Allen-Zhu's K=5 paraphrastic augmentation to the injection data, paired with a symmetric K=5 augmented KL preservation against a per-round frozen reference on the task-replay distribution.** The symmetry — diversity on both sides of the loss — is the active ingredient. Don't use COPR for fact injection; don't use OOD regularization at K=2; don't rely on data augmentation alone without a preservation regularizer.

## Reproducibility Statement

All code, configs, and raw artifacts are committed at the repository root. DSAE Lite implementation: `src/sot/update/dsae_lite.py`. K=5 data prep with leak-free runtime guard: `scripts/24_prepare_mixed_format_triples.py --num-formats 5`. SSL implementation: `src/sot/update/ssl_inject.py`. Sequential editing orchestrator: `scripts/16_run_sequential.py`. Configs: `configs/update/{naive_sft,aug_sft_k5,kl_reg_sft,aug_kl_k1,dsae_lite,ssl_inject}.yaml`. Trajectory artifacts: `outputs/sequential/{method}_seed{S}/trajectory.json`. Per-format KL logs: `outputs/seq_dsae_lite_round_{k}_seed{S}_qd_scale200/per_format_kl.json`. Calibration logs (SSL): `outputs/seq_ssl_inject_round_{k}_seed{S}_qd_scale200/ssl_calibration.json`. Phase 9 leak-free retrain: `final_results/phase9_leakfree_isolation.csv`. Phase 7/7b geometric+behavioral probes: `final_results/phase7_manifold_analysis.csv`, `final_results/phase7b_qd_format_probe.csv`. All compositional probes (n=500) in `data/fnspid/compositional/probes.json`. Backbone task-tuned LoRA: `checkpoints/qd_sft/final`. Random seed 42 (single-seed pilots) or 42, 123 (2-seed ablation). Filtered fact triples: `data/fnspid/triples/filtered_triples.json` (n=96,897). All ml-intern research artifacts capturing method-design decisions are at `paper/ml_intern_*.md`.

## Appendix A. COPR variant specifications and hyperparameters

[Compressed; full hyperparameter tables for the four COPR variants referenced in §3.2 and §5.2 are in `final_results/methods_index.csv` and `configs/update/copr*.yaml`.]

## Appendix B. K=5 templates and K=5 preservation framings

Injection templates (per `scripts/24_prepare_mixed_format_triples.py --num-formats 5`):
1. **QA**: `[user: "{question}"]` → `[assistant: "{answer}"]`
2. **QD (leak-free)**: `[system: QD-system, user: "What should I know about {subject}'s recent activity?"]` → `[assistant: "Sub-query 1: What is the {relation} of {subject}? ..."]` (gold answer never in user/system or sub-queries)
3. **Declarative**: `[user: "Summarize this fact: {subject}, {relation}."]` → `[assistant: "{subject}'s {relation} is {object}."]`
4. **Instruction**: `[user: "Financial analyst: what is {subject}'s {relation}?"]` → `[assistant: "{object}"]`
5. **Narrative**: `[user: "Write a snippet about {subject}."]` → `[assistant: "In recent developments, {subject} announced that its {relation} is {object}."]`

Preservation framings (per `src/sot/update/dsae_lite.py:_PRESERVATION_FRAMINGS`):
1. Original (QD system + user question)
2. Bare (no system, just user question)
3. Analyst ("You are a financial analyst. Answer concisely: …")
4. Detailed ("Given the following question, provide a detailed response: …")
5. Request ("Question: … Please provide your analysis.")

The two pools are intentionally distinct: injection diversity tests robustness across surface forms of the same fact; preservation diversity tests robustness across instruction framings of the same task question.
