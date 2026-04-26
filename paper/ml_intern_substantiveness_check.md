# Substantiveness Check: Are DSAE Lite's Pilot Results Real?

**Date:** 2025-04-26
**Status:** Pre-Plan-B assessment. Pilot only: 5 rounds × 1 seed × n=200, conditions (a) `naive_sft` and (e) `dsae_lite`.

---

## Q1: Is the (a) vs (e) +24pp Gap Likely Real?

**Yes. The gap is real. It is not a single-seed artifact.**

The statistical case is straightforward. At n=200 facts per round, even with conservatively high per-item SD of 0.25 (typical for binary-ish token-F1 where many naive_sft items score 0), the SE of the between-condition difference is ~0.025. The observed Δ=0.236 at round 5 is z≈9.4 — not remotely within noise. The gap is also *temporally stable*: it appears at round 1 (+0.121), widens through round 3 (+0.257), and plateaus at rounds 4-5 (+0.232 to +0.236). An artifact would not produce this smooth, monotonically increasing trajectory across 5 independent training rounds.

The single-seed concern is about *generalization across random initializations*, not about within-seed sampling noise. Published between-seed variance for LoRA methods at 4B scale is typically SD≈2-5pp (TRACE benchmark, arXiv:2310.06762; LoRA knowledge packing, arXiv:2502.14502). Even at an extremely generous between-seed SD of 10pp, the gap (23.6pp) is still 2.4 seed-SDs — significant.

**Calibration against published results.** The +24pp lift is within the expected range for K=5 augmentation effects on a near-zero baseline:

| Paper | Setting | Augmentation lift | Scale |
|---|---|---|---|
| Allen-Zhu 2309.14316 | Pre-training, bioS, K=1→K=5 | 9.7%→96.6% (+87pp) | GPT-2 124M-682M |
| Masked FT (2510.09885) | LoRA fine-tune, Qwen3-4B, no-para→permute | 0.509→0.693 fwd (+18pp); 0.353→0.639 bwd (+29pp) | Qwen3-4B |
| PIT (2402.12847) | Continued pre-training + IT, Llama-2-7B | 30.3%→45.4% EM (+15pp) | Llama-2 7B |

Our baseline (0.064) is far below the Masked FT baseline (0.509) and closer to Allen-Zhu's near-zero regime. At low baselines, augmentation has outsized proportional impact. A lift from 0.064 to 0.300 (+24pp, ~5×) is entirely consistent with what the Masked FT paper shows at Qwen3-4B scale when the starting point is low, and *less* extreme than Allen-Zhu's 9.7%→96.6% in pretraining.

**Verdict on Q1:** The gap is real. It would survive 2-seed replication. The question is not whether (e) beats (a) — it does, by a margin that cannot be noise. The question is *what component of (e) is doing the work*, which is Q2.

---

## Q2: How Likely Is It That (b) ≈ (e)?

**This is the crux. The literature gives a clear central estimate but wide uncertainty.**

The DSAE Lite method bundles two ingredients: K=5 augmented injection (Allen-Zhu, well-validated) and K=5 augmented KL preservation (novel, no precedent). The pilot compares (e) to (a), but the paper's contribution lives or dies on (e) vs (d), and the critical unknown is where (b) `aug_sft_k5` lands.

**What the literature says about K=5 injection alone in our regime:**

Allen-Zhu's mechanism is validated at pretraining scale. The Masked FT paper (2510.09885) — the most directly comparable study — tests paraphrase augmentation during LoRA fine-tuning on Qwen3-4B specifically. Their key finding: permute-order paraphrases (functionally equivalent to Allen-Zhu K=5) boost forward QA from 0.509→0.693 (+18pp) and backward from 0.353→0.639 (+29pp). But their baseline is much higher than ours (general QA, not domain-specific sequential injection), meaning the *absolute* augmentation lift is large even from a moderate baseline.

PIT (2402.12847) shows +15pp from pre-instruction-tuning (a curriculum trick, not pure augmentation) on Llama-2-7B. Their mechanism is ordering-based, not K-augmentation, so the number is a loose bound.

**No paper tests K=5 injection without KL preservation in multi-round continual LoRA.** This is the exact gap DSAE Lite's ablation fills. The closest data point is LoRA Knowledge Packing (2502.14502), which shows that at 3000 facts with 10 paraphrases, reliability *collapses to 0.48* — aggressive augmentation without preservation degrades in multi-round settings. This is indirect evidence that K=5 injection alone (condition b) may plateau or degrade at later rounds without KL preservation, while (e) maintains via the preservation anchor.

**My central estimate for (b) at round 5:**

Given (a) our near-zero baseline, (b) the Masked FT paper's +18-29pp at Qwen3-4B scale, and (c) the LoRA packing paper's evidence that unregulated augmentation degrades at scale, I estimate:

- **Conservative (b) ≈ 0.15** (2.3× lift) — augmentation helps but multi-round degradation eats most of the gain
- **Central (b) ≈ 0.20-0.22** (3.1-3.4× lift) — augmentation delivers strongly but plateaus below (e) without preservation
- **Aggressive (b) ≈ 0.28** (4.4× lift) — augmentation alone does almost everything; KL preservation is cosmetic

**The central estimate implies (b) ≈ 0.20-0.22, (d) ≈ 0.22-0.24 (adding single-format KL), and (e) ≈ 0.30.** Under this scenario, (e)−(d) ≈ 6-8pp — a meaningful, publishable gap where the novel K=5 KL preservation contributes ~25-35% of the total effect.

**Probability assessment:**
- P(b) < 0.18, making (e)−(d) > 8pp: ~30% (strong novel ingredient)
- P(b) ∈ [0.18, 0.25], making (e)−(d) ∈ [3pp, 8pp]: ~45% (solid but modest novel ingredient)
- P(b) > 0.25, making (e)−(d) < 3pp: ~25% (marginal or null novel ingredient)

The scenario where (b) ≈ (e) (augmentation alone explains everything) requires augmentation to deliver ~5× lift at Qwen3-4B in multi-round continual LoRA without any preservation mechanism. The LoRA packing paper's collapse at 3000 facts + heavy augmentation argues against this. My best guess is ~25% probability that augmentation alone gets within 3pp of DSAE Lite.

---

## Q3: What Does "Substantive" Mean? Expected Outcome Assessment

Three benchmarks, stated plainly:

**(a) Substantive contribution (e)−(d) ≥ 4pp at 2-seed, consistent across rounds.** The novel K=5 KL preservation is the active ingredient beyond augmentation. The paper's claim — "format-diverse preservation is the missing half" — is empirically supported. This justifies Option A (positive-result paper) and the current title. **Probability: ~55%.**

**(b) Marginal contribution (e)−(d) ∈ [1pp, 4pp].** Augmentation does 70-85% of the work; K=5 KL preservation adds a consistent but small margin. The paper becomes: "Allen-Zhu's augmentation transfers to continual LoRA at 4B scale (replication); format-diverse KL preservation adds a modest further improvement." The novel ingredient is real but not the headline — the replication of Allen-Zhu in a new setting becomes the main finding. **Probability: ~25%.**

**(c) Collapses (e)−(d) < 1pp.** Augmentation alone explains the entire effect. The paper's novel ingredient does nothing. Paper becomes: "Allen-Zhu replicates in continual LoRA; we tried a symmetric KL construction and it didn't help." This is an honest negative result for the K=5 KL preservation but not a novel positive method. **Probability: ~20%.**

**The most likely outcome is (a) or strong-(b).** My central estimate is (e)−(d) ≈ 5-7pp, based on:
1. The LoRA packing paper's evidence that unregulated augmentation degrades at multi-round scale
2. The structural argument that single-format KL cannot detect format-selective forgetting (the motivation for K=5 KL is sound, not just a trick)
3. The per-format KL diagnostic already showing differential drift across framings — the mechanism the K=5 KL preservation targets is empirically active

---

## Q4: Is the Preservation Claim Defensible?

**"Preservation maintained" — yes. "Preservation improved" — no.**

The pilot preservation data: dsae_lite Recall@10 is +0.009 to +0.060 above naive_sft across 5 rounds, mean +0.036. At n=104 items with SE≈0.032, the mean delta is z=1.12 — barely 1 SE. The 95% CI is [−0.027, +0.099].

This supports "comparable to or slightly above" but not "significantly improved." With 2 seeds, the CI narrows by √2 to SE≈0.023, giving CI [−0.009, +0.081] — still straddling zero. The honest claim is: **"DSAE Lite's K=5 KL preservation does not sacrifice task ability; preservation is maintained within noise of the unregulated baseline."** This is the right claim. For a continual injection method, "absorption goes up 5× and preservation doesn't degrade" is the useful finding.

Do not claim preservation *improves*. The data cannot support it, and it doesn't need to — "maintained" is the goal.

---

## Q5: What's the Paper If Plan B Returns Scenario (b)?

**Even at (e)−(d) ≈ 2pp (marginal), the paper has a real contribution. Here's why.**

The paper's contribution stack, independent of (e)−(d):

1. **COPR negative transfer to fact injection** — no precedent. COPR (Zhang et al. 2025) was proposed for continual preference alignment; showing it fails for knowledge injection due to K-sample-all-wrong pathology is a novel negative result. This is well-grounded (mechanistic explanation, 9-3 head-to-head, gold-injection convergence analysis).

2. **V-REx K=2 degeneracy** — the theoretical case (Arjovsky 2019, Rosenfeld 2021) is established, but nobody has tested V-REx at K=2 on prompt formats for knowledge injection. The empirical null (z=0.26) confirms the theory in a new setting. The leak-confound disclosure is a credibility signal.

3. **Format diversity without regularization hurts** — replicates PIT/UIT in continual LoRA injection. Their finding (−6pp EM from naive format mixing) parallels our result (format gap 0.100 vs 0.072). This is a useful extension to a new setting.

4. **Geometric-behavioral disagreement** — gold-injection COPR has the smallest geometric shift ratio but the *largest* behavioral format gap. This warns against hidden-state cosine proxies for knowledge integration, which are increasingly used uncritically in the editing literature. This observation stands regardless of DSAE Lite's performance.

5. **Allen-Zhu K=5 augmentation transfers to continual LoRA at 4B scale** — if (b) shows ~0.22 absorption vs (a) 0.064, that's a 3.4× lift, confirming Allen-Zhu's mechanism in a new regime. Nobody has shown this for multi-round sequential injection with LoRA on a task-tuned model.

6. **DSAE Lite as a concept** — even at (e)−(d) ≈ 2pp, the symmetric construction is novel (confirmed by exhaustive search). The K=5 KL preservation has no precedent. A 2pp marginal improvement with a clean mechanism explanation is a defensible positive result, just not the headline.

**Is this enough for a class research paper?** Yes, clearly. Items 1-5 each have standalone value. The negative results (COPR, V-REx, format-mixing) constrain the design space for other researchers. The Allen-Zhu replication extends a foundational result. The geometric-behavioral disagreement is a methodological contribution to the editing subfield. DSAE Lite's marginal contribution is the cherry on top, not the cake.

**What changes in the paper:** If (e)−(d) ≈ 2pp, reframe from "DSAE Lite's novel ingredient closes the gap" to "K=5 augmentation is the primary driver (Allen-Zhu replication); format-diverse KL preservation adds a modest further improvement." The paper's spine becomes diagnostic + negative results + augmentation-transfer finding, with DSAE Lite as one positive signal among several contributions. This is still a good paper — just different emphasis.

---

## Q6: Final Verdict

**One-sentence answer:** The +24pp (a)-vs-(e) gap is real and not a seed artifact, but it is the wrong contrast for the paper's novel contribution — the paper's substance depends on (e)−(d), which I estimate at ~5-7pp (55% probability of ≥4pp), making the most likely outcome a publishable positive result with the novel K=5 KL preservation as a genuine but secondary active ingredient alongside the dominant Allen-Zhu augmentation effect.

**Decision rule for Plan B:**

| Plan B outcome | (e)−(d) at round 15, 2-seed mean | Interpretation | Paper framing |
|---|---|---|---|
| **Win** | ≥ 4pp, consistent across rounds | Novel K=5 KL preservation is active ingredient | Option A: DSAE Lite as headline positive method |
| **Modest win** | 2-4pp, present but noisy | KL preservation adds real but small value | Hybrid: Allen-Zhu replication is the big finding; DSAE Lite's KL preservation contributes modestly |
| **Draw** | < 2pp | Augmentation alone explains the effect | Option B: diagnostic + negative results + Allen-Zhu replication paper |

**All three outcomes produce a defensible paper.** The negative results, geometric-behavioral disagreement, and Allen-Zhu replication in a new regime have standalone value regardless.

**What to do now:** Let Plan B run. It costs ~21 of 24 remaining GPU-hours. The 4-condition ablation (a, b, d, e) at 2 seeds × 15 rounds will resolve the uncertainty. Don't add conditions or change the design — the ablation as locked in Iteration 3 is exactly the right experiment for this question.

---

*Literature citations verified against paper text:*
- *Allen-Zhu 2309.14316 §4.2: K=1 bioS_single → 9.7% OOD QA; K=5 multi5 → 96.6%. The 0%→70-87% range often cited refers to the mixed-training regime (§3), not pretrain-then-finetune.*
- *Masked FT 2510.09885 Table 2: Qwen3-4B no-paraphrase 0.509/0.353 (fwd/bwd) → permute-order 0.693/0.639 (+18pp/+29pp).*
- *PIT 2402.12847 Table 1: Llama-2-7B standard IT 30.3% EM → PIT 45.4% EM (+15.1pp).*
- *LoRA packing 2502.14502 Table 2: 3000 facts + 10 paraphrases → reliability collapses to 0.48.*
- *Smith/Tian et al. 2203.17269 Table 2: PredKD 25.2% / 3.2% forgetting vs EWC 7.7% / 64.0% forgetting (CIFAR-100, 10-task, vision — NOT LLM; the draft cites this correctly as Kiyasseh et al. but the actual first author is Smith).*
- *Nayak 2504.07097 Table 1: Adaptive SVD 75.9% vs LwF 52.3% on T5-Large 5-task (+23.6pp).*

*Note: The draft's §5.7 cites "Kiyasseh et al. 2022 (arXiv:2203.17269)" — the actual paper is Smith, Tian, Halbe et al. 2022. Kiyasseh is not the first author. Verify and correct this attribution before submission.*
