# Collapse Verdict: DSAE Lite Fails, Synergy Emerges

**Date:** 2026-04-27  
**Status:** Post-Plan-B assessment. Full data: 4 conditions × 2 seeds × 15 rounds (a/b/d/e); condition (c) from prior phase.

---

## The Results in One Table

| ID | Condition | Injection | Preservation | R15 mean F1 | Half-spread |
|---|---|---|---|---|---|
| (a) | `naive_sft` | K=1 | none | 0.089 | 0.001 |
| (b) | `aug_sft_k5` | K=5 | none | 0.125 | 0.002 |
| (c) | `kl_reg_sft` | K=1 | KL on K=1 | 0.118 | (1 seed) |
| (d) | `aug_kl_k1` | K=5 | KL on K=1 | 0.411 | 0.006 |
| (e) | `dsae_lite` | K=5 | KL on K=5 | 0.405 | 0.018 |

**The novel ingredient (K=5 KL preservation) does exactly nothing.** (e) − (d) = −0.006 with higher variance. This is the iter-9 "collapse" scenario, assigned 20% probability. It happened.

**But the 2×2 interaction is extraordinary.** (b) alone: +0.036. (c) alone: +0.029. (d) = (b)+(c) combined: +0.322. The additive prediction is +0.065; the observed value is 5× that. K=5 injection and KL preservation are not independent treatments — they interact super-linearly.

---

## Q1: Is the Synergy Novel and Publishable?

**Yes. It is novel. No prior work has characterized this interaction.**

I ran an exhaustive literature search across the seven most relevant paper families. Summary:

| Paper | Tested augmentation? | Tested KL preservation? | Tested both together? | Quantified interaction? |
|---|---|---|---|---|
| Allen-Zhu 2309.14316 | ✅ (K=1→5) | ❌ | ❌ | ❌ |
| Masked FT 2510.09885 | ✅ (permute) | ❌ | ❌ | ❌ |
| PIT 2402.12847 | ✅ (curriculum) | ❌ | ❌ | ❌ |
| LoRA Packing 2502.14502 | ✅ (paraphrases) | ❌ | ❌ | ❌ |
| SEFE/ASD 2505.02486 | ✅ (format div.) | ~(weight reg, not KL) | ✅ | ❌ (additive framing) |
| GeRe 2508.04676 | ❌ (generic replay) | ✅ (KL vs alternatives) | ✅ | ❌ (subadditive) |
| KORE 2510.19316 | ✅ (multimodal) | ~(null-space, not KL) | ✅ | ❌ |
| RECAP 2510.21978 | ❌ | ~(KL discussed) | ❌ | ❌ |

**No paper runs the 2×2 ablation** — {no-aug, K=5 aug} × {no-KL, KL} — in text-only LoRA knowledge injection and reports the interaction magnitude. The two closest comparisons are:

- **SEFE** combines format diversity (ASD) with parameter-level regularization (RegLoRA, not KL). They report additive gains, not super-linear. Different mechanism, different setting (continual VQA instruction tuning).
- **GeRe** tests replay + KL distillation on generic SlimPajama data. Their result is *subadditive* — KL adds marginal value over vanilla replay. Making our super-linear finding a genuine contrast.

Allen-Zhu's own work never tests augmentation with any preservation regularizer. The Masked FT paper frames masked objectives as a *substitute* for paraphrasing, not a complement. LoRA Packing shows augmentation *degrades* at scale without preservation but never tests the fix. Nobody has closed the loop.

**The claim that survives peer review:** "We are the first to document a super-linear interaction between format-diverse data augmentation (Allen-Zhu K=5) and KL preservation regularization in continual LoRA knowledge injection. Each ingredient individually yields +3-4pp absorption F1 over naive SFT; together they yield +32pp — approximately 5× the additive prediction."

**Confidence level:** High on the effect's existence (2 seeds, 0.006 half-spread on (d), both seeds show the same pattern, n=200 per round). Moderate on the exact magnitude (could be 4× or 6× with more seeds). Low on the mechanistic explanation (we know *that* they interact, not *why*).

---

## Q2: How to Reframe the Paper

**Option C (Hybrid) is the correct framing. Here's why the other two are wrong.**

### Why not Option A (pure synergy paper)

Option A — "K=5 injection × KL preservation produces super-linear synergy" — oversells the depth of understanding. We have one model (Qwen3-4B), one LoRA rank (r=16), one dataset domain (FNSPID financial facts), 2 seeds. We documented a striking interaction effect but have no mechanistic explanation. A paper that leads with "super-linear synergy" as if it's a general law will invite the obvious reviewer question: "Is this just an artifact of your specific setup?" We cannot answer that yet.

### Why not Option B (negative-results paper)

Option B — "We test a wide method space; nothing novel works" — buries the strongest finding. The synergy is the most informative result in the entire paper. A negative-results framing would mention it in passing ("incidentally, the combination works well") while spending the headline on "DSAE Lite doesn't help." Reviewers will rightly ask why the paper emphasizes the null increment of one ingredient over the 5× super-linear interaction of the other two.

### Option C: What the paper actually says

The paper has three tiers of contribution, cleanly ordered by evidentiary strength:

**Tier 1 (strong, novel):** The 2×2 ablation reveals super-linear synergy between K=5 injection and KL preservation. Each alone gives +3-4pp; together +32pp. This is a first-of-its-kind factorial ablation in continual LoRA knowledge injection. Practical recommendation: combine Allen-Zhu K=5 augmentation with standard K=1 KL preservation (`aug_kl_k1`).

**Tier 2 (strong, negative):** Three families of methods fail to close the absorption-integration gap: COPR (preference alignment, negative transfer via K-sample-all-wrong pathology), V-REx at K=2 (theoretically degenerate, empirically null), format diversity without regularization (actively harmful, consistent with PIT/UIT). Plus the geometric-behavioral disagreement finding.

**Tier 3 (honest null):** Our proposed novel ingredient — K=5 augmented KL preservation (DSAE Lite) — does not improve over standard K=1 KL preservation when K=5 injection is already present. (e) ≈ (d) within noise. The extra preservation-side diversity adds compute (~3× more KL evaluations) without measurable benefit. The synergy is gated on the *injection* side, not the *preservation* side.

**Proposed title:** "Super-Linear Synergy Between Format Augmentation and KL Preservation in Continual LoRA Knowledge Injection"

**Proposed abstract structure:** Gap exists → three method families fail → 2×2 ablation reveals super-linear synergy → practical method (`aug_kl_k1`) → novel extension (K=5 KL preservation) adds nothing further → contributions are the synergy finding, the negative results, and the factorial ablation methodology.

**What changes from the current draft:**

- §3.3 (DSAE Lite as proposed method) → demoted to §5.4 tested extension, 1 paragraph
- §5.4 headline becomes the 2×2 interaction table, not the (e) vs (d) contrast
- §6.1 (symmetric K=5 mechanism explanation) → rewritten: the mechanism that matters is injection-side diversity being *anchored* by KL preservation; preservation-side diversity is redundant
- Abstract: rewrite around synergy, not DSAE Lite
- Conclusion: practical recommendation is `aug_kl_k1`, not `dsae_lite`
- Title: changed from "Format-Diverse Preservation Closes…" to synergy framing

The negative results (§5.1–5.3) survive unchanged. The compositional degradation finding (§6.2) survives. The geometric-behavioral disagreement survives. The leak confound disclosure survives. These are all independent of the headline.

---

## Q3: Future Research Directions — Top 3

From the seven candidates listed, ranked by expected information value per GPU-hour:

### 1. Mechanism: Why is the synergy super-linear? (HIGHEST PRIORITY)

This is the obvious reviewer question and the most valuable follow-up. Two testable hypotheses:

**Hypothesis A (Feature-space enrichment):** K=5 injection forces entity-anchored, format-invariant internal representations (Allen-Zhu's mechanism). Without KL preservation, these representations drift during subsequent rounds — they're created but not maintained. KL preservation anchors the model's output distribution, which *indirectly* anchors the internal representations that generate those outputs. Neither ingredient alone changes the representation space enough: K=5 alone creates good representations but lets them drift; KL alone prevents drift but on representations that were format-specific to begin with. The interaction is super-linear because you need both the *creation* and the *maintenance* of format-invariant features.

**Hypothesis B (Gradient interference cancellation):** K=5 injection produces gradient updates that pull in multiple format-specific directions simultaneously. Without an anchor, these gradients partially cancel or produce a muddled average. KL preservation provides a consistent "return vector" that prevents the multi-directional gradients from destructively interfering. The synergy is super-linear because the preservation term doesn't just add value — it changes the *effective direction* of the injection gradient by suppressing the destructive components.

**Test:** Linear probing at entity-token hidden states across the 4 conditions. If Hypothesis A is correct, (d) should show format-invariant linear features that (b) creates-then-loses and (c) never creates. This costs ~1-2 GPU-hours and is the single highest-value experiment for a revision or follow-up.

### 2. K-ablation on injection side: K=3, K=5, K=10 (with KL fixed)

We tested exactly one point on the injection-diversity axis (K=5). Allen-Zhu shows monotonic improvement with K in pre-training; LoRA Packing shows degradation at K=10 without preservation. The interaction with KL preservation may change the optimal K. Specifically:

- Does K=3 + KL already capture most of the synergy? (Would reduce compute ~40%)
- Does K=10 + KL recover from the LoRA Packing degradation? (Would be a direct test of whether KL preservation *fixes* the scaling failure)

This is a clean 3-condition experiment (K=3/5/10, all with KL K=1) at ~8-10 GPU-hours. Second priority because it extends the finding along a clean axis with direct practical implications.

### 3. Scale: Does the synergy hold at 7-8B? (HIGHEST IMPACT, HIGHEST COST)

The synergy at 4B could be a capacity phenomenon — LoRA r=16 on a 4B model may be too constrained to store format-invariant representations without both ingredients, while at 7B+ there's enough capacity that K=5 alone suffices. The Burns/Marks & Tegmark argument (truth directions become more linearly separable at larger scale) suggests the synergy magnitude should *shrink* at larger scale. Testing whether it's still present — even at reduced magnitude — at 7-8B would dramatically increase the finding's importance.

Cost: ~15-20 GPU-hours for 2 conditions (b, d) × 1 seed × 15 rounds at 7B. This exceeds current budget but is the clear next-grant experiment.

**Deprioritized follow-ups:**
- *Different LoRA ranks (r=8/32/64):* Useful but lower information density than K-ablation. The rank axis is orthogonal to the synergy mechanism.
- *Other preservation regularizers (EWC, R-Drop):* Interesting mechanistically — does the synergy require *output-distribution* preservation (KL) or any anchor (EWC)? — but ~12 GPU-hours for a clean comparison and not immediately publishable.
- *Compositional/multi-hop:* The compositional gap is real but orthogonal to the synergy finding. This is a different paper.
- *Streaming ingestion:* Production-relevant but requires substantial engineering to test properly.

---

## Q4: Should We Re-Run Anything?

**No. Do not spend the 5 GPU-hours on a 3rd seed.**

The decision breaks down to: what claim does the 3rd seed support, and is the evidence already sufficient without it?

**The null claim — (e) ≈ (d) — is already well-supported.** The observed difference is −0.006 with (d) half-spread 0.006 and (e) half-spread 0.018. The (e) distribution clearly encompasses (d). At 2 seeds, this is thin for a classical hypothesis test, but we don't need to *prove* (e) = (d) beyond doubt. We need to show that (e) does not meaningfully exceed (d), which the data already show: even in the most favorable reading (take the upper bound of (e) at 0.423 vs the lower bound of (d) at 0.405), the gap is +0.018 — trivial compared to the +0.322 synergy effect. The paper's claim is not "(e) = (d) with p < 0.05"; the paper's claim is "(e) does not extend the synergy beyond (d), and the synergy itself is the finding." The evidence supports this.

**The synergy claim — the headline — is rock-solid at 2 seeds.** (d) at 0.411 ± 0.006 vs (a) at 0.089 ± 0.001 is z > 50 at n=200. This does not need more seeds.

**The 3rd seed is most valuable on (b) and (d)**, not on (e). If anything, a skeptical reviewer would want to verify that (b) really is only 0.125 (confirming that K=5 injection alone is weak). But at half-spread 0.002, (b) is already tight.

**Spend the 5 GPU-hours on the linear probing experiment (Q3 #1) instead.** One probing experiment that explains *why* the synergy is super-linear adds more publication value than a 3rd seed confirming what 2 seeds already show. If you must spend compute, probe the hidden states of conditions (a), (b), (c), (d) at entity tokens and measure format-invariance of linear features. This is the experiment that turns the paper from "we found an interesting interaction" into "we found an interesting interaction and here's the mechanism."

---

## Updated Paper Skeleton

For reference — here's how the reframed paper reads:

**Title:** "Super-Linear Synergy Between Format Augmentation and KL Preservation in Continual LoRA Knowledge Injection"

| § | Content | Key claim |
|---|---------|-----------|
| 1 | Introduction | Absorption-integration gap exists. Three method families fail. 2×2 ablation reveals super-linear synergy. |
| 2 | Related Work | Allen-Zhu augmentation, KL preservation in CL, the gap between them — nobody has tested the interaction. |
| 3 | Methods | 3.1 Problem. 3.2 Baselines (naive, KL-reg, COPR, V-REx). 3.3 Augmented methods (aug_sft_k5, aug_kl_k1, dsae_lite). 3.4 The 2×2 factorial design. |
| 4 | Setup | Qwen3-4B, FNSPID, 15 rounds, 200 facts, 2 seeds, metrics. |
| 5 | Results | 5.1 Gap is real (Table 1). 5.2 COPR negative transfer. 5.3 V-REx K=2 + format diversity without regularization. **5.4 The synergy: 2×2 table, interaction plot, trajectory comparison.** 5.5 DSAE Lite null: K=5 KL preservation adds nothing over K=1. 5.6 Compositional degradation persists. |
| 6 | Discussion | Why the synergy is super-linear (hypotheses A and B). Geometric-behavioral disagreement. Practical recommendation: `aug_kl_k1`. |
| 7 | Limitations | Single model, 2 seeds, one domain, mechanism unexplained, K only at 5. |
| 8 | Conclusion | Use K=5 injection + standard KL preservation. The symmetric K=5 extension is unnecessary. Three method families don't work. |

The DSAE Lite null result moves from §5.4 (current headline) to §5.5 (one paragraph: "we tested whether extending K=5 to the preservation side helps; it does not"). The 2×2 interaction table becomes the centerpiece of §5.4.

---

## Summary of Verdicts

| Question | Answer |
|---|---|
| Q1: Is the synergy novel? | **Yes.** No prior work runs the 2×2 augmentation × KL-preservation ablation in LoRA knowledge injection. The ~5× super-linear interaction is unreported. |
| Q2: Which framing? | **Option C (Hybrid).** Lead with synergy as the empirical contribution. Report DSAE Lite as a tested null. Keep all negative results. |
| Q3: Top 3 follow-ups? | 1. Linear probing for mechanism (1-2 GPU-h). 2. K-ablation on injection side (8-10 GPU-h). 3. Scale to 7B (15-20 GPU-h). |
| Q4: Re-run anything? | **No.** Spend the 5 GPU-h on linear probing, not a 3rd seed. The (e) ≈ (d) null and the (d) ≫ (a)+(b) synergy are both clear at 2 seeds. |

---

*The iter-9 prediction assigned 20% to collapse. It collapsed. But the prediction framework was right to flag the synergy as the load-bearing result — the `(d) − (a)` gap was always the paper's strongest number. The mistake was framing the paper around the marginal (e) − (d) increment rather than the foundational (d) − (a) interaction. That's fixed now.*
