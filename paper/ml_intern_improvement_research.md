# Improvement Research: Cross-Format Absorption in Continual Knowledge Injection

**Date:** 2026-04-25
**Context:** Phase 9 leak-free retrain showed FI-SFT (V-REx on K=2 prompt formats) produces +0.014 absolute QD F1 over baseline — within noise at n=50. Format gap 0.068 vs baseline 0.072. The mean-variance trade-off mechanism story did not pan out. This document researches what could genuinely work.

---

## Q1: Why V-REx Is Weak at K=2

The short answer: K=2 is theoretically expected to be insufficient, and the literature is unambiguous about this.

**The degeneration.** At K=2, V-REx's variance penalty reduces to `(CE_qa - CE_qd)² / 4` — zero whenever the two per-format losses are equal. A model satisfies this by finding *any* representation that equalizes the two losses, including format-agnostic shortcuts that don't require genuine invariance. The penalty cannot distinguish a truly format-invariant predictor from one that is equally mediocre on both formats.

**IRM theory requires K to scale.** Arjovsky et al. (2019, arXiv:1907.02893) Theorem 9 requires `|E_tr| > d - r + d/r` environments. K=2 satisfies this only when non-invariant solutions are already excluded by construction. Their §4.2 concedes "two environments are sufficient" only when "E[Y|Φ(X)] cannot match for two different environments unless Φ extracts the causal invariance" — a condition our QA/QD templates almost certainly violate.

**Rosenfeld et al. (2021, arXiv:2010.05908)** prove IRM's penalty can be made arbitrarily small for non-invariant representations when `|E_tr| ≤ d_e` (spurious feature dimension). Our formats differ in system message, phrasing, output structure, and length — plausibly d_e > 1, making K=2 provably insufficient. Abroshan et al. (2021, arXiv:2106.09777) formalize this: when K=2 environments share similar statistical structure, the Gram matrix is nearly rank-deficient. Ahuja et al. (2021, arXiv:2106.06607) prove K = Ω(2^d_e) in the nonlinear worst case.

**DomainBed (Gulrajani & Lopez-Paz, 2021, arXiv:2007.01434) never tests K=2.** Minimum is K=3. Even at K=3–6, IRM/V-REx don't reliably beat ERM under realistic model selection.

**Bottom line:** K=2 V-REx is a pairwise consistency loss, not an invariance constraint. The +0.014 we observed is exactly what the theory predicts: near-zero signal from a degenerate penalty. This is not a tuning problem. It is a structural limitation.

---

## Q2: What Would Actually Close Cross-Format Gaps

The literature points to data augmentation, not loss engineering, as the high-leverage intervention. Effect sizes are stark.

### Allen-Zhu & Li (2024, arXiv:2309.14316) — The mechanistic answer

Their controlled experiments on synthetic biographies show the core result: knowledge trained in one surface form is stored as adjacent-token associations, not entity→attribute bindings. Without augmentation, extraction accuracy is **0%** regardless of subsequent instruction-tuning. With 5 paraphrase variants + sentence permutation ("multi5+permute"), extraction jumps to **70–87%**. The augmentation forces re-encoding as entity-anchored knowledge. Critically, §4.2 notes that "even with LoRA fine-tuning... test accuracy only slightly improves" when augmentation is absent. The fix must be in the data.

Their recipe: M=5 distinct text renderings per fact using ~50 sentence templates per attribute type, plus random permutation of attribute order across renderings, plus full-name repetition replacing pronouns. This is the strongest demonstrated effect in the literature.

### PIT (Jiang et al., 2024, arXiv:2402.12847) — Curriculum matters

Pre-Instruction Tuning achieves +17.8% absolute EM on Llama-2-7B (30.3% → 48.1%) via a two-phase curriculum: Phase 1 trains on QA pairs (3 epochs, lr=5e-6), Phase 2 jointly trains QA + source documents (10 epochs, lr=3e-5). The insight: QA exposure *first* establishes attention patterns for fact retrieval, so subsequent document encoding stores facts in query-addressable form. The "perplexity curse" they identify — minimizing document loss does not mean extractable knowledge — maps directly to our absorption-integration gap.

### PAFT (Wei et al., 2025, arXiv:2502.12859) and SKI (Ji et al., 2024, arXiv:2410.09629)

PAFT achieves large prompt-robustness gains (GSM8K adversarial +63.87pp) via rotating 400 diverse prompts during training, but addresses prompt-phrasing robustness, not factual storage→extraction. Composable with augmentation but not the primary lever. SKI is the closest domain match: on FiQA (finance), 1-gram QA decomposition lifts F1 from 0.123 to 0.357 (+19.3pp). For FNSPID triples, atomic QA decomposition is the most directly validated recipe.

### Why loss-level consistency fails; largest reported effects

Seleznyov et al. (2025, arXiv:2508.11383) tested LoRA-JS across 8 models/52 tasks: "higher spread and lower accuracy for all models" vs plain augmentation. JS divergence penalizes legitimate format-specific differences. V-REx avoids this specific failure but inherits K=2 degeneration. The lesson: output-distribution consistency losses at K=2 are structurally too weak.

Largest reported cross-format effects: Allen-Zhu's 0% → 70%+ (controlled pretraining), PIT's +17.8% EM (7B full FT). No paper reports controlled effects at exactly 4B/LoRA/200 facts.

---

## Q3: Smallest Viable Next Experiment

**Recommendation: 5-seed replication of kl_reg_sft at n=200, plus a K=5 augmentation variant.**

Rationale: the single highest-leverage change is *statistical power*, not a new method. At n=50/single seed, the Phase 9 CI on QD F1 delta is [−0.09, +0.12] — we literally cannot tell if the effect is positive or negative.

**Protocol:**
1. Run kl_reg_sft × 5 seeds × 15 rounds × n=200 (~3.75 GPU-hrs). This gives us the baseline distribution.
2. Run kl_reg_sft with K=5 augmented formats × 5 seeds × 15 rounds × n=200 (~3.75 GPU-hrs if augmentation is data-level, not loss-level). The 5 formats: direct QA, declarative statement, cloze completion, reversed cloze, instruction-style — following Allen-Zhu's recipe. No variance penalty. Pure augmentation.
3. Evaluate all 10 runs on the n=50 behavioral probe (or better: expand probe to n=200+).
4. Compare seed-level means with proper CIs.

**Cost:** ~7.5 GPU-hrs. **This is the minimum experiment that gives real CIs on the baseline AND tests the literature's primary recommendation (data augmentation > loss engineering).**

If augmentation at K=5 shows a consistent +3pp QD F1 shift across 5 seeds, that's your paper's improvement result. If it doesn't, you've killed the augmentation hypothesis at this scale — also publishable.

---

## Q4: Scaling Axes

At ~0.05 GPU-hrs/round for kl_reg_sft: 5 seeds × 15 rounds = 3.75 GPU-hrs (n=200), 9.38 GPU-hrs (n=500). Full 2×2 design (n∈{200,500} × method∈{kl_reg, augmented}, 5 seeds) ≈ 26 GPU-hrs — **feasible within 50 GPU-hours with headroom.**

With σ_seed ≈ 0.02 (LoRA is more stable than full FT; Dodge et al. 2020, arXiv:2002.06305), 5 seeds gives CI half-width ±2.5pp. Detects +3pp effects, not +2pp. Run n=200 at 5 seeds first; scale to n=500 only if seed variance σ > 0.025.

---

## Q5: Is K=2 with Templates Fundamentally Inadequate?

**Yes, for V-REx's intended mechanism.** The theory is clear (§Q1): K=2 creates a degenerate penalty landscape. But the question has a deeper layer.

**Would K=8 fix it?** Probably not *via V-REx specifically*. DomainBed shows IRM/V-REx don't beat ERM even at K=4–6. The deeper issue: prompt formats aren't "environments" in V-REx's sense. V-REx environments must share an invariant causal mechanism with varying spurious correlations. QA and QD differ in *what the model is asked to do* — genuinely different tasks, not spurious variations of one task.

**The better framing: formats are data augmentations, not environments.** Allen-Zhu & Li achieve 0% → 70%+ purely through diversity, no penalty. The gradient signal from diverse formats *directly trains* diverse retrieval pathways — a qualitatively different mechanism from environment-invariance.

**Alternative objectives if a penalty is still desired:**
- **Group DRO (Sagawa et al., 2020, arXiv:1911.08731):** Minimize worst-group loss across formats. Consistently beats IRM/V-REx when group labels are available (DomainBed evidence). LoRA's frozen backbone provides implicit regularization — natural fit. Negligible cost over augmented SFT.
- **Representation-level consistency (Kim et al., 2024, arXiv:2511.13052):** MSE on hidden states (not output distributions like LoRA-JS); reported 92.1% lower prompt-variant std. Overhead: ~1.5× forward passes.
- **PIT-style curriculum:** No penalty — curriculum alone achieves +17.8% EM by structuring *when* formats appear.

**The honest assessment:** V-REx on prompt formats was a reasonable hypothesis, but the K=2 case was always degenerate and the environment-invariance framing doesn't naturally fit prompt formats. The literature's answer is: use diverse augmentation (Allen-Zhu), possibly with curriculum structure (PIT), possibly with worst-group upweighting (Group DRO), but don't try to make an OOD-generalization penalty do the work of data diversity.

---

## Recommended Path Forward

**Priority 1 (7.5 GPU-hrs):** 5-seed baseline + K=5 augmentation comparison at n=200. This either validates augmentation as the real lever or kills it.

**Priority 2 (if augmentation works, ~10 GPU-hrs):** Group DRO on the K=5 augmented data — does upweighting the hardest format help beyond pure augmentation? This is the minimal-viable test of whether *any* loss modification helps on top of data diversity.

**Priority 3 (if everything is noise, ~5 GPU-hrs):** PIT-style curriculum (QA-first, then mixed) at K=5. Tests whether sequencing matters for LoRA injection specifically.

**Dead ends to avoid:** Further V-REx tuning (μ sweeps, etc.) — the K=2 degeneration is structural, not parametric. LoRA-JS or output-distribution KL — empirically negative in the literature. Any single-seed experiment on n=50 — statistically uninterpretable.

**What the paper becomes:** If augmentation at K=5 works, the paper gains a positive method result grounded in Allen-Zhu's theory. If it doesn't, the paper is an honest diagnostic contribution: "at 4B/LoRA, neither V-REx loss engineering nor Allen-Zhu-style augmentation closes the format gap at n=200 facts/round — the gap may be a fundamental scale limitation."
