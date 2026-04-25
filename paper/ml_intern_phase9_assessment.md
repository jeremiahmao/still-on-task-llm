# Phase 9 Assessment: What the Leak-Free Retrain Actually Shows

**Date:** 2026-04-25  
**Source data:** `final_results/phase9_leakfree_isolation.csv`, `phase8c_chained_comparison.csv`, `phase8d_variance_isolation.csv`  
**Template code:** `scripts/24_prepare_mixed_format_triples.py`  
**All numbers cited verbatim from CSVs. n=50 facts, single seed, 15 chained rounds.**

---

## (1) What did the leak contribute?

The leaky QD template (`_qd_assistant_target_leaky`) embedded the gold answer directly in Sub-query 1: `"What is {answer}'s role related to {subject}?"`. The leak-free template (`_qd_assistant_target_leakfree`) replaces this with `"What is the {relation} of {subject}?"` — the answer string never appears in any user or assistant turn.

Comparing fi_sft_leaky vs fi_sft_leakfree:

| Metric | Leaky | Leak-free | Δ |
|--------|-------|-----------|---|
| QA F1 | 0.129 | 0.153 | +0.024 |
| QD F1 | 0.092 | 0.086 | −0.006 |
| Format gap | 0.037 | 0.068 | +0.031 |

The leak contributed almost exactly to the headline format-gap claim and nothing else. The leaky template inflated QD F1 by ~0.006 (the model was partly parroting answer tokens it saw during QD training) and simultaneously *suppressed* QA F1 by 0.024 (possibly because the variance penalty was pulling the model toward a QD-format training signal that contained the answer for free, creating a shortcut that competed with genuine QA absorption). Net effect: the leak manufactured a 0.037 format gap that looked like "gap closing" but was actually an artifact of the QD template handing the answer to the model.

Once the leak is removed, the format gap (0.068) snaps back to within 0.004 of the KL-reg baseline (0.072). **The "gap halving" result from Phase 8 was almost entirely a confound.**

---

## (2) Which earlier claims survive?

### (a) Preserve as-is

- **COPR negative transfer (Claim 5 in honesty audit).** Unaffected by the leak; these experiments used single-format QA training. The COPR family's failure on knowledge editing, the K-sample-all-wrong pathology, the anchored-without-gold collapse — all survive intact.
- **Absorption-integration gap diagnosis.** The gap itself is real: all methods show QA F1 >> QD F1. The kl_reg baseline has a 0.072 format gap; no tested method eliminates it. This finding stands regardless of the FI-SFT results.
- **Geometric/behavioral disagreement (Phase 7).** Gold-injection variants had smallest shift ratio but largest behavioral gap. This observation is independent of the FI-SFT leak.
- **"Format diversity without regularization hurts" (Phase 8d).** Mixed-format SFT without variance penalty: QD F1 = 0.067, format gap = 0.100 — strictly worse than single-format kl_reg (QD F1 = 0.072, gap = 0.072). This survives; it never depended on the QD template.

### (b) Downgrade in magnitude

- **"FI-SFT narrows the format gap by roughly half."** The Phase 8 leaky result showed gap 0.037 vs baseline 0.072 (−48%). The leak-free result shows gap 0.068 vs baseline 0.072 (−6%). This is not "roughly half." It is within noise of zero improvement on format gap. Downgrade from a headline result to "no detectable format-gap reduction."
- **"+29% QD absorption."** Leaky: +28.8% (0.092 vs 0.072). Leak-free: +19.9% (0.086 vs 0.072). The QD improvement shrinks and is now a +0.014 absolute delta on n=50 — statistically indistinguishable from zero (z=0.26, see §4 below).

### (c) Drop entirely

- **Any claim that V-REx "closes" or "halves" the format gap.** The leak-free format gap (0.068) is 94% of the baseline gap (0.072). The variance penalty does not close the format gap. It does not meaningfully narrow it.
- **The "mean-variance trade-off" mechanism explanation.** The original Phase 8 story was: V-REx penalizes cross-format loss variance, forcing the model to equalize QA and QD absorption, thereby closing the gap. The leak-free data shows the gap is not closed. If the mechanism were working as described, the gap should have shrunk. It didn't. The variance penalty may be doing something, but it is not producing format-equalized absorption.

### (d) Reframe (different mechanism)

- **FI-SFT provides a uniform absorption lift, not selective gap-closing.** Leak-free FI-SFT: QA +7.0% (0.153 vs 0.143), QD +19.9% (0.086 vs 0.072). Both go up. The gap barely moves (0.068 vs 0.072). This looks like mixed-format training with a variance penalty gives the model slightly more total information about each fact, lifting both formats roughly in proportion. It does *not* selectively boost the weaker format. The mechanism is "more training signal" not "format equalization."

---

## (3) Is the user's proposed reframe accurate?

The proposed reframe: *"V-REx provides a +20% improvement on out-of-format (QD) absorption, +7% on in-format (QA), without closing the format gap. The mechanism is uniform absorption lift, not mean-variance trade-off."*

This is **directionally accurate but overstates the confidence**. The numbers are correct from the CSV. The "uniform lift" reading is the most parsimonious explanation of the data pattern. But it frames +20% and +7% as established facts, when the statistical analysis (§4) shows these deltas are well within noise for n=50. A more honest version:

*"Under leak-free conditions, V-REx shows a directional lift on both QA F1 (+0.010) and QD F1 (+0.014) without reducing the format gap. The effect magnitudes are small enough to be consistent with noise at n=50. If real, the mechanism is uniform absorption lift rather than selective format equalization — but we cannot confirm it is real from this experiment."*

The "not mean-variance trade-off" part is well-supported: the format gap literally did not close, so whatever the variance penalty is doing, it is not producing the predicted effect of that mechanism story.

---

## (4) Statistical power: is +20% on n=50 real?

No, it is not distinguishable from noise.

For QD F1 around 0.08, the standard error of a two-sample difference at n=50 is approximately 0.054 (binomial approximation). The observed delta of 0.014 yields z = 0.26. For QA F1 around 0.14, SE of difference ≈ 0.069, observed delta 0.010, z = 0.14. Both are nowhere near conventional significance thresholds.

A rough 95% CI on the QD delta: [−0.09, +0.12]. The true effect could easily be zero or negative.

**The paper should frame this as "directional/suggestive" — explicitly not a confirmed effect.** Something like: "At n=50 and single seed, the observed +0.014 QD F1 improvement (z=0.26) is consistent with noise. A 3-seed replication at n≥200 is needed before this can be treated as a real effect." The +20% relative framing is actively misleading when the absolute delta is 0.014 on a scale where the SE is 0.054. Relative percentages on tiny base rates exaggerate small absolute differences.

---

## (5) What does the paper now contribute?

Being brutal:

**The FI-SFT method contribution is gone.** The headline result was "V-REx halves the format gap." It doesn't. The leak-free numbers show no meaningful gap reduction. The remaining effect (+0.014 QD F1) is within noise. There is no statistically defensible method contribution from FI-SFT.

**What survives and has genuine value:**

1. **The COPR negative-transfer result** — novel, well-executed, first test of COPR on knowledge editing, with a clean mechanistic diagnosis. This is an honest contribution to the knowledge-editing literature.

2. **The absorption-integration gap measurement** — not a novel discovery (Berglund 2023, Allen-Zhu & Li 2023), but a quantified measurement in the continual-LoRA-injection setting with a format-gap metric. Worth reporting as a setting extension.

3. **"Format diversity without regularization hurts"** — mixed-format SFT (Phase 8d, format gap 0.100) is worse than single-format (0.072). This negative result is useful and parallels UIT/PIT findings in a new setting.

4. **The geometric/behavioral disagreement** — the specific inversion pattern (gold injection = best geometry, worst behavior) is a genuinely novel empirical observation, even if the general principle is documented.

5. **The V-REx experiment itself, even as a negative result** — "we applied V-REx to prompt formats; it did not close the format gap; the original leaky template was a confound" is honest and publishable as a negative finding. It saves the next researcher from trying the same thing.

**What the paper is, honestly:** A thorough empirical study of continual knowledge injection that produces two genuinely novel findings (COPR negative transfer, geometric/behavioral inversion), two useful negative results (format diversity hurts, V-REx doesn't close the gap), and a quantified measurement of a known phenomenon. The contribution is real but it is a diagnostic/negative-results paper, not a method paper. If the framing matches that reality, it is publishable and useful. If it still tries to sell FI-SFT as a working method, it is not honest.

The hardest sentence to write in a paper is "our proposed method does not work as hypothesized." This paper earned the right to write that sentence by running the confound test. That integrity is itself worth something in a field where most papers would have shipped the leaky result.
