# Paper Cleanup Plan: From FI-SFT to DSAE Lite

**Date:** 2025-04-25
**Status:** Pre-experiment paper restructure. DSAE Lite 5-way ablation locked but not yet run.

---

## 1. Two Paper Outlines

### Option A — DSAE Lite Works

**Title (working):** "Format-Diverse Preservation: A Symmetric Augmentation for Continual Knowledge Injection in LoRA"

**Contribution narrative:** The absorption-integration gap is real and measurable under LoRA continual editing. We tried five methods that don't close it (naive SFT, KL-reg SFT, three COPR variants, V-REx on formats). We then applied DSAE Lite — K=5 augmented injection (Allen-Zhu) + K=5 augmented KL preservation (novel) — and it does close it. The 5-way ablation isolates the novel ingredient (format-diverse KL preservation) from the known ingredient (format-diverse injection).

**Structure:**

| § | Title | Content | Length |
|---|-------|---------|--------|
| 1 | Introduction | Absorption-integration gap exists. Prior work documents it (Berglund, Allen-Zhu, Cohen, Zhong). We measure it under LoRA continual editing. Five methods fail to close it. DSAE Lite closes it. Contributions: (1) gap quantified at 4B/n=200/15 rounds; (2) three negative methods (COPR, V-REx K=2, format diversity without regularization); (3) DSAE Lite as the positive method; (4) ablation isolating format-diverse KL preservation. | 1.5pp |
| 2 | Related Work | Parametric editing (ROME, MEMIT, MEND). Continual learning for LLMs (InsCL, O-LoRA). Format robustness (PAFT, UIT, PIT). Knowledge storage vs extraction (Allen-Zhu, Berglund). OOD regularization (V-REx, IRM, Fish) — now positioned as "methods we tried that didn't work." DSAE Lite's ancestry: Allen-Zhu augmentation + novel preservation-side augmentation. | 1.5pp |
| 3 | Methods | 3.1 Problem formulation (preserve from current draft). 3.2 Baselines: naive SFT, KL-reg SFT, COPR family (compressed to 1 paragraph + table). 3.3 Failed intervention: FI-SFT / V-REx on formats (1 paragraph: what it is, why it failed, leak confound disclosure). 3.4 DSAE Lite (NEW, 1 page): K=5 augmented injection + K=5 augmented KL preservation, formal definition, relationship to Allen-Zhu & PAFT, what's novel (the KL side). 3.5 5-way ablation design (table from iteration3_verdict §4). | 2pp |
| 4 | Experimental Setup | Backbone, data, metrics, compute. Mostly preserved from current §4. Add: 3 seeds, format templates, decision rules. | 0.75pp |
| 5 | Results | 5.1 Diagnostic: absorption-integration gap (current §5.6 Phase 7b table, compressed). 5.2 Negative methods: COPR (current §5.1–5.2, heavily compressed to key numbers + one trajectory figure). 5.3 Negative methods: V-REx/FI-SFT (leak confound story — this is a *strength*). 5.4 Negative methods: format diversity without regularization hurts (Phase 8d result, 1 paragraph). 5.5 DSAE Lite 5-way ablation results (NEW — the main event). 5.6 Compositional degradation persists (1 paragraph, shared limitation). | 2.5pp |
| 6 | Discussion | What the negative methods teach. Why format-diverse KL preservation works (if it does). Geometric/behavioral disagreement as a methodological warning. | 1pp |
| 7 | Limitations | Single model (4B). 3 seeds. n=200/round. 15 rounds. Synthetic format templates. Compositional gap unsolved. Scale unknown. | 0.5pp |
| 8 | Conclusion | 0.5pp |

**Total:** ~10pp venue format.

**Limitations:** Single model (4B); 3 seeds; n=200/round; synthetic templates; frozen round-0 KL reference (staleness uncharacterized); compositional gap unsolved; FI-SFT leak confound disclosed in main text.

**Decision rule for Option A:** Condition (e) `dsae_lite` QD F1 ≥ (d) `aug_kl_k1` + 2pp across 3 seeds (p<0.05 paired t-test); format gap ≤ 0.04; preservation within 1pp of baseline (a). The (e) vs (d) contrast isolates the novel ingredient.

---

### Option B — DSAE Lite Fails

**Title (working):** "What Doesn't Close the Format Gap in Continual Knowledge Injection: Five Negative Methods at 4B Scale"

**Contribution narrative:** The absorption-integration gap is real but appears unclosable by data-side methods at 4B/LoRA/n=200 scale. We tested five interventions spanning three families (preference-based: COPR; OOD-regularization: V-REx; data augmentation: PAFT-style, Allen-Zhu-style, DSAE Lite). None closes the gap. The paper contributes the diagnostic, five negative results, a leak-confound disclosure, and the geometric/behavioral disagreement finding.

**Structure:**

| § | Title | Content | Length |
|---|-------|---------|--------|
| 1 | Introduction | Gap exists. We measure it. We tried five things. None worked. This is useful because it saves others from trying the same things. | 1pp |
| 2 | Related Work | Same as Option A, minus DSAE Lite positioning as positive. | 1pp |
| 3 | Methods | 3.1 Problem formulation. 3.2 All methods as a table (7 rows: naive SFT, KL-reg SFT, 3 COPR variants, FI-SFT/V-REx, DSAE Lite). Each gets 1 paragraph. 3.3 5-way DSAE Lite ablation design (to show the negative result is well-controlled). | 2pp |
| 4 | Experimental Setup | Same as Option A. | 0.75pp |
| 5 | Results | 5.1 Diagnostic: the gap. 5.2 COPR family (compressed). 5.3 V-REx + leak confound. 5.4 Format diversity without regularization. 5.5 DSAE Lite ablation (the fifth negative). 5.6 Compositional. 5.7 Geometric/behavioral disagreement. | 2.5pp |
| 6 | Discussion | What the pattern across five negative methods tells us. Hypothesis: the gap at 4B/LoRA/n=200 is a capacity or scale phenomenon, not an objective-design problem. Pointers to what might work (full fine-tuning, larger models, retrieval). | 1pp |
| 7 | Limitations + Conclusion | Combined. | 0.75pp |

**Total:** ~9pp.

**Limitations (beyond Option A):** "Nothing works at this scale" may not generalize upward — the gap might close at 14B or with full fine-tuning. Only data-side methods tested; architecture-side and inference-side interventions remain open.

---

## 2. What Dies, What Gets Rewritten, What Survives

### Dies (delete entirely)

- **Abstract (lines 3–5).** 600+ words selling FI-SFT. All of it goes.
- **§3.4 FI-SFT formal definition (lines 78–105).** Full-page V-REx formalism. Compress to 1 paragraph: what it was, why it failed, leak confound.
- **§5.7 Phase 8 V-REx isolation (lines 312–331).** The +19%/+18%/+16% table used leaky data. Replace with leak-confound disclosure (1 paragraph).
- **§6.0 "Arc of the experiment" (lines 335–367).** 2 pages of COPR→V-REx narrative; V-REx conclusion dead. Salvage COPR lessons into negative-methods section.
- **§6.1 "V-REx environments" (lines 369–375).** The analogy doesn't hold. Kill.
- **§6.2 KL-reg baseline analysis (lines 377–387).** Compress to 1 paragraph in Discussion.
- **§6.3 "Locality: two routes" (lines 389–398).** FI-SFT is dead; rewrite around COPR locality only.
- **§6.5 Format-coupling discussion (lines 405–415).** Keep the diagnostic; kill the FI-SFT-closes-the-gap conclusion.
- **§8 Conclusion (lines 454–461).** Entirely about FI-SFT. Rewrite from scratch.

### Rewrite substantially

- **§1 Introduction (lines 7–25).** Currently organized around COPR→FI-SFT arc. Rewrite to: diagnostic → negative methods → DSAE Lite (or negative-results framing). The "four contributions" list at line 15 needs to become 3–4 new contributions matching whichever option materializes.
- **§2 Related Work (lines 27–45).** Currently 2 pages including V-REx and cross-view regularizer ancestry. Compress to 1.5 pages. Demote V-REx/Fish/IRM to a "methods we tried" paragraph. Add DSAE Lite ancestry: Allen-Zhu augmentation, PAFT, SEFE/ASD, RECAP, GeRe. Position the novelty (format-diverse KL preservation has no precedent — confirmed by the DSAE review's exhaustive search).
- **§3.2–3.3 COPR methods + gold injection (lines 59–76).** Currently 1+ page of formal definitions for 4 COPR variants. Compress to a summary table (one row per variant, columns: what it adds, absorption F1 at round 15, compute) and one paragraph explaining gold injection / K-sample-all-wrong.
- **§5.1–5.2 Batch + Sequential results (lines 128–213).** Currently 3+ pages of tables. Compress batch to one paragraph + one table. Compress sequential to the trajectory-average table + the head-to-head tally (which is the strongest evidence). Drop the round-by-round table (it's in supplementary / appendix).
- **§5.4–5.5 Mechanistic + Hidden-state (lines 237–291).** Currently 2 pages. The Frobenius norm / subspace overlap analysis is interesting but secondary. Move to appendix. Keep the Phase 7 hidden-state result (shift ratio + cosine) as 1 table + 1 paragraph — it feeds the geometric/behavioral disagreement finding.
- **§7 Limitations (lines 423–452).** Currently 1.5 pages. Good content but bloated. Compress to 0.5 pages. The most important limitations to keep: single seed → now 3 seeds (improved); single model scale; synthetic templates; compositional unsolved; the leak confound disclosure (move to main text, it's a feature not a bug).

### Preserve as-is (or near as-is)

- **§3.1 Problem formulation (lines 49–56).** Clean and correct.
- **§5.3 Compositional / Phase 4 (lines 219–235).** The bridging-entity degradation result is real, method-invariant, and important. Keep the table and the two observations.
- **§5.6 Behavioral format gap / Phase 7b (lines 293–309).** The table showing gold-injection = worst behavioral gap despite best geometry. This is the paper's most novel empirical observation. Keep verbatim.
- **Reproducibility Statement (lines 463–467).** Keep, update paths.

---

## 3. Abstract Rewrites

### Option A Abstract (~150 words)

Continual knowledge injection via LoRA fine-tuning produces an absorption-integration gap: models absorb facts under the training prompt format but fail to deploy them under different downstream formats. We measure this gap (0.05–0.14 F1) across seven methods on Qwen3-4B over 15 sequential rounds of 200 edits each. Five methods fail to close it: three COPR variants exhibit negative transfer from preference alignment; V-REx applied to prompt formats appeared effective but was confounded by a template data leak (retracted upon discovery); format-diverse training without explicit regularization actively widens the gap. We then test DSAE Lite, which pairs K=5 augmented injection (Allen-Zhu & Li, 2023) with a novel K=5 augmented KL preservation constraint. A 5-way ablation across 3 seeds isolates the novel ingredient — format-diverse KL preservation — as the active component that narrows the gap from [X] to [Y]. Code and all artifacts are public.

### Option B Abstract (~150 words)

Continual knowledge injection via LoRA fine-tuning produces a measurable absorption-integration gap: models store injected facts but cannot deploy them under prompt formats different from training. We quantify this gap (0.05–0.14 F1) on Qwen3-4B across 15 sequential rounds of 200 edits. We test five interventions spanning preference-based objectives (COPR), OOD-regularization (V-REx on prompt formats), and data augmentation (format-diverse SFT, Allen-Zhu augmentation, DSAE Lite symmetric augmentation). None closes the gap. COPR exhibits negative transfer from preference alignment. V-REx appeared to halve the gap but was confounded by a template data leak, retracted here. Format diversity without regularization actively hurts. DSAE Lite provides no significant improvement over injection-side augmentation alone. We additionally report a geometric-behavioral disagreement: gold-injection COPR produces the best hidden-state alignment but the worst behavioral format transfer — cautioning against geometric proxies for knowledge integration.

---

## 4. Figures the Paper Needs

### Figure 1: Absorption-Integration Gap Diagnostic
**Description:** Bar chart showing QA F1 (in-format) vs QD F1 (out-of-format) side by side for each method at round 15.
**X-axis:** Method (7 bars). **Y-axis:** Token F1 (0–0.20). **Two bars per method** (QA, QD) with the gap annotated.
**Shows:** The gap is universal (0.05–0.14) and no single-environment method closes it.
**Source:** `phase7b_qd_format_probe.csv`.

### Figure 2: Sequential Absorption Trajectories
**Description:** Line plot, one line per method, showing absorption F1 over 15 rounds.
**X-axis:** Round (1–15). **Y-axis:** Absorption F1 (0–0.20).
**Shows:** kl_reg_sft dominates COPR family. The round-10 copr_gi spike is a one-off, not a trend.
**Source:** `phase3_sequential_trajectory.csv`.

### Figure 3: Geometric-Behavioral Disagreement
**Description:** Scatter plot. X-axis: hidden-state shift ratio (geometric proxy, lower = better integration). Y-axis: behavioral format gap F1 (lower = better transfer). Each point is one method, labeled.
**Shows:** The inversion — gold-injection methods are in the lower-left geometrically (best) but upper-right behaviorally (worst). Warns against geometric proxies.
**Source:** `phase7_manifold_analysis.csv` + `phase7b_qd_format_probe.csv`.

### Figure 4: DSAE Lite 5-Way Ablation (Option A only)
**Description:** Grouped bar chart. 5 conditions × 3 metrics (QA F1, QD F1, format gap).
**X-axis:** Condition (baseline_sft, aug_sft_k5, kl_sft_k1, aug_kl_k1, dsae_lite). **Y-axis:** F1 / gap metric. Error bars: ±1 SE across 3 seeds.
**Shows:** Which ingredient drives the improvement. The (e) vs (d) contrast is the money shot.
**Source:** NEW experiment results.

### Figure 5: Leak Confound Disclosure (both options)
**Description:** Simple 2×2 table or paired bar chart. Rows: leaky vs leak-free. Columns: QA F1, QD F1, format gap.
**Shows:** The leak inflated the headline result. Leak-free format gap (0.068) ≈ baseline (0.072). Presented as a transparency element, not hidden in supplementary.
**Source:** `phase9_leakfree_isolation.csv`.

---

## 5. Missing Experiments for Option A's Affirmative Claim

The 5-way DSAE Lite ablation is locked and sufficient for a method-contribution claim. Four additional experiments would strengthen it, in priority order:

**(a) K-ablation on the KL side (K=3/5/10).** Does KL-side diversity have diminishing returns like injection-side (Allen-Zhu)? ~6 GPU-hours.

**(b) Linear probing for format-invariant features.** Allen-Zhu showed K=5 injection stores facts as format-invariant linear features at entity tokens. Does DSAE Lite's KL term *also* maintain format-invariant features for preserved knowledge? Train a linear probe on entity-name hidden states across formats, compare DSAE Lite vs aug_sft_k5. ~1 GPU-hour.

**(c) Leak-free V-REx as a 6th ablation condition.** Direct A/B: variance penalty (V-REx) vs preservation-side augmentation (DSAE Lite), same K=5 formats. Resolves whether V-REx truly fails or just needed leak-free data. ~3 GPU-hours.

**(d) Per-format KL divergence logging.** Log per-format KL from frozen reference each round. Under single-format KL, format-B should drift while format-A stays anchored; under DSAE Lite, both stay anchored. This *is* the mechanism visualization. ~0 extra GPU-hours (add logging).

Priorities: (b) and (d) are free — include regardless. (a) is highest-value if the main result is positive. (c) clarifies the Option A/B boundary.

---

## 6. Honest Title Proposals

### Option A (DSAE Lite works)

1. **"Format-Diverse Preservation Closes the Absorption-Integration Gap in Continual LoRA Editing"**
   — Direct. Says what the method does. Doesn't oversell scope.

2. **"Symmetric Format Augmentation for Continual Knowledge Injection: Closing the Gap That Five Methods Couldn't"**
   — Emphasizes the negative-results backdrop. Slightly dramatic but honest.

3. **"What Finally Closes the Format Gap in Continual Knowledge Injection (And What Doesn't)"**
   — Signals both the positive and negative contributions. Best for a venue that values negative results.

### Option B (DSAE Lite fails)

1. **"Five Methods That Don't Close the Format Gap in Continual Knowledge Injection at 4B Scale"**
   — Direct negative-results title. Clear what the paper is.

2. **"The Absorption-Integration Gap Under Continual LoRA Editing: Diagnosis, Five Failed Interventions, and a Geometric-Behavioral Disagreement"**
   — Long but complete. Works for a workshop submission where the title is the pitch.

3. **"Why Format-Invariant Knowledge Injection Is Hard at Small Scale"**
   — Clean, honest, slightly opinionated. The "small scale" qualifier is important — it doesn't claim the problem is unsolvable, just that data-side methods at 4B/LoRA don't solve it.

---

## 7. The Leak Confound: How to Present It

The leak confound disclosure should be **in the main text, not buried in limitations.** Suggested placement: §5.3 in both options. Frame it as follows:

> **V-REx on prompt formats: a retracted result.** We applied V-REx (Krueger et al., 2021) to prompt formats, treating each format as an environment and penalizing cross-format loss variance (§3.3). Initial 15-round results showed format gap 0.037 vs baseline 0.072 — an apparent halving. However, the synthetic QD template embedded the gold answer in the assistant target ("What is {answer}'s role related to {subject}?"). A leak-free retrain with the answer removed from the template produced format gap 0.068 vs baseline 0.072 — within noise (z=0.26, n=50). The "gap halving" was almost entirely a template artifact. We retract the FI-SFT format-gap claim. The surviving V-REx finding is that format diversity *without* a variance penalty actively widens the gap (0.100 vs 0.072), a negative result consistent with UIT (2023) and PIT (Jiang et al., 2024).

This is a strength of the paper. Most groups would have shipped the leaky result. Catching and reporting the confound is a credibility signal.
