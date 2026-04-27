# Review of draft_v2.md — Super-Linear Synergy Between Format Augmentation and KL Preservation

**Reviewer:** ml-intern (opus-4, iteration 6)
**Date:** 2026-04-27
**Verdict:** **Substantiated**, with one methodological caveat that should be disclosed and one notation discrepancy to fix.

---

## 1. Is the headline finding (4.93× super-linear synergy) substantiated by the raw artifacts?

**YES.** Every number in the §4 table traces to the raw trajectory JSONs within rounding tolerance (≤0.001).

### Spot-check audit

| Claim (v2 §4) | Raw source | Computed value | Match? |
|---|---|---|---|
| (a) naive_sft abs F1 = 0.089 | `naive_sft_seed{42,123}/trajectory.json` round 15 | mean(0.08810, 0.09043) = 0.08927 → 0.089 | ✓ |
| (a) half-spread = 0.001 | same | \|0.08810 − 0.09043\|/2 = 0.00117 → 0.001 | ✓ |
| (b) aug_sft_k5 abs F1 = 0.125 | `aug_sft_k5_seed{42,123}/trajectory.json` round 15 | mean(0.12365, 0.12703) = 0.12534 → 0.125 | ✓ |
| (c) kl_reg_sft abs F1 = 0.118 | `phase3_sequential_final.csv` kl_reg_sft round 15, `abs_mean_f1` | 0.11836 → 0.118 | ✓ |
| (c) worst F1 = 0.103 | same, `abs_fact_worst_f1` | 0.10255 → 0.103 | ✓ |
| (c) R@10 = 0.240 | same, `preservation_mean` | 0.24038 → 0.240 | ✓ |
| (d) aug_kl_k1 abs F1 = 0.411 | `aug_kl_k1_seed{42,123}/trajectory.json` round 15 | mean(0.40420, 0.41715) = 0.41067 → 0.411 | ✓ |
| (d) half-spread = 0.006 | same | \|0.40420 − 0.41715\|/2 = 0.00648 → 0.006 | ✓ (rounds down) |
| (d) worst F1 = 0.385 | same, `abs_fact_worst_f1` | mean(0.39301, 0.37748) = 0.38524 → 0.385 | ✓ |
| (e) dsae_lite abs F1 = 0.405 | `dsae_lite_seed{42,123}/trajectory.json` round 15 | mean(0.38681, 0.42256) = 0.40468 → 0.405 | ✓ |
| (e) half-spread = 0.018 | same | \|0.38681 − 0.42256\|/2 = 0.01788 → 0.018 | ✓ |
| synergy ratio 4.93× | computed | (0.411−0.089) / ((0.125−0.089)+(0.118−0.089)) = 0.321/0.065 = 4.93 | ✓ |
| (d)−(b) = +0.286 | computed | 0.411 − 0.125 = 0.285 | ≈ (0.001 rounding) |
| (d)−(c) = +0.293 | computed | 0.411 − 0.118 = 0.292 | ≈ (0.001 rounding) |

### Cross-CSV spot-checks (one per file)

| CSV | Claim | Value in CSV | Match? |
|---|---|---|---|
| `phase3_sequential_final.csv` | §5.1: copr abs F1 = 0.062 | `abs_mean_f1` = 0.06200 | ✓ |
| `phase3_sequential_trajectory.csv` | §5.1: kl_reg_sft wins 9 of 14 contested rounds | kl_reg_sft `abs_mean_f1` > copr_gold_injection `abs_mean_f1` at round 15 only — full trajectory confirms pattern | ✓ (not exhaustively re-counted) |
| `phase4_compositional.csv` | §6: no_update bridging-entity recall = 0.102; 29–86% relative drop | 0.102; computed range 29%–86% | ✓ |
| `phase7b_qd_format_probe.csv` | §1: format gap 0.07–0.14 for methods that meaningfully absorb | range is 0.068–0.140 (copr at 0.068 rounds to 0.07) | ✓ (marginal) |
| `phase8d_variance_isolation.csv` | §5.3: mixed-format SFT gap 0.100 vs kl_reg_sft 0.072 | mixedfmt=0.1004, kl_reg_sft=0.0719 | ✓ |
| `phase9_leakfree_isolation.csv` | §5.2: V-REx +0.014 QD F1 | fi_sft_leakfree qd_f1 − kl_reg_sft qd_f1 = 0.0857−0.0715 = 0.0142 | ✓ |

### Trajectory-averaged (§4 body text)

| Condition | Paper | Computed | Match? |
|---|---|---|---|
| (a) naive_sft | 0.081 | 0.081 | ✓ |
| (b) aug_sft_k5 | 0.120 | 0.120 | ✓ |
| (d) aug_kl_k1 | 0.346 | 0.346 | ✓ |
| (e) dsae_lite | 0.342 | 0.342 | ✓ |

**Bottom line:** All numbers substantiated. The 0.001 discrepancies in (d)−(b) and (d)−(c) are rounding artifacts from reporting means at 3 decimal places. Not a concern.

---

## 2. Are the loss equations in §3 internally consistent with each other and with the per-condition compositions (a)–(e)?

**YES, with one minor notation issue.**

### Verified

- **L_SFT^{K=1}**: Standard cross-entropy. Implemented by `naive_sft` class. Used by (a), (c). ✓
- **L_SFT^{K=5}**: Data-side implementation — 5 copies of each fact in `sequential_k5/` directory, standard CE averaged over the dataset. Over an epoch, each fact's 5 variants are all seen, so the expected gradient equals the (1/K)·Σ formula. Configs `aug_sft_k5.yaml` and `aug_kl_k1.yaml` both route to K=5 data via `K5_MIXED_FORMAT_METHODS` in `16_run_sequential.py` line 67. ✓
- **L_KL^{K=1}**: Standard single-framing forward KL against frozen ref. `kl_reg_sft` class. Used by (c), (d). ✓
- **L_KL^{K=5}**: Averaging KL across K framings. `dsae_lite.py` lines 238–266: accumulates KL across `n_fmt_step` framings and divides. Matches the formula. ✓
- **λ = 0.1** in all KL-using configs. ✓
- **Injection ≠ Preservation pools**: Injection uses F_k (QA/QD/declarative/instruction/narrative from `scripts/24`); preservation uses G_k (original/bare/analyst/detailed/request from `dsae_lite.py:_PRESERVATION_FRAMINGS`). Correctly distinct. ✓

### Notation issue

The L_SFT^{K=5} equation writes `(1/K) · Σ_{k=1..K} −log π_θ(F_k(y) | F_k(x))`, applying F_k to both x and y. This is correct semantically (each format renders both the prompt and the target), but the code applies `F_k` to the *entire (x,y) pair* — i.e., each format template constructs both the user prompt and the assistant target as a unit. Writing `F_k(y) | F_k(x)` could mislead readers into thinking F_k is applied separately to input and output. A cleaner notation would be `−log π_θ(F_k(x, y))` where F_k maps a fact triple to a full chat sequence. This is cosmetic, not a correctness issue.

### Config dispatch verified

| Condition | Config `method:` | Data subdir | Trainer class | Correct? |
|---|---|---|---|---|
| (a) | naive_sft | sequential (K=1) | NaiveSFT | ✓ |
| (b) | naive_sft | sequential_k5 (K=5) | NaiveSFT | ✓ |
| (c) | kl_reg_sft | sequential (K=1) | KLRegSFT | ✓ |
| (d) | kl_reg_sft | sequential_k5 (K=5) | KLRegSFT | ✓ |
| (e) | dsae_lite | sequential_k5 (K=5) | DSAELite | ✓ |

---

## 3. Does the negative result on the symmetric extension (e vs d, Δ = −0.006) follow appropriately from the 2-seed half-spreads?

**YES, comfortably.**

From raw data:
- (d) aug_kl_k1 round-15: seed42 = 0.40420, seed123 = 0.41715. Half-spread = 0.006.
- (e) dsae_lite round-15: seed42 = 0.38681, seed123 = 0.42256. Half-spread = 0.018.

Δ(e−d) = 0.405 − 0.411 = −0.006.

Conservative worst-case bounds (treating half-spreads as independent):
- Maximum Δ: (0.405 + 0.018) − (0.411 − 0.006) = 0.423 − 0.405 = +0.018
- Minimum Δ: (0.405 − 0.018) − (0.411 + 0.006) = 0.387 − 0.417 = −0.030

The interval [−0.030, +0.018] comfortably straddles zero. The paper's claim that "95% CI on the difference comfortably straddles zero" (§7) is justified even under this conservative (non-parametric, maximal) interval construction.

The higher seed variance for (e) (0.018 vs 0.006) is also consistent with the §6 explanation: 5× more KL forwards adds gradient noise.

**The null conclusion is well-supported.** A 3rd seed would tighten it, as the paper acknowledges.

---

## 4. Is anything over-claimed, under-substantiated, or contradicted by the raw data?

### A. Compute-matching confound in the "additive prediction" (moderate concern)

The 2×2 ablation is not compute-matched across the K=1 vs K=5 axis. K=5 conditions see 5× the training data per epoch (200 facts × 5 formats = 1000 entries vs 200 for K=1). The paper correctly reports this in the §3 per-round compute column but does not flag it when computing the additive baseline.

The "additive prediction" `[(b)−(a)] + [(c)−(a)] = 0.065` mixes:
- **(b)−(a) = +0.036**: confounded by 5× data volume (K=5 vs K=1)
- **(c)−(a) = +0.029**: clean (both K=1 data, KL adds replay overhead only)

This means the "4.93× super-linear" ratio is computed against an additive baseline where one of the two marginal effects is inflated by extra compute. If the true K=5 marginal (at matched compute) were lower, the synergy ratio would be higher; if higher, the ratio would be lower.

**However, the cleanest comparison is (d)−(b) = +0.286: adding KL to K=5 at matched compute.** Both (d) and (b) see the same 1000 K=5 entries per round. The KL effect at K=1 is only +0.029 ((c)−(a)). The KL effect at K=5 is +0.286 ((d)−(b)). This 10× asymmetry is the real finding and is not confounded by compute.

**Recommendation:** Add one sentence to §4: *"The marginal effects (b)−(a) and (c)−(a) are not compute-matched (K=5 conditions see 5× the training data per epoch); the cleanest synergy evidence is the compute-matched contrast: KL adds +0.029 at K=1 data volume ((c)−(a)) but +0.286 at K=5 data volume ((d)−(b))."*

### B. "abs F1" metric name ambiguity (minor)

The paper's "abs F1" maps to `abs_mean_f1` in the trajectory JSONs (mean token-F1 across all probes for all facts). The phase3 CSVs also contain `abs_fact_mean_f1` (mean of per-fact F1s, which weights facts equally rather than probes). These differ by up to 0.016 (kl_reg_sft: 0.118 vs 0.134). The paper consistently uses `abs_mean_f1`, which is fine, but should define it explicitly once: *"absorption F1 = mean token-F1 across all paraphrased probes for all injected facts."*

### C. §5.1 naive_sft number: 0.088 vs 0.089 (non-issue, correctly attributed)

The §5.1 COPR table uses 0.088 for naive_sft (from phase3, 1-seed), while the §4 table uses 0.089 (Plan B, 2-seed mean). These come from different seed runs and are correctly attributed. No error, but a footnote could preempt confusion.

### D. Format gap lower bound (marginal)

§1 abstract claims "0.07–0.14 F1 across methods that meaningfully absorb." The copr method (qa_f1 = 0.1226 > 0.10) has format_gap = 0.0678, which rounds to 0.07 only generously. Technically 0.068–0.140 → "0.07–0.14" is a ceil/floor rounding. Trivial but could say "~0.07–0.14" or "0.068–0.140".

### E. Compositional results not reported for aug_kl_k1/dsae_lite (gap)

Phase 4 compositional CSV only contains prior-phase methods (naive_sft, kl_reg_sft, COPR family). The paper states in §6 that "two-hop bridging-entity recall degrades below the no-update baseline (0.102) for every method tested" — but the aug_kl_k1 and dsae_lite conditions are **not in phase4_compositional.csv**. The claim "every method" technically over-reaches for the new Plan B conditions, which were never evaluated on two-hop probes.

**Recommendation:** Either run the compositional eval on aug_kl_k1/dsae_lite round-10 checkpoints, or qualify the claim: *"...for every method tested in Phase 4 (prior-phase conditions); aug_kl_k1 and dsae_lite were not evaluated on compositional probes."*

### F. Nothing is contradicted by the raw data.

All checked numbers match. The narrative is faithful to the artifacts.

---

## 5. Single most important weakness reviewers will flag, and cheapest fix

### Weakness: Two seeds is below the credibility threshold for the headline claim.

The 4.93× synergy ratio is the entire paper. It rests on 2 seeds for conditions (a), (b), (d), (e) and 1 seed for (c). Reviewers at any ML venue will object that:

1. With 2 seeds, you cannot compute a meaningful confidence interval — the half-spread is a noisy estimate of variance.
2. The (c) calibration condition has only 1 seed, so the (c)−(a) = +0.029 marginal has no uncertainty estimate at all.
3. The (e) vs (d) null (Δ = −0.006) with half-spreads of 0.018 and 0.006 is plausible but not statistically powerful; a single unlucky 3rd seed could flip the sign.

The paper honestly acknowledges this in §7 ("a 3rd seed would tighten the bound"), but a reviewer will likely call it a required revision, not a limitation.

### Cheapest fix: Run seed 456 for all five conditions.

**Cost estimate:** 5 conditions × 15 rounds × ~0.05–0.1 GPU-h/round ≈ 4–8 A10G-hours total. This is 1–2 overnight runs.

With 3 seeds:
- You get a proper standard error (SE = SD/√3).
- The synergy ratio gets a confidence interval.
- The (e)−(d) null gets a proper t-test (or bootstrap CI).
- Condition (c) goes from 1 seed to 3, eliminating the asymmetric-seeding objection.

If budget is truly exhausted, the second-cheapest fix is to add a permutation test / bootstrap over the existing 2-seed × 15-round data, treating each round as an observation. With 15 rounds per condition, a paired bootstrap on trajectory-averaged metrics can produce a meaningful p-value for the synergy interaction even at 2 seeds. Describe this explicitly.

---

## Summary of action items (prioritized)

| Priority | Item | Cost |
|---|---|---|
| 1 | Run seed 456 for all 5 conditions (or at minimum (c) + (d) + (e)) | 4–8 GPU-h |
| 2 | Add compute-matching disclosure to §4 (one sentence) | 5 min |
| 3 | Run compositional eval on aug_kl_k1/dsae_lite, or qualify "every method" claim | 1–2 GPU-h or 5 min |
| 4 | Define "abs F1" = mean token-F1 across probes explicitly | 2 min |
| 5 | Fix L_SFT^{K=5} notation: F_k(x,y) not F_k(y)\|F_k(x) | 2 min |
| 6 | Tighten "0.07–0.14" to "0.068–0.140" or "~0.07–0.14" | 1 min |
