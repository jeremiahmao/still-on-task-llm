# Iteration 10 — Rigorous Audit of draft.md Against Data Artifacts

**Date:** 2025-07-26  
**Scope:** Every numerical claim, citation tuple, and structural argument in `paper/draft.md` cross-checked against raw artifacts.

---

## Check 1: Numerical Claim Verification

### Abstract & §5.5 — Pilot trajectory (5 rounds × 1 seed × n=200)

| Claim in draft | Source artifact | Actual value | Verdict |
|---|---|---|---|
| naive_sft r1 abs_f1 = 0.064 | `outputs/sequential/naive_sft/trajectory.json` → r1 `abs_mean_f1` | 0.06364 | ✅ Rounds correctly to 0.064 |
| naive_sft r2 abs_f1 = 0.070 | same → r2 | 0.06995 | ✅ Rounds to 0.070 |
| naive_sft r3 abs_f1 = 0.048 | same → r3 | 0.04789 | ✅ Rounds to 0.048 |
| naive_sft r4 abs_f1 = 0.078 | same → r4 | 0.07839 | ✅ Rounds to 0.078 |
| naive_sft r5 abs_f1 = 0.064 | same → r5 | 0.06397 | ✅ Rounds to 0.064 |
| dsae_lite r1 abs_f1 = 0.184 | `outputs/sequential/dsae_lite/trajectory.json` → r1 `abs_mean_f1` | 0.18438 | ✅ Rounds to 0.184 |
| dsae_lite r2 abs_f1 = 0.288 | same → r2 | 0.28765 | ✅ Rounds to 0.288 |
| dsae_lite r3 abs_f1 = 0.305 | same → r3 | 0.30476 | ✅ Rounds to 0.305 |
| dsae_lite r4 abs_f1 = 0.310 | same → r4 | 0.31021 | ✅ Rounds to 0.310 |
| dsae_lite r5 abs_f1 = 0.300 | same → r5 | 0.30005 | ✅ Rounds to 0.300 |

**All 10 pilot trajectory numbers are correct.**

### Abstract — "lifts absorption F1 from 0.064 to 0.300 (~5×)"

Actual: 0.064 → 0.300 = 4.69×. Draft says "~5×." **Acceptable as "approximately 5×"** but 4.7× would be more precise. Recommend keeping "~5×" since it's hedged, but be aware a reviewer could note 4.7× ≠ 5×.

### Abstract — "worst-fact F1 from 0.040 to 0.265 (~6×)"

Actual worst-fact ranges (from eval_results.json `abs_fact_worst_f1`):
- naive_sft: r1=0.040, r2=0.046, r3=0.024, r4=0.048, r5=0.046 → range 0.024–0.048
- dsae_lite: r1=0.168, r2=0.291, r3=0.267, r4=0.300, r5=0.259 → range 0.168–0.300

The abstract picks 0.040 (naive r1) and 0.265 (dsae r5? actually r5=0.259). **Flag: 0.265 doesn't match any single round.** The closest is r3=0.267. At round 5, worst-fact F1 is 0.259, not 0.265. The draft body (§5.5) says "dsae_lite 0.168-0.300" which is correct; the abstract's 0.265 appears to be a cherry-pick or rounding error. **The "~6×" multiplier (0.040 → 0.265 = 6.6×) is overstated relative to the round-5 endpoint (0.040 → 0.259 = 6.5×) but roughly defensible.**

⚠️ **Fix required:** Abstract should cite 0.259 (round-5 endpoint) or 0.168–0.300 (full range), not 0.265. Or use "~5–7×" for the range.

### §5.5 — Preservation claims

Draft: "dsae_lite's preservation is +0.009 to +0.060 above naive_sft across the 5 rounds (mean +0.036), but at SE≈0.032 (n=104) this is roughly 1 SE."

Actual preservation differences (dsae_lite − naive_sft Recall@10):
- R1: +0.059, R2: +0.010, R3: +0.037, R4: +0.046, R5: +0.026
- Range: +0.010 to +0.059. Mean: +0.036.

Draft says "+0.009 to +0.060" — actual is +0.010 to +0.059. **Minor rounding discrepancy, not materially wrong.** Mean +0.036 is exact. ✅

### §5.1 — Format gap table (phase7b)

| Method | Draft qa_f1 | CSV qa_f1 | Draft qd_f1 | CSV qd_f1 | Draft gap | CSV gap |
|---|---|---|---|---|---|---|
| no_update | 0.041 | 0.041 | 0.013 | 0.0128 | 0.028 | 0.0282 |
| naive_sft | 0.150 | 0.1503 | 0.044 | 0.0435 | 0.107 | 0.1068 |
| kl_reg_sft | 0.143 | 0.1434 | 0.072 | 0.0715 | 0.072 | 0.0719 |
| copr | 0.123 | 0.1226 | 0.055 | 0.0547 | 0.068 | 0.0678 |
| copr_gi | 0.184 | 0.1844 | 0.044 | 0.044 | 0.140 | 0.1404 |
| copr_gi_anchored | 0.168 | 0.1675 | 0.047 | 0.0468 | 0.121 | 0.1207 |
| copr_anchored | 0.085 | 0.0849 | 0.039 | 0.0391 | 0.046 | 0.0458 |

**All match to rounding precision.** ✅

### §5.2 — COPR Table (phase3_sequential_final.csv)

All six round-15 absorption and locality F1 values in the §5.2 table verified exactly against `phase3_sequential_final.csv`. ✅

### §5.2 — "9-3 with 2 ties" head-to-head claim

Computed from trajectory JSONs (kl_reg_sft vs copr_gold_injection, 14 valid rounds, threshold ±0.005):
- kl wins: 9, cgi wins: 3, ties: 2. **Exact match.** ✅

### §5.2 — "+47% locality" claim

`copr_gi_anchored` r15 loc_overall_f1 = 0.0709 vs `kl_reg_sft` r15 = 0.0482.
(0.0709 − 0.0482) / 0.0482 = +47.1%. **Matches.** ✅

### §5.2 — "copr_anchored absorption F1 0.066, below the no-update baseline"

🚨 **ERROR.** `phase3_sequential_final.csv` round-15 `copr_anchored` abs_mean_f1 = **0.076**, not 0.066. The value 0.066 appears to be from an intermediate round (rounds 10, 11, 12 all show ~0.065–0.066). **Fix: change to 0.076** or specify which round is being cited.

### §5.3 — V-REx "+0.014 absolute QD F1 (z=0.26 at n=50)"

Source: `phase9_leakfree_isolation.csv`. fi_sft_leakfree qd_f1=0.0857 − kl_reg_sft qd_f1=0.0715 = +0.0142. Rounds to +0.014. ✅
z=0.26 is consistent with SE ≈ 0.054. ✅
**Confirmed this uses phase9 (leak-free), NOT phase8d (leaky).** ✅

### §5.4 — "format gap 0.100 versus single-format kl_reg_sft 0.072"

Source: `phase8d_variance_isolation.csv`. mixedfmt format_gap_f1 = 0.1004, kl_reg_sft = 0.0719. **Matches to rounding.** ✅

However, §5.4 cites phase8d data (which contains the "leaky" fi_sft row). The claim itself is about `mixedfmt` vs `kl_reg_sft`, neither of which is leaky, so **no contamination concern here.** But worth noting: phase8d also contains the fi_sft row with the leak; the draft correctly avoids citing that row and directs V-REx claims to phase9 only.

### §1 — "36-40% [underperformance] in absorption at 3K batch edits"

Phase 2 (3K batch): kl_reg_sft=0.173 vs copr family 0.104–0.111. Underperformance: 35.8–39.9%. **Matches "36-40%".** ✅

### §1 — "10-20× the compute"

Draft §3.2 table says COPR is ~10-12× per-round compute. The "10-20×" in §1 is a slight overclaim if the measured wall-clock is 10-12×. **Minor — tighten to "10-12×" for consistency.**

### §5.7 — SSL pilot numbers

| Round | Draft dsae_lite | Draft ssl | Actual dsae_lite | SSL (from chat) |
|---|---|---|---|---|
| 1 | 0.184 | 0.168 | 0.184 ✅ | 0.168 (no artifact) |
| 2 | 0.288 | 0.219 | 0.288 ✅ | 0.219 (no artifact) |
| 3 | 0.305 | 0.261 | 0.305 ✅ | 0.261 (no artifact) |
| 4 | 0.310 | 0.236 | 0.310 ✅ | 0.236 (no artifact) |
| 5 | 0.300 | 0.265 | 0.300 ✅ | 0.265 (no artifact) |

⚠️ **SSL numbers have no committed JSON artifact.** They were captured from the SageMaker run in chat. The dsae_lite comparison column is verified; SSL itself cannot be verified from local data. **The draft does NOT flag this provenance gap.** It should include a footnote: "SSL pilot numbers from a SageMaker run; artifact pending commit to repository."

### §5.7 — "underperformed DSAE Lite on absorption by 3.5-7.4pp (mean 4.8pp)"

Gaps: R1=0.016, R2=0.069, R3=0.044, R4=0.074, R5=0.035. That's 1.6–7.4pp, mean 4.8pp. **Draft says "3.5-7.4pp" — the lower bound is wrong.** R1 gap is 1.6pp, not 3.5pp. R5 gap is 3.5pp, so the draft appears to have dropped R1.

🚨 **ERROR.** The gap range should be **1.6–7.4pp**, not 3.5–7.4pp. Mean 4.8pp is correct (average of 1.6+6.9+4.4+7.4+3.5 = 23.8 / 5 = 4.76 ≈ 4.8pp).

### Abstract — "format gap (0.05-0.14 F1)"

From phase7b: gaps range from 0.046 (copr_anchored) to 0.140 (copr_gi). The 0.05-0.14 range cited in abstract is correct for "methods that meaningfully absorb" (§5.1 restricts to qa_f1 > 0.10, excluding no_update's 0.028 and copr_anchored's 0.046). **Defensible.** ✅

---

## Check 2: Overclaims and Underclaims

### "lifts absorption F1 from 0.064 to 0.300 (~5×)" — defensible at 1 seed × 5 rounds?

The number is real (4.7×). The concern is **statistical power**: 1 seed, 5 rounds, n=200 facts. The effect size is enormous (0.064 → 0.300), so the qualitative finding ("DSAE Lite helps a lot") is almost certainly real even at 1 seed. But the exact multiplier "~5×" could shift to ~3× or ~7× at a different seed. **The draft correctly hedges with "pilot" and "[Pending full Plan B results]".** This is defensible as stated. Recommend adding: "effect size is large enough that the qualitative direction is robust to seed variance; the precise multiplier awaits 2-seed confirmation."

### "without degrading task preservation"

Preservation Recall@10: dsae_lite runs +0.010 to +0.059 above naive_sft (mean +0.036, ~1 SE at n=104). This is "not degraded" at best and "marginally improved" at generous reading. The draft's framing is honest: "We claim 'not degraded,' not 'improved.'" **Defensible.** ✅

### §3.3 "K=5 augmented KL preservation has no precedent"

The draft says SEFE/ASD uses K format variants to replay data via SFT (not KL), RECAP uses KL but on single-format data, GeRe anchors activations on fixed-format texts. This was verified in iteration 6 literature search and the citation research here confirms these three papers exist and do what the draft says. **The "no precedent" claim is specific and narrow enough to be defensible**: format-diverse augmentation applied to the *KL preservation constraint* specifically. No false claim detected.

### "DSAE Lite is a deployment-friendly novel method"

The abstract says "deployment-friendly novel method" but DSAE Lite requires a **per-round frozen teacher snapshot** (§3.3, §6.4). The *deployment-friendly* variant is FDR (§6.4), which is explicitly flagged as future work and not validated.

🚨 **OVERCLAIM.** The abstract should NOT call DSAE Lite "deployment-friendly." It should say something like "a novel method" or "a principled method." The deployment story is FDR, which is speculative. Currently the abstract line reads: "a deployment-friendly novel method whose K=5 KL preservation is the active ingredient." **Remove "deployment-friendly" from the abstract.**

### COPR 9-3 head-to-head and +47% locality

Both verified exactly (see Check 1). The 9-3 record is across 14 contested rounds (round 9 skipped for COPR due to gap). +47% locality is for `copr_gold_injection_anchored` vs `kl_reg_sft` at round 15 specifically. **Both are correctly scoped.** ✅

### §4 "Seeds: 2 seeds (42, 123) for the DSAE Lite ablation conditions"

The current pilot only has 1 seed (42) for conditions (a) and (e). Plan B (4 conditions × 2 seeds × 15 rounds) hasn't run yet. **This section anticipates Plan B but currently overstates what's available.** Add a note that current pilot is 1 seed only; 2-seed results pending.

---

## Check 3: Citation Accuracy

### 🚨 Critical errors requiring immediate fix

| Citation | Issue | Fix |
|---|---|---|
| **Dalvi et al. 2025 (2602.03493)** | **Hallucinated authorship.** Paper is by Quercia, Bangun, Assent, Scharr — "Least but not Last: Fine-tuning Intermediate Principal Components." | Change to "Quercia et al. 2025, arXiv 2602.03493" |
| **Burns et al. 2310.06824** (§7) | **Wrong arXiv ID and/or wrong authors.** 2310.06824 is Marks & Tegmark, "The Geometry of Truth." Burns et al. (2022) is arXiv 2212.03827, "Discovering Latent Knowledge." | Fix ID to 2212.03827 if citing Burns, or fix authors to Marks & Tegmark if citing that paper |
| **Cohen et al. 2024 RippleEdits** | **Wrong year.** ArXiv is 2307.12976, July 2023. | Change to "Cohen et al. 2023" |
| **Rosenfeld et al. 2021 (2010.05908)** | **Wrong arXiv ID** — 2010.05908 returns 404. Correct Rosenfeld IRM paper is likely 2010.02922. | Fix to 2010.02922 |

### ⚠️ Unverifiable citations (may be real but couldn't confirm)

| Citation | Issue |
|---|---|
| Zhang et al. 2025 (COPR) | No paper found matching this description. May be a very recent preprint or workshop paper. If real, add arXiv ID. |
| Rozner et al. 2024 (KDPO) | Not found. Same concern. |
| Wan et al. 2025 (Mirage) | Not found across multiple search strategies. |
| Buzaaba et al. 2024 | Not found. (Cited once in §1 for "standard single-format KL preservation.") |
| Wei et al. 2024 (EVOKE) | **Wrong authors.** EVOKE (2410.07819) is by Mengqi Zhang, Xiaotian Ye et al. — not "Wei et al." |

### ⚠️ Minor issues

| Citation | Issue |
|---|---|
| Allen-Zhu & Li 2309.14316 | The "K=5 paraphrastic renderings" characterization simplifies the paper's mechanism (which is about mixed-training structure, not a simple paraphrase count). Acceptable shorthand for a related-work citation but slightly imprecise. |
| Ahuja et al. 2021 | Multiple Ahuja papers exist; provide arXiv ID to disambiguate. Likely 2010.16412. |
| MEMIT "Meng et al. 2023" | ArXiv preprint is Oct 2022 (2210.07229); ICLR publication is 2023. Either year is defensible, but be consistent. |

---

## Check 4: Logical Coherence / Flow

### §1 narrative: 3 negative results → DSAE Lite

The flow works: (1) COPR doesn't transfer (preference methods fail for fact injection), (2) V-REx at K=2 is degenerate (OOD regularization needs more environments), (3) format diversity alone hurts (augmentation without regularization backfires). Therefore → need K≥5 augmentation WITH a regularizer → DSAE Lite. **Logically sound.** ✅

### §3.2 / §5.5 ablation conditions

The 5-way ablation (a–e) in §5.5 is consistent with the §3.2 table. Condition IDs match. The decisive contrast (e) vs (d) is correctly identified. **Consistent.** ✅

However, §5.5 currently says "5 conditions × 2 seeds × 15 rounds" but the actual data is only (a) and (e) at 1 seed × 5 rounds. The section acknowledges this with "[Pending full Plan B results]" but the table header is misleading. **Fix: move the "5 conditions × 2 seeds × 15 rounds" description to a "Plan B design" paragraph, and clearly label the current table as "Confirmed pilot (2 of 5 conditions, 1 seed, 5 rounds)."**

### §5.7 SSL scoping

Currently 1 full paragraph + comparison table + 1 paragraph of post-hoc diagnosis + literature interpretation. This is ~350 words for a negative result with no committed artifact. **This is too long for a pilot with no artifact.** Recommend compressing to: (a) 1 sentence on SSL design, (b) comparison table, (c) 2 sentences on the calibration collapse, (d) 1 sentence pointing to Smith et al. for the structural-asymmetry explanation. Move the detailed calibration discussion to §6.2 or an appendix. Currently §5.7 bloats the results section with speculative interpretation that belongs in Discussion.

### §6.2 structural-asymmetry argument

§6.2 follows from §5.7 cleanly IF §5.7 is trimmed. The argument chain is: SSL underperforms by ~5pp → literature says KL has output-distribution sensitivity that weight-space methods lack → this explains the gap. The Smith et al. evidence (PredKD 25.2% vs EWC 7.7%) supports this, and the Nayak et al. counterpoint (dynamic projection beats KL) correctly distinguishes *static* from *dynamic* architectural constraints. **Logically sound.** ✅

### §6.4 deployment story

§6.4 is honestly scoped: FDR is described as "the most natural follow-up," not as a validated result. The "estimated ~3pp" cost is explicitly flagged as "not validated." **Appropriately hedged.** ✅

But the **abstract** still says "deployment-friendly" (see Check 2 overclaim). Fix the abstract; §6.4 is fine.

---

## Check 5: Missing Content

### Should we include phase4_compositional.csv?

The data shows all methods degrade below the no_update baseline (0.102 bridging-entity recall). DSAE Lite isn't in this data (it's from phase 4, pre-DSAE). §6.3 already discusses compositional degradation qualitatively. **Including a 1-row table** showing the no_update baseline and the best-performing method (e.g., copr_gi_anchored at 0.052) would strengthen §6.3 with a concrete number. Currently §6.3 cites "0.102" as the no_update baseline but no other numbers. **Recommend: add a brief table or inline citation** of the actual compositional F1 values from phase4_compositional.csv.

### Should we include kl_reg_sft 15-round trajectory as calibration?

The kl_reg_sft 15-round data exists in the trajectory CSV. Currently §5.5 only shows naive_sft (a) and dsae_lite (e) at 5 rounds. **Including kl_reg_sft (c) as a 3rd row in the pilot table** would be very useful — it shows what "standard KL preservation" does over 5 rounds, which is the natural comparison point for DSAE Lite (both have KL preservation; DSAE Lite has K=5 vs K=1). From the trajectory data:
- kl_reg_sft: r1=0.082, r2=0.112, r3=0.086, r4=0.121, r5=0.110

This shows kl_reg_sft is ~2× naive but ~3× below dsae_lite — a clean hierarchy (a < c << e) that previews the full ablation. **Strongly recommend adding this row.**

### Should the abstract mention 5 standalone contributions hold even if (e)≈(d)?

Yes. Currently the abstract is structured around the DSAE Lite positive result. But contributions (1)–(3) (format-gap diagnosis, COPR negative transfer, V-REx degeneracy) and (5) (SSL pilot) hold regardless of the (e) vs (d) outcome. If Plan B shows (e) ≈ (d), the paper's story shifts from "DSAE Lite's novel KL preservation is the active ingredient" to "Allen-Zhu augmentation alone is sufficient, and format-diverse KL preservation is not needed — but the diagnostic, negative results, and SSL comparison remain." **Recommend adding 1 sentence to the abstract:** "Contributions (1)–(3) and (5) hold independently of the (e)–(d) contrast."

---

## Check 6: Final Assessment

### Does the paper accurately represent what we did and found?

**Mostly yes.** The core experimental narrative is honest and well-grounded. The pilot trajectory numbers are all correct. The negative results (COPR, V-REx, format diversity without regularization) are rigorously supported by the data. The hedging around Plan B is appropriate. The leak-free guarantee and the transparent footnote about the Phase 9 confound are exemplary — this is how honest research should handle methodological errors.

**The main integrity risks are:** (a) several citation errors (some serious — hallucinated authorship for Dalvi/2602.03493, wrong ID for Burns/2310.06824); (b) one numerical error (copr_anchored 0.066 → should be 0.076); (c) the SSL gap range (1.6–7.4pp, not 3.5–7.4pp); (d) "deployment-friendly" overclaim in abstract.

### Top 3 things to fix before Plan B comes back

1. **Fix citation errors immediately.** The Dalvi→Quercia, Burns→Marks&Tegmark, Cohen 2024→2023, Rosenfeld arXiv ID, and EVOKE authorship errors are the kind of thing that tanks credibility in peer review. A reviewer who checks one citation and finds it wrong will distrust every number in the paper. Fix all six flagged errors and add arXiv IDs to the unverifiable citations (COPR, KDPO, Mirage, Buzaaba) or remove them.

2. **Fix the three numerical errors.** (a) copr_anchored: 0.066 → 0.076 in §5.2. (b) SSL gap range: 3.5–7.4pp → 1.6–7.4pp in §5.7. (c) Abstract worst-fact: 0.265 → 0.259 (or cite the range 0.168–0.300).

3. **Remove "deployment-friendly" from abstract.** DSAE Lite requires a per-round teacher snapshot. FDR (the deployment-friendly variant) is explicitly future work and unvalidated. The abstract should not claim deployment-friendliness for the proposed method.

### What to update once Plan B is in

- Replace the §5.5 pilot table with the full 5-condition × 2-seed × 15-round table
- Report the (e) vs (d) contrast with effect size and seed variance
- If (e) ≈ (d): rewrite the abstract to foreground the negative results and the augmentation-alone finding; demote the K=5 KL claim to "tested but null"
- If (e) > (d) by ≥2pp: keep current abstract framing but upgrade from "pilot" to "confirmed"
- Add kl_reg_sft, aug_sft_k5, and aug_kl_k1 trajectories to §5.5
- Commit SSL trajectory JSON to the repo and add the artifact path to §5.7
- Update §4 Seeds section to reflect actual seed counts (currently anticipates Plan B)
- Update §7 Limitations to reflect 2-seed validation rather than "pending"

### Summary of all errors found

| Section | Issue | Severity | Fix |
|---|---|---|---|
| Abstract | "deployment-friendly" for DSAE Lite | 🟡 Overclaim | Remove |
| Abstract | worst-fact 0.265 — no matching round | 🟡 Inaccuracy | Use 0.259 or range |
| §1 | "10-20× the compute" — measured is 10-12× | 🟢 Minor | Tighten to 10-12× |
| §5.2 | copr_anchored "absorption F1 0.066" — actual round-15 is 0.076 | 🔴 Wrong number | Fix to 0.076 |
| §5.7 | SSL gap "3.5-7.4pp" — R1 gap is 1.6pp | 🔴 Wrong number | Fix to 1.6-7.4pp |
| §5.7 | SSL numbers have no committed artifact | 🟡 Provenance | Add footnote |
| §4 | "2 seeds" anticipates Plan B but only 1 seed exists | 🟡 Misleading | Clarify pilot vs planned |
| Citations | Dalvi et al. 2602.03493 → Quercia et al. | 🔴 Hallucinated authorship | Fix |
| Citations | Burns et al. 2310.06824 → Marks & Tegmark | 🔴 Wrong ID or wrong authors | Fix |
| Citations | Cohen et al. 2024 → 2023 | 🔴 Wrong year | Fix |
| Citations | Rosenfeld 2010.05908 → likely 2010.02922 | 🔴 Wrong arXiv ID | Fix |
| Citations | Wei et al. 2024 EVOKE → Zhang et al. | 🔴 Wrong authors | Fix |
| Citations | COPR, KDPO, Mirage, Buzaaba | 🟡 Unverifiable | Add arXiv IDs or remove |

**Bottom line:** The experimental work is solid and honestly reported. The paper's main vulnerability is citation hygiene — five serious errors that would undermine reviewer confidence. Fix those, fix the three numerical errors, trim the SSL discussion, and the draft is in good shape for Plan B integration.
