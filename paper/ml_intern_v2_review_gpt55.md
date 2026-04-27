Substantiated, with two important caveats: (i) the 4.93× headline is numerically supported by the artifacts, but one COPR dominance sentence is false (9/14 should be 11/14), and (ii) the loss equations idealize the implementation because SFT is computed on full chat tokens, not assistant-target tokens only.

1. Headline finding / artifact check

- The 2×2 endpoint numbers in §4 match the raw trajectories:
  - `naive_sft`: seed42 0.088104, seed123 0.090433 → mean 0.089269, half-spread 0.001164; paper rounds to 0.089 / 0.001.
  - `aug_sft_k5`: 0.123647, 0.127028 → mean 0.125338, half-spread 0.001691; paper 0.125 / 0.002.
  - `aug_kl_k1`: 0.404197, 0.417151 → mean 0.410674, half-spread 0.006477; paper 0.411 / 0.006.
  - `dsae_lite`: 0.386806, 0.422558 → mean 0.404682, half-spread 0.017876; paper 0.405 / 0.018.
  - `kl_reg_sft` from `phase3_sequential_trajectory.csv` round 15 is 0.118358; paper 0.118.
- Headline arithmetic is correct after rounding:
  - K=5 alone: 0.125338 − 0.089269 = +0.036069.
  - KL alone: 0.118358 − 0.089269 = +0.029089.
  - Combined: 0.410674 − 0.089269 = +0.321405 ≈ +0.322.
  - Additive prediction: 0.065158; ratio 0.321405 / 0.065158 = 4.93×.
- Worst-F1 / preservation values in §4 also match rounding: `aug_kl_k1` worst mean 0.385245, preservation mean 0.237179; `dsae_lite` worst 0.377091, preservation 0.236378.
- Trajectory averages match: (a) 0.081068, (b) 0.120132, (c) 0.113962, (d) 0.345988, (e) 0.342329.
- Spot-checks for every cited CSV:
  - `phase3_sequential_final.csv`: table §5.1 values match: `copr` abs 0.062005 / locality 0.028818; `copr_gold_injection_anchored` abs 0.118877 / locality 0.070864; `kl_reg_sft` abs 0.118358 / locality 0.048187.
  - `phase3_sequential_trajectory.csv`: `kl_reg_sft` round 15 0.118358 and trajectory mean 0.113962 match §4.
  - `phase4_compositional.csv`: no-update bridging 0.102; method bridging recall range is 0.014-0.072, i.e. 29-86% relative drop, matching §6. Token F1 mostly improves except `copr_anchored` (0.0246 < baseline 0.0517), so “most updates” is accurate.
  - `phase7b_qd_format_probe.csv`: n=50; `naive_sft` gap 0.1068, `kl_reg_sft` 0.0719, COPR GI gaps 0.1404 / 0.1207; matches §1/§5.1.
  - `phase8d_variance_isolation.csv`: `mixedfmt` gap 0.1004 vs `kl_reg_sft` 0.0719; matches §5.3.
  - `phase9_leakfree_isolation.csv`: `fi_sft_leakfree` QD F1 0.0857 vs `kl_reg_sft` 0.0715 → +0.0142; matches §5.2. The stated z/CI are not in the CSV, so they are not independently auditable from that artifact alone.

2. Loss equations / condition consistency

- The condition composition (a)-(e) is consistent with configs and dispatch:
  - `naive_sft.yaml` → `NaiveSFTUpdate`, ordinary sequential data → K=1 SFT.
  - `aug_sft_k5.yaml` → same class, but `scripts/16_run_sequential.py` routes to `sequential_k5`; K=5 lives in data.
  - `kl_reg_sft.yaml` → `KLRegSFTUpdate`, K=1 facts + K=1 replay KL, λ=0.1.
  - `aug_kl_k1.yaml` → `KLRegSFTUpdate` + K=5 injection data + single-format replay KL.
  - `dsae_lite.yaml` → `DSAELiteUpdate` + K=5 injection data + `num_kl_formats: 5` replay KL.
- KL direction is internally consistent with code and text: `KL(π_ref || π_θ)` is exactly implemented as `ref_lp.exp() * (ref_lp - cur_lp)` in both `kl_reg_sft.py` and `dsae_lite.py`.
- K=5 preservation equation is consistent with `dsae_lite.py`: five `_PRESERVATION_FRAMINGS`, per-format KLs averaged as `kl_accum_step / n_fmt_step`.
- Main caveat: the SFT equations write `−log πθ(y|x)` / `−log πθ(F_k(y)|F_k(x))`, but implementation labels the entire chat sequence except padding (`labels = input_ids`; no masking of user/system tokens). Thus the realized loss is full-chat next-token LM loss, not assistant-only conditional NLL. This probably does not invalidate the ablation because all SFT-family conditions share it, but reviewers may flag the mismatch. Cheapest correction: state “implemented as full-chat causal LM loss over rendered chat transcripts” or mask prompts and rerun if you want the displayed equation literally true.
- Minor equation notation issue: `F_k(x)` / `F_k(y)` suggests separable transforms of input and output; actual templates transform the whole `(subject, relation, object)` triple into a chat. Prefer `F_k(x,y) -> (prompt_k, target_k)`.

3. Symmetric-extension negative (e vs d)

- The negative result follows descriptively from the two-seed summaries:
  - Round 15: `dsae_lite − aug_kl_k1 = 0.404682 − 0.410674 = −0.005992` ≈ −0.006.
  - Trajectory average: 0.342329 − 0.345988 = −0.003659 ≈ −0.004.
  - Half-spreads: e 0.017876 vs d 0.006477, so e is visibly noisier.
- But “clean negative” should be interpreted as “no observed benefit,” not a statistically sharp negative. With only two seeds, e’s interval [0.3868, 0.4226] overlaps d’s [0.4042, 0.4172]. The draft mostly handles this in §7, but the abstract’s “clean negative” wording is a little strong. Say “clean null/no-benefit result” or “negative for added value.”

4. Over-claims / contradictions / under-substantiation

- Incorrect: “`kl_reg_sft` wins absorption 9 of 14 contested rounds vs `copr_gold_injection`.” Recomputing `phase3_sequential_trajectory.csv` gives 11 wins, 0 ties, 3 losses over common rounds [1-8,10-15]. The direction strengthens the claim; fix the count.
- Over-strong: “closed” absorption-integration gap. `aug_kl_k1` greatly improves absorption F1, but v2 does not show a format-gap probe for the new K=5/KL conditions analogous to `phase7b`; it infers closure from paraphrased absorption. Use “substantially closes” unless you add a behavioral QA-vs-QD n=50 probe for `aug_kl_k1`/`dsae_lite`.
- Under-substantiated: “K-sample-all-wrong pathology; worst-F1 ≈ 0 in pilot” is plausible and code-consistent, but no raw pilot artifact is listed. Add the pilot CSV/log or soften.
- Under-substantiated: z=0.26 and CI [−0.09,+0.12] for V-REx are not present in `phase9_leakfree_isolation.csv`. Add a small bootstrap/stat script output or remove the inferential stats.
- Over-strong mechanistic language: §6 explanations (“forces entity-anchored linear features,” “anchors enriched feature space”) are hypotheses, not shown by current raw artifacts. §7 admits this; keep §6 phrased as hypothesis/model, not established mechanism.
- Compute table: §5.1 GPU-h/round values are supported by metadata, not by `phase3_sequential_final.csv`. That is fine, but reproducibility should point to `outputs/seq_*_round_*/metadata.json` or aggregate it into a CSV.
- Related-work claims with future-looking arXiv IDs (e.g. 2510.*) may attract skepticism unless references are fully bibliographically resolved; not checked here.

5. Single most important weakness + cheapest fix

- Biggest reviewer flag: only two seeds for the headline cells and only one seed for the calibration cell (c), while the headline ratio depends on (c) and on small deltas in the denominator. This makes the “4.93×” look numerically precise beyond the experimental support.
- Cheapest fix: run one more seed for `kl_reg_sft` and, if possible, one more seed for the four new cells; then report the interaction term with bootstrap/randomization CI. If compute is too tight, at minimum rerun only `kl_reg_sft` seed123 plus add a table reporting per-seed endpoints and an interaction CI using existing half-spreads; this directly defuses the precision/variance critique without changing the story.
