# paper/TODO.md

Unresolved items for `paper/draft.md`. Loosely ordered by severity.

## Related Work (sister agent in progress)

- [x] Initial Related Work (Section 2) spliced from sister agent's `paper/related_work_notes.md`. Re-verify once the sister agent finalizes:
  - [ ] **Zhang et al. 2025 (COPR)**: confirm exact ACL Findings 2025 citation, arXiv ID, and that the characterization of COPR's KL-replay anchor in Section 2 is faithful to the paper. See `refs.bib` note `% VERIFY`.
  - [ ] **Rozner et al. 2024 (KDPO)**: confirm author list, EMNLP 2024 vs Findings, and that our description (new fact chosen / old fact rejected; pre-edit model as KL anchor) is correct.
  - [ ] **Zhang et al. 2024 (CPPO)**: confirm arXiv ID and whether it is the same author group as COPR.
  - [ ] **Yao et al. 2023 (editing survey)**: confirm title/venue; several surveys exist.
  - [ ] **Wu et al. 2024 (CL-for-LLM survey)**: confirm — several contemporaneous surveys exist.
  - [ ] **Yu et al. 2024 (MELO)**: confirm AAAI 2024 venue and title.
  - [ ] **Zhu et al. 2024 (LoRA asymmetry)**: confirm title/arXiv ID.
  - [ ] **Shah 2023 (FNSPID)**: fill in exact arXiv ID.
- [ ] If any sister-agent entry shows that a paper already considered **candidate-set augmentation** with a gold answer in a COPR/DPO ranking setup, adjust the gold-injection novelty claim in Sections 1, 3.3, and 6 accordingly. (Current draft positions it as novel.)
- [ ] If sister agent finds a reference-policy-anchoring variant of COPR in the literature, cite it in Section 3.2 next to `copr_anchored` and adjust the "novel" label if appropriate.

## Figures (placeholders in draft)

- [ ] **Figure 1**: Per-round absorption F1 (y-axis 0-0.20, x-axis rounds 1-10) for all six methods. Source: `final_results/phase3_sequential_trajectory.csv`, column `abs_mean_f1`. Suggested script: `scripts/plot_trajectory.py` (not yet written).
- [ ] **Figure 2**: Per-round locality overall F1 for all six methods, same format. Source: same CSV, column `loc_overall_f1`.
- [ ] **Figure 3**: Per-layer Frobenius norm by LoRA site across the four COPR variants (layers 0-31 on x-axis). Source: `outputs/seq_*_round_10_qd_scale200/update_adapter/` tensors, aggregated by layer. Requires re-loading the adapter files to aggregate (`final_results/phase6_lora_deltas.csv` has per-module means but not per-layer decomposition). Consider adding a `phase6_lora_deltas_by_layer.csv` snapshot.

Plots are deferred to a follow-up commit. All placeholders in the draft name the exact CSV / adapter source and the claim the figure is supposed to support.

## Numbers cross-checked

- [x] All Section 5 table numbers verbatim from `final_results/phase{1,2,3,4,6}_*.csv`.
- [x] Phase 3 GPU-h-per-round computed from `run_metadata.csv` (per-round means across rounds 1-10).
- [ ] Double-check the `kl_reg_sft` per-round GPU-h of 0.051: `run_metadata.csv` shows rounds 1-10 in the 0.0508-0.0513 band; picking 0.051 as a rounded figure.
- [ ] Confirm `copr_gi` per-round GPU-h of 0.58: round-1 0.715, round-9 0.394, round-10 0.505; mean ~0.58. Verify rounding convention.

## Deferred analyses

- [ ] **Multi-seed replication** (seed in {7, 13, 42}). Current draft flags single-seed in Limitations.
- [ ] **Significance tests** (bootstrap over probe-level F1 for absorption / locality). Draft does not claim significance and uses SEs qualitatively.
- [ ] **Temporal-contrast metric** (Phase 4 drop): Gemini-generated probes were missing gold answers; either regenerate with a stricter template or drop the metric permanently (current: dropped).
- [ ] **`same_sector` locality stratum (n=1)**: broaden the probe set to 50-200 so the stratum becomes meaningful. Currently acknowledged in Limitations.
- [ ] **Phase 3 trajectory backfill** for `copr_anchored` compositional at rounds 6/7/9 is intentionally not done; compositional is by design only at round 10 for Phase 4. Leave as-is.
- [ ] **Scale sweep**: 10K-edit and 100K-edit batch runs to see whether `kl_reg_sft`'s dominance persists at larger batch scales.

## Draft polish

- [ ] Abstract 200 words target --- current draft ~260 words. Tighten by one or two sentences if journal-style is required.
- [ ] Verify word counts against the target in the plan:
  - Introduction ~800 (current ~640)
  - Related Work ~600 (current ~700)
  - Methods ~1200 (current ~900)
  - Experimental setup ~800 (current ~500)
  - Results ~1800 (current ~1800)
  - Discussion ~1000 (current ~900)
  - Limitations ~400 (current ~500)
  - Conclusion ~300 (current ~280)
- [ ] Decide whether to include the per-round trajectory tables in an appendix or keep them as CSV pointers only (current: CSV pointer + qualitative description).

## Reproducibility

- [x] Commit hash placeholder in the reproducibility statement resolves to `1c28ce0` at draft time.
- [ ] Re-resolve commit hash at final submission.
- [ ] Verify that `scripts/20_snapshot_results.py` regenerates all CSVs in `final_results/` from `outputs/` + `data/` on a clean checkout.
