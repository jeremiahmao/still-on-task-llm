Verdict: The main synergy result is understandable, but the metric layer is still too compressed; several columns look like unexplained dashboard artifacts rather than claims. The biggest clarity risk is QD-vs-fact: the paper says facts are rendered in QD format, but not precisely what is scored, so “fact accuracy from a QD model” remains confusing.

1. **Metric definitions.**
- **Absorption F1:** Defined in §3 line 27 as “mean token-F1 across all paraphrased probes for all injected facts.” Better than before, but incomplete: it does not say which probe formats are included at eval (all five F_k? QA+QD only? held-out paraphrases?), whether the target is object string `y_i`, and whether averaging is macro over facts then probes or flat over probe instances.
- **Preservation R@10:** Defined only as “Recall@10 on the QD test split (n=104)” (§3). Not enough. For a generative QD task, R@10 needs an explanation: are 10 generated sub-queries/items matched against gold decompositions? Is it retrieval over candidate facts? What counts as recalled?
- **Locality F1:** §3 says “stratified locality probes (n=2000)” but does not define strata. §6/limitations mentions `same_sector` degeneracy, but §3 never defines `same_entity`, `same_sector`, `other_sector`, etc. Reviewer cannot know what locality means.
- **Worst-fact F1:** Appears in §4 table but is not defined in §3. Presumably min over injected facts of that fact’s mean probe F1 at round 15; say that explicitly.
- **Format gap:** Defined in §3 as `|F1_QA − F1_QD|` on n=50. This is mostly clear, but should say these are behavioral probes over the same facts and what QA/QD targets are.

2. **QD-vs-fact confusion.**
Real ambiguity. §3 says base model is task-tuned for QD, while injection triples are rendered as full chat sequences including QA, QD, declarative, instruction, narrative. That partly answers it, but the scoring side is unclear. Are absorption probes direct QA probes asking for `object`? QD-formatted probes whose expected output includes a decomposition containing `object`? Both? Per fact? The paper should add one concrete example: triple → QA training target, QD training target, QA eval probe, QD eval probe, scored span.

3. **Metric noise floor / signal.**
- **Format gap 0.07–0.14 at n=50:** useful as motivation/methodological warning, not as a fine-grained outcome. Keep one short diagnostic paragraph; don’t foreground small differences.
- **Locality F1 0.046–0.080:** weak signal and low absolute values. Since it does not show a tradeoff, move out of headline table or label “sanity check: no obvious locality collapse.”
- **Preservation R@10 0.236–0.267:** same: keep as a guardrail only. Do not invite interpretation.
- **Compositional bridging:** §6 is useful because it states “unsolved and not evaluated for Plan B.” Keep, but shorten; the 29–86% spread is less actionable than the qualitative conclusion.

4. **Table redundancy.**
Yes, confusing. §4 and §5.1 repeat `naive_sft`/`kl_reg_sft` with 0.089 vs 0.088. Fix by making §4 the canonical 2×2 table and §5.1 the COPR-only table with a single “KL-reg SFT reference” row, footnoted as prior-phase single seed. Or explicitly label §4 values as 2-seed Plan B means and §5.1 values as phase3 single-seed artifacts.

5. **Numbers / contradictions.**
- “+0.286 compute-matched lift” is consistent with §4: 0.411−0.125.
- “~10× asymmetry” is approximate: 0.286/0.029 = 9.86, fine.
- COPR locality “+47% over kl_reg_sft” conflicts with table math depending denominator: 0.071 vs 0.048 is +48% relative, +0.023 absolute. Say “+0.023 absolute / +48% relative” or drop.
- §5.2 says V-REx QD F1 0.0857 vs `kl_reg_sft` 0.0715, while earlier sections mostly discuss absorption F1; label this as QD-format behavioral F1 to avoid cross-metric comparison.

6. **Top 3 fixes.**
1. Add a compact “Metrics” boxed paragraph/table in §3 with exact eval set, unit of averaging, target, and interpretation for each metric; remove low-signal metrics from claim language.
2. Add one worked QD-vs-fact example showing how a fact triple is rendered/trained and how QA/QD probes are scored.
3. Make §4 the only canonical 2×2 result table; demote preservation/locality to guardrails and consolidate COPR comparison to avoid duplicated baseline numbers.