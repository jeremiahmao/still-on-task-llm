# Clarity Review: draft_v2.md §3–§6

**Summary verdict:** The results narrative is significantly tighter than draft_v1 (cutting ~150 lines of redundant subsections), but metric definitions are still scattered and incomplete — a reviewer will hit "what is absorption F1?" in §4's table before the definition in §3 has told them *what probes are scored*. The QD-vs-fact confusion is real: the paper never explicitly states that absorption probes are direct QA/QD questions about injected facts, not query-decomposition outputs.

---

## 1. Metric definitions

All four metrics are "defined" in §3 ¶1 in a single compressed sentence. Problems:

- **Absorption F1.** §3 says "mean token-F1 across all paraphrased probes for all injected facts." Missing: (a) what probes? The reader doesn't learn until Appendix A that there are 5 injection templates — are absorption probes the *same* 5 formats, or a held-out probe set? (b) F1 of what against what? Token-level overlap between generation and gold `y_i`, presumably, but never stated. (c) "Mean across all paraphrased probes" — mean over K formats first, then mean over facts? Or flat mean? The parenthetical "`abs_mean_f1` in trajectory artifacts" is a code pointer, not a definition.

- **Preservation R@10.** §3 says "Recall@10 on the QD test split (n=104)." This is opaque. R@10 is a retrieval metric; this is a generation task. The reader must guess: is this "among the top-10 generated tokens, does the gold token appear?" Or is it something else entirely? The draft_v1 §4 says "Recall@10 on QD test split" with no further explanation either. This metric needs a one-sentence operational definition (e.g., "fraction of 104 held-out QD test prompts whose greedy generation contains the gold answer within the first 10 tokens" — or whatever it actually measures).

- **Locality F1.** §3 says "mean token-F1 on stratified locality probes (n=2000)." §7 (Limitations) reveals `same_sector` is n=1 (degenerate), `same_entity` n=182, `other_sector` n=1817. These strata are never defined in §3. What is a "locality probe"? A QA question about a *non-injected* fact? The stratification names (`same_entity`, `same_sector`, `other_sector`) suggest entity/sector proximity to injected facts, but this is never stated.

- **Worst-fact F1.** §4 table has a column `worst F1` with no definition anywhere. Context implies min over facts of per-fact mean-across-probes F1, but this should be stated once.

**Fix:** Add a "Metrics" subheading to §3 with 1–2 sentence operational definitions for each. Move the parenthetical code pointer to a footnote.

---

## 2. QD-vs-fact confusion

This is the core ambiguity. The model is task-tuned on QD (query decomposition for financial news). Facts are injected as triples. The paper never cleanly answers: **what do absorption probes look like?**

- §3 says facts are formatted via K=5 templates (QA, QD, declarative, instruction, narrative) for *training*. But are absorption *evaluation* probes the same templates? Different ones? The phrase "paraphrased probes" in §3's definition doesn't connect to the K=5 injection templates.
- The format gap is defined as `|F1_QA − F1_QD|` on n=50 behavioral probes. So the QD probe presumably asks a QD-formatted question about the injected fact — but QD's output format is sub-query decomposition, not a direct factual answer. How is token-F1 computed against a gold answer when the model is expected to output sub-queries? Is the gold target the sub-query containing the answer, or the direct answer? This is never stated.
- A reader's natural confusion: "If the model was trained to decompose queries, how does it answer fact questions?" The paper needs one sentence: e.g., "Absorption probes are direct questions about injected facts; the QD-formatted probe wraps the question in the QD system prompt and expects the answer embedded in the model's sub-query output."

**Fix:** One paragraph in §3 that shows an example absorption probe for QA format and QD format, with what the gold target is for each.

---

## 3. Metric noise floor and signal

- **Format gap (0.07–0.14 on n=50).** At n=50, a bootstrap 95% CI on token-F1 differences is wide. The paper uses this only to establish "the gap exists" (§4 draft_v1's §5.1), not to rank methods. Acceptable as a diagnostic, but the paper should state the CI or SE once.

- **Locality F1 (0.046–0.080, ~0.03 spread).** With n=2000 probes this has reasonable power, but the absolute values are tiny and the spread is small. The paper correctly doesn't over-claim ("locality comparable"), but the column adds visual clutter to §4's table without driving any conclusion. **Candidate for cutting from the main table** and relegating to a footnote ("locality F1 is stable across conditions; see Appendix").

- **Preservation R@10 (0.236–0.267, ~0.03 spread).** Same: the paper says "preserved, not improved." The column is defensive (showing nothing broke) but never drives a claim. Could be a footnote.

- **Compositional bridging recall (29–86% degradation).** §6 reports all methods are bad. This is useful as a scoping statement ("the synergy doesn't solve compositionality") but the 6-row table in draft_v1 §6.2 could be a single sentence in draft_v2 §6 with the range. draft_v2 handles this fine — it's already compressed to prose.

**Recommendation:** In the §4 main table, keep only abs F1, half-spread, and worst F1. Move preservation R@10 and locality F1 to a footnote or a supplementary "safety check" table. This sharpens the narrative to the metric that actually has signal.

---

## 4. Table redundancy

**§4's 5-row table** lists `naive_sft` abs F1 = 0.089. **§5.1's 6-row COPR table** lists `naive_sft` abs F1 = 0.088. The difference (seed variance) is unexplained and will confuse a careful reviewer.

**Fix:** (a) State explicitly that §5.1's numbers come from single-seed prior-phase runs while §4's are 2-seed means. (b) Better: remove `naive_sft` and `kl_reg_sft` rows from the §5.1 COPR table and say "baselines from Table 1" with a cross-reference. The COPR table only needs the 4 COPR rows + a reference line.

---

## 5. Numbers that contradict each other

- **"+0.286 compute-matched lift"** (Abstract) vs §4 body: §4 computes (d)−(b) = 0.411−0.125 = 0.286. This checks out. ✓

- **COPR locality "+47% over kl_reg_sft"** (§5.1): `copr_gold_injection_anchored` locality = 0.071, `kl_reg_sft` = 0.048. Ratio: 0.071/0.048 = 1.479, so +47.9%. Checks out. ✓ But "+47%" is a *relative* lift on a tiny absolute number (0.023 absolute). Reporting a relative percentage on values this small is misleading. Should say "+0.023 absolute (0.071 vs 0.048)."

- **No outright contradictions found**, but the Abstract says "worst-fact 0.385" while the table shows 0.385 for (d) — consistent. The trajectory-averaged numbers in §4 (0.346 for aug_kl_k1) don't appear in the Abstract, which only cites round-15 endpoint — fine but could be flagged.

---

## 6. Top 3 changes to improve clarity (ranked)

1. **Add a Metrics subsection to §3 with operational definitions.** One sentence per metric. Include what a probe looks like, what the gold target is, and what "F1" means (token-overlap F1 between generated text and gold answer). This is the single highest-impact fix — it resolves the author's own confusion about "how do we get changed fact accuracy from a QD model."

2. **Cut locality F1 and preservation R@10 from the §4 main table.** Move to a footnote: "Preservation R@10 and locality F1 are stable across conditions (0.24–0.27 and 0.05–0.08 respectively); full numbers in Appendix." This eliminates two columns that add noise without signal and lets the reader focus on the absorption F1 story.

3. **Deduplicate the baseline rows across §4 and §5.1.** Remove `naive_sft` and `kl_reg_sft` from the COPR table; cross-reference §4's Table 1. Add a one-line note explaining the 0.089/0.088 seed discrepancy if both must remain.
