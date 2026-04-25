# Honesty Audit: Novelty Claims in draft.md (commit 38ac789)

**Auditor:** ML Intern (automated literature audit)
**Date:** 2026-04-25
**Scope:** Five specific claims of contribution/novelty. For each: closest prior work, relevant quotes, and a verdict.

---

## Claim 1: The "Absorption-Integration Gap" Framing

**What the paper claims (Abstract / §1 / §5.6 / §6.5):** The paper claims to "expose" a universal *absorption-integration gap* — the phenomenon where a model absorbs an injected fact (measurable via in-format probes) but cannot use it under a different prompt format (behavioral format-transfer failure). The paper frames this as a diagnostic contribution.

### Prior work that already documents this phenomenon

#### Berglund et al. (2023), "The Reversal Curse" (2309.12288)

This is the cleanest prior demonstration of the exact phenomenon. Models trained on "A is B" achieve ~97% accuracy when prompted in the training format, but **0% accuracy** (no better than random baseline) when prompted in the reversed format "Who is B?"

> "If a model is trained on a sentence of the form 'A is B', it will not automatically generalize to the reverse direction 'B is A'. This is the Reversal Curse." (Abstract)

> "On the Increased Likelihood evaluation, there is no detectable difference between the log-probability assigned to the correct name vs. a random name." (Section 2, Experiment 1 Results)

> "When a model is updated on 'A is B', this gradient update may slightly alter the representation of A such that it contains information about B... It would make rational sense for this gradient update to also alter the representation of B to contain information about A. However, the gradient update is myopic." (Section 4, Discussion)

**This IS the absorption-integration gap:** the fact is absorbed (near-perfect same-direction accuracy) but behaviorally unavailable under format shift (reversed prompt → 0% accuracy). Berglund et al. measured it, quantified it, and proposed a mechanistic explanation ("myopic gradient update"). They just didn't call it by this name.

#### Allen-Zhu & Li (2023), "Physics of Language Models: Part 3.1, Knowledge Storage and Extraction" (2309.14316)

This is the deepest mechanistic account of the gap in the literature. Using controlled synthetic biographies, they show that 99+% token memorization (storage) coexists with **0% QA extraction accuracy** unless the training distribution includes augmented prompts:

> "A model pretrained to word-by-word memorize knowledge may never be fine-tuned to extract knowledge. …perfect BIO token memorization + perfect QA answers for half the people ⟹̸ correct QA answers for the other half. (knowledge extraction does not come for free)" (Section 4.1, Result 2)

> "Despite a 99+% first-token accuracy during pretraining, the model exhibits zero-zero QA accuracy on P_test for all finetuning parameters. This indicates that while the model can memorize BIO data token-by-token, it struggles to extract the underlying knowledge." (Section 4.1)

> "Without such augmentation, knowledge may be memorized but not extractable, leading to 0% accuracy, regardless of subsequent instruction fine-tuning." (Abstract)

The paper's **formal terminology** for this gap is "knowledge storage" vs. "knowledge extraction." They use probing to show *why* the gap exists: knowledge is stored attributed to positional/sequential tokens rather than to the entity name, making it structurally unavailable under QA-style prompts. This is a mechanistic account of the absorption-integration gap avant la lettre.

#### Allen-Zhu & Li (2023), "Physics of Language Models: Part 3.2, Knowledge Manipulation" (2309.14402)

Extends the storage/extraction distinction to manipulation tasks (classification, comparison, inverse search):

> "Our primary contribution is a controlled, synthetic experiment that confirms these weaknesses are inherent to language models: they cannot efficiently manipulate knowledge from pre-training data, **even when such knowledge is perfectly stored in the models**, despite adequate training and sufficient model size." (Abstract, emphasis added)

> "In conclusion, our findings underscore a fundamental limitation of generative language models: they cannot perform inverse knowledge search, period." (Section 5)

The phrase "even when such knowledge is perfectly stored" is semantically equivalent to "absorption succeeds but integration fails." This paper also documents same-fact/different-format failure: Result 1 shows that `"Which company and where did Anya work"` succeeds but `"Where and which company did Anya work"` fails — a word-order/format shift within the same semantic fact.

#### Zhong et al. (2023), "MQuAKE" (2305.14795)

Documents the gap at the multi-hop level. After successful single-hop injection (edit-wise success ~100%), multi-hop questions that require the injected fact as an intermediate step fail catastrophically:

> "While we find that current knowledge-editing approaches can recall edited facts accurately, they fail catastrophically on the constructed multi-hop questions." (Abstract)

> "Our findings suggest that current knowledge-editing techniques, **instead of integrating new facts into the model as new internal knowledge, are rather hard coding them** into the model by updating weights locally." (Section 4.2, emphasis added)

The phrase "instead of integrating… are rather hard coding" is nearly word-for-word the absorption-integration distinction. MQuAKE's failure mode is compositional (multi-hop), not pure prompt-template variation, but the underlying gap — successful injection does not produce genuinely integrated knowledge — is the same phenomenon.

#### Cohen et al. (2024), "Evaluating the Ripple Effects of Knowledge Editing" (2307.12976)

Documents the gap at the relational-propagation level. After a verified successful edit, logically entailed related facts are NOT updated. ROME achieves Logical Generalization = 20.2%, Compositionality I = 35.6%.

> "We evaluate prominent editing methods on RippleEdits, showing that they fail to introduce consistent changes in the model's knowledge." (Abstract)

**Important distinction:** RippleEdits' failure mode is **semantic/logical generalization** (editing Messi's team doesn't update what league Messi plays in), not **format/surface generalization** (same question, different prompt template). The paper explicitly notes it does not include paraphrase tests: "With RippleEdits focusing on the ripple effect of edits, it does not include tests, such as paraphrasing of the edit and subject specificity" (Section 6, Limitations). This is the weakest match to the draft's claimed gap among the four papers audited.

### What the paper does vs. doesn't add

**What the paper adds:**
1. A specific measurement in the **continual injection** setting (15 rounds of chained edits, not one-shot) under LoRA fine-tuning — this specific setting is not covered by any of the four papers above.
2. A quantified "format gap" metric (QA F1 − QD F1) that directly measures the magnitude of the phenomenon.
3. The specific term "absorption-integration gap" as a consolidating label.

**What the paper does NOT add:**
- The phenomenon itself. It is extensively documented under different names: "Reversal Curse" (Berglund), "storage vs. extraction" (Allen-Zhu & Li 3.1), "storage vs. manipulation" (Allen-Zhu & Li 3.2), "hard coding vs. integrating" (MQuAKE), "ripple effect failure" (Cohen).
- The insight that single-format training causes format-coupled knowledge. Allen-Zhu & Li 3.1 explicitly show that augmentation (multi-format training data) is necessary for extraction — the same insight the draft arrives at.

### The paper's own honesty on this point

Credit where due: §6.5 of the draft cites Berglund, Allen-Zhu & Li, Cohen, and Zhong explicitly and says "This phenomenon is well-attested in closely-related literatures." The abstract and §1 are less careful — phrases like "This paper exposes that gap" (abstract) and "localized the failure mode to a universal *absorption-integration gap*" (§1) read as discovery claims when the gap is well-documented.

### Verdict: **EXTENSION** (honest) / **Borderline DUPLICATE** (if framing language isn't tightened)

The phenomenon is not novel. Four independent prior works document it under different names and in different settings. What the paper adds is: (a) measurement in the specific continual-LoRA-injection setting, (b) a quantified format-gap metric, and (c) a consolidating term. This is a legitimate setting extension, but the abstract/introduction language ("This paper exposes that gap") overstates the contribution. The §6.5 discussion is appropriately honest; the abstract should match it.

**Recommendation:** Replace "exposes" with "characterizes" or "measures" in the abstract. Add a sentence like: "The gap between storage and behavioral availability is well-documented under different names (Berglund et al. 2023; Allen-Zhu & Li 2023, 2024; Zhong et al. 2023); our contribution is a quantified measurement of it under continual LoRA injection with a format-gap metric, not a discovery of the phenomenon itself."

---

## Claim 2: Geometric/Behavioral Disagreement (Hidden-State Proximity ≠ Behavioral Transfer)

**What the paper claims (§5.5–5.6, §6.5):** Gold-injection variants exhibit the *smallest* geometric hidden-state shift ratio (1.44–1.60, closest to the integration target of 1.0) but the *largest* behavioral format gap (+0.121 to +0.140). The paper calls this disagreement "itself a contribution" and recommends the field "complement geometric probes with behavioral cross-format probes when claiming integration."

### Prior work on this type of disagreement

#### "The Mirage of Model Editing" (Wan et al., 2025, 2502.11177)

Documents a ~57-point gap between teacher-forced evaluation metrics (~96%) and real-world autoregressive generation accuracy (38.5%) for knowledge editing methods. Teacher-forced evaluation operates on hidden states in a structurally similar way to cosine-similarity measures: it feeds ground-truth tokens and measures next-token probability, which reflects the geometry of the hidden-state space at each step:

> "Current editing methods perform substantially worse than previously reported (38.5% vs. ~96%). One key issue is the inappropriate use of teacher forcing in testing prevents error propagation by feeding ground truth tokens (inaccessible in real-world scenarios) as input."

This is the broadest documentation that "geometric-level" (teacher-forced / probability-based) evaluation metrics dramatically overstate behavioral success.

#### EtCon (Scialanga et al., 2025, 2512.04753)

Explicitly names a "parametric-behavioral gap":

> "Recent studies reveal a critical gap: while existing methods achieve high success under teacher-forcing evaluation, they can fail in realistic autoregressive generation. Without a knowledge consolidation stage, edited knowledge may remain superficially encoded at the parametric level, failing to propagate to the model's generation behavior."

This documents that parametric-level success (which is geometrically adjacent to cosine-based measurement) disagrees with behavioral generation — structurally the same pattern as the audited claim.

#### "Uncovering Overfitting in LLM Editing" / EVOKE (Wei et al., 2024, 2410.07819)

> "Edited models assign disproportionately high probabilities to the edit target, hindering the generalization of new knowledge in complex scenarios... common overfitting mitigation strategies are ineffective in knowledge editing."

The EVOKE benchmark explicitly measures the gap between Recall Tasks (P(target), geometrically adjacent to cosine probes) and Overfit Tasks (complex behavioral scenarios). FT-based methods — which resemble gold-injection fine-tuning — show the **most extreme** overfit: highest direct-recall metrics paired with worst complex-behavioral generalization.

#### Hernandez et al. (2023), "Linearity of Relation Decoding in Transformer Language Models" (2308.09124)

> "For a subset of relations, this computation is well-approximated by a single linear transformation on the subject representation... However, we also identify many cases in which **LM predictions capture relational knowledge accurately, but this knowledge is not linearly encoded in their representations**."

This is the most explicit prior documentation that geometric representation structure can decouple from behavioral output. A model can behave correctly on a relation without the relation being linearly encoded — and by symmetry, a hidden state that "looks right" geometrically may not produce the correct behavioral output.

#### Inference-Time Intervention (Li et al., 2023, 2306.03341)

From Appendix B.1:

> "Looking at the complicated geometry in Figure 2(B), we realize that whichever direction we choose, shifting the activation along it will inevitably cause other encoded information to be distorted. It's the complexity of Transformer architecture and its activation space that make pure geometric analysis insufficient."

Practitioner-level acknowledgment that activation-space geometry is too complex for simple cosine metrics to reliably predict behavioral outcomes.

#### ROME (Meng et al., 2022, 2202.05262) — implicit evidence

ROME's causal tracing uses geometric probes (Average Indirect Effect of restoring hidden states). The value vector v* is optimized specifically with random-prefix diversification to improve behavioral robustness under format change:

> "We select v* by optimizing across N random prefix texts to encourage robustness under differing contexts."

The fact that ROME's authors added prefix randomization explicitly to combat format sensitivity implies they already knew that geometrically optimal edits (without diversification) would fail behaviorally under format change. A gold-injection variant that bypasses this diversification — directly injecting a clean gold state — would have exactly the property the draft claims: minimal geometric shift but maximal behavioral format gap.

#### CoRE / CHED (2025, 2505.23026)

CoRE operates on hidden-state geometry (minimizing variance of hidden-state vectors) as a proxy for behavioral robustness:

> "CoRE...strengthens context robustness by minimizing context-sensitive variance in hidden states of the model for edited knowledge."

The paper's evaluation measures behavioral degradation under context change. The fact that CoRE *needs* to explicitly regularize hidden-state variance to get behavioral robustness implies prior methods' hidden-state geometry does NOT reliably predict behavioral outcomes.

### What is novel vs. what is known

**Known (individual components):**
- Parametric/teacher-forced metrics overstate behavioral success (Mirage, EtCon, EVOKE)
- Geometric representations can decouple from behavioral predictions (Hernandez et al.)
- Pure geometric analysis is insufficient for behavioral claims (ITI Appendix B.1)
- FT-based methods show worst behavioral generalization despite high direct-recall (EVOKE)

**Novel (the specific combination):**
- The draft uses cosine similarity / shift-ratio as the specific geometric probe (not teacher-forced probability or probing accuracy).
- The draft provides a method-by-method comparison showing **gold-injection as the outlier** — smallest geometric shift, largest behavioral gap — across six methods.
- The draft measures the disagreement specifically on **same-fact cross-format transfer** (QA → QD), not on general multi-hop or complex scenarios.
- No prior paper packages this as: "here are the geometric numbers, here are the behavioral numbers for the same facts and same methods, and they disagree in a specific, counterintuitive pattern (gold injection looks best geometrically but is worst behaviorally)."

### Verdict: **EXTENSION** — with a genuinely novel empirical observation

The *principle* that geometric/parametric metrics can disagree with behavioral metrics is documented (Mirage, EtCon, EVOKE, Hernandez). The *specific observation* — that gold-injection variants have the smallest geometric shift ratio but the largest behavioral format gap, creating a counterintuitive inversion pattern across methods — is not documented. No prior paper runs both geometric and behavioral probes on the same facts and methods and reports this specific inversion pattern.

The recommendation to "complement geometric probes with behavioral cross-format probes" is a reasonable practical suggestion that follows from the data, but is substantially anticipated by Wan et al. (2025) recommending against teacher-forced evaluation.

**Recommendation:** Tone down "this disagreement is itself a contribution" to something like: "This specific inversion pattern — geometric integration proxies being worst where behavioral transfer is worst — has not been previously reported, though the general principle that geometric metrics can overstate behavioral success is documented (Wan et al. 2025; Scialanga et al. 2025; Wei et al. 2024)." Cite Mirage and EtCon.

---

## Claim 3: Cross-Format Consistency Regularization (V-REx on Prompt Formats)

**What the paper claims (§3.4, §5.7, §8):** The paper applies V-REx (Krueger et al. 2021) to prompt formats, treating each format as a V-REx "environment" and adding `μ * Var_k[CE_k]` to the SFT loss. The paper explicitly states "the mathematical novelty of this paper is therefore zero" and claims only application novelty: V-REx applied to continual knowledge injection via LoRA SFT, not image OOD classification.

### Closest prior work checked

#### PAFT (Wei et al., 2025, 2502.12859)

**No variance penalty.** PAFT generates diverse synthetic prompts via LLM ensemble, then randomly samples one prompt per step for standard CE training. The loss function is standard SFT with stochastic prompt sampling — purely data-side augmentation. No cross-format regularizer of any kind.

#### UIT (2307.15504)

**No regularization.** UIT converts all formats to a unified target format via GPT-3.5 before training, then trains with standard SFT on the unified data. The approach eliminates format diversity entirely rather than regularizing across formats.

#### R-Drop (Wu et al., 2021, 2106.14448)

**Correct that it operates on dropout variation, not format variation.** No paper in R-Drop's citation graph applies R-Drop's KL consistency objective to different prompt formats (as opposed to different dropout masks on the same input).

#### CoRE / CHED (2025, 2505.23026)

**Structurally closest — but operates in a fundamentally different space.** CoRE minimizes pairwise L2 distance between hidden-state vectors across prefix contexts:

```
L_prefix = (λ/LD) Σ_{ℓ∈L} Σ_{i<j} ||h_i^ℓ − h_j^ℓ||²
```

Key differences from V-REx on prompt formats:

| Dimension | CoRE | FI-SFT (V-REx on formats) |
|-----------|------|---------------------------|
| What varies | Prefix contexts before a fixed prompt | Prompt format templates (QA vs. QD) |
| What's penalized | Pairwise L2 in hidden-state space | Variance of scalar CE losses |
| Optimization target | A single value vector v* (MEMIT paradigm) | Model parameters θ (SFT paradigm) |
| Training paradigm | Locate-then-edit | Full gradient SFT |

The conceptual motivation overlaps (both want format/context-invariant representations), but the implementation and training paradigm are distinct. The draft paper already distinguishes from CoRE in §3.4: "CoRE (2025) applies a variance penalty in MEMIT-style editing but over hidden-state value vectors, not output losses."

#### LoRA-JS (Qiang et al., 2024, cited in 2508.11383 "When Punctuation Matters")

**This is the closest unacknowledged neighbor.** LoRA-JS adds a Jensen-Shannon divergence consistency loss between outputs of different prompt variants:

> "We add a Jensen-Shannon consistency loss between outputs of different prompt variants, encouraging format-invariant predictions. The total loss combines standard cross-entropy with a divergence term."

Differences from V-REx on prompt formats:
- JS divergence penalizes **output distribution divergence** between format pairs
- V-REx penalizes **scalar variance of CE losses** across format environments
- LoRA-JS is applied to general instruction-following robustness, not knowledge injection
- LoRA-JS does not cite V-REx or frame formats as "environments"

LoRA-JS's result in 2508.11383 was actually **negative**: "higher spread and lower accuracy for all models" (Appendix A), unlike FI-SFT's positive result.

#### MetaICL, LIMA, VAT on prompts, IRM/Group DRO on prompt formats

No evidence of any of these applying cross-format consistency regularization to prompt formats in knowledge editing. LIMA is pure data curation. MetaICL is meta-training with diverse tasks but no explicit regularizer. No paper found applying VAT, IRM, or Group DRO with prompt formats as environments in an NLP/LLM knowledge-editing setting.

### What is novel vs. what is known

**Known:**
- V-REx loss formulation (Krueger et al. 2021) — verbatim
- CoRE applies a variance-like penalty in knowledge editing, but on hidden states in MEMIT paradigm
- LoRA-JS applies JS divergence across prompt formats in SFT (Qiang et al. 2024)
- PAFT applies format diversification without a penalty term

**Novel (the specific three-way combination):**
- V-REx formulation (`Var_k[CE_k]`) + prompt formats as environments + knowledge injection/editing via LoRA SFT
- No prior paper combines all three elements

### Verdict: **GENUINELY NOVEL** (application novelty, as claimed)

The paper's own assessment is correct: the mathematical novelty is zero, but no prior work applies V-REx to prompt formats in knowledge injection. The closest neighbors are CoRE (variance-like penalty in editing, but on hidden states not CE) and LoRA-JS (cross-format divergence in SFT, but JS not V-REx and not in knowledge editing). The three-way combination is unoccupied.

**Recommendation:** The paper should cite LoRA-JS (Qiang et al. 2024, referenced in 2508.11383 "When Punctuation Matters") as the closest SFT-level cross-format consistency regularizer. Distinguish on (a) JS divergence vs. variance of CE, (b) general instruction-following vs. knowledge injection, and (c) LoRA-JS's negative result vs. FI-SFT's positive result. This strengthens the paper's argument that the V-REx formulation (scalar variance) matters vs. alternative consistency penalties.

---

## Claim 4: "Format Diversity Without Regularization Actively Hurts"

**What the paper claims (§5.7, §6.5, Phase 8b/8d):** Mixed-format SFT (training on QA + QD formats without variance penalty) is *worse* than single-format SFT on behavioral format transfer. At one round: format gap 0.106 vs 0.082. At 15 rounds: behavioral QD F1 0.067 vs 0.072. The paper calls this "the knowledge-injection analog of a task-level finding by UIT, 2023."

### What UIT (2307.15504) actually found

UIT's core finding is about instruction-tuning across task types (NLP benchmarks on T5-LM-xl), not knowledge injection:

> "Without format unification, src+same performs better than src+diff, which indicates that **increasing task diversity may be inferior without format consistency**." (Section 7)

UIT's numbers: raw mixed-format training = EM 29.3, unified format training = EM 30.8. The difference is ~1.5 EM points — modest. More importantly:

- UIT measures **task generalization accuracy** (EM on NLP benchmarks), not **behavioral format transfer** (a "format gap" metric).
- UIT's formats are structural template-level variations across task types (PromptSource/FLAN/CrossFit/Ni-v2), not semantic format types within a single knowledge-injection task.
- UIT's solution is **format unification** (convert all data to one format via GPT-3.5), not a loss-level regularizer.
- UIT never directly compares mixed-format vs. single-format in a knowledge injection context.

### Additional relevant prior work

#### PIT — Pre-Instruction-Tuning (2402.12847)

This is closer to the draft's finding than UIT:

> "We found that LLMs struggle to adhere to the QA format after training on raw documents for multiple epochs. Therefore, we include a small set of QA pairs (64) during continued pre-training to prevent LLMs from forgetting the QA format."

PIT shows: "mix all data" (QA + documents jointly) = 39.4 EM vs. PIT (sequential QA → documents) = 45.4 EM — a 6-point gap from naive format mixing during knowledge acquisition. This is about format interference during knowledge injection (continued pre-training), which is closer to the draft's setting than UIT's instruction-tuning setting.

#### Multi-task negative transfer literature

The general principle that mixing diverse tasks can cause negative transfer is well-documented:
- TaskWeb (2305.13256): pairwise task transfer analysis shows many task pairs hurt each other.
- ForkMerge (2301.12618): negative transfer from auxiliary tasks in MTL.

However, none of these study *format-as-task* in a knowledge injection context. They treat tasks as semantically distinct NLP tasks (NER vs. QA vs. summarization), not as format variants of the same knowledge injection objective.

### What is novel vs. what is known

**Known:**
- Format inconsistency hurts instruction tuning (UIT, ~1.5 EM)
- Naive format mixing during knowledge acquisition hurts (PIT, 6 EM)
- Multi-task mixing can cause negative transfer (TaskWeb, ForkMerge)

**Novel (the specific finding):**
1. The **knowledge-injection-specific setting** — UIT never tests format mixing during factual knowledge injection from documents.
2. The **"format gap" as the outcome metric** — no prior work measures behavioral format-transfer degradation as a distinct metric separate from task accuracy.
3. The **"mixed < single-format" ordering** for behavioral format transfer specifically — UIT shows mixed is slightly worse than *unified*, but never directly compares mixed vs. single-format in knowledge injection.
4. The specific magnitude (format gap 0.106 vs. 0.082, a >25% relative increase) in the continual-LoRA setting.

### Verdict: **EXTENSION** — from task-level instruction tuning to knowledge injection

The general principle (format mixing without consistency can hurt) is documented in UIT and PIT. The draft's finding extends this to: (a) knowledge injection specifically, (b) LoRA-based continual editing, (c) with a quantified "format gap" metric, and (d) with a regularization-based solution (variance penalty) rather than a data-normalization solution (UIT) or a sequencing solution (PIT).

The paper's own characterization — "the knowledge-injection analog of a task-level finding by UIT, 2023" — is honest and accurate. This is an extension, not a duplication and not a wholly independent discovery.

**Recommendation:** Also cite PIT (2402.12847) as closer prior work to the knowledge-injection setting than UIT. The paper should say: "UIT (2023) showed format inconsistency hurts instruction-tuning generalization by ~1.5 EM; PIT (2024) showed naive format mixing during continued pre-training hurts by ~6 EM. Our finding extends this to LoRA-based continual knowledge injection, where the degradation manifests as a 25% relative increase in behavioral format gap."

---

## Claim 5: COPR Negative Transfer to Knowledge Editing

**What the paper claims (§1, §3.2–3.3, §5.1–5.2, §6.0–6.2):** COPR (Zhang et al. 2025) does not transfer from continual alignment to continual knowledge editing. KL-regularized SFT leads the COPR family by 36–40% on absorption at batch scale and 9-3 (2 ties) across 14 sequential rounds. `copr_anchored` (anchoring without gold injection) collapses below the no-update baseline (F1 0.066). Gold injection collapses COPR toward SFT at 10–12x compute.

### What the COPR paper actually is

COPR exists in two versions (Zhang, Gui, et al., Harbin Institute of Technology):
- Original: 2310.15694 (Oct 2023)
- Revised: 2402.14228 (Feb 2024), "Continual Human Preference Learning via Optimal Policy Regularization"

**COPR was designed exclusively for continual alignment**, not knowledge editing. It regularizes the current policy against a historically optimal policy to prevent catastrophic forgetting of previous preference tasks. Its evaluation benchmarks are alignment benchmarks (MT-Bench style helpfulness/harmlessness), never factual QA or knowledge injection.

### Has anyone else tested COPR on knowledge editing?

**No.** COPR 2402.14228 has 8 citations total. All are surveys or alignment papers with zero knowledge-editing experiments. The draft paper is the **first** to evaluate COPR on knowledge editing.

### Has anyone reported similar preference-based methods failing on knowledge editing?

#### KDPO (Rozner et al., 2024, 2406.09920) — the closest comparison point

KDPO is a DPO variant **redesigned from scratch** for knowledge editing, with three specific adaptations: online negative generation (the model generates its current wrong belief as the negative), teacher-forced negative construction, and continual reference update. At 500 sequential edits on LLaMA3-8B, KDPO achieves Edit Success 87.4%, Portability 47.1%, Locality 35.2% — significantly outperforming vanilla DPO, which degrades on locality at scale.

**Key implication:** Even KDPO, which was purpose-built for knowledge editing, required substantial redesign from vanilla DPO. That COPR — which was designed for alignment, not editing — fails on editing is consistent with the broader pattern: preference-based methods need fundamental structural adaptation to work on knowledge injection.

#### OVERTONE (Liu et al., 2025, 2502.00602) — DPO challenges in KE

> "As highlighted in Rozner et al. (2024), another challenge of applying DPO to KE is that determining win-loss data pairs can be unstraightforward in KE."

This directly validates the draft paper's K-sample-all-wrong pathology diagnosis: COPR's self-sampled candidates for a novel fact are all wrong because the fact has near-zero probability under the pre-update policy. The win-loss definition problem is well-known in the DPO-for-editing literature but has never been articulated in COPR-specific terms.

#### EtCon (Li et al., 2025, 2512.04753) — SFT for injection, RL for consolidation

Uses GRPO (not DPO/preference ranking) for knowledge consolidation, but SFT for the actual knowledge injection step. This is consistent with the finding that preference-based objectives don't help injection.

### What is novel vs. what is known

**Known (indirectly):**
- Preference-based methods (DPO) struggle with knowledge editing without structural adaptation (KDPO, OVERTONE)
- Win-loss pair definition is "unstraightforward" for knowledge editing (OVERTONE citing KDPO)
- SFT-based methods are the default for the actual injection step (EtCon)

**Novel:**
1. **No prior paper has tested COPR on knowledge editing.** This is the first cross-domain evaluation.
2. **No prior paper has tested COPR variants** (gold-injected, anchored, gold-injected-anchored) in any knowledge-editing setting.
3. **The anchored-COPR-without-gold collapse** (absorption F1 0.066, below no-update baseline) is a specific new negative result not reported anywhere.
4. **The gold-injection-collapses-COPR-toward-SFT diagnosis** — backed by LoRA-delta analysis (Phase 6) and hidden-state geometry (Phase 7) — is a novel mechanistic analysis.
5. **The K-sample-all-wrong pathology** as a specific COPR failure mode for knowledge editing has not been previously described (though the general win-loss problem in DPO-for-editing has been noted by OVERTONE/KDPO).

### Verdict: **GENUINELY NOVEL**

The negative result is real and novel. No one has tested COPR on knowledge editing. The variants are novel. The mechanistic analysis of why COPR fails (K-sample-all-wrong, gold-injection collapsing toward SFT, anchored-without-gold collapse) is novel. The broader principle (preference-based methods struggle on knowledge injection) is anticipated by the KDPO/OVERTONE literature but has never been demonstrated for COPR specifically, and the draft provides significantly more mechanistic detail than prior work.

**Recommendation:** Cite KDPO (2406.09920) and OVERTONE (2502.00602) as prior work documenting structural challenges of preference-based methods for knowledge editing. Frame the COPR negative result as: "KDPO (Rozner et al. 2024) showed that DPO requires fundamental redesign for knowledge editing; OVERTONE (Liu et al. 2025) noted that win-loss pair definition is 'unstraightforward' in this setting. Our COPR experiments provide the first direct evidence that a reference-anchored MSE-ranking objective designed for continual alignment does not transfer to continual knowledge injection, and diagnose the specific failure modes: K-sample-all-wrong pathology, gold-injection-collapses-to-SFT, and anchored-without-gold collapse."

---

## Summary Table

| # | Claim | Verdict | Explanation |
|---|-------|---------|-------------|
| 1 | Absorption-integration gap framing | **EXTENSION** (borderline DUPLICATE on the phenomenon) | The phenomenon is documented by Berglund 2023, Allen-Zhu & Li 2023/2024, Zhong 2023 under different names. The paper adds: continual-LoRA setting, format-gap metric, consolidating term. The §6.5 discussion is honest; the abstract overstates. |
| 2 | Geometric/behavioral disagreement | **EXTENSION** with a genuinely novel empirical observation | The principle (geometric metrics overstate behavioral success) is documented by Wan 2025, Wei 2024, Hernandez 2023. The specific inversion pattern (gold injection = smallest geometric shift but largest behavioral gap) across six methods is new. |
| 3 | V-REx on prompt formats (FI-SFT) | **GENUINELY NOVEL** (application novelty) | No prior work applies V-REx formulation to prompt formats in knowledge injection. Closest: CoRE (hidden-state variance in MEMIT), LoRA-JS (JS divergence across prompts in SFT). The paper's own claim of zero mathematical novelty is accurate and honest. |
| 4 | Format diversity without regularization hurts | **EXTENSION** from instruction tuning to knowledge injection | UIT (2023) showed ~1.5 EM degradation from mixed formats in instruction tuning. PIT (2024) showed ~6 EM degradation from format mixing during continued pre-training. The draft extends to LoRA-based continual knowledge injection with a format-gap metric. The paper's own characterization ("knowledge-injection analog") is accurate. |
| 5 | COPR negative transfer | **GENUINELY NOVEL** | First test of COPR on knowledge editing. Novel variants. Novel mechanistic diagnosis. Consistent with but not duplicating KDPO/OVERTONE observations about preference-method challenges in editing. |

---

## Overall Assessment

The paper is more honest than most. It explicitly states "the mathematical novelty of this paper is therefore zero" and correctly identifies V-REx as the source of its loss function. It cites the relevant prior work on the Reversal Curse and storage/extraction distinction. The limitations section is unusually thorough.

**Where honesty could be improved:**
1. The abstract's "This paper exposes that gap" overstates Claim 1. The gap is well-documented; the paper *measures* it in a new setting.
2. The geometric/behavioral disagreement claim (Claim 2) should cite Wan et al. 2025 ("Mirage of Model Editing") and EtCon (2025) as documenting the general principle, while claiming the specific inversion pattern as novel.
3. Claim 3 should cite LoRA-JS (Qiang et al. 2024) as the closest SFT-level cross-format consistency regularizer.
4. Claim 4 should cite PIT (2402.12847) alongside UIT.
5. Claim 5 should cite KDPO (2406.09920) and OVERTONE (2502.00602) for the broader context of preference-method challenges in editing.

**What the paper gets right that most papers don't:**
- Transparent about zero mathematical novelty
- Reports negative results (COPR failure, anchored collapse) prominently
- Self-corrects from an earlier headline (round-10 COPR advantage that didn't replicate)
- Flags the QD template content-leak confound in Limitations
- Reports standard errors and warns against over-interpreting small differences

The contribution is real but bounded: two genuinely novel results (COPR negative transfer, V-REx on prompt formats), two legitimate setting extensions (absorption-integration gap measurement, format-diversity-hurts), and one extension with a novel empirical detail (geometric/behavioral inversion pattern).
