# Iteration 3 Verdict: Scrutiny of Fish-DSAE-NSI Hybrid

**Prior verdict:** (d) Implement "Fish-on-ΔW with null-space init + DSAE Lite KL" hybrid.
**This verdict:** The hybrid does not survive scrutiny. Fish is the wrong tool for prompt-format invariance. The null-space init at r=8 is inoperable. The simpler method is also the correct one.

---

## Section 1: Does Fish Actually Work for Prompt-Format Invariance?

**No. The mechanism is vacuous for paraphrastic surface variations.**

Fish (Shi et al., 2104.09937) maximizes the inter-domain gradient inner product (IDGM): $\mathcal{L}_{\text{idgm}} = \mathcal{L}_{\text{erm}} - \gamma \cdot \frac{2}{S(S-1)}\sum_{i \neq j} G_i \cdot G_j$. The penalty rewards gradient *alignment* across domains and penalizes *disagreement*. This is useful only when, absent the penalty, per-domain gradients would conflict — i.e., when ERM would exploit a spurious feature that correlates with the label in one domain but anti-correlates (or is absent) in another.

Prompt formats are not such domains. "What is the capital of France?" and "Please tell me: the capital of France is ___" teach the same $P(y|x)$ over the same knowledge triple. Both formats produce gradients that push model weights in approximately the same direction — toward associating "France" with "Paris." There is no spurious format-specific cue that would produce gradient *disagreement* between format A and format B. The GIP term $G_A \cdot G_B$ is already large (gradients already aligned), so the IDGM penalty adds zero useful constraint beyond ERM.

Shi et al.'s own ablation (§4.4, Table 5) confirms this directly: Fish with **random** minibatch groupings (no domain structure) collapses to ERM-level performance. They write: *"the effectiveness of our algorithm largely benefited from the domain grouping strategy, and maximising the gradient inner product between random batches of data does not achieve the same domain generalization performance."* Paraphrastic prompt formats, which carry no systematic gradient divergence, are functionally equivalent to random groupings from Fish's perspective.

**The K=2 "success case" does not exist.** The prior review's verdict (d) assumed Fish works at K=2, citing Shi et al.'s CdSprites results. Verified against the paper: Fish at K=2 domains on CdSprites achieves ~50% test accuracy — *random chance*. Fish requires N≥10 domains for meaningful generalization on that benchmark, reaching ~100% only at N≥20. Theorem 3.1 uses S=2 as a *pedagogical simplification* in the proof's Taylor expansion, not as an empirical sufficiency claim. The theorem proves only that Fish's update direction approximates the IDGM gradient — it does not prove GIP maximization identifies invariant features at any particular number of domains. That is an empirical question, and the empirical answer at K=2 is negative.

**Consequence for the hybrid:** The entire rationale for including Fish inner loops — "GIP maximization across prompt formats extracts format-invariant gradient components" — collapses. Prompt formats don't produce the gradient disagreement Fish needs. Even if they did, K=2–5 formats is far too few. Fish is designed for image-style domain generalization with many statistically distinct environments, not for paraphrastic text variation.

---

## Section 2: Frozen-A Null-Space + Fish — Do They Compose?

**Moot given Section 1, but the null-space component has its own fatal problem at r=8.**

KORE (Zhao et al., 2510.19316, §3.2) computes $C = XX^T$ from calibration activations, takes the bottom-$r$ singular vectors of $C$ as an approximate null space, initializes $A$ in this subspace, and **hard-freezes $A$ during training** — only $B$ is learned. The formal guarantee (Appendix C, Theorem 2) is $AX \approx 0$ for pre-training inputs $X$, which holds only if $A$ remains in the null space. Freezing is non-negotiable for the theorem.

**The rank arithmetic kills this at r=8.** KORE's own experiments use $r \in \{235, 256\}$ for 7B models (LLaVA-v1.5) and achieve their best knowledge-adaptation scores at these ranks. At $r = 8$ with $d_\text{in} = 3584$ (4B model), the frozen $A$ spans 8 directions out of 3584 — 0.22% of the weight space — and these are the 8 directions where pre-training activations have the *least* energy. This is simultaneously the most restrictive possible capacity constraint and the least informative subspace for new fact storage.

KORE never tested $r = 8$. Their lowest reported rank is $r = 64$, and even there performance degrades substantially compared to $r = 235$. Extrapolating to $r = 8$ with frozen $A$: the LoRA update $\Delta W = BA$ is rank-8, confined to the bottom-8 activation directions, with only $B$ trainable. The per-layer trainable parameter count is $8 \times d_\text{out}$ (B only) versus $8 \times (d_\text{in} + d_\text{out})$ (standard LoRA). This is not enough capacity for meaningful fact injection across 200 facts per round, let alone 15 rounds of sequential chaining.

**The "approximate null space" problem.** With 256+ calibration samples at sequence length ≥32, the calibration set produces $\geq 8192$ activation vectors, making $C$ effectively full-rank ($\text{rank}(C) = d_\text{in} = 3584$). The true null space is *empty*. KORE's "null space" is really "the 8 directions of smallest covariance eigenvalue" — an approximation whose quality depends entirely on the spectral gap between $\sigma_{3576}$ and $\sigma_{3577}$. KORE does not analyze this gap. If the spectrum decays slowly (plausible for a well-trained 4B model whose representations are distributed across all dimensions), $\sigma_{3576} \approx \sigma_{3577}$, and the "null space" is not meaningfully more null than the adjacent directions. The preservation guarantee becomes noise.

**Verdict on composition:** Even setting aside Fish's inapplicability, the null-space initialization at $r=8$ is too constrained for fact injection and too approximate for reliable preservation. To use KORE meaningfully, you'd need $r \geq 64$, which is a different experiment design entirely (and requires confirming the A10G memory budget handles a 64-rank LoRA with frozen A and trainable B on a 4B model — feasible but untested).

---

## Section 3: DSAE-Lite KL + Fish — Interaction Effects

**This section is mooted by Section 1 (Fish is the wrong tool), but the DSAE-Lite KL component is sound on its own.**

The question of whether injection and preservation should use the same or different format pools is answered by Allen-Zhu & Li (2309.14316, §4–5): format diversity on the *injection* side is what drives format-invariant encoding. The probing results (§5) show that at K=5 diverse wordings, the model stores facts as linear features at the entity-name token — format-invariant by construction. Separately, diverse KL preservation (DSAE Lite's ingredient #2) monitors that the model hasn't forgotten old capabilities *across formats*. These are independent mechanisms: injection diversity creates format-invariant new knowledge; preservation diversity detects format-selective forgetting. Using overlapping format pools is fine — the two losses operate on different data (new facts vs. preservation prompts) and serve different purposes (learning vs. monitoring).

The compute cost without Fish is straightforward: K=5 augmented SFT is ~5× the data volume (5 format renderings per fact), and K=5 KL preservation adds 5 forward passes per preservation example per step. Total overhead versus single-format SFT: ~5–6× per step. At n=200 facts/round with standard LoRA, one round takes ~3 minutes on A10G. With 5× augmentation: ~15 min/round. For 15 rounds × 3 seeds: $15 \times 15 \times 3 / 60 \approx 11$ GPU-hours. Add baselines (~3 hours) + evaluation (~2 hours) = ~16 GPU-hours. This leaves 14 GPU-hours for debugging and additional seeds — a comfortable margin that the Fish hybrid never had.

---

## Section 4: The Smallest Preregistered Experiment

### Method ablation conditions (5-way)

| Condition | Injection | Preservation | Novel component |
|-----------|-----------|-------------|-----------------|
| (a) `baseline_sft` | SFT, K=1 format | No KL | None (control) |
| (b) `aug_sft_k5` | SFT, K=5 formats | No KL | Allen-Zhu augmentation |
| (c) `kl_sft_k1` | SFT, K=1 format | KL, K=1 format | Standard KL preservation |
| (d) `aug_kl_k1` | SFT, K=5 formats | KL, K=1 format | Augmented injection only |
| (e) `dsae_lite` | SFT, K=5 formats | KL, K=5 formats | Full DSAE Lite |

The 5-way design separates injection diversity (b vs. a), preservation KL (c vs. a), their combination (d), and the novel symmetric augmentation of the KL term (e vs. d). The contrast (e) vs. (d) isolates DSAE Lite's genuinely novel contribution: does format-diverse KL preservation add value beyond format-diverse injection alone?

### Seeds and budget

At ~16 GPU-hours for 5 conditions × 3 seeds × 15 rounds + eval, this fits within 30 GPU-hours with margin. Running 5 seeds per condition would cost ~27 GPU-hours — feasible if the 3-seed pilot shows promising results and no debugging is needed. **Start with 3 seeds. Upgrade to 5 only if condition (e) shows a signal worth powering up.**

### Per-round specification

- n = 200 facts/round, 15 rounds chained sequential (cumulative injection, re-evaluating after each round).
- K = 5 format templates: QA, cloze, declarative, instruction-response, multi-sentence narrative. Constructed from existing templates.
- Preservation set: 100 task-relevant prompts (held fixed across rounds), each rendered in K=5 formats for condition (e), K=1 for conditions (c) and (d).
- KL reference: frozen copy of the base model at round 0.

### Concrete decision rules

**Positive result — "DSAE Lite works":**
- Condition (e) achieves QD F1 ≥ (d) + 2pp, statistically significant across 3 seeds ($p < 0.05$, paired $t$-test per round, Bonferroni-corrected across rounds).
- Condition (e) achieves format gap $|\text{QA F1} - \text{QD F1}| \leq 0.04$, and this gap is smaller than (d)'s gap by ≥ 1.5pp.
- Task preservation: condition (e) accuracy within 1pp of baseline (a) on held-out task benchmark.

**Negative result — pivot:**
- (e) ≈ (d) (within 1pp) on both QD F1 and format gap → the diverse KL term adds nothing beyond diverse injection. Write up as: "format-diverse injection is the active ingredient; format-diverse preservation monitoring is redundant." This is still a useful finding.
- (b) ≈ (a) → augmentation at K=5 doesn't help at this scale (contradicts Allen-Zhu, notable finding). Investigate whether the formats are too similar or the model is too small.
- All conditions within 2pp of each other → no intervention helps at n=200/round on a 4B model. This is the clean negative result: format invariance at this scale/budget is not achievable by data-side methods alone.

### Pre-registered early stopping criteria

- **Kill after round 3** if condition (e) shows *worse* QD F1 than condition (a) by > 3pp. This indicates the KL term is actively interfering with injection (the preservation constraint is too tight). Debug the KL coefficient before continuing.
- **Kill after round 5** if conditions (b), (d), (e) are indistinguishable (all within 1pp on QD F1). Five rounds is enough to detect a trend; if there is no separation by round 5, there won't be at round 15. Switch to a diagnostic investigation (probe the internal representations) rather than running 10 more rounds of null results.
- **Kill immediately** if any condition shows task preservation degradation > 5pp after round 1. The KL coefficient or learning rate needs adjustment.

---

## Section 5: Are There Simpler Alternatives We're Overlooking?

**Yes. The simplest thing is the right thing.**

The evidence chain is now clear:

1. **Allen-Zhu (2309.14316):** K=5 diverse wordings → 9.7% → 96.6% OOD QA accuracy. The format invariance comes from data distribution coverage, not optimization objectives. Probing confirms the model encodes facts as format-invariant linear features at the entity-name token when trained with diverse formats.

2. **PAFT (2502.12859, §5.3):** Accuracy is insensitive to prompt-switching frequency (K=1 vs. K=8 steps/prompt: ~87% either way). The scheduling of format exposure — which is exactly what Fish-style inner loops control — is irrelevant. Pool diversity is all that matters.

3. **Lin et al. (2510.00237):** Prompt-diverse SFT matches or beats RL baselines on generalization benchmarks. The "failure of SFT" narrative is an artifact of fixed-prompt training, not a fundamental limitation.

4. **Fish (2104.09937, §4.4):** Fish with random groupings (no domain structure) = ERM. Prompt formats, which lack systematic gradient divergence, are functionally random groupings.

**What does Fish + null-space buy over K=5 augmented SFT + K=5 KL?** Fish buys: nothing (Section 1). Null-space at r=8 buys: negative value (Section 2 — it constrains capacity without reliable preservation). The hybrid was constructed by taking the prior review's correct diagnosis (Reptile ≠ Fish, fix the inner loop) and building a mechanism around it — but the fix assumes Fish *would* work for prompt formats if implemented correctly. It wouldn't. The gradient disagreement across prompt formats that Fish needs to exploit simply does not exist.

DSAE Lite is the correct method. Its novel contribution — format-diverse KL preservation — is mechanistically sound, computationally cheap, has no precedent in the literature (confirmed in the DSAE review's exhaustive search), and composes naturally with Allen-Zhu-style injection augmentation. It does not require meta-learning inner loops, Riemannian geometry, null-space initialization, or any machinery beyond a multi-format KL loss term.

---

## Section 6: Final Verdict

**(b) Implement DSAE Lite as the headline method.**

The Fish-DSAE-NSI hybrid fails scrutiny on two independent grounds: (1) Fish's inter-domain gradient matching is vacuous for paraphrastic prompt formats, which lack the gradient disagreement the mechanism requires; (2) KORE's null-space initialization at r=8 is too capacity-constrained for fact injection and too approximate for reliable preservation. These are not tuning problems — they are structural mismatches between the mechanisms and the task.

DSAE Lite avoids both problems. Its two ingredients are: (i) K=5 augmented SFT for injection (Allen-Zhu, well-validated), (ii) K=5 augmented KL for preservation (novel, no precedent). The 5-way ablation design in Section 4 cleanly isolates each ingredient's contribution within a 16 GPU-hour budget, leaving 14 hours of margin for debugging and additional seeds. The pre-registered decision rules specify exactly what numbers constitute a positive result, a partial result (injection diversity is the active ingredient), or a clean negative result. All three outcomes are publishable.

Option (c) — implementing both Fish-DSAE-NSI and DSAE Lite as ablations — is rejected because Fish-DSAE-NSI's failure modes are not empirically interesting. Fish failing on prompt formats is a *predicted* null result with a clear theoretical explanation (no gradient disagreement), not a surprising finding. It would consume ~12 GPU-hours to demonstrate what the theory already tells us. Those hours are better spent on additional seeds, K-ablations (K=3 vs. K=5 vs. K=10), or probing experiments that characterize *how* the model stores format-invariant facts under DSAE Lite.

**Recommended next action:** Implement the 5-way DSAE Lite experiment from Section 4. Start with a 1-seed pilot of conditions (a) and (e) only (2 conditions × 1 seed × 5 rounds = ~2 GPU-hours) to verify the pipeline works, KL coefficients are sane, and the format gap is measurable. If the pilot is clean, launch the full 5-condition × 3-seed × 15-round experiment. Reserve 5 GPU-hours for post-hoc probing of the trained models' internal representations to characterize the mechanism by which format-diverse KL preservation reduces format-selective forgetting.

---

*Verified citations: Shi et al./Fish (2104.09937, §3.1–3.4, §4.4, Algorithm 1); Zhao et al./KORE (2510.19316, §3.2, §4, Appendix C); Allen-Zhu & Li (2309.14316, §4.2, §5); Cheng et al./PAFT (2502.12859, §5.3, Table 5); Lin et al. (2510.00237, §5); Opiełka et al. (2602.22424). All section numbers and claims verified against paper text via direct reading.*
