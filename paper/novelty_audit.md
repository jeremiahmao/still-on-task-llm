# Novelty Audit — draft_v2.md (audit date 2026-04-28)

Five claimed contributions audited against 2023-2026 literature. Citations as arXiv IDs. Verdicts are calibrated to a skeptical NeurIPS/ICLR reviewer.

---

## Contribution 1 — Quantitative diagnosis of an "absorption-integration gap" under continual LoRA at 4B

**Status: PARTIAL OVERLAP.** The phenomenon (knowledge stored in one format fails to surface in another) is documented at multiple levels of the stack: pre-training (Allen-Zhu & Li, 2309.14316; Berglund Reversal Curse 2309.12288), one-shot editing (Cohen RippleEdits 2307.12976; Zhong MQuAKE 2305.14795), continued pre-training/IFT (Jiang PIT 2402.12847; Pletenev 2502.14502), and prompt robustness (Wei PAFT 2502.12859; Pan masked-FT 2510.09885). SEFE (2505.02486) explicitly names "superficial forgetting" — a near-isomorphic concept — and "essential forgetting" in MCIT.

**Closest prior work.** Jiang 2402.12847 (perplexity↓ but QA fails — same gap, naming free); SEFE 2505.02486 (superficial vs essential forgetting under MCIT); Pletenev 2502.14502 (LoRA fact-injection generalization gap).

**What's actually new.** (i) The explicit name and operational metric `|F1_QA − F1_QD|` on a fixed n=50 behavioral probe set; (ii) the gap measured under sequential LoRA editing at 200 facts × 15 rounds in a *task-tuned* 4B base (most prior work measures on a generic base or pre-training corpus). The 0.07-0.14 magnitude across single-environment methods is a fresh data point.

**What's NOT new.** The phenomenon itself, the framing as "absorbed but not integrated", and the diagnosis-via-paraphrase-probe pattern.

**Reviewer-rejection risk: MEDIUM.** A reviewer can fairly say "this is superficial forgetting (SEFE) renamed." Mitigation: cite SEFE/Jiang/Pletenev explicitly as prior namings and frame the contribution as the *quantification* under continual-LoRA-on-task-tuned-4B specifically.

---

## Contribution 2 — Super-linear synergy of K=5 paraphrastic injection × K=1 KL preservation (4.93× / ~10× asymmetry)

**Status: NOVEL (strongest claim).** A targeted 2×2 factorial of {K=1,K=5} × {no-KL, K=1 KL} with the headline showing each ingredient ≈ +0.03 and combined = +0.32 has no direct precedent. Allen-Zhu (2309.14316) shows K=5 in *pre-training*; Pletenev (2502.14502) shows paraphrasing helps LoRA injection but doesn't pair it with KL-against-frozen-reference; PAFT (2502.12859) does prompt augmentation without per-round KL anchor; LiLoRA / SEFE-RegLoRA use cosine/parameter regularizers, not KL on a per-round teacher. Allen-Zhu's industry impact note (paraphrasing as standard) and Pletenev's "paraphrasing improves integration" are the two closest priors and neither runs the factorial.

**Closest prior work.** Allen-Zhu & Li 2309.14316; Pletenev 2502.14502; Wei PAFT 2502.12859.

**What's actually new.** The interaction itself, isolated by compute-matched contrast `(d)−(b) = +0.286 vs (c)−(a) = +0.029`. The synergy ratio (~5×, ~10× compute-matched) is a quantitatively concrete result that no prior paper reports because no prior paper runs this 2×2.

**What's NOT new.** Each marginal ingredient: K=5 paraphrastic injection (Allen-Zhu, Pletenev); KL-to-frozen-reference (RL's Razor 2509.04259, KL-regularized SFT generally). Showing each marginal alone is small is the easy part; the synergy is the contribution.

**Reviewer-rejection risk: LOW-MEDIUM.** Main risk is "you should also test EWC/replay/L2 in place of KL" — addressed in §7 as future work but the reviewer will still ask. The 2-seed n is a separate methodological vulnerability, not a novelty issue.

---

## Contribution 3 — Negative on symmetric K=5-on-preservation extension (DSAE Lite)

**Status: NOVEL (narrow but clean).** No prior work tests K=5 augmentation on the *KL preservation distribution* in continual fact injection; the (e) condition is a construct unique to this paper. SEFE/RECAP/GeRe diversify replay data but not the KL distribution; PAFT diversifies inference-time prompts. The null result `Δ(e−d) = −0.006` is informative because it isolates the active mechanism to "K=5 injection × any KL anchor."

**Closest prior work.** RECAP 2510.21978 (notes KL is computed on current task and doesn't guarantee broader knowledge — motivation for an (e)-style fix); SEFE/ASD 2505.02486 (data-side diversification on replay).

**What's actually new.** The empirical null on the symmetric construction at this specific scale — and the §6 conjecture tying the null to LoRA's shared low-rank update subspace.

**What's NOT new.** "Diversify the preservation distribution" as an *intuition* (RECAP raises it explicitly).

**Reviewer-rejection risk: MEDIUM.** A reviewer may push back on n=2 seeds and the claim's narrowness ("null at one scale"). The §7 limitation acknowledges this, which is the right move.

---

## Contribution 4 — Three negatives constraining loss-engineering design space

**Status: PARTIAL OVERLAP (4a NOVEL, 4b SUBSUMED-by-theory, 4c PARTIAL).**

(4a) **COPR doesn't port to fact injection.** No prior paper tests COPR (2402.14228) in fact-injection regime; the K-sample-all-wrong pathology framing is original. **NOVEL** as an applied negative. *Risk: LOW.*

(4b) **V-REx at K=2 is degenerate.** This follows immediately from Arjovsky 1907.02893, Rosenfeld 2010.05761, Ahuja 2010.16412 — all cited in §2. Establishing degeneracy at K=2 is theory-internal; the empirical confirmation on language data is new but the theoretical claim is **SUBSUMED** by the IRM identifiability literature. *Risk: MEDIUM* — a reviewer may say "you didn't need to run this experiment to know it would fail." Reframing as "we ran it because no one had on LMs" is defensible but mild.

(4c) **Naïve format mixing widens the gap.** PIT (2402.12847) already shows naïve format mixing degrades EM by ~6pp; the paper's 0.100 vs 0.072 finding is a near-replication on a different task. **PARTIAL OVERLAP** — confirmatory, not novel. *Risk: MEDIUM-HIGH* if pitched as a finding; LOW if pitched as a replication that motivates §4.

**Closest prior work.** Zhang COPR 2402.14228; Krueger V-REx 2003.00688 + Rosenfeld 2010.05761; Jiang PIT 2402.12847.

---

## Contribution 5 — Methodological warning: hidden-state cosine disagrees with behavioral cross-format availability

**Status: PARTIAL OVERLAP (likely the weakest novel claim).** That representation-geometry probes underspecify behavior is well-established: Belinkov "Probing Classifiers" (1909.03368), Hewitt-Liang control tasks, and recent LLM-specific work on probe-vs-behavior gaps. In the *editing* literature specifically, RippleEdits (2307.12976) and "Towards a Principled Evaluation of Knowledge Editors" (2507.05937) push behavioral evaluation precisely because geometric/probe-localized signals don't translate.

**Closest prior work.** RippleEdits 2307.12976; Cohen et al. on consistent-edit behavior; Belinkov probing surveys.

**What's actually new.** The specific finding that within the COPR family, gold-injection variants exhibit smallest geometric shift but largest behavioral gap — a within-method-family disagreement on n=50 facts. The compactness ("cosine doesn't predict cross-format availability") is sharp but not novel as a methodological position.

**What's NOT new.** The general claim that geometric proxies and behavioral metrics disagree.

**Reviewer-rejection risk: MEDIUM-HIGH** if pitched as a methodological contribution in its own right; LOW if scoped to "we observed this within COPR variants and recommend behavioral probes alongside geometric ones."

---

## Overall verdict

- **Strongest novel claim:** Contribution 2 (the 2×2 super-linear synergy). The compute-matched ~10× asymmetry is a concrete, quantitative, hard-to-replace finding with no direct precedent.
- **Weakest / most replicable:** Contribution 4c (naïve format mixing widens the gap) — replicates PIT 2402.12847 on a different task. Should be re-pitched as a replication.
- **Overstated as novel:** Contribution 1 (gap diagnosis) and Contribution 5 (geometric-behavioral disagreement). Both exist in the literature under different names; the paper frames them more strongly than warranted.
- **Missing citations the paper should add:**
  - SEFE 2505.02486 already cited but not engaged on the "superficial forgetting = absorption-integration gap" point — this should be acknowledged in §1.
  - "Towards a Principled Evaluation of Knowledge Editors" 2507.05937 — directly relevant to Contribution 5.
  - RL's Razor 2509.04259 — relevant prior on KL-minimal solutions reducing forgetting; supports the §4 mechanism story for the KL ingredient.
  - Belinkov probing-classifier survey 1909.03368 — to ground Contribution 5's methodological framing.
  - SLACK 2306.09998 (KL-regularized augmentation policy learning) — adjacent prior on augmentation × KL composition.
  - STABLE 2510.16089 — gated continual LoRA editing; relevant baseline framing for §2.

**Bottom line.** Contribution 2 carries the paper. Contributions 3 and 4a are clean supporting negatives. Contributions 1, 4b, 4c, and 5 should be re-pitched as quantification / replication / methodological reminders rather than novel findings, with the missing citations added.
