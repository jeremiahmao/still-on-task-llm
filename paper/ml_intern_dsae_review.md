# Technical Review: Distribution-Symmetric Augmented Editing (DSAE)

**Verdict: one genuinely novel piece, one useful-but-incremental piece, and one piece that is likely broken. The optimization framing is hand-wavy. Implement DSAE Lite; shelve Full DSAE until the subspace assumption is empirically validated.**

---

## (1) Optimization Framing: Hand-Wavy in Two Specific Places

The objective $\max_\theta\; \mathbb{E}_F[\log P_\theta(y_i \mid F(x_i))]$ s.t. $\mathbb{E}_G[\mathrm{KL}(P_\theta \| P_{\theta_0})(\cdot \mid G(x_t))] \le \varepsilon$ has the right shape but two problems.

**Problem A: the constraint is backwards.** You write $\mathrm{KL}(P_\theta \| P_{\theta_0})$ (mode-seeking), but preservation constraints in the continual-learning literature universally use the forward KL $\mathrm{KL}(P_{\theta_0} \| P_\theta)$ (mean-seeking), because you want the fine-tuned model to *cover* everything the reference does, not to *concentrate* on a subset of it. Mode-seeking KL lets $P_\theta$ collapse to a narrow mode of $P_{\theta_0}$ while satisfying the constraint — the opposite of preservation. This may be a notational slip, but it matters: the Lagrangian behaves differently.

**Problem B: $F$ and $G$ are discrete finite pools, not true distributions.** Writing $\mathbb{E}_F$ implies integration over a continuous format distribution; in practice you have $K=20$ templates. The "Monte Carlo estimator" is minibatch SGD over a finite set. PAFT (Cheng et al., 2502.12859) already does this with $|\mathbb{P}| \approx 400$ prompts. Serviceable for a workshop paper; not rigorous enough for a venue that will scrutinize it.

---

## (2) Ingredient #1: Allen-Zhu's K=5 With Bigger K — Yes, Mostly

Allen-Zhu & Li (2309.14316) test $K \in \{1,2,3,5\}$ with monotonic accuracy gains and **no plateau visible at $K=5$**. They never tested $K > 5$. Part 3.3 (2404.05405) shows 40× rewrites preserve the 2-bit/param capacity law — larger $K$ helps *extractability* but not *capacity*.

**Does $N$-sampling-per-step buy anything?** PAFT's Table 5 ablation shows accuracy is **insensitive to switching frequency** ($K=1$ vs $K=8$ steps/prompt: $\approx 87\%$ either way). The benefit comes from pool *diversity*, not sampling schedule. This is evidence against any regime where $K=20, N=2$ beats $K=20, N=1$.

Honest summary: ingredient #1 is Allen-Zhu with a larger (untested) $K$ and explicit per-step sampling. PAFT's evidence suggests this is approximately equivalent to just putting all variants in the dataset. Reasonable design choice, not a novel contribution.

---

## (3) Ingredient #2: Symmetric Task-Side Augmentation — Genuinely Novel

After exhaustive search, **no paper applies format-diverse augmentation to the KL preservation constraint.** The nearest misses: SEFE/ASD (2505.02486) applies 5 format variants to replay data but via SFT loss, no KL; RECAP (2510.21978) uses KL against a frozen reference but on single-format data (the authors note: *"KL terms are calculated on the current task, thus they do not guarantee broader knowledge"*); GeRe (2508.04676) anchors activation states but uses fixed-format generic texts. Standard EWC/replay-KL all use a single format per example.

DSAE's proposal — $\mathcal{L}_{\mathrm{KL}} = \frac{1}{K}\sum_k \mathrm{KL}(P_{\mathrm{ref}}(\cdot \mid G_k(x)) \| P_\theta(\cdot \mid G_k(x)))$ — has no direct precedent. The motivation is clean: if format-coupling causes selective forgetting under format B while retaining under format A, a single-format KL anchor cannot detect it. This is the strongest novel contribution in DSAE, and the cheapest to test.

---

## (4) Ingredient #3: Subspace-Constrained LoRA — Not Novel in Mechanism, Possibly Novel in Data Source

"PCA on activations → constrain LoRA's $A$" is thoroughly precedented. **EVA** (2410.07170) does exactly this: SVD on activation batches, $A_{\mathrm{init}} = V_{:r,:}$ to maximize $\mathrm{Tr}(A \Sigma A^T)$. **CorDA** (2406.05223) builds $C = XX^T$ from task activations for context-aware decomposition. **EigenLoRAx** (2502.04700) runs PCA over stacked adapter matrices → constrains new adapters to the shared subspace. **O-LoRA** (2310.14152) constrains $A$ via soft orthogonality penalty against previous tasks.

DSAE's difference: the covariance is computed over *multiple format renderings of the same fact*. This is a new data construction, not a new algorithm. A reviewer will write: *"How is this different from EVA with a specific data sampling strategy?"* The answer must be empirical — cross-format PCA must yield a qualitatively different subspace than single-format PCA.

---

## (5) The Key Technical Question: Does the Shared Subspace Exist and Is It Broad Enough?

The literature gives a clear, unfavorable answer.

**Cauderay et al. (2602.22424)** — the definitive paper — show function vectors (causally potent representations) extracted from different input formats of the *same concept* are **nearly orthogonal**. Separate "concept vectors" (format-invariant, RSA-identified) exist but have **smaller causal effects** — they modulate behavior but cannot drive it. The two systems are dissociated: different attention heads, minimal overlap.

**Sclar et al. (2310.11324):** PCA of hidden states across format variants clusters **by format, not by content**. **Burns et al. (2310.06824):** shared "truth directions" are **scale-dependent** — large at 70B+, small-to-moderate at 7B–13B. **Xie et al. (2402.05827):** ROME edits degrade 99.9% → 74.7% (paraphrase) → 21.3% (different format). **Hernandez et al. (2308.09124)** offer the one positive note: linear relational embeddings at the *subject token* are template-stable for simple relations (capital-of: 0.84–0.87 across 4 templates) — but only *within* a format type, not across types.

**Synthesis:** PCA of 20 format renderings at 7B middle layers will show moderate shared variance — enough to identify, not enough to build a useful constraint on. The format-invariant subspace is (a) low-dimensional, (b) relation-specific not globally shared, (c) causally weaker than format-specific representations, and (d) scale-dependent. Constraining $A$ to it risks starving the LoRA update of the format-specific directions it actually needs.

---

## (6) Failure Modes, Ranked by Probability

1. **Subspace too narrow (p ≈ 0.6).** The shared format-invariant subspace at 7B scale is low-rank and relation-specific. $A$ constrained to it cannot express the weight updates needed to inject new facts. Training loss plateaus; the model memorizes nothing.

2. **Subspace captures surface-level shared features, not knowledge (p ≈ 0.4).** PCA finds directions where formats agree, but these may encode entity-name token features (shared across all renderings because the entity name is literally the same string), not factual associations. The constraint becomes trivially satisfied but useless.

3. **Compute overhead of subspace identification exceeds the edit itself (p ≈ 0.3).** Encoding $K=20$ renderings × $N_{\mathrm{facts}}$ facts through the frozen base, computing per-layer PCA, aggregating across facts — this is expensive. For a class experiment on 1000 facts, that is 20,000 forward passes before training starts.

4. **Ingredient #2 works but #3 cancels it out (p ≈ 0.25).** The symmetric augmentation provides genuine preservation benefit, but the subspace constraint prevents the model from learning in the format-specific directions that actually matter, zeroing out the knowledge injection gains.

---

## (7) Implementation Recommendation

**DSAE Lite (ingredients 1+2 only): yes, implement it.** ~10 GPU-hours, 5 seeds is reasonable. Ingredient #2 (format-diverse KL preservation) is genuinely novel, well-motivated, and cheap. The ablation structure writes itself: (a) standard single-format replay, (b) format-diverse learning only (≈ PAFT), (c) format-diverse preservation only (novel), (d) both (DSAE Lite). If (c) outperforms (a), you have a clean result.

**Full DSAE (add ingredient #3): no, not yet.** The subspace assumption is empirically shaky at 7B scale (Cauderay et al., 2025; Sclar et al., 2023). Before investing 15–20 GPU-hours + nontrivial implementation, run one diagnostic: PCA 20 format renderings of 100 facts through the frozen base, measure the explained variance ratio. If PC1 explains < 40% of cross-format variance, the subspace is too thin and ingredient #3 will not work. This diagnostic costs < 1 GPU-hour and should precede any full DSAE implementation.

If the diagnostic is positive, implement ingredient #3 **as EVA (2410.07170) with your specific data construction** — the math is identical, and EVA already has reference code. Frame the novelty correctly: the mechanism is EVA, the data construction (cross-format PCA) is yours, and the burden is on you to show these produce different subspaces than standard single-format EVA.

---

*References: Allen-Zhu & Li (2309.14316); Cauderay et al. (2602.22424); Cheng et al./PAFT (2502.12859); EVA (2410.07170); CorDA (2406.05223); EigenLoRAx (2502.04700); Hernandez et al. (2308.09124); Jiang et al./PIT (2402.12847); Meng et al./ROME (2202.05262); O-LoRA/Wang et al. (2310.14152); Burns et al. (2310.06824); Park et al. (2311.03658); Sclar et al. (2310.11324); SEFE (2505.02486); RECAP (2510.21978); GeRe (2508.04676); Xie et al. (2402.05827).*
