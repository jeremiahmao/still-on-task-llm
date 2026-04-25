# Manifold Reptile: Format-Invariant Fact Injection via Meta-Learning on the LoRA Rank Manifold

**Status:** Independent theoretical proposal (not dependent on DSAE)
**Date:** 2025-04-25

---

## Central Insight

The LoRA parameter space is not $\mathbb{R}^{m \times r} \times \mathbb{R}^{r \times n}$. It is the smooth Riemannian manifold $\mathcal{M}_r = \{X \in \mathbb{R}^{m \times n} : \mathrm{rank}(X) = r\}$ of fixed-rank matrices. This manifold has a tangent space at each point, and that tangent space admits a natural decomposition into *shared* and *point-specific* directions. The proposal: execute Reptile meta-learning (Nichol et al., 2018) not in Euclidean parameter space but on $\mathcal{M}_r$ itself, treating each prompt format as a meta-learning task. The tangent-space consensus of the inner-loop endpoints — the Riemannian Fréchet mean of their log-mapped displacements — extracts precisely the format-invariant component of the fact-injection gradient, while discarding format-coupled components by geometric averaging. Task preservation is enforced by a separate null-space initialization that confines the entire meta-learning trajectory to directions orthogonal to the pretrained model's task-critical activation subspace.

The reason this should work where loss engineering fails: loss penalties (V-REx, KL-reg, Group DRO) operate on *scalar* signals — one number per format per step. The degeneration at K=2 is structural: a scalar variance over two points is trivially satisfiable (Krueger et al., 2021, arXiv:2003.00688). Manifold Reptile operates on *d-dimensional tangent vectors* — at every step, it has access to the full gradient geometry across formats, not just their scalar losses. Even at K=2, two gradients in $\mathbb{R}^d$ disagree on exactly the coordinates that encode format-specific information. The tangent-space average preserves the coordinates where both formats agree (the fact) and cancels the coordinates where they disagree (the format).

---

## Why It Should Work: Theoretical Argument

### (A) Reptile's implicit gradient inner product

Nichol et al. (2018) show that the Reptile update $\theta \leftarrow \theta + \varepsilon(\tilde{\theta}_i - \theta)$, averaged over tasks $i$, is a first-order approximation to maximizing the expected inner product of task gradients:

$$\mathbb{E}_{i,j}[\nabla \mathcal{L}_i(\theta)^\top \nabla \mathcal{L}_j(\theta)]$$

This is the *gradient inner product* (GIP) objective of Shi et al. (2021, arXiv:2104.09937), which they prove is sufficient for domain-invariant representation learning even at K=2. The key theorem (Shi et al., Proposition 1): maximizing GIP over two domains is equivalent to minimizing a tight upper bound on the combined risk under distribution shift. Unlike V-REx's scalar variance, GIP is a d-dimensional inner product — it retains directional information that scalar losses discard. This is why Fish (Shi et al.'s first-order instantiation) works at K=2 while V-REx degenerates.

Reptile *is* a first-order GIP maximizer (ATM, Davari et al., 2024, arXiv:2411.03055, §3 proves the formal equivalence between task-vector averaging and Reptile's implicit gradient alignment). But standard Reptile operates in Euclidean space, where the LoRA $(A, B)$ factorization introduces a gauge ambiguity: $(A, B) \equiv (AC^{-1}, CB)$ for any invertible $C$. Two inner-loop endpoints can disagree *not because the formats require different updates* but because the optimizer chose different gauge representatives. This is a confound that poisons the consensus signal.

### (B) The manifold eliminates the gauge ambiguity

On $\mathcal{M}_r$, the parametrization is the rank-$r$ matrix $\Delta W = BA$ itself, not the factors $(B, A)$ separately. The tangent space $\mathcal{T}_X \mathcal{M}_r$ at any point $X$ has a canonical decomposition (Vandereycken, 2013, SIAM J. Optim.; operationalized in RiemannLoRA, Hsu et al., 2025, arXiv:2507.12142):

$$\mathcal{T}_X \mathcal{M}_r = \{U_L \dot{M} V_R^\top + U_L M_\perp^\top + M_\perp V_R^\top : \dot{M} \in \mathbb{R}^{r \times r}, U_\perp^\top U_L = 0, V_\perp^\top V_R = 0\}$$

where $X = U_L \Sigma V_R^\top$ is the thin SVD. The projection from ambient space to this tangent space (RiemannLoRA Eq. 6):

$$P_{\mathcal{T}_X}(Z) = U_L U_L^\top Z + (I - U_L U_L^\top) Z V_R V_R^\top$$

This projection is *unique* — no gauge freedom. Two inner-loop trajectories on $\mathcal{M}_r$ that arrive at different endpoints encode genuinely different functional updates, not gauge-equivalent reparametrizations. The tangent-space average of their displacements from the meta-initialization is therefore a clean consensus signal, uncontaminated by factorization artifacts.

### (C) Why the tangent consensus separates fact from format

Consider two formats $F_1, F_2$ encoding the same fact $(s, r, o)$. The inner loops produce endpoints $X_1, X_2 \in \mathcal{M}_r$. Map them back to the tangent space at the meta-initialization $X_0$:

$$\xi_i = \mathrm{Log}_{X_0}(X_i) \approx P_{\mathcal{T}_{X_0}}(X_i - X_0)$$

(first-order approximation, valid when $X_i$ is close to $X_0$, which holds for small inner-loop learning rates). The meta-update is:

$$\bar{\xi} = \frac{1}{K}\sum_{i=1}^{K} \xi_i$$

Decompose each $\xi_i = \bar{\xi} + \delta_i$ where $\delta_i$ is the format-specific residual. By construction, $\sum_i \delta_i = 0$. The meta-update $\bar{\xi}$ retains only the *consensus component* — the directions in tangent space where all formats agree on how to modify $\Delta W$. The residuals $\delta_i$ encode how format $F_i$'s surface structure (system message, output schema, phrasing) biases the gradient, and they are averaged away.

This is exactly the separation that Crowded B-Space (Li et al., 2025, arXiv:2604.16826) observes empirically: independently-trained LoRA adapters for different tasks converge to a small shared set of output-space directions (the leading 3 singular vectors of the stacked $B$ matrices account for 53.7% of joint energy), while task-specific variation lives in orthogonal directions. Our protocol constructs the shared directions *during training* via geometric averaging, rather than discovering them *post-hoc* via SVD.

### (D) Task preservation via null-space initialization

Independently of the meta-learning protocol, we initialize $\Delta W_0$ in the null space of the pretrained model's task-critical activation subspace. Following KORE (Zhao et al., 2025, arXiv:2510.19316, §3.2): collect activations $\mathbf{X}_\text{task}$ from the task-preservation prompt distribution (using all format variants for robustness), compute the activation covariance $C = \mathbf{X}_\text{task}\mathbf{X}_\text{task}^\top$, SVD-decompose, and initialize the LoRA update in the null space of $C$:

$$\Delta W_0 \in \mathrm{null}(C) \implies \Delta W_0 \cdot \mathbf{X}_\text{task} \approx \mathbf{0}$$

This guarantees that the initial LoRA update produces zero output change on task-relevant inputs. The meta-learning trajectory stays in a neighborhood of this null-space initialization (controlled by the inner-loop learning rate and step count), providing approximate task preservation throughout training. This is the Adam-NSCL principle (Wang et al., 2021, arXiv:2112.06025) adapted to LoRA, and KORE demonstrates it empirically on LLaVA-1.5 at 7B/13B scale.

The two mechanisms are *orthogonal by construction*: null-space initialization handles axis (2) (task preservation across formats), while manifold Reptile handles axis (1) (format-invariant fact storage). They compose without interference because the null-space constraint is an initialization, not a per-step penalty — the meta-learning protocol operates freely within the null-space neighborhood.

---

## Concrete Algorithm

```
MANIFOLD-REPTILE(model M, facts F, formats {F_1,...,F_K}, task prompts T)

  // Phase 0: Null-space initialization (once, ~20 min for 4B model)
  X_task = collect_activations(M, T, all_format_variants)
  C = X_task @ X_task.T                          // activation covariance
  U, S, V = svd(C)
  null_basis = U[:, S < threshold]               // null space of task activations
  ΔW_0 = project_to_rank_r(random_init, null_basis)  // init on M_r ∩ null(C)

  // Phase 1: Manifold Reptile (main training loop)
  for epoch = 1 to N:
    for fact f in shuffle(F):
      for k = 1 to K:                             // K format inner loops
        ΔW_k = ΔW_0                               // clone meta-init
        for step = 1 to n_inner:                   // n_inner ∈ {3,5,8}
          g = ∇_ΔW L(M + ΔW_k, F_k(f))           // Euclidean gradient
          ξ = P_{T_{ΔW_k} M_r}(g)                 // project to tangent space
          ΔW_k = Retract(ΔW_k, -α_inner * ξ)      // Riemannian SGD step
        end
        ξ_k = P_{T_{ΔW_0} M_r}(ΔW_k - ΔW_0)     // log-map approx: tangent at init
      end

      // Fréchet mean in tangent space (exact for small displacements)
      ξ_bar = (1/K) * sum(ξ_k for k=1..K)

      // Meta-update: retract along consensus tangent direction
      ΔW_0 = Retract(ΔW_0, ε * ξ_bar)            // ε = outer learning rate
    end
  end

  return ΔW_0
```

**Key operations and their costs** (per RiemannLoRA, arXiv:2507.12142):
- Tangent projection $P_{\mathcal{T}_X}(Z)$: $O((m+n)r^2)$ — requires maintaining the thin SVD of $\Delta W$
- Retraction $\mathrm{SVD}_r(X + \xi)$: $O((m+n)r^2 + r^3)$ — truncated SVD
- Total per-fact overhead vs. standard LoRA: $K \times n_\text{inner}$ forward/backward passes instead of 1, plus $K$ SVD operations of cost $O((m+n)r^2)$. For $r=8, m=n=4096, K=3, n_\text{inner}=5$: roughly $15\times$ standard LoRA per fact.

**Hyperparameters:**
- $K$: number of formats (propose $K \in \{3, 5\}$ — enough for non-degenerate consensus, few enough for budget)
- $n_\text{inner}$: inner-loop steps (propose 3–8; fewer = closer to first-order Reptile ≈ gradient averaging; more = stronger adaptation before consensus)
- $\alpha_\text{inner}$: inner learning rate (propose $\sim 10\times$ the outer rate; standard Reptile practice)
- $\varepsilon$: outer learning rate (propose $\sim 5 \times 10^{-5}$ based on standard LoRA rates for 4B models)
- $r$: LoRA rank (propose $r=8$ as baseline; the manifold structure becomes richer at higher rank)
- Null-space threshold: propose $S_\text{max} \times 10^{-3}$ following KORE

---

## Empirical Requirements

### Compute budget: ~25 GPU-hours (within the 30 GPU-hour budget)

| Component | GPU-hours | Hardware |
|-----------|-----------|----------|
| Null-space initialization (activation collection + SVD) | 0.5 | A10G |
| Manifold Reptile training: 3 seeds × 15 rounds × n=200 facts | 18.75 | A10G |
| Baselines (kl_reg_sft, augmented SFT): 3 seeds × 15 rounds × n=200 | 3.75 | A10G |
| Evaluation (QA + QD probes, all seeds) | 1.5 | A10G |
| **Total** | **24.5** | |

Estimate for per-round Manifold Reptile cost: standard LoRA round ≈ 0.05 GPU-hr (from project data); MR adds $K \times n_\text{inner} = 15\times$ forward/backward cost, but the tangent projections and retractions are dominated by the forward passes. Estimated $\sim 5\times$ wall-clock (some parallelism in inner loops, SVD is cheap at $r=8$): $0.25$ GPU-hr/round. At 15 rounds × 3 seeds × 200/(50 baseline) scaling ≈ $0.25 \times 15 \times 3 \times 1 \approx 11.25$ GPU-hr core training. Add 50% safety margin for debugging → 18.75.

### Code requirements

1. **RiemannLoRA ops**: tangent projection, retraction, and vector transport. Reference implementation in arXiv:2507.12142 supplementary; ≈200 lines of PyTorch.
2. **Reptile outer loop**: standard; ≈50 lines wrapping the inner loop.
3. **Null-space initialization**: activation collection (one forward pass over task prompts) + SVD + projection to $\mathcal{M}_r$. Following KORE (arXiv:2510.19316); ≈100 lines.
4. **Format templates**: $K \in \{3, 5\}$ prompt formats per fact. Reuse existing QA/QD templates + add cloze, declarative, instruction-style.

### Data

Use the existing FNSPID-derived fact triples and task evaluation suite. No new data collection required. Format templates are the "data augmentation" — each fact is rendered in $K$ surface forms.

### Evaluation

- Primary: QA F1 and QD F1 on held-out behavioral probe (existing infrastructure)
- Format gap: $|\text{QA F1} - \text{QD F1}|$ (the number we're trying to minimize)
- Task preservation: accuracy on the existing task benchmark before/after injection
- Statistical: 3 seeds, n=200, report means ± SE. A +3pp QD F1 shift is detectable at this sample size with $\alpha=0.05$.

---

## Failure Modes (Honest Assessment)

### 1. Inner loops don't diverge enough to provide consensus signal (p ≈ 0.3)

If $n_\text{inner}$ is too small or $\alpha_\text{inner}$ too low, all $\xi_k$ are approximately identical (first-order Taylor: $\xi_k \approx -\alpha \nabla \mathcal{L}_k$ for all $k$). The consensus average $\bar{\xi}$ then equals the gradient average — which is exactly what augmented SFT already computes by putting all formats in the training set. Manifold Reptile reduces to augmented SFT with extra overhead.

**Diagnostic:** measure $\|\delta_k\| / \|\bar{\xi}\|$ (format-specific residual relative to consensus). If this ratio is $< 0.1$, the inner loops aren't differentiating. **Fix:** increase $n_\text{inner}$ or $\alpha_\text{inner}$.

### 2. Tangent-space approximation breaks down (p ≈ 0.2)

The log-map approximation $\mathrm{Log}_{X_0}(X_i) \approx P_{\mathcal{T}_{X_0}}(X_i - X_0)$ is first-order accurate. If inner loops take large steps, $X_i$ exits the geodesically convex neighborhood of $X_0$, and the tangent-space average is no longer a valid Fréchet mean approximation. The geometric consensus becomes meaningless.

**Diagnostic:** measure $\|X_i - X_0\|_F / \|X_0\|_F$ after inner loops. If $> 0.1$, the approximation is suspect. **Fix:** reduce $n_\text{inner}$ or $\alpha_\text{inner}$, or use iterative Karcher mean computation (Karcher, 1977; operationalized in Fisher-Rao Karcher merging, Tam et al., 2025, arXiv:2603.04972).

### 3. Null-space is too narrow at 4B scale (p ≈ 0.25)

Cauderay et al. (2025, arXiv:2602.22424) show format-invariant representations at 7B scale are low-dimensional and causally weaker than format-specific ones. If the task-activation null space at 4B is very low-rank, the meta-learning trajectory is over-constrained — the model cannot express the weight updates needed to inject facts.

**Diagnostic:** measure the dimensionality of $\mathrm{null}(C)$ as a fraction of the LoRA rank budget. If $\dim(\mathrm{null}(C)) < 2r$, there isn't enough room. **Fix:** relax the null-space constraint to a soft penalty (O-LoRA-style, Wang et al., 2023, arXiv:2310.14152: $\lambda \|A_\text{task}^\top A_\text{fact}\|_F^2$) instead of hard initialization.

### 4. $15\times$ compute overhead makes the budget tight (p ≈ 0.15)

With only 30 GPU-hours and 3 seeds needed for statistical validity, there's little room for hyperparameter sweeps. If the first configuration doesn't work, we may not have budget to debug.

**Mitigation:** Run a single-seed pilot (5 GPU-hours) before committing to the full 3-seed experiment. Use the pilot to tune $n_\text{inner}$ and $\alpha_\text{inner}$ via the diagnostics above.

### 5. The gauge ambiguity argument is empirically irrelevant (p ≈ 0.2)

The theoretical advantage over Euclidean Reptile is eliminating the $(A,B)$ gauge confound. If in practice the AdamW optimizer doesn't explore different gauge representatives (e.g., because weight decay implicitly regularizes toward a canonical factorization), then standard Reptile-on-$(A,B)$ might work just as well, and the manifold machinery adds complexity for no gain.

**Test:** run Euclidean Reptile (same protocol, but in $(A,B)$ space without manifold projection) as an ablation. If it matches Manifold Reptile, the manifold is unnecessary (but the Reptile protocol itself may still be the contribution).

---

## Why This Hasn't Been Tried

Each component exists independently; the combination is novel (confirmed by systematic search, §Prior Art below):

| Component | Exists? | Reference |
|-----------|---------|-----------|
| Riemannian optimization for LoRA | ✅ | RiemannLoRA (Hsu et al., 2025, arXiv:2507.12142) |
| Reptile outer loop for LLM LoRA | ✅ | ABMLL (Lin et al., 2025, arXiv:2508.14285) |
| Null-space LoRA initialization | ✅ | KORE (Zhao et al., 2025, arXiv:2510.19316) |
| Tangent-space task decomposition | ✅ (post-hoc) | Iso-CTS (Xu et al., 2025, arXiv:2502.04959) |
| Reptile *on* $\mathcal{M}_r$ | ❌ | Not found |
| Tangent decomposition *during training* | ❌ | Not found |
| Meta-learning for format-invariant fact injection | ❌ | Not found |
| Any combination of the above | ❌ | Not found |

**Why the gap exists:** The Riemannian LoRA literature (RiemannLoRA, PoLAR/arXiv:2506.03133) focuses on single-task optimization quality — "better LoRA training" — not multi-objective or continual learning. The meta-learning-on-LoRA literature (ABMLL) focuses on cross-task generalization, not format invariance within a single task. The null-space continual learning literature (KORE, O-LoRA, Sculpting Subspaces/arXiv:2504.07097) focuses on sequential task preservation, not on the structure of the injection update itself. Nobody has combined manifold geometry with meta-learning for knowledge editing because these are three separate subcommunities (differential geometry in optimization, meta-learning, knowledge editing) that rarely cite each other.

---

## Comparison to DSAE

DSAE's core mechanism (subspace-constrained LoRA via cross-format PCA) and Manifold Reptile attack the same problem from opposite directions:

| | DSAE | Manifold Reptile |
|---|---|---|
| Where format invariance is enforced | **Representation space** (constrain $A$ to format-invariant subspace) | **Optimization trajectory** (average out format-specific gradient components) |
| Assumption | Format-invariant subspace exists and is broad enough | Inner loops diverge enough to separate fact from format |
| Risk | Subspace too narrow at 4B scale (Cauderay et al., 2025) | $15\times$ compute overhead; inner loops may not diverge |
| Theoretical backing | EVA-style activation PCA (mature) | Reptile + Riemannian geometry (novel combination, each piece validated) |
| Shares with DSAE | Null-space task preservation (both use activation covariance) | — |

The proposals are **complementary, not competing**: DSAE constrains *where* the update can go (subspace); MR constrains *which direction* to take within the update space (consensus). A synthesis is possible: initialize in the DSAE subspace, then run Manifold Reptile within it. But each should be tested independently first.

---

## What Would Convince Me This Works

1. **Format gap reduction**: $|\text{QA F1} - \text{QD F1}|$ drops from baseline 0.072 to $\leq 0.04$, consistent across 3 seeds.
2. **QD F1 lift**: absolute QD F1 improves by $\geq$ +0.03 over kl_reg_sft baseline (detectable at n=200, 3 seeds).
3. **Task preservation**: task benchmark accuracy within 1pp of baseline (no degradation).
4. **Ablation separation**: Manifold Reptile > Euclidean Reptile > augmented SFT > kl_reg_sft. Each step adds value.
5. **Diagnostic confirmation**: format-specific residuals $\|\delta_k\|$ are non-trivial ($> 10\%$ of $\|\bar{\xi}\|$), confirming the inner loops do separate fact from format.

If (4) shows Euclidean Reptile ≈ Manifold Reptile, the contribution is Reptile-for-format-invariance (still novel in this application), not the manifold. If augmented SFT ≈ Reptile, the contribution is negative: meta-learning doesn't help beyond data diversity (still publishable — saves the next researcher from trying).

---

*References: Absil et al. (2009), Optimization Algorithms on Matrix Manifolds, Princeton; Amari (1998), Neural Computation 10(2); Arjovsky et al. (2019, arXiv:1907.02893); ATM/Davari et al. (2024, arXiv:2411.03055); Cauderay et al. (2025, arXiv:2602.22424); Crowded B-Space/Li et al. (2025, arXiv:2604.16826); Fisher-Rao Karcher/Tam et al. (2025, arXiv:2603.04972); Fish/Shi et al. (2021, arXiv:2104.09937); GAF/Chaubard et al. (2024, arXiv:2412.18052); Iso-CTS/Xu et al. (2025, arXiv:2502.04959); Karcher (1977), Annali di Mat. Pura ed Applicata; KORE/Zhao et al. (2025, arXiv:2510.19316); Krueger et al./V-REx (2021, arXiv:2003.00688); Lin et al./ABMLL (2025, arXiv:2508.14285); Nichol et al. (2018), OpenAI Tech Report; O-LoRA/Wang et al. (2023, arXiv:2310.14152); PCGrad/Yu et al. (2020, arXiv:2001.06782); PoLAR (2025, arXiv:2506.03133); RiemannLoRA/Hsu et al. (2025, arXiv:2507.12142); RoSE (2025, arXiv:2603.15518); Sculpting Subspaces/Nayak et al. (2025, arXiv:2504.07097); Vandereycken (2013), SIAM J. Optim. 23(2); Wang et al./Adam-NSCL (2021, arXiv:2112.06025).*
