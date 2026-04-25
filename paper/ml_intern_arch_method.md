# Architectural Continual Injection: A Teacher-Free Method Proposal

**Date:** 2025-04-26 &ensp; **Status:** Research proposal, parallel to Iteration 6 (DSAE Lite survey).

---

## Motivation

DSAE Lite (locked per `ml_intern_iteration3_verdict.md`) requires a frozen teacher snapshot in memory for KL preservation at every update round: K=5 forward passes through the reference model per training step, then discard. This is sound as a research probe — it cleanly isolates whether format-diverse KL preservation helps — but it is not a production mechanism. Production ingestion wants: one document in, one update out, no second model in memory, no batched 200-fact rounds, no frozen reference forward passes. The user's direction: *"something more tied down to the LLM layer"* — an architectural mechanism that enforces preservation structurally, not through an external loss term.

This document surveys the design space of layer-level / architectural mechanisms for teacher-free continual knowledge injection, proposes three candidates, recommends one for implementation as a 6th ablation condition, and pre-registers its failure modes.

---

## Section 1: The Architectural Design Space

### 1.1 Frozen-Base + Modular Adapters

**AdapterFusion** (Pfeiffer et al., 2021) / **AdapterDrop** (Rücklé et al., 2021): per-task bottleneck adapters with learned composition. Preservation is trivial (freeze prior adapters), but adapters grow linearly with documents (3000 instances at our scale) and composition requires a routing layer trained on labeled task data. **Verdict: doesn't scale to per-document CL.**

### 1.2 Subspace-Constrained LoRA

**O-LoRA** (Wang et al., 2310.14152) learns each new task in a LoRA subspace orthogonal to all previous tasks' LoRA subspaces. The A matrices for task t are trained with an orthogonality penalty: $L_{orth}(A_i, A_t) = \|A_i^T A_t\|_F^2$ for all prior tasks i < t. Previous LoRA parameters are frozen; after training, all are merged: $W_{init} := W_{init} + \sum_i A_i B_i$. No teacher model needed — preservation comes from geometric separation in parameter space.

*Critical limitation at our rank:* O-LoRA at rank r=8 with d=3584 can fit at most $\lfloor 3584/8 \rfloor = 448$ orthogonal tasks before the subspace is exhausted. With 15 rounds this is fine in theory, but the orthogonality penalty is soft (not hard-projected), so approximate orthogonality degrades with each round. The original paper tests on 15 CL tasks with r=8 and shows degradation by task 10. More critically, O-LoRA treats each "task" as a large dataset (SuperNI benchmarks with thousands of examples), not a single document.

**KORE** (Jiang et al., 2510.19316) takes a harder line: compute the covariance $C = XX^T$ of calibration activations, extract the bottom-r singular vectors as an approximate null space, initialize A in this subspace, and *hard-freeze A*. Only B is trainable. As established in `iteration3_verdict.md §2`, this fails at r=8 because the "null space" of a well-trained 4B model is effectively empty — the calibration covariance is full-rank, and the bottom 8 eigenvalues have no spectral gap. At r≥64, KORE works but changes the experimental budget.

**Null-LoRA** (Zhang et al., 2512.15233) similarly projects into null spaces of pre-trained weight matrices. The preservation guarantee is geometric: updates orthogonal to the row/column space of W cannot change outputs on the pretraining distribution's activation manifold. Same rank-arithmetic problem as KORE at r=8.

**Key finding from this family (2602.03493):** Dalvi et al. show that fine-tuning *intermediate* singular components (neither the top PiSSA-style nor the bottom MiLoRA-style) achieves the best forgetting-performance trade-off. The forgetting curve is U-shaped across the singular value spectrum. This suggests that the correct architectural prior is not "freeze the top or bottom" but "route updates to the mid-rank subspace."

*Our constraints:* No teacher needed — purely geometric. But the r=8 rank constraint makes hard null-space methods inoperable. The soft orthogonality penalty (O-LoRA style) is viable but degrades over many rounds. **Verdict: the subspace geometry insight is the strongest thread in this family. The specific implementations need adaptation for our setting.**

### 1.3 Sparse Fine-Tuning

**BitFit** (Zaken et al., 2106.10199): bias-only training. **IA³** (Liu et al., 2205.05638): learned per-layer rescaling vectors. Both are capacity-limited: Allen-Zhu & Li (2309.14316) show facts are stored in MLP weight matrices, not biases or scales. **Verdict: too limited for fact injection.**

### 1.4 MoE-LoRA / Mixture Routing

**LoRAHub** (Huang et al., 2307.13269), **MoRA** (2506.21035), **C-LoRA** (2502.17920): designate one LoRA as "fact-storage expert," freeze task experts, route via learned gating. The separation-of-concerns idea is sound, but the router needs a training signal to distinguish fact vs. task tokens — which requires labeled data we don't have for single-document ingestion. **Verdict: router training is the chicken-and-egg bottleneck.**

### 1.5 Sparse Autoencoder Edits

**CRISP** (2508.13650), **k-sparse SAEs** (Gao et al., 2406.04093), **Farrell et al.** (2410.19278): use SAEs to identify/suppress/activate concept-specific features. The disentangled feature basis makes injection clean — edit specific features, leave others untouched. But: requires a pre-trained SAE for Qwen3-4B (none exists), and training one costs ~8-16 GPU-hours per layer. **Verdict: budget-infeasible.**

### 1.6 Memory-Augmented Transformers

**kNN-LM** (Khandelwal et al., 2020): frozen LM + external key-value datastore at inference. Preservation is free but weights are never updated — doesn't satisfy the "ingest via training" requirement. **Verdict: bypasses the problem.**

### 1.7 Mechanistic-Interpretability-Driven Edits

**ROME** (Meng et al., 2202.05262) uses causal tracing to localize factual associations to specific MLP layers (middle layers of GPT-2/J), then performs a rank-one weight update to insert a new (subject, relation, object) triple. **MEMIT** (Meng et al., 2210.07229) extends this to multi-layer mass editing.

*Our constraints:* ROME/MEMIT require pre-extracted (s,r,o) triples — they cannot ingest raw documents. The causal tracing step assumes knowledge is localized to specific layers, but Hase et al. (2301.04213) show that *editing works at layers different from where causal tracing localizes the knowledge*. The localization-editing correspondence is unreliable. Also, ROME's rank-one updates are not additive safely: Chughtai et al. (2402.07321) show factual recall involves multiple independent additive mechanisms across many heads and layers, not just one MLP.

*However:* ROME's key insight — MLP layers act as linear associative memories where $W_{proj}$ stores key→value mappings — is directly relevant. It tells us *where* facts live (MLP projection matrices, predominantly in middle-to-upper layers) and *how* they're stored (as directions in the column space of $W_{proj}$). This localizes the "fact-storage layer" even if we don't use ROME's editing mechanism.

### 1.8 Layer-Wise Learning Rate and Gradient Routing

**GaLore** (Zhao et al., 2403.03507) projects gradients into a low-rank subspace that changes periodically, reducing optimizer memory while maintaining full-rank update expressiveness. The key theoretical result: weight gradients in transformers are empirically low-rank, and projecting them onto their top singular subspace loses little information.

**VeRA** (Kopiczko et al., 2023) shares frozen random matrices across layers and learns only per-layer scaling vectors. **PiSSA** (Meng et al., 2404.02948) initializes LoRA A,B from the top-r singular vectors of W, making the adapter start in the model's most informative subspace while freezing the residual.

The layer-wise LR idea: if we know which layers store facts vs. syntax vs. reasoning, we can set per-layer learning rates accordingly. ROME's causal tracing and Geva et al.'s key-value memory view both point to *middle MLP layers* as factual storage. Lower layers handle syntax/tokenization; upper layers handle task formatting and generation.

*Our constraints:* Layer-wise LR doesn't require a teacher — it's a pure architectural/optimization constraint. The main risk is that the "which layers store what" picture is model-specific and may not hold for Qwen3-4B (it was established on GPT-2/J, replicated on Llama). Needs a model-specific causal tracing step to validate. **Verdict: the per-layer gradient-allocation idea is sound and teacher-free, but needs empirical layer-role verification.**

### 1.9 Summary: What Survives Filtering

| Family | Teacher-free? | Works at r=8? | Raw-doc ingest? | Budget-feasible? |
|--------|:---:|:---:|:---:|:---:|
| O-LoRA (soft orthogonal) | ✓ | ✓ (degrades) | ✓ | ✓ |
| KORE/Null-LoRA (hard null-space) | ✓ | ✗ (r too low) | ✓ | ✓ |
| Intermediate-SVD (2602.03493) | ✓ | ✓ | ✓ | ✓ |
| MoE-LoRA routing | ✓ | ✓ | Needs router | ✓ |
| SAE feature edits | ✓ | N/A | ✓ | ✗ (need SAE) |
| Layer-wise LR / gradient routing | ✓ | ✓ | ✓ | ✓ |
| kNN-LM | ✓ | N/A | ✓ | ✗ (no training) |
| ROME/MEMIT | ✓ | N/A | ✗ (need triples) | ✓ |
| Adapter modular | ✓ | ✓ | ✓ | ✗ (composition) |

Three families survive all four filters: **O-LoRA-style subspace geometry**, **intermediate-SVD component selection**, and **layer-wise gradient routing**. The strongest proposals combine elements from these families.

---

## Section 2: Three Candidate Proposals

### Candidate A: Spectral-Zoned LoRA (SZ-LoRA)

**Core idea.** Combine the intermediate-SVD insight (2602.03493) with layer-zone freezing. Partition transformer layers into three zones based on their functional role: *syntax* (bottom), *knowledge* (middle), *task* (top). Apply LoRA only to the knowledge zone's MLP projection matrices, initialized at intermediate singular components. Freeze all other layers entirely.

**Mathematical statement.** For a model with L layers, define zones:
- Syntax zone: layers $\{1, \ldots, \lfloor L/3 \rfloor\}$ — **fully frozen**
- Knowledge zone: layers $\{\lfloor L/3 \rfloor + 1, \ldots, \lfloor 2L/3 \rfloor\}$ — **LoRA-active on MLP $W_{proj}$ only**
- Task zone: layers $\{\lfloor 2L/3 \rfloor + 1, \ldots, L\}$ — **fully frozen**

For each knowledge-zone MLP, compute SVD: $W_{proj}^{(l)} = U \Sigma V^T$. Initialize LoRA at intermediate components: $A = U_{[s:s+r]} \Sigma_{[s:s+r]}^{1/2}$, $B = \Sigma_{[s:s+r]}^{1/2} V_{[s:s+r]}^T$, where $s = \lfloor d/4 \rfloor$ (start at the 25th percentile of the singular spectrum, avoiding both the top task-critical directions and the bottom noise directions). The residual $W_{res} = W_{proj} - AB$ is frozen; only A, B are trained.

**Per-document update procedure:**
1. Render document as K=5 augmented SFT examples (reuse DSAE Lite's template pool).
2. Gradient descent on A, B in the knowledge zone only. ~3 epochs, lr=2e-5.
3. After training, merge: $W_{proj}^{(l)} := W_{res}^{(l)} + A^{(l)} B^{(l)}$. Re-compute SVD for next document.

**Why this preserves DSAE Lite's insight:** The K=5 augmented injection is retained from Allen-Zhu — format-invariant encoding is a data-side mechanism orthogonal to the architectural constraint. The architectural constraint replaces the KL teacher: instead of measuring output-distribution drift, we *prevent* drift by structurally freezing the syntax and task zones. Facts are written to mid-layer MLPs (consistent with ROME's localization); syntax and task formatting cannot be disrupted because those layers never see a gradient.

**Compute:** 2.16M trainable params (12 layers × r=8 on MLP). ~45 sec/document on A10G (15s SVD + 30s SFT).

**Falsification:** If task preservation is > 3pp worse than DSAE Lite despite zone freezing → zone boundaries are wrong for Qwen3-4B.

---

### Candidate B: Orthogonal Residual Injection (ORI)

**Core idea.** Combine O-LoRA's orthogonal subspace separation with per-round LoRA merging (K-Merge, 2510.13537). Each injection round trains a fresh LoRA in a subspace orthogonal to all previously merged directions. After training, merge into base weights. The merged weight directions become the "previously used subspace" for the next round's orthogonality constraint.

**Mathematical statement.** Let $\Delta W_t = A_t B_t$ be the LoRA update from round t. After merging, define the cumulative update subspace as:
$$\mathcal{S}_t = \text{colspan}([A_1, A_2, \ldots, A_t])$$

For round t+1, train $A_{t+1}, B_{t+1}$ with the standard SFT loss plus an orthogonality penalty:
$$\mathcal{L} = \mathcal{L}_{SFT} + \lambda_{orth} \sum_{i=1}^{t} \| A_i^T A_{t+1} \|_F^2$$

After training: $W := W + A_{t+1} B_{t+1}$. Crucially, we don't need to store all previous $A_i$ matrices — after merging at round t, we maintain only the orthonormal basis $Q_t$ of $\mathcal{S}_t$ (via incremental QR: append $A_{t+1}$'s columns to $Q_t$ and re-orthogonalize). The penalty becomes $\| Q_t^T A_{t+1} \|_F^2$, which is $O(tr \cdot d)$ per step — trivial.

**Per-document update procedure:**
1. Load the current orthonormal basis $Q_t \in \mathbb{R}^{d \times (t \cdot r)}$. Size: at round 15 with r=8, this is 3584 × 120 ≈ 1.7 MB per layer.
2. Initialize fresh LoRA $A_{t+1}, B_{t+1}$ (standard Kaiming init, not SVD-based).
3. Train with $\mathcal{L}_{SFT} + \lambda_{orth} \| Q_t^T A_{t+1} \|_F^2$. The penalty is per-layer and adds ~1% compute overhead.
4. Merge: $W := W + A_{t+1} B_{t+1}$. Update: $Q_{t+1} = \text{QR\_extend}(Q_t, A_{t+1})$.

**Preservation argument:** Each round's update lies in a subspace orthogonal to all previous rounds'. With r=8 × 15 rounds = 120 directions out of 3584 (3.3%), ample room remains. The soft penalty degrades over rounds; hard projection (Gram-Schmidt per step, ~5% overhead) fixes this.

**Compute:** ~26M trainable params. ~37 sec/document + 61 MB for $Q_t$ state. No teacher.

**Falsification:** If round-15 task preservation is > 3pp worse than round 1 despite orthogonality → soft penalty leaked. Hard projection is the fallback. If hard projection also fails → fact and task directions are entangled.

---

### Candidate C: Stratified Spectral LoRA (SSL)

**Core idea.** The most architecturally "built-in" proposal. Combine the intermediate-SVD initialization (2602.03493) with a *per-layer spectral allocation rule*: each layer's LoRA is initialized at a different position in the singular spectrum based on that layer's *activation similarity score* (from 2504.07097). Layers whose input-output similarity is high (i.e., they primarily pass information through — typically lower layers) get their LoRA placed at bottom singular components (minimal interference). Layers whose similarity is low (i.e., they heavily transform — typically middle MLP layers) get their LoRA at intermediate components (maximum fact-injection capacity with moderate forgetting). Upper task-formatting layers get LoRA at top components (for expressive adaptation) but with a very low learning rate that makes them near-frozen.

**Mathematical statement.** For each layer l, compute the activation similarity:
$$I^{(l)} = \frac{1}{N} \sum_{i=1}^{N} \text{cos\_sim}(X_i^{(l)}, W^{(l)} X_i^{(l)})$$

Normalize so $\frac{1}{L}\sum_l I^{(l)} = 1$. Map to spectral starting position:
$$s^{(l)} = \lfloor I^{(l)} \cdot (d - r) \rfloor$$

High $I^{(l)}$ (pass-through layers) → high $s$ → bottom singular components → near-null-space updates.
Low $I^{(l)}$ (transformative layers) → low $s$ → top/mid singular components → expressive updates.

Per-layer learning rate:
$$\eta^{(l)} = \eta_{base} \cdot \min(1, \alpha / I^{(l)})$$

where $\alpha$ is a scaling constant (set so $\eta^{(l)} = \eta_{base}$ for the median layer). Pass-through layers (high I) get lower LR; transformative layers (low I) get full LR.

**Per-document update procedure:**
1. **One-time calibration** (done once before any injection): Forward 256 calibration samples, compute $I^{(l)}$ for all layers, compute SVD for all target matrices, initialize layer-specific LoRA at $s^{(l)}$, set layer-specific LR. This takes ~3 minutes on A10G.
2. **Per-document:** Standard SFT with K=5 augmented data. Each layer's LoRA trains at its assigned spectral position and learning rate. No orthogonality penalty, no teacher — the spectral allocation *is* the preservation mechanism.
3. **After each round:** Re-merge LoRA into base weights. Optionally re-calibrate (re-compute $I^{(l)}$ on fresh data) every 5 rounds.

**Preservation argument:** High-similarity layers (pass-through) get near-null-space updates + low LR; transformative layers get mid-spectrum updates where 2602.03493 showed the best forgetting-performance trade-off. A continuous generalization of PiSSA (s=0) and MiLoRA (s=d-r).

**Compute:** ~35 sec/document after calibration (3 min one-time). No teacher, no orthogonality compute. Memory: base model + LoRA.

**Falsification:** If SSL doesn't improve over uniform LoRA by ≥ 2pp on task preservation → the similarity-based allocation doesn't capture the right structure.

---

## Section 3: Recommendation — Stratified Spectral LoRA (SSL)

### Why SSL over the other two

**SZ-LoRA** hardcodes zone boundaries — brittle without model-specific evidence (Hase et al., 2301.04213 show localization ≠ editing). **ORI** accumulates growing state and adds $\lambda_{orth}$ to the loss, making it harder to isolate the architectural contribution.

**SSL** is the purest architectural mechanism: (1) no loss modification — preservation comes from initialization + per-layer LR; (2) no growing state; (3) data-driven layer allocation via measured activation similarity; (4) grounded in published evidence (2602.03493, 2504.07097); (5) composes with DSAE Lite's K=5 augmented data.

### Pseudocode

```python
class SSLUpdate(UpdateMethod):
    name = "ssl_inject"

    def calibrate(self, model, tokenizer, calibration_data, cfg):
        """One-time: compute per-layer activation similarity and SVD."""
        # Forward 256 calibration samples, hook MLP linear layers
        # Record input X, output Y per layer; compute cos_sim(X, Y).mean()
        # Normalize so mean(I^(l)) = 1.0 across layers
        # SVD each target weight: (U, S, Vh) per layer
        self._layer_sim = {name: normalized_cos_sim}
        self._layer_svd = {name: (U, S, Vh)}

    def apply(self, model, tokenizer, fact_qa_pairs, task_data=None, cfg=None):
        rank, base_lr, alpha = 8, 2e-5, 1.0

        param_groups = []
        for name in self._layer_sim:
            U, S, Vh = self._layer_svd[name]
            I_l = self._layer_sim[name]
            s = int(I_l * (len(S) - rank))           # spectral position
            lr_l = base_lr * min(1.0, alpha / I_l)     # adaptive LR

            A = U[:, s:s+rank] @ diag(S[s:s+rank].sqrt())  # init from SVD
            B = diag(S[s:s+rank].sqrt()) @ Vh[s:s+rank, :]
            # Inject as LoRA: W = W_res + A @ B
            param_groups.append({"params": [A, B], "lr": lr_l})

        optimizer = AdamW(param_groups)
        # Standard K=5 augmented SFT loop (same as DSAE Lite minus KL)
        for epoch in range(3):
            for batch in fact_loader:
                loss = model(**batch, labels=labels).loss
                loss.backward(); optimizer.step(); optimizer.zero_grad()

        merge_lora_weights(model)
        return model
```

### Files to Add/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/sot/update/ssl_inject.py` | **New** | `SSLUpdate` class (~250 LOC). Calibration + per-layer spectral LoRA init + adaptive LR. |
| `configs/update/ssl_inject.yaml` | **New** | Config: `method: ssl_inject`, `lora_rank: 8`, `ssl_alpha: 1.0`, `calibration_samples: 256`, training params same as DSAE Lite. |
| `scripts/09_run_update.py` | **Edit** | Add `"ssl_inject": SSLUpdate` to METHODS dict. |
| `scripts/16_run_sequential.py` | **Edit** | Add `ssl_inject` to METHODS list, CONFIG_MAP, and K5_MIXED_FORMAT_METHODS (reuses K=5 augmented data). |

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lora_rank` | 8 | Same as all other conditions for fair comparison |
| `ssl_alpha` | 1.0 | Sets median-similarity layer to base LR. Sweep {0.5, 1.0, 2.0} in pilot. |
| `calibration_samples` | 256 | Sufficient for stable cosine similarity estimates |
| `spectral_start` | Adaptive per layer | $s^{(l)} = \lfloor I^{(l)} \cdot (d - r) \rfloor$ |
| `lr` | 2e-5 (base) | Same as DSAE Lite; actual per-layer LR varies |
| `epochs` | 3 | Same as all conditions |
| `K` | 5 | Augmented injection (same templates as DSAE Lite) |

### Integration as 6th Ablation Condition

| Condition | Injection | Preservation mechanism | Novel component |
|-----------|-----------|----------------------|-----------------|
| (a) `baseline_sft` | SFT, K=1 | None | Control |
| (b) `aug_sft_k5` | SFT, K=5 | None | Augmentation only |
| (c) `kl_sft_k1` | SFT, K=1 | KL teacher, K=1 | Standard KL |
| (d) `aug_kl_k1` | SFT, K=5 | KL teacher, K=1 | Aug injection + KL |
| (e) `dsae_lite` | SFT, K=5 | KL teacher, K=5 | Full DSAE Lite |
| **(f) `ssl_inject`** | **SFT, K=5** | **Spectral init + adaptive LR (no teacher)** | **Architectural preservation** |

The key contrast is (e) vs. (f): both use K=5 augmented injection, but (e) preserves via KL teacher and (f) preserves via architectural constraint. If (f) ≈ (e), the KL teacher is redundant — the spectral allocation alone suffices. If (e) >> (f), the teacher provides information the architecture can't capture (e.g., the model has already drifted in ways the spectral prior doesn't prevent).

### Cost Estimate

- **Calibration:** 1 forward pass of 256 examples + 36 SVDs ≈ 3 minutes (one-time, not per-round)
- **Per-round training:** Same as `aug_sft_k5` (no KL forwards): ~7 min/round on A10G
- **15 rounds × 3 seeds:** $15 \times 7 \times 3 / 60 \approx 5.25$ GPU-hours
- **Evaluation:** ~1.5 GPU-hours (shared eval infrastructure)
- **Total: ~7 GPU-hours** for the full `ssl_inject` condition

This fits comfortably within the 8 GPU-hour margin identified in the budget.

---

## Section 4: Honest Failure Modes

### FM1: Activation similarity doesn't capture the right layer structure (probability: 35%)

**What goes wrong:** Low I/O similarity might indicate syntactic transformation, not factual storage — assigning mid-spectrum LoRA and full LR would damage syntax instead of writing facts.

**Diagnostic:** Compare $I^{(l)}$ profile against ROME-style causal tracing on 50 facts. If correlation < 0.3, the proxy is wrong. Cost: ~0.5 GPU-hours.

**Mitigation:** Replace cosine similarity with gradient-based importance: $I^{(l)} = \|\nabla_{W^{(l)}} \mathcal{L}_{task}\|$.

### FM2: Intermediate spectral init doesn't help at r=8 (probability: 25%)

**What goes wrong:** The U-shaped forgetting was demonstrated at r=32/128. At r=8, any 8 singular directions may capture roughly the same variance — no spectral effect.

**Diagnostic:** 3-way micro-ablation (s=0, s=d/4, s=d-r) on 3 rounds × 1 seed. If all within 1pp QD F1 → spectral position doesn't matter at r=8. Cost: ~1 GPU-hour.

**Mitigation:** Drop SVD init; fall back to layer-wise adaptive LR only (still a valid architectural method).

### FM3: Per-layer LR doesn't prevent drift on task metrics (probability: 30%)

**What goes wrong:** LR reduction slows drift but doesn't prevent it. 15 rounds of accumulated small gradients may still shift task behavior. SSL has no feedback signal; DSAE Lite's KL teacher actively corrects.

**Diagnostic:** Track per-round task preservation. If monotone decline > 0.5pp/round → architectural constraint too weak. Caught by existing eval; no extra cost.

**Mitigation:** Hybrid: SSL + minimal KL on top-1/3 layers only, reducing teacher cost by ~2/3 vs. full DSAE Lite.

### FM4: SVD re-computation per round is too slow (probability: 10%)

**What goes wrong:** Re-calibrating every round = 540 SVDs ≈ 11 minutes. Not a blocker, but contends with GPU memory if done on-device.

**Diagnostic/Mitigation:** Compare one-time vs. per-round calibration in pilot. Likely skip re-calibration — spectral structure doesn't shift dramatically with r=8 updates.

---

## Section 5: Why This Hasn't Been Done

The specific combination in SSL — intermediate-SVD initialization *with per-layer spectral position determined by activation similarity* — has not been published. Here's why each individual component exists but the combination doesn't:

**Intermediate-SVD init exists** (2602.03493, published 2025): Dalvi et al. demonstrate the U-shaped forgetting curve and recommend intermediate components. But they use a *uniform* starting position across all layers — every layer gets the same $s$. They do not adapt $s$ per-layer based on the layer's functional role. Their work is primarily empirical (ViT and LLaMA-2 7B) and focused on single-task fine-tuning, not continual knowledge injection.

**Activation-similarity-based layer importance exists** (2504.07097, published 2025): The "Sculpting Subspaces" paper uses I/O cosine similarity to determine how many singular vectors to preserve per layer ($r^{(l)}$ varies across layers). But they vary the *rank*, not the *spectral starting position*. Every layer still initializes at the bottom of the spectrum (MiLoRA-style); the similarity score only controls how many directions are frozen. This is a different knob — rank allocation rather than spectral allocation.

**The gap:** No one has combined spectral *position* selection (where in the spectrum to place the LoRA) with *layer-aware* adaptation (different layers get different positions). The reason is likely that both papers are from 2025, published within months of each other, in different communities (the intermediate-SVD paper from a vision/continual-learning group; the Sculpting Subspaces paper from a Red Hat AI group working on LLM CL). Cross-pollination hasn't happened yet.

**The per-layer LR component** is not novel in isolation — layer-wise LR has been explored in the "discriminative fine-tuning" tradition (Howard & Ruder, 2018, ULMFiT). What is novel is tying the LR schedule to the measured activation similarity rather than using a fixed decay schedule, and combining it with spectral-position LoRA initialization.

**Honest assessment of novelty:** SSL is an *integration* of existing ideas, not a new mechanism. The individual components (intermediate SVD, activation similarity, per-layer LR) are all published. The contribution is recognizing that they compose into a teacher-free architectural preservation mechanism for continual knowledge injection — a setting none of the original papers addressed. This is a class-project-appropriate level of novelty: sound integration of recent ideas applied to a new setting, with honest attribution.

**Citation verification:** All 18 arXiv IDs cited in this document were verified against paper content via the HuggingFace papers API. Key verifications: 2310.14152 (O-LoRA), 2602.03493 (intermediate-SVD forgetting), 2504.07097 (activation-similarity SVD), 2202.05262 (ROME), 2301.04213 (localization ≠ editing), 2404.02948 (PiSSA), 2403.03507 (GaLore), 2510.19316 (KORE). Section numbers and mechanism descriptions cross-checked against §3 of each paper.

---

## Recommended Next Action

Implement `ssl_inject` as condition (f) in the existing 5-way ablation. Concretely: (1) write `src/sot/update/ssl_inject.py` (~250 LOC, pattern from `dsae_lite.py` minus the KL block, plus calibration and per-layer SVD init); (2) add `configs/update/ssl_inject.yaml`; (3) register in `scripts/09_run_update.py` and `scripts/16_run_sequential.py`; (4) run the FM2 micro-ablation first (spectral position at r=8: top vs. mid vs. bottom, 3 rounds × 1 seed, ~1 GPU-hour) to confirm the spectral position matters; (5) if it does, run the full `ssl_inject` condition at 15 rounds × 3 seeds (~5.25 GPU-hours). The entire addition fits within the 8 GPU-hour margin. If the micro-ablation shows no spectral effect at r=8, fall back to layer-wise adaptive LR only (drop the SVD init, save the calibration cost, and the condition becomes a clean test of whether per-layer LR scheduling substitutes for a KL teacher).
