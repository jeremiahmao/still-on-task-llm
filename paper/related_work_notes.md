# Related Work Notes (for paper/draft.md)

## 1. Parametric knowledge editing
- **Meng et al. 2022 (ROME)**: Locates factual associations to specific middle-layer MLP modules via causal tracing and edits them with a rank-one weight update. **Relation to us:** Direct weight surgery at single-fact scale; we instead update an adapter continually via preference optimization, preserving the base weights.
- **Meng et al. 2023 (MEMIT)**: Extends ROME to mass-editing thousands of facts at once by solving a layer-wise least-squares update across several MLP layers. **Relation to us:** MEMIT is still "one-shot mass edit" — evaluated in batch, not under the continual, per-round injection setting we target.
- **Mitchell et al. 2022a (MEND)**: Learns a hypernetwork that maps a gradient on an edit example into a low-rank update to model weights, enabling fast single-edit application. **Relation to us:** Editor-as-network paradigm; we instead reuse standard SFT/preference losses over LoRA, keeping the pipeline identical to normal fine-tuning.
- **Mitchell et al. 2022b (SERAC)**: Stores edits in an external memory with a scope classifier and a counterfactual model, routing queries to the memory on hit. **Relation to us:** Non-parametric edit store; our method keeps all new knowledge inside parametric LoRA weights so it composes with downstream reasoning.
- **Dai et al. 2022 (Knowledge Neurons)**: Identifies individual MLP neurons whose activations are causally tied to factual predictions, enabling targeted suppress/amplify edits. **Relation to us:** Inspires the mechanistic-analysis phase, where we look at which LoRA subspaces absorb a new fact versus preserve task behavior.
- **Yao et al. 2023 (survey)**: Survey of knowledge editing for LLMs that taxonomizes memory-based, meta-learning, and locate-then-edit families and highlights evaluation gaps. **Relation to us:** Our evaluation suite (absorption, locality, compositional, mechanistic) is an explicit response to the gaps the survey calls out.

## 2. DPO/preference-based editing
- **Rozner et al. 2024 (KDPO, EMNLP)**: Casts a single fact edit as a DPO problem — the new fact is the "chosen" completion and the old fact the "rejected" — and applies standard DPO with the pre-edit model as reference. **Relation to us:** Closest prior art. KDPO (a) uses the pre-edit base as the KL anchor, which drifts as edits accumulate, and (b) does not inject gold answers into candidate pools for absorption. We swap the anchor for a task-replay reference and add gold-injection for monotonic absorption.
- **Rafailov et al. 2023 (DPO)**: Derives a closed-form policy from a Bradley-Terry preference model, eliminating the explicit reward/RL loop of RLHF. **Relation to us:** Foundational loss; both KDPO and COPR sit on top of DPO-style objectives.
- **Azar et al. 2024 (IPO)**: Shows DPO can overfit to deterministic preferences and proposes an identity-mapped objective with better regularization. **Relation to us:** Motivates KL-style regularization terms — our baseline KL-SFT and COPR's KL-replay both address the same overfitting risk.

## 3. COPR and continual preference alignment
- **Zhang et al. 2025 (COPR, ACL Findings)**: Continual Optimal Policy Regularization for RLHF-style alignment across a stream of preference tasks; maintains a KL-regularized replay buffer of past-task prompts to anchor the new policy toward the old optimal policy, avoiding catastrophic alignment drift. **Relation to us:** Seed paper. COPR was proposed for *preference* continual learning (helpfulness/harmlessness). We port its inductive bias — KL regularization anchored on task-replay rather than the frozen base — to *knowledge injection*, treating each new fact as a mini preference task.
- **Zhang et al. 2024 (CPPO)**: Continual proximal policy optimization variant with replay for sequential RLHF tasks. **Relation to us:** Parallel continual-RLHF line; confirms replay + KL is the dominant prescription.
- **Bai et al. 2022 (Constitutional AI / RLHF at scale)**: Foundational RLHF-from-preferences recipe with a reference-model KL term. **Relation to us:** The KL-to-reference template we inherit from; our contribution is *which* reference to use under continual editing.

## 4. LoRA / PEFT for knowledge editing
- **Hu et al. 2021 (LoRA)**: Low-rank adapters injected in parallel to frozen attention/MLP projections, reducing trainable parameters by orders of magnitude. **Relation to us:** Our entire training substrate — we edit only LoRA deltas, never the base.
- **Huang et al. 2023 (LoRAHub)**: Composes multiple task-specific LoRA modules via black-box optimization for cross-task generalization. **Relation to us:** Suggests LoRA subspaces for different facts are approximately additive; we stay within a single LoRA slot and let COPR's regularizer arbitrate conflicts.
- **Wu et al. 2024 (MELO / LoRA-based editing)**: Uses LoRA adapters as the update medium for knowledge editing with routing/gating. **Relation to us:** Validates LoRA as an editing substrate; our contribution is the *training objective* on top of it, not the adapter topology.
- **Aghajanyan et al. 2021 (intrinsic dimensionality)**: Empirical evidence that fine-tuning occupies a very low-dimensional subspace. **Relation to us:** Justifies why LoRA (rank 8-32) suffices for absorbing a fact and motivates the mechanistic phase.

## 5. Continual learning for LLMs
- **Wang et al. 2024 (InsCL)**: Instruction-tuning continual learning with a task-replay buffer selected by instruction similarity; shows replay beats EWC-style regularization on instruction streams. **Relation to us:** Our "task-replay-anchored reference" in COPR is in the spirit of InsCL — replay as the mechanism against forgetting.
- **Kirkpatrick et al. 2017 (EWC)**: Elastic Weight Consolidation penalizes movement of parameters important to old tasks via a Fisher-weighted quadratic term. **Relation to us:** Classical regularization baseline; COPR's KL term is its functional-space analogue.
- **Luo et al. 2023 (catastrophic forgetting in LLMs)**: Empirical study showing instruction-tuned LLMs forget general capabilities rapidly under continual SFT. **Relation to us:** Directly motivates the task-preservation axis of our evaluation (Phase 1).
- **Wu et al. 2024 (continual learning for LLMs survey)**: Survey covering replay, regularization, and architecture-based CL for LLMs. **Relation to us:** Places our method in the replay+regularization quadrant.

## 6. Compositional / multi-hop reasoning evaluation
- **Cohen et al. 2024 (Ripple Effects of Knowledge Editing, TACL)**: Shows that editing a single fact rarely propagates to its logical consequences (e.g., editing a spouse does not update "mother-in-law"); proposes RippleEdits benchmark. **Relation to us:** The direct motivation for Phase 4 (compositional 2-hop eval) — we measure whether COPR+gold injection ripples through a learned hop, not just the edited triple.
- **Yang et al. 2018 (HotpotQA)**: Multi-hop QA benchmark with supporting-fact supervision. **Relation to us:** Template for constructing our 2-hop eval items from financial fact triples.
- **Trivedi et al. 2022 (MuSiQue)**: Multi-step QA designed to resist single-hop shortcuts by explicit composition of primitive hops. **Relation to us:** Design inspiration for our "primitive fact + composition step" test construction.
- **Ho et al. 2020 (2WikiMultihopQA)**: 2-hop QA over Wikipedia with structured reasoning paths. **Relation to us:** Another template source; our financial 2-hop items follow the same "bridge entity" structure.
- **Zhong et al. 2023 (MQuAKE)**: Multi-hop benchmark specifically for evaluating knowledge editing beyond the edited fact. **Relation to us:** Methodologically nearest to our Phase 4; we adopt its "edited-fact-as-bridge" pattern on financial triples.

## 7. Mechanistic analysis of LoRA updates
- **Hu et al. 2021 (LoRA, rank analysis)**: Original paper shows that effective rank of the update is far below the nominal LoRA rank. **Relation to us:** Baseline for subspace occupancy metrics.
- **Zhu et al. 2024 (LoRA subspace analysis)**: Analyzes singular directions of LoRA B @ A and shows task-specific directions dominate. **Relation to us:** Methodology for our mechanistic phase — we compare singular spectra of knowledge-only vs. task-preserving LoRA slots. % VERIFY
- **Aghajanyan et al. 2021 (intrinsic dimensionality)**: See Section 4. **Relation to us:** Quantitative prior on why a rank-r LoRA can host a new fact without destroying the task subspace.

## Suggested prose structure for Section "Related Work" (600 words target)
- **Paragraph 1** — Parametric editing line: open with ROME/MEMIT/MEND/SERAC/KN, frame them as "single-shot edits into weights or side memory". Note Yao et al.'s survey flags continual + compositional eval as open gaps.
- **Paragraph 2** — Preference-based editing: introduce KDPO as the closest prior art. Describe its reference-model choice (frozen pre-edit base) and lack of gold-answer injection; position our COPR+gold-injection as addressing exactly these two design points.
- **Paragraph 3** — COPR's origin: sketch continual preference alignment (COPR, CPPO), emphasize the KL-to-replay-anchored-reference idea, and argue its inductive bias (protect past-optimal behavior while admitting new preferences) maps one-to-one onto continual knowledge updates.
- **Paragraph 4** — Continual-learning scaffolding: InsCL/EWC/Luo et al./Wu survey. Position our task-replay anchor as functional-space regularization that generalizes EWC and inherits InsCL's empirical replay wins.
- **Paragraph 5** — Compositional evaluation: Cohen et al.'s ripple effects + MQuAKE / HotpotQA / MuSiQue. Motivate Phase 4: an editing method that absorbs the edited triple but fails 2-hop composition has not really "learned" the fact.

*(Optional Paragraph 6, if space) — LoRA mechanics: tie Aghajanyan et al. and LoRA rank analyses to the mechanistic Phase 5, framing our subspace comparison as a diagnostic that complements behavioral metrics.*
