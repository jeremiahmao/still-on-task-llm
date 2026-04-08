# Final Execution Plan

**Last updated:** 2026-03-28
**Authors:** Jeremiah Mao, Mateo Juliani

---

## Scope Summary

This plan reconciles the interim report (March 25) with realistic compute and
time constraints. It is the single source of truth for what ships in the final
paper.

**Research question:** Does policy-space regularization (COPR-adapted) better
preserve task performance in task-tuned models during knowledge injection
compared to weight-space editing (AlphaEdit)?

**Model:** Qwen2.5-3B, LoRA fine-tuned for financial query decomposition (QD).

---

## Methods (4 total)

| Method | Type | Description | Status |
|--------|------|-------------|--------|
| No update | Lower bound | Task-tuned model, no knowledge injection | Core |
| Naive SFT | Baseline | SFT on fact QA pairs (= KL-reg with lambda=0) | Core |
| KL-reg SFT | Baseline | SFT + lambda * D_KL(pi_task \|\| pi_theta) | Core |
| COPR-adapted | Policy-space | Advantage-weighted fitting + task replay | Core |

**Dropped:**
- **AlphaEdit** — unlikely to work in repeated applications (per advisor
  feedback); better to focus GPU time on COPR and gradient-learning baselines.
- **Mixed replay** — similar to KL-reg SFT; adds a comparison axis (data mixing
  vs. loss regularization) that dilutes the core story. Can mention as future work.
- **Full retrain** — the review noted it's hard to interpret as an upper bound;
  compute-prohibitive; not needed for the core comparison.

---

## Edit Scales (2, down from 3)

| Scale | Role |
|-------|------|
| **1,000** | Primary comparison point — midpoint, enough to stress methods |
| **3,000** | Scaling stress test — AlphaEdit's documented limit |

**Dropped:** 200 edits. Too small to differentiate methods meaningfully and
does not add to the scaling story that 1K→3K already provides.

---

## Phases

### Phase 0: Data Pipeline
**Goal:** Produce all inputs needed for experiments.

| Step | Script | Output |
|------|--------|--------|
| Download FNSPID + FinQA | `01_download_data.py` | `data/fnspid/`, `data/finqa/` |
| Extract fact triples from post-cutoff FNSPID | (data pipeline) | `data/fnspid/triples/triples_{1000,3000}.json` |
| Build pre-cutoff retrieval corpus | `02_build_corpus.py` | `data/fnspid/corpus/` |
| Build FAISS index | `03_build_faiss_index.py` | `data/fnspid/index/` |
| Generate QD training data (teacher LLM) | (data pipeline) | `data/qd/train.json`, `data/qd/test.json` |
| Build locality test facts | (data pipeline) | `data/fnspid/locality_facts.json` |

### Phase 1: Core Comparison at 1K Edits (MUST SHIP)
**Goal:** The minimum viable result for the paper.

**Prerequisite:** Task-tune Qwen2.5-3B on QD via SFT.

**Experiments (3 methods + baseline):**

| Run | Method | Scale | Task | Metrics |
|-----|--------|-------|------|---------|
| 1 | No update | — | QD | preservation, locality |
| 2 | Naive SFT | 1K | QD | preservation, absorption, locality |
| 3 | KL-reg SFT | 1K | QD | preservation, absorption, locality |
| 4 | COPR-adapted | 1K | QD | preservation, absorption, locality |

**Success criteria:** All 4 runs complete with valid metrics. Results show
meaningful differentiation between at least two methods on task preservation.

### Phase 2: Scaling to 3K Edits (SHOULD SHIP)
**Goal:** Show how methods degrade as edit count increases.

**Experiments (3 methods at 3K):**

| Run | Method | Scale | Task | Metrics |
|-----|--------|-------|------|---------|
| 5 | Naive SFT | 3K | QD | preservation, absorption, locality |
| 6 | KL-reg SFT | 3K | QD | preservation, absorption, locality |
| 7 | COPR-adapted | 3K | QD | preservation, absorption, locality |

**Success criteria:** Scaling curves (1K vs 3K) plotted for each method.

### Phase 3: FinQA Forgetting Control (TIME PERMITTING)
**Goal:** Verify methods don't cause generic forgetting on unrelated tasks.

Run top 2 methods from Phase 1 on a FinQA-tuned model at 1K edits.
Measure execution accuracy before/after update. Expected outcome: no
degradation (knowledge updates shouldn't affect table arithmetic).

This is a guardrail, not the core result. If all methods preserve FinQA, report
as a one-line confirmation. If any method degrades it, that's a finding.

---

## Metrics

| # | Metric | What it measures | Primary? |
|---|--------|-----------------|----------|
| 1 | **Task preservation** | Recall@10 on held-out QD test set | Yes |
| 2 | **Knowledge absorption** | Exact match + F1 on fact-probe questions | Yes |
| 3 | **Locality** | Accuracy on untouched facts (same-entity / same-sector / unrelated) | Yes |
| 4 | **Compute cost** | GPU-hours, peak memory, wall time per method | Yes |
| 5 | **Generic forgetting** | FinQA execution accuracy (Phase 3 only) | No |

---

## Key Comparisons for the Paper

1. **Naive SFT vs KL-reg SFT:** Does KL regularization help at all?
2. **KL-reg SFT vs COPR-adapted:** Does advantage-weighted fitting help beyond
   plain KL regularization? (This is the ablation from Contribution 3.)
3. **1K vs 3K scaling:** Where does each method start to degrade?

---

## What's NOT in Scope

- No ablation sweeps (replay %, K, lambda values) — report fixed values only
- No DPO refinement of QD model (SFT-only is sufficient)
- No mixed replay method
- No full retrain upper bound
- No 200-edit scale
- No novel dataset claims (QD benchmark is operational scaffolding only)

---

## Total Experiment Budget

| Phase | Runs | Est. GPU-hours each | Total |
|-------|------|-------------------|-------|
| Phase 1 | 4 | ~2-4h | ~8-16h |
| Phase 2 | 3 | ~3-6h | ~9-18h |
| Phase 3 | 2 | ~2-4h | ~4-8h |
| **Total** | **9** | | **~21-42h** |

This is a ~75% reduction from the original plan (7 methods x 3 scales x 2 tasks
= 42 runs → 9 runs).
