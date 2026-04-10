# Final Execution Plan

**Last updated:** 2026-04-09
**Authors:** Jeremiah Mao, Mateo Juliani

---

## Research Question

Does policy-space regularization (COPR-adapted) better preserve task performance
in task-tuned models during knowledge injection compared to gradient-learning
baselines (naive SFT, KL-reg SFT)?

**Setting:** Qwen2.5-3B LoRA fine-tuned for financial query decomposition (QD)
on pre-cutoff FNSPID data, then updated with post-cutoff fact triples.

---

## Methods (4 total)

| Method | Type | Description |
|--------|------|-------------|
| No update | Lower bound | Task-tuned model, zero knowledge injection |
| Naive SFT | Baseline | SFT on fact QA pairs (KL-reg with lambda=0) |
| KL-reg SFT | Baseline | SFT + lambda * D_KL(pi_task \|\| pi_theta), lambda=0.1 |
| COPR-adapted | Policy-space | Advantage-weighted fitting + 5% task replay |

**Dropped from interim report:**
- **AlphaEdit** — advisor feedback: edits are fragile under prior fine-tuning;
  activations move in directions orthogonal to editing directions
- **Mixed replay** — too similar to KL-reg SFT; dilutes the core story
- **Full retrain** — compute-prohibitive and hard to interpret as upper bound
- **200-edit scale** — *reinstated* (see below)

---

## Edit Scales (3)

| Scale | Role |
|-------|------|
| **200** | Low-stress regime; shows where methods are equivalent |
| **1,000** | Primary comparison point |
| **3,000** | Scaling stress test |

200 edits reinstated: costs nothing extra (same checkpoint, subset of triples)
and gives a 3-point scaling curve rather than 2-point.

---

## Corpus & Data

### Tickers (50 stocks)
Selected from FNSPID by max balanced pre/post-cutoff coverage, ETFs excluded.
See `configs/data/fnspid.yaml` for the full list (~110K post-cutoff articles total).

### Triple extraction
- **Script:** `04_extract_triples.py`
- **Scope:** Post-cutoff articles for the 50 tickers above
- **Volume:** ~110K API calls
- **Approach:** OpenAI Batch API (overnight job, 50% cost reduction)
- **Teacher model:** TBD (gpt-4o-mini or gpt-4.1-mini)
- **Min cross-doc agreement:** 2 (production); 1 (debug)
- **Output:** `data/fnspid/triples/filtered_triples.json` + scaled subsets at 200/1K/3K

### QD training data
- **Script:** `05_generate_qd_data_foundational_model.py`
- **Scope:** Same 50 tickers; topic pairs require ≥5 articles per side of cutoff
- **Target:** 300-500 training examples (sufficient for LoRA SFT on 3B model)
- **Volume:** ~1,500 API calls (3 per pair: question + pre-decomp + post-decomp)
- **Teacher model:** TBD (same as above)
- **Output:** `data/qd_temporal/train.json`, `data/qd_temporal/test.json`

---

## Phases

### Phase 0: Data Pipeline
| Step | Script | Output |
|------|--------|--------|
| Download FNSPID + FinQA | `01_download_data.py` | `data/fnspid/`, `data/finqa/` |
| Build pre/post corpus | `02_build_corpus.py` | `data/fnspid/processed/` |
| Build FAISS index | `03_build_faiss_index.py` | `data/fnspid/index/` |
| Extract fact triples | `04_extract_triples.py` | `data/fnspid/triples/` |
| Generate QD training data | `05_generate_qd_data_foundational_model.py` | `data/qd_temporal/` |
| Build locality test facts | `06_build_locality_facts.py` | `data/fnspid/locality_facts.json` |

### Phase 1: Core Comparison at 1K Edits (MUST SHIP)
Task-tune Qwen2.5-3B on QD, then run all 4 methods.

| Run | Method | Scale | Metrics |
|-----|--------|-------|---------|
| 1 | No update | — | preservation, locality |
| 2 | Naive SFT | 1K | preservation, absorption, locality |
| 3 | KL-reg SFT | 1K | preservation, absorption, locality |
| 4 | COPR-adapted | 1K | preservation, absorption, locality |

**Success criteria:** All 4 runs complete. Meaningful differentiation on task preservation
between at least two methods.

### Phase 2: Scaling Curve (SHOULD SHIP)
Run the 3 update methods at 200 and 3K to complete the scaling picture.

| Run | Method | Scale | Metrics |
|-----|--------|-------|---------|
| 5 | Naive SFT | 200 | preservation, absorption, locality |
| 6 | KL-reg SFT | 200 | preservation, absorption, locality |
| 7 | COPR-adapted | 200 | preservation, absorption, locality |
| 8 | Naive SFT | 3K | preservation, absorption, locality |
| 9 | KL-reg SFT | 3K | preservation, absorption, locality |
| 10 | COPR-adapted | 3K | preservation, absorption, locality |

**Success criteria:** 3-point scaling curves (200/1K/3K) for each method.

### Phase 3: FinQA Forgetting Control (TIME PERMITTING)
Run top 2 methods from Phase 1 on a FinQA-tuned model at 1K edits.
Measure execution accuracy before/after update.

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

1. **Naive SFT vs KL-reg SFT** — does KL regularization help?
2. **KL-reg SFT vs COPR-adapted** — does advantage-weighted fitting help beyond plain KL?
3. **200 vs 1K vs 3K** — where does each method start to degrade?

---

## What's NOT in Scope

- No ablation sweeps (replay %, K, beta, lambda) — report fixed values only
- No DPO refinement of QD model
- No mixed replay method
- No full retrain upper bound
- No novel dataset claims (QD benchmark is operational scaffolding)
- No AlphaEdit

---

## Experiment Budget

| Phase | Runs | Est. GPU-hours each | Total |
|-------|------|-------------------|-------|
| Phase 1 | 4 | ~2-4h | ~8-16h |
| Phase 2 | 6 | ~2-4h | ~12-24h |
| Phase 3 | 2 | ~2-4h | ~4-8h |
| **Total** | **12** | | **~24-48h** |

### API cost estimate (data pipeline only)
| Step | Calls | Est. cost |
|------|-------|-----------|
| Triple extraction (Batch API) | ~110K | ~$8-9 |
| QD pair generation | ~1,500 | <$1 |
| **Total** | | **~$9-10** |
