# DSAE Lite: Code-Level Implementation Spec

**Date:** 2025-04-25 &ensp; **Status:** Pre-implementation. Locked method per `ml_intern_iteration3_verdict.md §4`.

---

## §1 New Method Classes and Configs

### Condition map (5-way ablation)

| ID | Condition | Injection | Preservation | New code? |
|----|-----------|-----------|-------------|-----------|
| (a) | `baseline_sft` | SFT, K=1 | None | **No.** Reuse `NaiveSFTUpdate` + `naive_sft.yaml`. |
| (b) | `aug_sft_k5` | SFT, K=5 | None | **No new class.** Reuse `NaiveSFTUpdate`. New config + K=5 data. |
| (c) | `kl_sft_k1` | SFT, K=1 | KL, K=1 | **No.** Reuse `KLRegSFTUpdate` + `kl_reg_sft.yaml`. |
| (d) | `aug_kl_k1` | SFT, K=5 | KL, K=1 | **No new class.** Reuse `KLRegSFTUpdate`. New config + K=5 data. |
| (e) | `dsae_lite` | SFT, K=5 | KL, K=5 | **Yes. New class + config.** |

**Rationale for (b) and (d):** Both classes are data-agnostic. K=5 augmentation is purely a data-prep concern; the pipeline feeds K=5 triples and the same class handles them. The KL in `KLRegSFTUpdate` operates on `task_data` replay (K=1 format), correct for condition (d).

### New file: `src/sot/update/dsae_lite.py`

SFT on K=5 augmented facts (data prep handles this). KL preservation on task-replay prompts rendered in K formats, averaged.

**Class signature:**

```python
class DSAELiteUpdate(UpdateMethod):
    """DSAE Lite: K=5 augmented injection + K=5 augmented KL preservation."""

    @property
    def name(self) -> str:
        return "dsae_lite"

    def apply(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        fact_qa_pairs: list[dict],
        task_data: list[dict] | None = None,
        cfg: DictConfig | None = None,
    ) -> PreTrainedModel: ...
```

**Subclassing:** Standalone (not subclassing `KLRegSFTUpdate`). The K-format KL loop rewrites enough of the training loop that inheritance would mean overriding it entirely. Copy pattern from `kl_reg_sft.py` (173 LOC) with modified KL block.

**Key difference from `KLRegSFTUpdate`.** In the KL step:
1. Takes a replay batch of task prompts (raw `messages` from `task_data`).
2. Renders each prompt in K=5 formats using the same template pool as injection.
3. Forward-passes all K renderings through both `ref_model` and `model`.
4. Computes per-format KL, averages: `kl_loss = (1/K) * Σ_k KL_k`.

**Estimated LOC:** ~220 (173 base from `kl_reg_sft.py` + ~50 for K-format rendering loop).

### New file: `configs/update/dsae_lite.yaml`

```yaml
method: dsae_lite
kl_lambda: 0.1        # inherited from kl_reg_sft; same default
num_kl_formats: 5     # K for preservation-side rendering
replay_pct: 0.05      # fraction of task_data used per round
training:
  lr: 2.0e-5
  epochs: 3
  batch_size: 8       # in individual examples (K=5 data = ~1000 per round)
  gradient_accumulation_steps: 2
  max_seq_length: 512
  bf16: true
```

### New file: `configs/update/aug_sft_k5.yaml`

```yaml
method: naive_sft     # same class
training:
  lr: 2.0e-5
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 2
  max_seq_length: 512
  bf16: true
```

### New file: `configs/update/aug_kl_k1.yaml`

```yaml
method: kl_reg_sft    # same class
kl_lambda: 0.1
training:
  lr: 2.0e-5
  epochs: 3
  batch_size: 8
  gradient_accumulation_steps: 2
  max_seq_length: 512
  bf16: true
```

### Registration changes

**`scripts/09_run_update.py`** — add to `METHODS` dict:

```python
from sot.update.dsae_lite import DSAELiteUpdate

METHODS = {
    ...  # existing entries unchanged
    "aug_sft_k5": NaiveSFTUpdate,         # same class, K=5 data
    "aug_kl_k1": KLRegSFTUpdate,          # same class, K=5 data, K=1 KL
    "dsae_lite": DSAELiteUpdate,           # new class
}
```

**`scripts/16_run_sequential.py`** — add to `METHODS` list and `CONFIG_MAP`:

```python
METHODS = [
    ...  # existing entries
    "aug_sft_k5",
    "aug_kl_k1",
    "dsae_lite",
]

CONFIG_MAP = {
    ...
    "aug_sft_k5": "configs/update/aug_sft_k5.yaml",
    "aug_kl_k1": "configs/update/aug_kl_k1.yaml",
    "dsae_lite": "configs/update/dsae_lite.yaml",
}

# Add to the mixed-format routing (new set):
K5_MIXED_FORMAT_METHODS = {"aug_sft_k5", "aug_kl_k1", "dsae_lite"}
```

The round-data routing in `run_method()` currently branches on `MIXED_FORMAT_METHODS` / `LEAKFREE_MIXED_FORMAT_METHODS` to select the triples subdir. Add a third branch:

```python
if method in K5_MIXED_FORMAT_METHODS:
    subdir = "sequential_k5"
elif method in LEAKFREE_MIXED_FORMAT_METHODS:
    subdir = "mixed_format_sequential_leakfree"
elif method in MIXED_FORMAT_METHODS:
    subdir = "mixed_format_sequential"
else:
    subdir = "sequential"
```

---

## §2 K=5 Mixed-Format Data Preparation

### The 5 templates

All templates are **leak-free**: the gold answer (`object`) NEVER appears in the `user` prompt or any part of the input the model conditions on. It appears ONLY in the `assistant` target (the supervised signal).

| k | Format name | User turn | Assistant turn |
|---|-------------|-----------|---------------|
| 1 | **QA** (cloze) | `"Complete this statement: {phrasing_prefix}"` | `"{object}"` |
| 2 | **QD** (query-decomp, leak-free) | `"What should I know about {subject}'s recent activity?"` | `"Sub-query 1: What is the {relation_human} of {subject}?\nSub-query 2: Recent updates from {subject}.\nSub-query 3: Latest announcements by {subject}."` |
| 3 | **Declarative** | `"Summarize this fact in one sentence: {subject}, {relation_human}."` | `"{subject}'s {relation_human} is {object}."` |
| 4 | **Instruction** | `"You are a financial analyst. A colleague asks: what is {subject}'s {relation_human}? Answer in one phrase."` | `"{object}"` |
| 5 | **Narrative** | `"Write a brief news snippet mentioning {subject}."` | `"In recent developments, {subject} announced that its {relation_human} is {object}. This was confirmed in the latest filing."` |

Template 1 (QA/cloze) reuses the existing `render_triple()` output. Template 2 reuses the existing leak-free QD renderer from `scripts/24_prepare_mixed_format_triples.py`. Templates 3–5 are new.

### Script changes

**Extend `scripts/24_prepare_mixed_format_triples.py`** with:

- New flag: `--num-formats {2,5}` (default 2 for backward compat).
- When `--num-formats 5`, emit 5 copies per triple instead of 2, each with `train_format` ∈ `{qa, qd, declarative, instruction, narrative}` and pre-rendered `messages` field.
- The `--leak-free` flag is **always on** when `--num-formats 5` (K=5 implies the new experiment; no reason to ever combine K=5 with leaky templates). Enforce this in argparse.

**New batch-prep helper: `scripts/26_prep_all_k5_rounds.sh`**

```bash
#!/bin/bash
# Prepare K=5 mixed-format triples for all sequential rounds.
for k in $(seq 1 15); do
  python scripts/24_prepare_mixed_format_triples.py \
    --input data/fnspid/triples/sequential/round_${k}.json \
    --output data/fnspid/triples/sequential_k5/round_${k}.json \
    --num-formats 5 --leak-free
done
```

### Output path

```
data/fnspid/triples/sequential_k5/round_{1..15}.json
```

Each file: `N_facts × 5` entries (e.g., 200 facts × 5 = 1000 entries per round).

---

## §3 Per-Format KL Preservation Loss (the novel ingredient)

### Preservation prompts source

`task_data` from `data/qd_temporal/train.json` — same as `KLRegSFTUpdate`. Sampled at `replay_pct` (5%). No new data needed.

### Format rendering for preservation

Preservation prompts are task examples (complex financial questions), not fact triples — no (subject, relation, object) structure. The K=5 rendering is **prompt-wrapping**: same question under K system-prompt/instruction framings:

| k | Preservation framing |
|---|---------------------|
| 1 | Original (QD system prompt + user question) |
| 2 | No system prompt, bare user question |
| 3 | `"You are a financial analyst. Answer concisely: {question}"` |
| 4 | `"Given the following question, provide a detailed response:\n\n{question}"` |
| 5 | `"Question: {question}\nPlease provide your analysis."` |

KL penalizes drift from `π_ref` under ALL framings, not just the training framing.

### Loss computation pseudocode

```python
# Inside DSAELiteUpdate.apply(), KL block per training step:
# replay_batch: list[str] of raw user questions (sampled from task_data)
# K: number of preservation formats (cfg.num_kl_formats, default 5)
# PRESERVATION_FRAMINGS: list of K template functions (question -> chat messages)

kl_accum = 0.0
for k, framing_fn in enumerate(PRESERVATION_FRAMINGS):
    # Render replay batch in format k
    formatted_texts = [
        tokenizer.apply_chat_template(
            framing_fn(q), tokenize=False, add_generation_prompt=False
        )
        for q in replay_questions
    ]
    tokens = tokenizer(
        formatted_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=max_seq_length
    ).to(device)

    # Forward through current model and frozen reference
    cur_logits = model(**tokens).logits
    with torch.no_grad():
        ref_logits = ref_model(**tokens).logits

    # Per-position KL: KL(π_ref || π_θ)
    cur_lp = F.log_softmax(cur_logits, dim=-1)
    ref_lp = F.log_softmax(ref_logits, dim=-1)
    kl_per_pos = (ref_lp.exp() * (ref_lp - cur_lp)).sum(dim=-1)
    n_real = tokens["attention_mask"].sum().clamp(min=1)
    kl_k = (kl_per_pos * tokens["attention_mask"]).sum() / n_real
    kl_accum += kl_k

kl_loss = kl_accum / K  # average over formats
(kl_lambda * kl_loss).backward()
```

### Memory / batch-size implications

K=5 loop is **sequential** (not batched), so peak memory ≈ same as `kl_reg_sft` (SFT forward + 2× single-format KL forward). Wall time: ~5× per KL step. 200 facts × 5 formats = 1000 examples, 3 epochs → ~375 steps × (1 SFT + 10 KL forwards) ≈ 4125 forwards/round. Estimated: ~15 min/round on A10G (vs. ~3 min for baseline SFT).

### KL coefficient β

`kl_lambda: 0.1` — same as `kl_reg_sft.yaml`. With K=5 averaging, each format-k KL is ≈ same scale. Start at 0.1; pilot (§7) will confirm.

---

## §4 Per-Round Per-Format KL Monitoring (free experiment d)

In `src/sot/update/dsae_lite.py`, inside the per-format KL loop:

```python
# After kl_k computation:
per_format_kl[k].append(kl_k.item())
```

At epoch end, write per-format KL averages:

```python
format_kl_log.append({
    "epoch": epoch,
    "per_format_kl": {
        FRAMING_NAMES[k]: np.mean(vals)
        for k, vals in per_format_kl.items()
    },
    "mean_kl": np.mean([np.mean(v) for v in per_format_kl.values()]),
})
```

**Output:** Saved to `outputs/<run_id>/metadata.json` (as `per_format_kl` key) and standalone:
```
outputs/seq_dsae_lite_round_{k}_qd_scale200/per_format_kl.json
```

**Plotting:** Load `per_format_kl.json` per round → line plot (x=round, y=KL, one line per format). Expect all K lines anchored near 0 under dsae_lite; under aug_kl_k1, non-training formats drift upward.

**Control comparison:** Add measurement-only (no gradient) per-format KL logging to `KLRegSFTUpdate` behind `log_per_format_kl: true` config flag, so condition (d) logs all 5 format KLs even though it only penalizes format 1.

---

## §5 Format-Invariant Linear Probe (free experiment b)

### New script: `scripts/30_format_invariant_probe.py`

**Protocol** (Allen-Zhu §5): Render each injected fact in K=5 formats, extract hidden state at the **last token of the subject span** (final layer), train logistic regression to predict relation:object from the hidden state. Per-format probe accuracy should be high and equal across formats if facts are stored format-invariantly.

**Inputs:**
- Trained model checkpoint (e.g., `outputs/seq_dsae_lite_round_15_qd_scale200/model`)
- Injected facts: `data/fnspid/triples/sequential/round_{1..15}.json`

**Outputs:**
- `final_results/format_invariant_probe.json`: per-format accuracy table
- Per-condition comparison: run on all 5 ablation conditions' round-15 checkpoints

**Implementation sketch (pseudocode):**

```python
# For each fact triple:
for triple in all_injected_facts:
    for k, template_fn in enumerate(K5_TEMPLATES):
        prompt = template_fn(triple)
        tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        # Find subject token span
        subj_token_ids = tokenizer(triple.subject, add_special_tokens=False).input_ids
        span_end = find_subword_span(tokens.input_ids[0], subj_token_ids)
        h = outputs.hidden_states[-1][0, span_end, :]  # (d_model,)
        features.append(h.cpu().numpy())
        labels.append(f"{triple.relation}:{triple.object}")
        format_ids.append(k)

# Train probe (80/20 split, stratified by label)
from sklearn.linear_model import LogisticRegression
X_train, X_test, y_train, y_test = train_test_split(...)
probe = LogisticRegression(max_iter=1000).fit(X_train, y_train)

# Evaluate per-format
for k in range(5):
    mask = [i for i, f in enumerate(format_ids_test) if f == k]
    acc_k = probe.score(X_test[mask], y_test[mask])
```

**Estimated LOC:** ~180.
**Estimated GPU-hours:** ~0.5 per condition (forward pass only, no training). 5 conditions × 0.5 = 2.5 GPU-hours total.

---

## §6 Engineering Risk Matrix

| Component | Most likely failure | Early detection (pilot) | Fallback |
|-----------|-------------------|------------------------|----------|
| K=5 data prep (§2) | Template 3–5 produce degenerate text (e.g., subject not matching tokenizer split) | Inspect 10 samples from `round_1.json` K=5 output. Check all 5 renderings are distinct, non-empty, and answer appears only in assistant turn. | Fix template strings. This is pure string manipulation; debugging is minutes. |
| `dsae_lite.py` KL loop (§3) | OOM from K=5 sequential forwards when `ref_model` + `model` + activations exceed 24GB | Run 1 round of condition (e) in pilot. Watch `peak_memory_gb` in metadata. Must be <22GB. | Reduce KL `batch_size` to 4 (independent of SFT batch size). Or: compute KL formats in micro-batches of 2-3 with gradient accumulation. |
| KL coefficient too tight | β=0.1 with K=5 formats overconstains learning; absorption collapses vs. baseline | After pilot round 3: if condition (e) absorption F1 < condition (a) − 3pp, β is too tight. | Halve β to 0.05. If still too tight, 0.01. This is the pre-registered early-stop criterion from `iteration3_verdict.md §4`. |
| KL coefficient too loose | β=0.1 with K=5 averaging makes effective penalty = β/K = 0.02 per format — may be negligible | After pilot round 5: if (e) ≈ (d) within 1pp on both QD F1 and format gap. | Raise β to 0.5 (effective 0.1/format). Also try removing the 1/K averaging (sum instead of mean). |
| Preservation framings (§3) too similar | If all 5 prompt framings produce near-identical logit distributions, the K=5 KL degenerates to K=1. | In pilot: check per-format KL variance. If std across 5 formats < 0.01 at round 1, framings are too similar. | Replace generic framings with more diverse ones: add a foreign-language framing, a bullet-point framing, or a Socratic-dialogue framing. |
| Registration / pipeline plumbing | Wrong triples subdir path, missing config reference, `METHODS` dict typo | Pilot end-to-end run of condition (e) for 1 round. Check `metadata.json` records correct `triples_path` pointing to `sequential_k5/round_1.json`. | Standard debugging; caught in the first 5 minutes of the pilot. |
| Linear probe (§5) | Subject span not found in tokenized input (subword misalignment) | Run probe on 20 facts before full experiment. Check span-detection hit rate. | Fall back to last-token-of-input hidden state (cruder but always available). |

---

## §7 Pilot → Full-Experiment Progression

### Pilot spec: 1-seed, conditions (a) + (e), 5 rounds

**Step 1: Data prep** (~2 min)
```bash
# Generate K=5 triples for rounds 1–5
for k in 1 2 3 4 5; do
  python scripts/24_prepare_mixed_format_triples.py \
    --input data/fnspid/triples/sequential/round_${k}.json \
    --output data/fnspid/triples/sequential_k5/round_${k}.json \
    --num-formats 5 --leak-free
done
```

Spot-check: `python -c "import json; d=json.load(open('data/fnspid/triples/sequential_k5/round_1.json')); print(len(d)); print(set(x['train_format'] for x in d))"` → expect `1000` and `{'qa', 'qd', 'declarative', 'instruction', 'narrative'}`.

**Step 2: Pilot run** (~2 GPU-hours)
```bash
python scripts/16_run_sequential.py \
  --methods baseline_sft,dsae_lite \
  --n-rounds 5 --per-round 200
```

`baseline_sft` aliases `naive_sft` in CONFIG_MAP → `naive_sft.yaml`, routed to `sequential` (K=1) triples subdir.

**Step 3: What to check**

| Check | Where | Healthy range | Action if out of range |
|-------|-------|--------------|----------------------|
| Pipeline completes without crash | stdout / `trajectory.json` | All 5 rounds have `abs_mean_f1` key | Debug crash, fix, re-run |
| Condition (e) absorption F1 | `trajectory.json` round 3 | Within 3pp of condition (a) | If (e) < (a) − 3pp: β too tight → halve β, re-pilot |
| Condition (e) preservation | `eval_results.json` round 5 | `preservation_recall_at_10` within 1pp of (a) | If degraded >5pp: β too loose → raise β |
| Peak GPU memory | `metadata.json` per round | < 22 GB | If OOM: reduce KL batch_size |
| Per-format KL log | `per_format_kl.json` round 1 | std across 5 formats > 0.01 | If too uniform: diversify preservation framings |
| Wall time per round | `elapsed_seconds` in metadata | < 20 min for (e), < 5 min for (a) | If >30 min for (e): profile for bottleneck |

**Step 4: Decision gate**

- **Pilot clean** → Prep K=5 data for rounds 6–15, launch full 5-condition × 3-seed × 15-round run.
- **β needs tuning** → Adjust, re-pilot conditions (a) + (e) for 5 rounds with new β. Budget: 2 more GPU-hours.
- **OOM** → Reduce KL batch_size, re-pilot. Budget: 2 more GPU-hours.
- **Pipeline bug** → Fix, re-pilot. No budget cost if caught in round 1.

### Full run

```bash
# Prep all 15 rounds
bash scripts/26_prep_all_k5_rounds.sh

# Full experiment: 5 conditions × 3 seeds × 15 rounds
for seed in 42 123 456; do
  python scripts/16_run_sequential.py \
    --methods baseline_sft,aug_sft_k5,kl_sft_k1,aug_kl_k1,dsae_lite \
    --n-rounds 15 --per-round 200 \
    --overrides "seed=${seed}"
done
```

`kl_sft_k1` aliases `kl_reg_sft` in CONFIG_MAP → `kl_reg_sft.yaml`, routed to `sequential` subdir.

**TODO (seed passthrough):** `seed_everything()` in `09_run_update.py` reads `base_cfg.seed`, but `--overrides` merges into `method_cfg`. Fix: add `--seed` arg to `16_run_sequential.py` that passes through to `09_run_update.py`, and amend `09_run_update.py` to check `method_cfg.seed` falling back to `base_cfg.seed`.

**Budget:** (a),(c): ~3 min/round → 4.5 h; (b): ~7 min/round → 5.25 h; (d): ~8 min/round → 6 h; (e): ~15 min/round → 11.25 h; eval: ~7.5 h. **Total: ~16 GPU-h main + ~2 pilot + ~2 probe = ~20 GPU-h.**

---

## §8 One-Screen Summary Table

| # | Component | File(s) | LOC (new) | GPU-h | Primary risk |
|---|-----------|---------|-----------|-------|-------------|
| 1 | K=5 data templates | `scripts/24_prepare_mixed_format_triples.py` (extend) | ~60 | 0 | Degenerate template text |
| 2 | K=5 batch prep | `scripts/26_prep_all_k5_rounds.sh` (new) | ~10 | 0 | Wrong paths |
| 3 | `DSAELiteUpdate` class | `src/sot/update/dsae_lite.py` (new) | ~220 | 0 | OOM from K=5 KL forwards |
| 4 | DSAE Lite config | `configs/update/dsae_lite.yaml` (new) | ~12 | 0 | β miscalibrated |
| 5 | `aug_sft_k5` config | `configs/update/aug_sft_k5.yaml` (new) | ~8 | 0 | — |
| 6 | `aug_kl_k1` config | `configs/update/aug_kl_k1.yaml` (new) | ~9 | 0 | — |
| 7 | Registration (09) | `scripts/09_run_update.py` (edit) | ~8 | 0 | Typo in METHODS dict |
| 8 | Registration (16) | `scripts/16_run_sequential.py` (edit) | ~15 | 0 | Wrong subdir routing |
| 9 | Seed passthrough | `scripts/09_run_update.py` + `16_run_sequential.py` (edit) | ~15 | 0 | Seed not propagated |
| 10 | Per-format KL logging | Inside `dsae_lite.py` + diagnostic in `kl_reg_sft.py` | ~30 | 0 | — |
| 11 | Pilot run | — | 0 | 2 | Pipeline crash |
| 12 | Full 5-way × 3-seed | — | 0 | ~16 | β tuning from pilot |
| 13 | Format-invariant probe | `scripts/30_format_invariant_probe.py` (new) | ~180 | 2.5 | Subject-span misalignment |
| 14 | QD format probe (existing) | `scripts/23_qd_format_probe.py` (reuse) | 0 | 1.5 | — |
| | **Total** | | **~567** | **~22** | |

**Budget remaining:** 30 − 22 = 8 GPU-hours for debugging, β re-tuning, optional K-ablation (K=3/10 on KL side, ~3 GPU-hours each).
