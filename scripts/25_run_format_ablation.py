"""Phase 8: Three-way format-transfer ablation (ml-intern-specified design).

Three conditions on the same 200 facts at round 15, same compute budget,
single seed:

  1. kl_reg_sft           -- baseline (QA-only training, already in outputs/)
  2. kl_reg_sft_mixedfmt  -- QA + QD-rendered SFT, no variance penalty
  3. fi_sft               -- QA + QD + variance penalty (mu * Var_k[log pi_k])

Evaluate each with scripts/23_qd_format_probe.py on the same 50-fact probe set.
Critical comparison: format_gap(fi_sft) vs format_gap(mixedfmt). If fi_sft's
gap is >= 0.03 absolute smaller, the variance penalty does non-trivial work
and we have a method contribution. Otherwise the penalty is inert and the
paper's contribution is the distribution hypothesis alone.

Each condition chains from the same starting checkpoint (the task-tuned
backbone) for a clean three-way comparison that avoids round-14-chain
confounds. The baseline kl_reg_sft checkpoint we already have was chained
through 14 prior rounds; we use it as a reference point but compute its
Phase 7b numbers from the existing outputs/seq_kl_reg_sft_round_15 checkpoint.

Usage on SageMaker (single GPU, ~2 GPU-hours):

  PY=/home/ec2-user/anaconda3/envs/pytorch/bin/python

  # 0. Data prep: build mixed-format training triples.
  $PY scripts/24_prepare_mixed_format_triples.py \\
    --input data/fnspid/triples/sequential/round_15.json \\
    --output data/fnspid/triples/mixed_format/round_15.json

  # 1. Mixed-format SFT (no variance penalty).
  CUDA_VISIBLE_DEVICES=0 $PY scripts/09_run_update.py \\
    --method kl_reg_sft_mixedfmt \\
    --run-name ablation_kl_reg_sft_mixedfmt_r15 \\
    --scale 200 --task qd \\
    --config configs/update/kl_reg_sft_mixedfmt.yaml \\
    --triples-path data/fnspid/triples/mixed_format/round_15.json \\
    --starting-checkpoint checkpoints/qd_sft/final

  # 2. FI-SFT (mixed-format + variance penalty).
  CUDA_VISIBLE_DEVICES=0 $PY scripts/09_run_update.py \\
    --method fi_sft \\
    --run-name ablation_fi_sft_r15 \\
    --scale 200 --task qd \\
    --config configs/update/fi_sft.yaml \\
    --triples-path data/fnspid/triples/mixed_format/round_15.json \\
    --starting-checkpoint checkpoints/qd_sft/final

  # 3. Evaluate all three with the QD-format probe.
  CUDA_VISIBLE_DEVICES=0 $PY scripts/23_qd_format_probe.py \\
    --checkpoints outputs/seq_kl_reg_sft_round_15_qd_scale200/model \\
                  outputs/ablation_kl_reg_sft_mixedfmt_r15_qd_scale200/model \\
                  outputs/ablation_fi_sft_r15_qd_scale200/model \\
    --names kl_reg_sft_r15 kl_reg_sft_mixedfmt fi_sft \\
    --n-facts 50 \\
    --out final_results/phase8_format_ablation.csv

Interpretation at the end:
  - If format_gap(fi_sft) < format_gap(mixedfmt) - 0.03 : variance penalty earns its keep.
  - If format_gap(mixedfmt) < format_gap(kl_reg_sft) - 0.03 : distribution hypothesis confirmed.
  - If neither separates: format-coupling is structurally deeper; paper stays as 'motivated hypothesis.'

This script exists as documentation / reproducibility only; it does not execute
the runs (each run takes ~20-30 min GPU compute and is gated behind cloud
credentials). Copy the commands above and run them directly.
"""

import sys


def main():
    print(__doc__)
    sys.exit(0)


if __name__ == "__main__":
    main()
