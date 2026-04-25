#!/usr/bin/env bash
# Prep K=5 augmented training triples for every sequential round.
#
# Takes data/fnspid/triples/sequential/round_{1..N}.json and writes
# data/fnspid/triples/sequential_k5/round_{1..N}.json with each fact rendered
# in 5 surface formats (qa, qd, declarative, instruction, narrative). Always
# leak-free.
#
# Used by the DSAE Lite ablation (paper/ml_intern_iteration3_verdict.md §4).
# Conditions aug_sft_k5, aug_kl_k1, dsae_lite all read from sequential_k5/.
#
# Usage:
#   bash scripts/27_prep_all_k5_rounds.sh
#   N_ROUNDS=5  bash scripts/27_prep_all_k5_rounds.sh   # pilot

set -e

N_ROUNDS=${N_ROUNDS:-15}
PY=${PY:-/home/ec2-user/anaconda3/envs/pytorch/bin/python}
IN_DIR=data/fnspid/triples/sequential
OUT_DIR=data/fnspid/triples/sequential_k5

mkdir -p "$OUT_DIR"

for k in $(seq 1 "$N_ROUNDS"); do
  in_path="$IN_DIR/round_${k}.json"
  out_path="$OUT_DIR/round_${k}.json"
  if [ ! -f "$in_path" ]; then
    echo "SKIP round $k: $in_path does not exist"
    continue
  fi
  echo "Round $k: $in_path -> $out_path"
  "$PY" scripts/24_prepare_mixed_format_triples.py \
    --input "$in_path" --output "$out_path" --num-formats 5 --leak-free
done

echo
echo "Done. Prepared $N_ROUNDS rounds of K=5 augmented triples under $OUT_DIR."
