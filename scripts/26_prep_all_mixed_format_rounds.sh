#!/usr/bin/env bash
# Prep mixed-format training triples for every sequential round.
#
# Takes data/fnspid/triples/sequential/round_{1..N}.json and writes
# data/fnspid/triples/mixed_format_sequential/round_{1..N}.json with each
# fact duplicated (QA-format + QD-format) via scripts/24_prepare_mixed_format_triples.py.
#
# Usage:
#   bash scripts/26_prep_all_mixed_format_rounds.sh          # default: 15 rounds
#   N_ROUNDS=10 bash scripts/26_prep_all_mixed_format_rounds.sh

set -e

N_ROUNDS=${N_ROUNDS:-15}
PY=${PY:-/home/ec2-user/anaconda3/envs/pytorch/bin/python}
IN_DIR=data/fnspid/triples/sequential
OUT_DIR=data/fnspid/triples/mixed_format_sequential

mkdir -p "$OUT_DIR"

for k in $(seq 1 "$N_ROUNDS"); do
  in_path="$IN_DIR/round_${k}.json"
  out_path="$OUT_DIR/round_${k}.json"
  if [ ! -f "$in_path" ]; then
    echo "SKIP round $k: $in_path does not exist"
    continue
  fi
  echo "Round $k: $in_path -> $out_path"
  "$PY" scripts/24_prepare_mixed_format_triples.py --input "$in_path" --output "$out_path"
done

echo
echo "Done. Prepared $N_ROUNDS rounds of mixed-format triples under $OUT_DIR."
