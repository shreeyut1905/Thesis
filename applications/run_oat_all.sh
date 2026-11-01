#!/bin/bash
# 8 GPUs x 4 configurations (concurrent processes per GPU); latency is measured separately on an idle GPU.
export HF_HOME=~/mech_apps/hf_cache
OUT=~/mech_apps/results/oat
for g in 0 1 2 3 4 5 6 7; do
  for cfg in "kalsea 0.3" "respect 0.3" "kalsea 0.15" "respect 0.15"; do
    set -- $cfg
    CUDA_VISIBLE_DEVICES=$g ../.venv/bin/python oat_cache.py --method $1 --delta $2 --steps 20 --guidance 2.0 \
      --n_samples 10 --start 0 --end 5000 --shard $g --num_shards 8 --out $OUT > $OUT/logs/gpu${g}_$1_$2.log 2>&1 &
  done
done
wait
echo ALL_DONE
