#!/bin/bash
# OAT's FEA evaluation (RunComplianceTest.py, unchanged) for each cached configuration; uses all 8 GPUs.
export HF_HOME=~/mech_apps/hf_cache
export LD_LIBRARY_PATH=$(ls -d ~/mech_apps/.venv/lib/python3.10/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
cd ~/mech_apps/cache
for cfg in "kalsea 0.3" "respect 0.3" "kalsea 0.15" "respect 0.15"; do
  set -- $cfg
  ../.venv/bin/python eval_oat.py --method $1 --delta $2 --workers_per_gpu 6 > ~/mech_apps/results/oat/logs/eval_$1_$2.log 2>&1
done
echo EVAL_DONE
