#!/usr/bin/env bash
# Wait until the running seed driver (neural 1.5B) has finished, then resume the
# 7B seeds (seed 1 cache is partial, seed 2 not started) with the fixed code.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS=1
until grep -q "all done" results/driver_all.log 2>/dev/null; do sleep 120; done
echo "$(date '+%F %T') driver finished; resuming neural7b seeds" | tee -a results/neural7b/driver.log
bash scripts/run_all_seeds.sh neural7b
