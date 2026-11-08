#!/usr/bin/env bash
# Run the neural backends over three seeds, one seed at a time, so that an
# intermediate results.json exists after every seed.  Intermediate runs use a
# small bootstrap; the final run over all seeds uses the full 2,000 resamples.
#
#   bash scripts/run_all_seeds.sh neural7b neural
#
# Logs: results/<backend>/run_seeds_<seeds>.log
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS=1
BACKENDS=${*:-neural7b neural}

model_for() {
  case "$1" in
    neural7b) echo "qwen2.5:7b-instruct" ;;
    neural)   echo "qwen2.5:1.5b-instruct-fp16" ;;
    *)        echo "" ;;
  esac
}

for b in $BACKENDS; do
  m=$(model_for "$b")
  if [ -n "$m" ]; then
    until ollama list 2>/dev/null | grep -q "^$m"; do
      echo "$(date '+%F %T') waiting for ollama model $m to finish downloading"; sleep 60
    done
  fi
  mkdir -p "results/$b"
  for s in "0" "0 1" "0 1 2"; do
    nb=200; [ "$s" = "0 1 2" ] && nb=2000
    tag=${s// /}
    echo "$(date '+%F %T') == $b seeds [$s] n_boot=$nb" | tee -a "results/$b/driver.log"
    python scripts/run_benchmark.py --backend "$b" --seeds $s --n-boot "$nb" > "results/$b/run_seeds_${tag}.log" 2>&1
    echo "$(date '+%F %T') == $b seeds [$s] exit $?" | tee -a "results/$b/driver.log"
  done
done
echo "$(date '+%F %T') all done" | tee -a results/driver_all.log
