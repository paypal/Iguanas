#!/usr/bin/env bash
# Multi-seed extension of the 15-dataset benchmark.
# On a 16-core machine this completes in roughly 40 minutes for 5 seeds.
#   bash benchmarks/run_seeds.sh 5 16 benchmarks/results/seedrun
set -euo pipefail
SEEDS="${1:-5}"; JOBS="${2:-8}"; OUT="${3:-benchmarks/results/seedrun}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
DATASETS=(creditcard mc1 mammography satellite_anomaly bank_marketing churn sick \
          ozone_level_8hr credit_default pc4 jm1 kc1 wilt pc1 pc3)
for ((s=0; s<SEEDS; s++)); do for d in "${DATASETS[@]}"; do echo "$d $s"; done; done \
  | xargs -P "$JOBS" -n 2 bash -c 'cd '"$REPO"' && python3 benchmarks/seed_driver.py --dataset "$0" --seed "$1" --out '"$OUT"''
echo "Done. $(ls "$OUT"/*.csv 2>/dev/null | wc -l) unit files in $OUT"
