#!/bin/bash
# Submit one SLURM job per fold on the RU HPC.
#
#   bash submit.sh CONFIG                 one job (config without {fold})
#   bash submit.sh CONFIG 0 1 2 3 4       one job per fold, they run in parallel if GPUs are free
#   GPU=l40 bash submit.sh CONFIG 1 2     use L40 nodes (48 GB) instead of the A100 (80 GB); also GPU=l40s
#   SMOKE=1 bash submit.sh CONFIG 0       quick test (about 10 min), writes to <out>/smoke
#
# Logs: <out_dir>/logs/<job name>_<job id>.log
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh" >/dev/null
CONFIG=$(readlink -f "$1"); shift
FOLDS=("$@"); [ ${#FOLDS[@]} -eq 0 ] && FOLDS=("")
case "${GPU:-a100}" in
  a100) PART=(--partition=hpc_a100_a --gres=gpu:a100:1) ;;   # 80 GB, 1 node x 4 GPUs, often busy
  l40s) PART=(--partition=hpc_l40s_b --gres=gpu:l40s:1) ;;   # 48 GB
  l40)  PART=(--partition=hpc_l40_a --gres=gpu:l40:1) ;;     # 48 GB, 10 nodes x 2 GPUs, usually free
  *) echo "GPU must be a100, l40s or l40" >&2; exit 1 ;;
esac
TIME=23:30:00; [ "${SMOKE:-0}" = 1 ] && TIME=01:00:00
for f in "${FOLDS[@]}"; do
  read -r NAME OUT < <($PY -c "import sys; sys.path.insert(0, '$HERE'); from agft.config import Config; \
c = Config('$CONFIG', ${f:-None}); print(c.name, c.out)")
  mkdir -p "$OUT/logs"
  JN="ag_${NAME}${f:+_f$f}"; [ "${SMOKE:-0}" = 1 ] && JN="${JN}_smoke"
  sbatch "${PART[@]}" --time=$TIME --job-name="$JN" --output="$OUT/logs/%x_%j.log" \
    --export=ALL,AGFT_HOME="$HERE",SMOKE="${SMOKE:-0}" "$HERE/slurm/run.sbatch" "$CONFIG" $f
done
