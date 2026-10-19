#!/bin/bash
# Run the whole pipeline for one config (and one fold): check, windows, train, evaluate, insertion.
# Steps already finished are skipped, so a crashed run can simply be started again.
#
#   bash run_pipeline.sh CONFIG [FOLD]            full run
#   SMOKE=1 bash run_pipeline.sh CONFIG [FOLD]    20 training steps, 3 test windows, 2 insertion sites
#   STEPS="03" bash run_pipeline.sh CONFIG        only some steps (00 01 02 03 04)
#
# On SLURM use slurm/run.sbatch (calls this). On a WSL2 machine like genos set AG_ALLOCATOR=cuda_async.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/env.sh"
CONFIG=$(readlink -f "$1"); FOLD=${2:-}
FA=(); [ -n "$FOLD" ] && FA=(--fold "$FOLD")
SM=(); [ "${SMOKE:-0}" = 1 ] && SM=(--smoke)
STEPS=${STEPS:-"00 01 02 03 04"}
OUT=$($PY -c "import sys; sys.path.insert(0, '$HERE'); from agft.config import Config; \
c = Config('$CONFIG', ${FOLD:-None}); print(c.out)")
RUN=$OUT; [ "${SMOKE:-0}" = 1 ] && RUN=$OUT/smoke
INS=$($PY -c "import sys; sys.path.insert(0, '$HERE'); from agft.config import Config; \
print(int(bool(Config('$CONFIG', ${FOLD:-None}).insertion['enabled'])))")
mkdir -p "$OUT"
echo "config $CONFIG  fold ${FOLD:-none}  out $OUT  smoke ${SMOKE:-0}  host $(hostname)  $(date)"
has() { [[ " $STEPS " == *" $1 "* ]]; }
t() { echo; echo "=== $1  $(date +%T)"; }

if has 00; then t "00 check inputs"; $PY "$HERE/scripts/00_check_inputs.py" --config "$CONFIG" "${FA[@]}"; fi
if has 01; then
  t "01 windows"
  if [ -s "$OUT/windows/eval_windows_131072.bed" ] && [ "${FORCE:-0}" != 1 ]; then echo "exists, skipped (FORCE=1 to redo)"
  else $PY "$HERE/scripts/01_make_windows.py" --config "$CONFIG" "${FA[@]}"; fi
fi
if has 02; then
  t "02 train"
  if [ -d "$RUN/checkpoints/best" ] && [ -s "$RUN/train_info.json" ] && [ "${FORCE:-0}" != 1 ]; then echo "exists, skipped"
  else $PY "$HERE/scripts/02_train.py" --config "$CONFIG" "${FA[@]}" "${SM[@]}"; fi
fi
if has 03; then
  t "03 evaluate"
  if [ -s "$RUN/test_metrics.tsv" ] && [ "${FORCE:-0}" != 1 ]; then echo "exists, skipped"
  else $PY "$HERE/scripts/03_evaluate.py" --config "$CONFIG" "${FA[@]}" "${SM[@]}"; fi
fi
if has 04 && [ "$INS" = 1 ]; then
  t "04 motif insertion"
  if [ -s "$RUN/insertion/summary.tsv" ] && [ "${FORCE:-0}" != 1 ]; then echo "exists, skipped"
  else $PY "$HERE/scripts/04_motif_insertion.py" --config "$CONFIG" "${FA[@]}" "${SM[@]}"; fi
fi
echo; echo "PIPELINE_DONE $(date)"
