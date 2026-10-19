#!/bin/bash
#SBATCH --job-name=agft
#SBATCH --partition=hpc_a100_a
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=23:30:00
# One GPU job = the whole pipeline for one config and one fold. Submit with submit.sh, which sets
# AGFT_HOME, the log path and the partition. Direct use:
#   AGFT_HOME=/path/to/alphagenome_ft_pipeline sbatch slurm/run.sbatch CONFIG [FOLD]
set -euo pipefail
: "${AGFT_HOME:?set AGFT_HOME to the pipeline folder (submit.sh does this)}"
echo "job ${SLURM_JOB_ID:-none} on $(hostname)"
bash "$AGFT_HOME/run_pipeline.sh" "$@"
