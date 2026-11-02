# Sourced by run_pipeline.sh and the SLURM scripts. Override any of these before running.
# AG_ENV: the python environment (see REPRODUCE.md). Default: the working env on the RU HPC.
export AG_ENV=${AG_ENV:-/lustre/fs4/fuchs_lab/scratch/gmunoz/pioneer_factors_scc_models/envs/alphagenome}
export PY=${PY:-$AG_ENV/bin/python}
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-4}} TF_CPP_MIN_LOG_LEVEL=2
# WSL2 (genos) needs AG_ALLOCATOR=cuda_async; normal Linux (the HPC) uses the default allocator.
[ -n "${AG_ALLOCATOR:-}" ] && export XLA_PYTHON_CLIENT_ALLOCATOR=$AG_ALLOCATOR
# TMPDIR on HPC: node /tmp is small; use your own scratch if TMPDIR is not set by SLURM.
export TMPDIR=${TMPDIR:-/tmp}
if [ ! -x "$PY" ]; then echo "python not found at $PY: set AG_ENV (see REPRODUCE.md)" >&2; exit 1; fi
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no GPU visible"
