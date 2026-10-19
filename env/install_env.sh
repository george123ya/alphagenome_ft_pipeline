#!/bin/bash
# Build the AlphaGenome fine-tuning environment with the exact versions that were tested.
#
#   bash env/install_env.sh /path/to/new/env [/path/to/conda]
#
# Takes about 10 minutes and about 6 GB. On the RU HPC run it inside a job (sbatch --wrap or
# srun), not on the login node. A GPU is not needed to install.
# What it does:
#  1. conda env with python 3.11
#  2. clones alphagenome_research and alphagenome_ft at the tested commits into <env>/src
#  3. applies our 2 bug fixes to alphagenome_ft (env/alphagenome_ft_2c261c4.patch)
#  4. pip installs everything pinned to env/requirements.lock.txt
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
E=$(readlink -f "$1")
CONDA=${2:-$(command -v conda || echo /lustre/fs4/fuchs_lab/scratch/gmunoz/miniconda3/bin/conda)}
RESEARCH_COMMIT=eecf3d60234dda0428f5f083d22e518b72487ce0
FT_COMMIT=2c261c48225c6c5d9c1185529b24517b82d8d174
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$E/.pip_cache} TMPDIR=${TMPDIR:-$E/.tmp}
mkdir -p "$TMPDIR"

[ -x "$E/bin/python" ] || "$CONDA" create -y -p "$E" python=3.11 pip
PIP="$E/bin/python -m pip"
$PIP install -U pip
mkdir -p "$E/src"
cd "$E/src"
[ -d alphagenome_research ] || git clone https://github.com/google-deepmind/alphagenome_research.git
[ -d alphagenome_ft ] || git clone https://github.com/genomicsxai/alphagenome_ft.git
git -C alphagenome_research checkout -q $RESEARCH_COMMIT
git -C alphagenome_ft checkout -q $FT_COMMIT
if git -C alphagenome_ft apply --check "$HERE/alphagenome_ft_2c261c4.patch" 2>/dev/null; then
  git -C alphagenome_ft apply "$HERE/alphagenome_ft_2c261c4.patch"; echo "patch applied"
else
  echo "patch already applied (or does not fit): check 'git -C $E/src/alphagenome_ft diff'"
fi
grep -v -E "^(-e|alphagenome_ft|alphagenome_research)| @ " "$HERE/requirements.lock.txt" > "$E/constraints.txt"
$PIP install -c "$E/constraints.txt" "jax[cuda12]==0.10.2"
$PIP install -c "$E/constraints.txt" -e ./alphagenome_research
$PIP install -c "$E/constraints.txt" -e ./alphagenome_ft
$PIP install -c "$E/constraints.txt" pyBigWig pyfaidx pandas PyYAML huggingface_hub
$PIP freeze > "$E/installed_freeze.txt"
"$E/bin/python" -c "import jax, alphagenome_ft, alphagenome_research, pyBigWig, pyfaidx, yaml; print('imports ok, jax', jax.__version__, jax.devices())"
echo "ENV_DONE: set AG_ENV=$E before running the pipeline"
