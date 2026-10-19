#!/bin/bash
# Download the AlphaGenome JAX weights (all_folds, about 750 MB) from Hugging Face.
#
# First, once per person (the licence is personal, non-commercial use only, and it also covers
# fine-tuned models and their outputs):
#   1. log in at https://huggingface.co and open https://huggingface.co/google/alphagenome-all-folds
#   2. accept the AlphaGenome model terms (access is granted right away)
#   3. make a read token at https://huggingface.co/settings/tokens
#   4. $AG_ENV/bin/hf auth login      (paste the token)
# Then:
#   bash env/download_weights.sh /path/to/weights/all_folds
# and put that path in the config under `weights:`.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$HERE/env.sh" >/dev/null
D=$1
mkdir -p "$D"
"$AG_ENV/bin/hf" download google/alphagenome-all-folds --local-dir "$D" --max-workers 4
ls -la "$D"; du -sh "$D"
