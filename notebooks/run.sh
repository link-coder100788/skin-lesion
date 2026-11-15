#!/usr/bin/env bash
# Convert the percent-format sources to .ipynb and execute them in place.
#   notebooks/run.sh 01 02    -> runs 01_*.py and 02_*.py
set -euo pipefail
cd "$(dirname "$0")"
for n in "$@"; do
  src=$(ls ${n}_*.py)
  ../.venv/bin/jupytext --to ipynb --set-kernel python3 --execute "$src"
done
