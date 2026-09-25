#!/bin/bash
#SBATCH --job-name=items
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=cpu
#SBATCH --mem=2GB

# set -euo pipefail

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

# ----------------------------------------------------
# t_main
# ----------------------------------------------------
MODEL="l8b"
METHOD="token_change"
PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.items.t_main \
    --model "$MODEL" \
    --method "$METHOD"
