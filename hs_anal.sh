#!/bin/bash
#SBATCH --job-name=zmgt_hs_anal
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=arc-hpc-comp054

# Runs src/hs_anal/norm_anal.py.
# Usage: sbatch hs_anal.sh

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

nvidia-smi

MODEL="l8b"
DATASET="drlXDomainLen_wiki_300_tokens"
SEED=42
SPLIT="test"

echo "Running norm_anal: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.hs_anal.norm_anal \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --seed "$SEED" \
    --split "$SPLIT"
