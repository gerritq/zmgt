#!/bin/bash
#SBATCH --job-name=zmgt_hs_anal_rest
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=arc-hpc-comp054

# Runs the scripts in src/hs_anal on every dataset in DATASETS.
# Usage: sbatch hs_anal.sh [script|all]
#   sbatch hs_anal.sh              # all scripts (default)
#   sbatch hs_anal.sh hs_across    # one script (file name in src/hs_anal, with or without .py)

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

ALL_SCRIPTS=("norm_anal" "hs_across" "hs_within" "hs_surprisal" "hs_attn")

TARGET="${1:-all}"
TARGET="${TARGET%.py}"
if [ "$TARGET" = "all" ]; then
    SCRIPTS=("${ALL_SCRIPTS[@]}")
elif [ -f "src/hs_anal/${TARGET}.py" ]; then
    SCRIPTS=("$TARGET")
else
    echo "Unknown script '${TARGET}'. Use one of: all ${ALL_SCRIPTS[*]}"
    exit 1
fi

nvidia-smi

MODEL="l8b"
DATASETS=(
    # "drlXDomainLen_wiki_300_tokens"
    "drlXDomainLen_academic_300_tokens"
    "drlXDomainLen_seo_300_tokens"
    "drlXDomainLen_news_300_tokens"
)
SEED=42
SPLIT="test"

for SCRIPT in "${SCRIPTS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        echo "Running ${SCRIPT}: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
        PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m "src.hs_anal.${SCRIPT}" \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --seed "$SEED" \
            --split "$SPLIT"
    done
done
