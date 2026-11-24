#!/bin/bash
#SBATCH --job-name=zmgt_hs_anal
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=arc-hpc-comp054

# Runs the scripts in src/hs_anal.
# Usage: sbatch hs_anal.sh <script|all>
#   sbatch hs_anal.sh norm_anal    # one script (file name in src/hs_anal, with or without .py)
#   sbatch hs_anal.sh all          # every .py file in src/hs_anal

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"
HS_DIR="src/hs_anal"

available() { for f in "${HS_DIR}"/*.py; do n=$(basename "$f" .py); [ "$n" != "__init__" ] && echo "$n"; done; }

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
    echo "Usage: sbatch hs_anal.sh <script|all>. Scripts: all $(available | tr '\n' ' ')"
    exit 1
fi
if [ "$TARGET" = "all" ]; then
    mapfile -t SCRIPTS < <(available)
else
    TARGET="${TARGET%.py}"
    if [ ! -f "${HS_DIR}/${TARGET}.py" ]; then
        echo "No script ${HS_DIR}/${TARGET}.py. Scripts: all $(available | tr '\n' ' ')"
        exit 1
    fi
    SCRIPTS=("$TARGET")
fi

nvidia-smi

MODELS=("l8b")
DATASETS=(
        #     "drlXDomainLen_wiki_300_tokens"
        #   "drlXDomainLen_academic_300_tokens"
          "drlXAttacks_character_deletion"
          "drlXAttacks_decoder_paraphrasing")
SEEDS=(42)
SPLIT="test"

for SCRIPT in "${SCRIPTS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            for DATASET in "${DATASETS[@]}"; do
                echo "------------------------------------------------"
                echo "Running ${SCRIPT}: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
                echo "------------------------------------------------"
                PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m "src.hs_anal.${SCRIPT}" \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --seed "$SEED" \
                    --split "$SPLIT"
            done
        done
    done
done
