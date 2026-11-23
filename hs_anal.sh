#!/bin/bash
#SBATCH --job-name=zmgt_hs_anal
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:25:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=arc-hpc-comp054

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("l8b")
DATASETS=(
        #     "drlXDomainLen_wiki_300_tokens" 
        #   "drlXDomainLen_academic_300_tokens"
          "drlXAttacks_character_deletion"
          "drlXAttacks_decoder_paraphrasing")
SEEDS=(42)
SPLIT="test"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running hs_anal: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.hs_anal.hs_anal \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --split "$SPLIT"
        done
    done
done
