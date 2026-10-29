#!/bin/bash
#SBATCH --job-name=zmgt_probe
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:45:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|a100|a100_40g|a100_80g
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040,erc-hpc-comp049

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("l8b")
DATASETS=(
    "drlXDomainLen_wiki_300_tokens" 
        # "drlXAttacks_character_deletion"
        # "drlXAttacks_decoder_paraphrasing"
    #   "raidDomain_wiki"
     "drlXDomain_academic"
#   "drlXAttacks_character_deletion"
#  "drlXDomain_academic"
#   "samplingTemperature_1_0"
# "editlens_amazon_reviews" "editlens_fineweb_edu"
#   "samplingTemperature_0_9"
#      "samplingTemperature_1_0"
)
SEEDS=(42)

# ==================================================================================================
# Spectral geometry per layer: top eigenvalue ratio, effective rank, mean cosine (± centering)
# ==================================================================================================
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running test: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.test \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED"
        done
    done
done
