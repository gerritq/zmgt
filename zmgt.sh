#!/bin/bash
#SBATCH --job-name=zmgt_ideas
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040
# SBATCH --partition=tier2_gpu 
# SBATCH --account=er_prj_inf_impact_llm_wikipedia

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


MODELS=("l8b") # "q8b" "q8bb" "l8b" "l8bb"

DATASETS=(
  "drlXAttacks_decoder_paraphrasing"
  "raidDomain_wiki"
  "drlXMix_mixed"
)

SEEDS=(42) # 42 43 44
OUTPUT_FOLDER="ideas"

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running token_change: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.1_token_change \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --output_folder "$OUTPUT_FOLDER"
        done
    done
done


