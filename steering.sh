#!/bin/bash
#SBATCH --job-name=zmgt_prefix_steer
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=10:00:00
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
    "drlXpaired_wiki_gpt_4o_256_tokens"
     "drlXpaired_wiki_gemini_2_5_flash_256_tokens"
    "drlXpaired_news_gpt_4o_256_tokens"
     "drlXpaired_news_gemini_2_5_flash_256_tokens"
    "drlXpaired_academic_gpt_4o_256_tokens"
     "drlXpaired_academic_gemini_2_5_flash_256_tokens"

)
SEEDS=(42)
CONTROLS=("random" "data_shuffle")

# ==================================================================================================
# Prefix steering: push the prefix along the mean-difference direction, measure suffix angles
# ==================================================================================================
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for CONTROL in "${CONTROLS[@]}"; do
                echo "------------------------------------------------"
                echo "Running prefix steering: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Control=$CONTROL"
                echo "------------------------------------------------"
                PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.steering.prefix \
                    --model "$MODEL" \
                    --dataset "$DATASET" \
                    --seed "$SEED" \
                    --control "$CONTROL" \
                    --split_frac 0.5 \
                    --max_layer 20
            done
        done
    done
done
