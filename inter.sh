#!/bin/bash
#SBATCH --job-name=zmgt_inter
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=arc-hpc-comp054

# set -euo pipefail

nvidia-smisbarc

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


MODELS=("l8b")
DATASETS=("drlXpaired_wiki_gpt_4o_256_tokens")
SEEDS=(42)
LAYER=7   # layer_i of src.zero (transformer layer i + 1); best layer for joint in t_main
TOP_K=5   # pairs with the largest joint divergence; output in output/inter

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running inter_sample: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Layer=$LAYER"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.inter_sample \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --layer "$LAYER" \
                --top_k "$TOP_K"
        done
    done
done
