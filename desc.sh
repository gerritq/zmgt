#!/bin/bash
#SBATCH --job-name=zmgt_desc3
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --constraint=h200|h100|a100|a100_40g|a100_80g|b200
#SBATCH --exclude=arc-hpc-comp054

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


MODELS=("q8b" "m8b" "g7b") # "q8b" "l8b" "q8b" "m8b"

DATASETS=(
 "drlXDomainLen_wiki_300_tokens"
"drlXDomainLen_academic_300_tokens"
"drlXDomainLen_novel_300_tokens"
"drlXDomainLen_seo_300_tokens"
"drlXDomainLen_webtext_300_tokens"
)

SEEDS=(42) # 42 43 44
SPLIT="test"

# ==================================================================================================
# ==================================================================================================
# LAYER SELECTION ID HEATMAP
# ==================================================================================================
# ==================================================================================================
# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running id_heatmap: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.desc.id_heatmap \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --split "$SPLIT"
#         done
#     done
# done

# ==================================================================================================
# LAYER SELECTION INFORMATION INBALANCE
# ==================================================================================================
# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running info_imbalance: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.layer_desc.info_imbalance \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --split "$SPLIT"
#         done
#     done
# done

# ==================================================================================================
# GEOMETRY: HUMAN VS MACHINE (src/geometric_metrics/desc3.py)
# ==================================================================================================
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running desc3: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.desc3 \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --split "$SPLIT"
        done
    done
done
