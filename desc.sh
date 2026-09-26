#!/bin/bash
#SBATCH --job-name=zmgt_desc_entropy
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g|l40s
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040,erc-hpc-comp222
# SBATCH --partition=tier2_gpu
# SBATCH --account=er_prj_inf_impact_llm_wikipedia

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


MODELS=("l8b") # "q8b" "l8b"

DATASETS=(
#   "drlXAttacks_decoder_paraphrasing"
#   "drlXAttacks_character_deletion"
  "raidDomain_wiki"
  "drlXDomain_academic"

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
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.desc.info_imbalance \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --split "$SPLIT"
#         done
#     done
# done

# ==================================================================================================
# LAYER SELECTION ENTROPY
# ==================================================================================================
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running layer_selection_entropy: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Split=$SPLIT"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.desc.layer_selection_entropy \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --split "$SPLIT"
        done
    done
done
