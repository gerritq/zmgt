#!/bin/bash
#SBATCH --job-name=zmgt_ideas
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
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
  "drlXAttacks_character_deletion"
)

SEEDS=(42) # 42 43 44
OUTPUT_FOLDER="ideas"

CURVATURE_METHODS=(
  "curvature_hs"
  "curvature_context_against_current"
  "curvature_full_context_against_current"
)

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for METHOD in "${CURVATURE_METHODS[@]}"; do
#                 echo "------------------------------------------------"
#                 echo "Running $METHOD: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#                 echo "------------------------------------------------"
#                 PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.2_curvature_tweaked \
#                     --model "$MODEL" \
#                     --dataset "$DATASET" \
#                     --seed "$SEED" \
#                     --output_folder "$OUTPUT_FOLDER" \
#                     --method "$METHOD"
#             done
#         done
#     done
# done

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running token_change_2m: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.3_token_change_2m \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --output_folder "$OUTPUT_FOLDER"
#         done
#     done
# done

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running representational_novelty: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.5_representational_novelty \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --output_folder "$OUTPUT_FOLDER" \
#                 --lam 1.0 \
#                 --skip_tokens 1
#         done
#     done
# done


# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running token layer curv 2m: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.6_curv_layer_2m \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --output_folder "$OUTPUT_FOLDER"
#         done
#     done
# done

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running layer token curv 2m: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.7_curv_tokens_2m \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --output_folder "$OUTPUT_FOLDER"
        done
    done
done


