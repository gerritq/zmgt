#!/bin/bash
#SBATCH --job-name=zmgt_ideas
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
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


# ==================================================================================================
# 2 Curvature Tweaked
# ==================================================================================================
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

# ==================================================================================================
# 2 Token change between 2m
# ==================================================================================================
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

# ==================================================================================================
# 5 Representational Novelty
# ==================================================================================================
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

# ==================================================================================================
# 6 Curvature across layers 2m
# ==================================================================================================
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

# ==================================================================================================
# 7 Curvature across tokens
# ==================================================================================================
# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running layer token curv 2m: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.7_curv_tokens_2m \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --output_folder "$OUTPUT_FOLDER"
#         done
#     done
# done

# ==================================================================================================
# 8 Information Balance as a metric
# ==================================================================================================
# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running info imbalance: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.8_info_imbalance \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --output_folder "$OUTPUT_FOLDER" \
#                 --layer_a 11 \
#                 --layer_b 10
#         done
#     done
# done

# ==================================================================================================
# 4 Layer Selection
# ==================================================================================================
OUTPUT_FOLDER="layer_selection"
LAYER_SELECTIONS=(
  "info_imbalance_first_local_max"
  "fluoroscopy_kl"
  "raw_global_max_token_entropy"
  "min_normalized_curvature"
  "fixed_layer"
)

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for SELECTION in "${LAYER_SELECTIONS[@]}"; do
#                 echo "------------------------------------------------"
#                 echo "Running curv layer selection: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, Selection=$SELECTION"
#                 echo "------------------------------------------------"
#                 PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.4_curv_layer_selection \
#                     --model "$MODEL" \
#                     --dataset "$DATASET" \
#                     --seed "$SEED" \
#                     --output_folder "$OUTPUT_FOLDER" \
#                     --selection "$SELECTION"
#             done
#         done
#     done
# done

# ==================================================================================================
# 9 CSD ngu
# ==================================================================================================
OUTPUT_FOLDER="csd_ngu"
for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running csd_ngu: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.9_csd_ngu \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --output_folder "$OUTPUT_FOLDER" \
                --rank 10 \
                --layer 10
        done
    done
done
