#!/bin/bash
#SBATCH --job-name=zmgt_desc_all
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=50GB
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --constraint=h200|h100|a100|a100_40g|a100_80g
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp048
# SBATCH --partition=tier2_gpu 
# SBATCH --account=er_prj_inf_impact_llm_wikipedia

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

MODELS=("q8b" "m8b" "l8b") # "q8b" "q8bb" "l8b" "l8bb"

DATASETS=(
            "drlXDomainLen_wiki_300_tokens" 
             "drlXDomainLen_academic_300_tokens"
             "drlXDomainLen_news_300_tokens"
             "drlXDomainLen_novel_300_tokens" 
            #   "drlXDomainLen_novel_500_tokens" 
            #     "drlXDomainLen_seo_500_tokens" 
        #   "drlXDomainLen_wiki_2000_chars"
        #    "drlXAttacks_character_substitution"
        #   "drlXAttacks_decoder_paraphrasing"

#           "drlXAttacks_decoder_paraphrasing"
#   "drlXAttacks_character_deletion"
#   "raidDomain_wiki"
          )

SEEDS=(42) # 42 43 44


# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running descriptive statistics: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.descriptives \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED"
#         done
#     done
# done


# ----------------------------------------------------
# Within-layer angle and norm per token position, layers 1, 3, ..., 17 (src/geometric_metrics/viz.py)
# ----------------------------------------------------
# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         echo "------------------------------------------------"
#         echo "Running within-layer per-token plots: Dataset=drlXDomainLen_wiki_500_tokens, Model=$MODEL, Seed=$SEED"
#         echo "------------------------------------------------"
#         PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.viz \
#             --model "$MODEL" \
#             --dataset "drlXDomainLen_wiki_500_tokens" \
#             --seed "$SEED"
#     done
# done


# ----------------------------------------------------
# Across-layer mean angle vs. mean norm per text, contour plot (src/geometric_metrics/viz_inter.py)
# ----------------------------------------------------
# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         echo "------------------------------------------------"
#         echo "Running across-layer contour: Dataset=drlXDomainLen_wiki_500_tokens, Model=$MODEL, Seed=$SEED"
#         echo "------------------------------------------------"
#         PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.viz_inter \
#             --model "$MODEL" \
#             --dataset "drlXDomainLen_wiki_500_tokens" \
#             --seed "$SEED"
#     done
# done


# ----------------------------------------------------
# Shared vs. token-specific part, within-text and corpus-level PCA (src/geometric_metrics/test_shared.py)
# ----------------------------------------------------
FIT_SPLIT="test" # split the global SVD is fitted on: test, val, train
SAVE_SCORES=1 # 1 = store per-text scores in the output json

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running shared-part tests: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.test_shared \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --fit_split "$FIT_SPLIT" \
#                 --save_scores "$SAVE_SCORES"
#         done
#     done
# done


# ----------------------------------------------------
# Per-text anisotropy at every hidden-state layer (src/geometric_metrics/aniso.py)
# ----------------------------------------------------
# computes all token variants (all, skip10pct, med5, med10) in one run, see the module docstring

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running anisotropy: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.aniso \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED"
#         done
#     done
# done


# ----------------------------------------------------
# Logit lens and attention vs. MLP updates per layer (src/geometric_metrics/across_exp.py)
# ----------------------------------------------------
# computes all token variants (all, skip10pct, med5, med10) in one run, see the module docstring

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running across-layer experiments: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.across_exp \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED"
#         done
#     done
# done


# ----------------------------------------------------
# Dispersion, norm, angle/magnitude, ppl and shared / per-layer PCA (src/geometric_metrics/desc.py)
# ----------------------------------------------------
# runs the model twice (pass 1: scores and PCA sums, pass 2: projections), output in output/desc2

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running desc2: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.geometric_metrics.desc_all \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED"
        done
    done
done
