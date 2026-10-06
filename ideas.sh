#!/bin/bash
#SBATCH --job-name=zmgt_perturb
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|a100|a100_40g|a100_80g
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040
# SBATCH --partition=tier2_gpu 
# SBATCH --account=er_prj_inf_impact_llm_wikipedia

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


MODELS=("l8b") # "q8b" "q8bb" "l8b" "l8bb"

DATASETS=(
#    "drlXAttacks_character_deletion"
     "drlXAttacks_decoder_paraphrasing"
#    "raidDomain_wiki"
    # "drlXDomain_academic"
#   "drlXAttacks_character_deletion"
#  "drlXDomain_academic"
# "samplingTemperature_1_0"
# "editlens_amazon_reviews" "editlens_fineweb_edu"
)

SEEDS=(42) # 42 43 44
OUTPUT_FOLDER="ideas"


# ==================================================================================================
# 2 Curvature Tweaked
# ==================================================================================================
CURVATURE_METHODS=(
   "curvature_hs"
#   "curvature_context_against_current"
#   "curvature_full_context_against_current"
#   "curvature_context_against_context"
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
# 2.1 curvature_hs decomposed into novelty (sin θ) and significance (residual norm r)
# ==================================================================================================
OUTPUT_FOLDER="curvature_decomp"
DECOMPOSE_CENTER=(0 1)                  # 1: subtract the text's mean hidden state per layer
DECOMPOSE_SIG_NORMS=("none" "std")      # std: divide r_t by its within-text standard deviation

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for CENTER in "${DECOMPOSE_CENTER[@]}"; do
#                 for SIG_NORM in "${DECOMPOSE_SIG_NORMS[@]}"; do
#                     echo "------------------------------------------------"
#                     echo "Running curvature_decompose: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, center=$CENTER, sig_norm=$SIG_NORM"
#                     echo "------------------------------------------------"
#                     PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.2_1_curvature_decompose \
#                         --model "$MODEL" \
#                         --dataset "$DATASET" \
#                         --seed "$SEED" \
#                         --output_folder "$OUTPUT_FOLDER" \
#                         --layer 10 \
#                         --center "$CENTER" \
#                         --significance_norm "$SIG_NORM"
#                 done
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
# 9 Surrounding-, preceding-context and cross-layer novelty
# ==================================================================================================
OUTPUT_FOLDER="cross_layer"
# K: context size, neighbours on either side (surrounding) and preceding states (preceding).
KS=(1)
CONTEXT_METHODS=("cross_layer")

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for METHOD in "${CONTEXT_METHODS[@]}"; do
#                 for K in "${KS[@]}"; do
#                     echo "------------------------------------------------"
#                     echo "Running $METHOD: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, K=$K"
#                     echo "------------------------------------------------"
#                     PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.9_csd_ngu \
#                         --model "$MODEL" \
#                         --dataset "$DATASET" \
#                         --seed "$SEED" \
#                         --output_folder "$OUTPUT_FOLDER" \
#                         --method "$METHOD" \
#                         --window "$K" \
#                         --layer 10
#                 done
#             done
#         done
#     done
# done


# ==================================================================================================
# 10 Two-model novelty ratio (Binoculars-style): novelty under MODEL / novelty under MODEL2
# ==================================================================================================
OUTPUT_FOLDER="novelty_m2"
MODEL2S=("l3b") # must share MODEL's tokenizer
M2_METHODS=("preceding_context")
M2_KS=(2)

# for MODEL in "${MODELS[@]}"; do
#     for MODEL2 in "${MODEL2S[@]}"; do
#         for SEED in "${SEEDS[@]}"; do
#             for DATASET in "${DATASETS[@]}"; do
#                 for METHOD in "${M2_METHODS[@]}"; do
#                     for K in "${M2_KS[@]}"; do
#                         echo "------------------------------------------------"
#                         echo "Running novelty ratio $METHOD: $MODEL / $MODEL2, Dataset=$DATASET, Seed=$SEED, K=$K"
#                         echo "------------------------------------------------"
#                         PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.novelty_m2 \
#                             --model "$MODEL" \
#                             --model2 "$MODEL2" \
#                             --dataset "$DATASET" \
#                             --seed "$SEED" \
#                             --output_folder "$OUTPUT_FOLDER" \
#                             --method "$METHOD" \
#                             --window "$K" \
#                             --layer 10
#                     done
#                 done
#             done
#         done
#     done
# done


# ==================================================================================================
# 11 Effective rank of the CSD novelty directions (unit-norm and ‖h_t‖-normalised residuals)
# ==================================================================================================
OUTPUT_FOLDER="cross_layer"
ERANK_KS=(1)

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for K in "${ERANK_KS[@]}"; do
#                 echo "------------------------------------------------"
#                 echo "Running novelty_direction_erank: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, K=$K"
#                 echo "------------------------------------------------"
#                 PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.11_dominant_curv \
#                     --model "$MODEL" \
#                     --dataset "$DATASET" \
#                     --seed "$SEED" \
#                     --output_folder "$OUTPUT_FOLDER" \
#                     --window "$K" \
#                     --min_sin 0.05 \
#                     --layer 10
#             done
#         done
#     done
# done


# ==================================================================================================
# 13 Distance of the final hidden state to the unembedding vector of the actual next token
# ==================================================================================================
OUTPUT_FOLDER="hs_unem"
UNEM_KS=(1000 10000) # nearest unembedding vectors (observed token excluded) for the ratio / difference / z scores; K >= 2

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for UNEM_K in "${UNEM_KS[@]}"; do
#                 echo "------------------------------------------------"
#                 echo "Running hs_unem: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, K=$UNEM_K"
#                 echo "------------------------------------------------"
#                 PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.hs_unem \
#                     --model "$MODEL" \
#                     --dataset "$DATASET" \
#                     --seed "$SEED" \
#                     --output_folder "$OUTPUT_FOLDER" \
#                     --k "$UNEM_K"
#             done
#         done
#     done
# done


# ==================================================================================================
# 14 Actual hidden-state curvature vs. angle to the expected (top-K logit-lens) unembedding direction
# ==================================================================================================
OUTPUT_FOLDER="curv_unem"
CURV_UNEM_KS=(100)
CURV_UNEM_VARIANTS=("readout")

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for K in "${CURV_UNEM_KS[@]}"; do
#                 for VARIANT in "${CURV_UNEM_VARIANTS[@]}"; do
#                     echo "------------------------------------------------"
#                     echo "Running curv_unem $VARIANT: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, K=$K"
#                     echo "------------------------------------------------"
#                     PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.curv_unem \
#                         --model "$MODEL" \
#                         --dataset "$DATASET" \
#                         --seed "$SEED" \
#                         --output_folder "$OUTPUT_FOLDER" \
#                         --k "$K" \
#                         --variant "$VARIANT"
#                 done
#             done
#         done
#     done
# done


# ==================================================================================================
# 15 curvature_hs under the instruct model vs. its base model, per layer (inst, base, diff, ratio)
# ==================================================================================================
OUTPUT_FOLDER="curv_hs_2m"
INSTRUCT_2M="l8b"  # key of cfg.model_dict, of BASE_MODELS in curv_hs_2m.py, or a Hugging Face id
BASE_2M="l8bb"     # same options; both models must share the tokenizer

# for SEED in "${SEEDS[@]}"; do
#     for DATASET in "${DATASETS[@]}"; do
#         echo "------------------------------------------------"
#         echo "Running curv_hs_2m: Dataset=$DATASET, Instruct=$INSTRUCT_2M, Base=$BASE_2M, Seed=$SEED"
#         echo "------------------------------------------------"
#         PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.curv_hs_2m \
#             --instruct "$INSTRUCT_2M" \
#             --base "$BASE_2M" \
#             --dataset "$DATASET" \
#             --seed "$SEED" \
#             --output_folder "$OUTPUT_FOLDER"
#     done
# done


# ==================================================================================================
# 16 Same two-model contrast as 15, with curvature across layers per token: ∠(h_t^ℓ, h_t^{ℓ+1})
# ==================================================================================================
# OUTPUT_FOLDER="cruv_hs_layer_2m"
# INSTRUCT_LAYER_2M="l8b"  # key of cfg.model_dict, of BASE_MODELS in curv_hs_2m.py, or a Hugging Face id
# BASE_LAYER_2M="l8bb"     # same options; both models must share the tokenizer

# for SEED in "${SEEDS[@]}"; do
#     for DATASET in "${DATASETS[@]}"; do
#         echo "------------------------------------------------"
#         echo "Running cruv_hs_layer_2m: Dataset=$DATASET, Instruct=$INSTRUCT_LAYER_2M, Base=$BASE_LAYER_2M, Seed=$SEED"
#         echo "------------------------------------------------"
#         PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.cruv_hs_layer_2m \
#             --instruct "$INSTRUCT_LAYER_2M" \
#             --base "$BASE_LAYER_2M" \
#             --dataset "$DATASET" \
#             --seed "$SEED" \
#             --output_folder "$OUTPUT_FOLDER"
#     done
# done


# ==================================================================================================
# 17 Last-layer curvature in the "remaining" subspace of W_U (smallest singular directions)
# ==================================================================================================
# OUTPUT_FOLDER="harp_idea"

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             echo "------------------------------------------------"
#             echo "Running harp_idea: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
#             echo "------------------------------------------------"
#             PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.harp_idea \
#                 --model "$MODEL" \
#                 --dataset "$DATASET" \
#                 --seed "$SEED" \
#                 --output_folder "$OUTPUT_FOLDER"
#         done
#     done
# done


# ==================================================================================================
# 18 Per-text layer selection for curvature_hs / curvature_context_against_current (raw, delta scores)
# ==================================================================================================
OUTPUT_FOLDER="curv_layer_selection"
SELECTION_STATISTICS=( "id_mle") # "information_imbalance" "id_mle"
SELECTION_CRITERIA=("first_local_max" "first_effective_peak")
SELECTION_SMOOTH=(0 1)
SELECTION_HORIZON=3

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            for STATISTIC in "${SELECTION_STATISTICS[@]}"; do
                for CRITERION in "${SELECTION_CRITERIA[@]}"; do
                    for SMOOTH in "${SELECTION_SMOOTH[@]}"; do
                        echo "------------------------------------------------"
                        echo "Running curv_layer_selection $STATISTIC $CRITERION smooth=$SMOOTH: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
                        echo "------------------------------------------------"
                        PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.curv_layer_selection \
                            --model "$MODEL" \
                            --dataset "$DATASET" \
                            --seed "$SEED" \
                            --output_folder "$OUTPUT_FOLDER" \
                            --statistic "$STATISTIC" \
                            --criterion "$CRITERION" \
                            --smooth "$SMOOTH" \
                            --horizon "$SELECTION_HORIZON"
                    done
                done
            done
        done
    done
done


# ==================================================================================================
# 19 Projection of each token on its predecessor / mean of its 3 predecessors: angle and residual per layer
# ==================================================================================================
OUTPUT_FOLDER="project"
PROJECT_VARIANTS=("within_layer")
PROJECT_KS=(6) # within_layer context size (preceding tokens); across_layers always uses 3 layers

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for VARIANT in "${PROJECT_VARIANTS[@]}"; do
#                 for K in "${PROJECT_KS[@]}"; do
#                     echo "------------------------------------------------"
#                     echo "Running project $VARIANT: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, K=$K"
#                     echo "------------------------------------------------"
#                     PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.project \
#                         --model "$MODEL" \
#                         --dataset "$DATASET" \
#                         --seed "$SEED" \
#                         --output_folder "$OUTPUT_FOLDER" \
#                         --variant "$VARIANT" \
#                         --context_k "$K"
#                 done
#             done
#         done
#     done
# done


# ==================================================================================================
# 20 StALT (arXiv:2605.01853), cosine distances: across-token change weighted by softmax over across-layer change,
#    τ ∈ {0.01, 0.1, 1, uniform}
# ==================================================================================================
OUTPUT_FOLDER="stalt"
STALT_SKIP_FIRST=(1) # 1: drop the first token (attention sink) before computing the deltas

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for SKIP in "${STALT_SKIP_FIRST[@]}"; do
#                 echo "------------------------------------------------"
#                 echo "Running stalt: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, skip_first=$SKIP"
#                 echo "------------------------------------------------"
#                 PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.stalt \
#                     --model "$MODEL" \
#                     --dataset "$DATASET" \
#                     --seed "$SEED" \
#                     --output_folder "$OUTPUT_FOLDER" \
#                     --skip_first "$SKIP"
#             done
#         done
#     done
# done


# ==================================================================================================
# 21 Causal perturbation probe: nudge h_t^l within span(h_{t-k..t-1}) (contrast: orthogonal to it), re-run layers
#    l+1..L, measure ‖Δh^j‖/ε per layer, KL(p_clean ‖ p_pert) and |ΔH| of the next-token distribution
# ==================================================================================================
OUTPUT_FOLDER="perturb"
PROBE_LAYERS=(5)
CSD_WINDOWS=(3)
REL_EPSS=(0.02 0.05) # nudge size ε = rel_eps · ‖h_t‖

# for MODEL in "${MODELS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         for DATASET in "${DATASETS[@]}"; do
#             for LAYER in "${PROBE_LAYERS[@]}"; do
#                 for K in "${CSD_WINDOWS[@]}"; do
#                     for REL_EPS in "${REL_EPSS[@]}"; do
#                         echo "------------------------------------------------"
#                         echo "Running perturb: Dataset=$DATASET, Model=$MODEL, Seed=$SEED, layer=$LAYER, k=$K, rel_eps=$REL_EPS"
#                         echo "------------------------------------------------"
#                         PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.ideas.perturb \
#                             --model "$MODEL" \
#                             --dataset "$DATASET" \
#                             --seed "$SEED" \
#                             --output_folder "$OUTPUT_FOLDER" \
#                             --probe_layer "$LAYER" \
#                             --csd_window "$K" \
#                             --rel_eps "$REL_EPS" \
#                             --n_dirs 4 \
#                             --token_stride 4 \
#                             --readout norm
#                     done
#                 done
#             done
#         done
#     done
# done
