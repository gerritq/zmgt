#!/bin/bash
#SBATCH --job-name=items
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:05:00
#SBATCH --partition=cpu
#SBATCH --mem=2GB

# set -euo pipefail

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


# ----------------------------------------------------
# t_metrics_desc
# ----------------------------------------------------
# MODEL="l8b"
# DESC_DATASETS=("drlXDomainLen_wiki_500_tokens" "drlXDomainLen_wiki_2000_chars")
# for DATASET in "${DESC_DATASETS[@]}"; do
#     PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.items.t_metrics_desc \
#         --model "$MODEL" \
#         --dataset "$DATASET"
# done



# ----------------------------------------------------
# t_main
# ----------------------------------------------------
# SCORE: key in metrics_by_score (e.g. full_log, joint, withinacross_rel, intra_mean_norm)
# VERSION: raw | ratio (ratio = relative to layer_0)
# LAYER: layer index (layer_i = transformer layer i + 1)
MODEL="l8b"
SCORE="joint"
VERSION="ratio"
LAYER=7
PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.items.t_main \
    --model "$MODEL" \
    --score "$SCORE" \
    --version "$VERSION" \
    --layer "$LAYER"


# ----------------------------------------------------
# f_robustness
# ----------------------------------------------------
# SCORE: key in metrics_by_score (e.g. full_log, joint, withinacross_rel, intra_mean_norm)
# VERSION: raw | ratio (ratio = relative to layer_0)
# LAYER: layer index (layer_i = transformer layer i + 1)
MODEL="l8b"
SCORE="joint"
VERSION="ratio"
LAYER=7
PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.items.f_robustness \
    --model "$MODEL" \
    --score "$SCORE" \
    --version "$VERSION" \
    --layer "$LAYER"


# ----------------------------------------------------
# f_length
# ----------------------------------------------------
# SCORE: key in metrics_by_score (e.g. full_log, joint, withinacross_rel, intra_mean_norm)
# VERSION: raw | ratio (ratio = relative to layer_0)
# LAYER: layer index (layer_i = transformer layer i + 1)
MODEL="l8b"
SCORE="joint"
VERSION="ratio"
LAYER=7
PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.items.f_length \
    --model "$MODEL" \
    --score "$SCORE" \
    --version "$VERSION" \
    --layer "$LAYER"


# ----------------------------------------------------
# f_metrics_stats_corr
# ----------------------------------------------------
# MODEL="l8b"
# for DATASET in "drlXDomainLen_wiki_500_tokens" "drlXDomainLen_wiki_2000_chars"; do
#     PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.items.f_metrics_stats_corr \
#         --model "$MODEL" \
#         --dataset "$DATASET"
# done
