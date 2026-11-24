#!/bin/bash
#SBATCH --job-name=b1_ts
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB
#SBATCH --constraint=h100
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp040
#SBATCH --partition=gpu,nmes_gpu

# Runtime analysis (latency and GPU memory, batch size 1) of the base1 detectors and our joint score.
# Runs with uv and the project's default environment, as base1.sh. Output: output/time_space/
# Use the same GPU type (--constraint) as base_2_ts.sh so the numbers are comparable.
# CUDA_LAUNCH_BLOCKING is not set (base1.sh sets it): it makes every GPU call wait and would distort the latency.

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

DATASETS=("drlXDomainLen_wiki_300_tokens")
SEEDS=(42)
MODELS=("joint" "fdgpt" "irm")

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Timing: Model=$MODEL, Dataset=$DATASET, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run -m src.baseline_time.run_time \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED"
        done
    done
done
