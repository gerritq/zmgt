#!/bin/bash
#SBATCH --job-name=b2_ts
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --partition=gpu,nmes_gpu
#SBATCH --constraint=h100
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp054,erc-hpc-comp250

# Runtime analysis (latency and GPU memory, batch size 1) of the base2 detectors. Output: output/time_space/
# Needs the .venv_bino environment, which lives in BASE_ZERO (as in base2.sh); the code runs from BASE_ZMGT.
# Use the same GPU type (--constraint) as base_1_ts.sh so the numbers are comparable.
# CUDA_LAUNCH_BLOCKING is not set (base2.sh sets it): it makes every GPU call wait and would distort the latency.

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"

cd "${BASE_ZERO}"
source .venv_bino/bin/activate
which python
python --version

cd "${ROOT_DIR}"

DATASETS=("drlXDomainLen_wiki_300_tokens")
SEEDS=(42)
MODELS=("lastde" "detectllm" "binoculars")

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Timing: Model=$MODEL, Dataset=$DATASET, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" python -m src.baseline_time.run_time \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED"
        done
    done
done
