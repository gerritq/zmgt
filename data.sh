#!/bin/bash
#SBATCH --job-name=data_drlXDomains_len_balan
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=6
# SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
# sSBATCH --gres=gpu:1
#SBATCH --mem=50GB
# SBATCH --constraint=h200|a100|b200
#SBATCH --exclude=erc-hpc-comp054

# nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"

SEEDS=(42) # 42 43 44 45 46

for SEED in "${SEEDS[@]}"; do
    echo "------------------------------------------------"
    echo "Running Data Generation: Seed=$SEED"
    echo "------------------------------------------------"

    PYTHONPATH="${ROOT_DIR}" uv run -m src.data \
        --name detectRLX_domain_length_balanced \
        --seed "${SEED}"
done
