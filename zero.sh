#!/bin/bash
#SBATCH --job-name=zmgt_save_scores
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem=30GB
#SBATCH --partition=gpu,nmes_gpu,interruptible_gpu
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=erc-hpc-comp054,erc-hpc-comp048
# SBATCH --partition=tier2_gpu 
# SBATCH --account=er_prj_inf_impact_llm_wikipedia

# set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZMGT:-$(pwd)}"
cd "${ROOT_DIR}"


MODELS=("l8b") # "q8b" "q8bb" "l8b" "l8bb"

# detectRLX - Attacks
# DATASETS=(
#   "drlXAttacks_backtranslation" "drlXAttacks_character_deletion"
#   "drlXAttacks_character_insertion" "drlXAttacks_character_substitution"
#   "drlXAttacks_condensing" "drlXAttacks_decoder_paraphrasing"
#   "drlXAttacks_encoder_paraphrasing" "drlXAttacks_expanding"
#   "drlXAttacks_general_64" "drlXAttacks_general_128"
#   "drlXAttacks_general_256" "drlXAttacks_general_512"
#   "drlXAttacks_polishing" "drlXAttacks_seq2seq_paraphrasing"
#   "drlXAttacks_zero_width_insertion"
# )

# detectRLX - Models
# DATASETS=(
#   "drlXModel_deepseek_v3" "drlXModel_gemini_2_5_flash"
#   "drlXModel_gpt_4o" "drlXModel_qwen_max"
# )

# detectRLX - Domains
# DATASETS=(
#   "drlXDomain_academic" "drlXDomain_news" "drlXDomain_novel"
#   "drlXDomain_seo" "drlXDomain_webtext" "drlXDomain_wiki"
# )

# detectRLX - Languages
# DATASETS=("drlXLang_chinese" "drlXLang_german" "drlXLang_portuguese" "drlXLang_russian")


# detectRLX - All
# DATASETS=(
#   "drlXDomain_academic" "drlXDomain_news" "drlXDomain_novel"
#   "drlXDomain_seo" "drlXDomain_webtext" "drlXDomain_wiki"
#   "drlXLang_arabic" "drlXLang_chinese"
#   "drlXLang_german" "drlXLang_portuguese" "drlXLang_russian"
#   "drlXModel_deepseek_v3" "drlXModel_gemini_2_5_flash"
#   "drlXModel_gpt_4o" "drlXModel_qwen_max"
#   "drlXAttacks_backtranslation" "drlXAttacks_character_deletion"
#   "drlXAttacks_character_insertion" "drlXAttacks_character_substitution"
#   "drlXAttacks_condensing" "drlXAttacks_decoder_paraphrasing"
#   "drlXAttacks_encoder_paraphrasing" 
#   "drlXAttacks_expanding"
#   "drlXAttacks_general_64" "drlXAttacks_general_128"
#   "drlXAttacks_general_256" "drlXAttacks_general_512"
#   "drlXAttacks_polishing" "drlXAttacks_seq2seq_paraphrasing"
#   "drlXAttacks_zero_width_insertion"
#   "drlXLang_chinese" "drlXLang_german" "drlXLang_portuguese" "drlXLang_russian"
# )

# RAID - Domains
# DATASETS=(
#   "raidDomain_abstracts" "raidDomain_books" "raidDomain_news"
#   "raidDomain_poetry" "raidDomain_recipes" "raidDomain_reddit"
#   "raidDomain_reviews" "raidDomain_wiki"
# )

# RAID - Models
# DATASETS=(
#   "raidModel_chatgpt" "raidModel_cohere_chat" "raidModel_cohere"
#   "raidModel_gpt2" "raidModel_gpt3" "raidModel_gpt4"
#   "raidModel_llama_chat" "raidModel_mistral_chat" "raidModel_mistral"
#   "raidModel_mpt_chat" "raidModel_mpt"
# )

# Sampling - Temperature
# DATASETS=(
#   "samplingTemperature_0_5" "samplingTemperature_0_7"
#   "samplingTemperature_0_9" "samplingTemperature_1_0"
#   "samplingTemperature_1_1" "samplingTemperature_1_2"
#   "samplingTemperature_1_3"
# )

# Sampling - Top-k
# DATASETS=(
#   "samplingTopK_10" "samplingTopK_20" "samplingTopK_50"
#   "samplingTopK_75" "samplingTopK_100" "samplingTopK_1000"
# )

# Sampling - Top-p
# DATASETS=(
#   "samplingTopP_0_3" "samplingTopP_0_5" "samplingTopP_0_7"
#   "samplingTopP_0_8" "samplingTopP_0_9" "samplingTopP_0_95"
# )

# Sampling - Eta
# DATASETS=(
#   "samplingEta_1e_4" "samplingEta_1e_3" "samplingEta_5e_3"
#   "samplingEta_0_01" "samplingEta_0_05" "samplingEta_0_1"
# )

# Sampling - Repetition penalty
# DATASETS=(
#   "samplingRepetitionPenalty_1_05" "samplingRepetitionPenalty_1_1"
#   "samplingRepetitionPenalty_1_15" "samplingRepetitionPenalty_1_2"
#   "samplingRepetitionPenalty_1_25" "samplingRepetitionPenalty_1_3"
# )

# One full run (all datasets, one seed) takes about 14 hrs
# DATASETS=("drlXMix_mixed")


# RAID - Domains - Models
# DATASETS=(
# "raidModel_chatgpt" "raidModel_cohere_chat" "raidModel_cohere"
# "raidModel_llama_chat" "raidDomain_recipes" "raidDomain_reddit"
# "raidDomain_reviews" "raidDomain_wiki"
# )

# # EditLens - sources
# DATASETS=(
#     "editlens_amazon_reviews" "editlens_fineweb_edu"
#     "editlens_google_reviews" "editlens_news"
#     "editlens_reddit_writing_prompts"
# )

# "drlXAttacks_decoder_paraphrasing" "drlXDomain_academic" "drlXAttacks_character_substitution"
# DATASETS=("drlXDomain_seo" "drlXAttacks_decoder_paraphrasing" "editlens_fineweb_edu")
DATASETS=("raidDomain_wiki")

SEEDS=(42) # 42 43 44 45 46
OUTPUT_FOLDER="sandbox" # output/zero/sandbox
BENCHMARK=0
SAVE_SCORES=1 # 1 = store per-text scores in the output json

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
            echo "------------------------------------------------"
            echo "Running angle_norm: Dataset=$DATASET, Model=$MODEL, Seed=$SEED"
            echo "------------------------------------------------"
            PYTHONPATH="${ROOT_DIR}" uv run --project "${BASE_ZERO}" -m src.zero \
                --model "$MODEL" \
                --dataset "$DATASET" \
                --seed "$SEED" \
                --output_folder "$OUTPUT_FOLDER" \
                --benchmark "$BENCHMARK" \
                --save_scores "$SAVE_SCORES"
        done
    done
done
