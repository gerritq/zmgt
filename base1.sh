#!/bin/bash
#SBATCH --job-name=b1_curv_SALL
#SBATCH --output=logs/%j.log
#SBATCH --error=logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --constraint=h200|h100|b200|a100|a100_40g|a100_80g
#SBATCH --exclude=erc-hpc-comp035,erc-hpc-comp054,erc-hpc-comp040,erc-hpc-comp050
#SBATCH --partition=gpu,nmes_gpu
# SBATCH --partition=tier2_gpu 
# SBATCH --account=er_prj_inf_impact_llm_wikipedia

set -euo pipefail

nvidia-smi

ROOT_DIR="${BASE_ZERO:-$(pwd)}"
cd "${ROOT_DIR}"

export CUDA_LAUNCH_BLOCKING=1

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
#   "drlXModel_deepseek_v3" "drlXModel_gemini_2_5_flash"
#   "drlXModel_gpt_4o" "drlXModel_qwen_max"
#   "drlXAttacks_backtranslation" "drlXAttacks_character_deletion"
#   "drlXAttacks_character_insertion" "drlXAttacks_character_substitution"
#   "drlXAttacks_condensing" "drlXAttacks_decoder_paraphrasing"
#   "drlXAttacks_encoder_paraphrasing" "drlXAttacks_expanding"
#   "drlXAttacks_polishing" "drlXAttacks_seq2seq_paraphrasing"
#   "drlXAttacks_zero_width_insertion" "drlXMix_mixed"
# )

# Sampling - Temperature and Top-p
# DATASETS=(
#     "samplingTemperature_0_5" "samplingTemperature_0_7"
#     "samplingTemperature_0_9" "samplingTemperature_1_0"
#     "samplingTemperature_1_1" "samplingTemperature_1_2"
#     "samplingTemperature_1_3"
#     "samplingTopP_0_3" "samplingTopP_0_5" "samplingTopP_0_7"
#     "samplingTopP_0_8" "samplingTopP_0_9" "samplingTopP_0_95"
# )

# RAID - Domains - Models
# DATASETS=(
# "raidModel_chatgpt" "raidModel_cohere_chat" "raidModel_cohere"
# "raidModel_llama_chat" "raidDomain_recipes" "raidDomain_reddit"
# "raidDomain_reviews" "raidDomain_wiki"
# )

# DATASETS=(
#             "samplingTemperature_0_5" "samplingTemperature_0_7"
#     "samplingTemperature_0_9" "samplingTemperature_1_0"
#     "samplingTemperature_1_1" "samplingTemperature_1_2"
#     "samplingTemperature_1_3"
#     "samplingTopP_0_3" "samplingTopP_0_5" "samplingTopP_0_7"
#     "samplingTopP_0_8" "samplingTopP_0_9" "samplingTopP_0_95"
#     "ntsraidDomain_abstracts" "ntsraidDomain_books" "ntsraidDomain_news"
#     "ntsraidDomain_poetry" "ntsraidDomain_recipes" "ntsraidDomain_reddit"
#     "ntsraidDomain_reviews" "ntsraidDomain_wiki"
#     "samplingTopK_10" "samplingTopK_20" "samplingTopK_50"
#     "samplingTopK_75" "samplingTopK_100" "samplingTopK_1000"
#     "samplingEta_1e_4" "samplingEta_1e_3" "samplingEta_5e_3"
#     "samplingEta_0_01" "samplingEta_0_05" "samplingEta_0_1"
# )

# DATASETS=(
#     "editlens_amazon_reviews" "editlens_fineweb_edu"
#     "editlens_google_reviews" "editlens_news"
#     "editlens_reddit_writing_prompts"
# )

# Sampling - Repetition penalty
# DATASETS=(
#   "samplingRepetitionPenalty_1_05" "samplingRepetitionPenalty_1_1"
#   "samplingRepetitionPenalty_1_15" "samplingRepetitionPenalty_1_2"
#   "samplingRepetitionPenalty_1_25" "samplingRepetitionPenalty_1_3"
# )


DATASETS=(
  "drlXAttacks_decoder_paraphrasing"
  "drlXAttacks_character_deletion"
  "raidDomain_wiki"
)

SEEDS=(42)

# Full ZERO-SHOT run (all datasets, one seed) takes about 10 hrs
MODELS=(
        # ZERO-SHOT
        # "llr" 
        # "likelihood" 
        # "entropy" 
        # "rank"
        # "irm"
        # "fdgpt"
        #  "revise"
        #  "gecscore"
        "curvature"
        # TRAINED
        # "radar"
        # "openai_roberta"
        # "editlens"
        # TO TRAIN
        # "repreguard"
        # "id"
        # "text_fluoroscopy"
         )

for MODEL in "${MODELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for DATASET in "${DATASETS[@]}"; do
        
            echo "------------------------------------------------"
            echo "Running Baseline: Model=$MODEL, Dataset=$DATASET, Seed=$SEED"
            echo "------------------------------------------------"

            PYTHONPATH="${ROOT_DIR}" uv run src/baseline/baseline.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --seed "$SEED"
        done
    done
done

