#!/bin/bash
#SBATCH --job-name=chunkner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/%x-%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/%x-%j.err

set -eo pipefail

LOG_ROOT=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs
mkdir -p "${LOG_ROOT}"

# Cache directories on scratch
CACHE_ROOT=/n/netscratch/tambe_lab/Lab/msong300
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"

# ---- user-configurable variables ----
EXPERIMENT_NAME=${EXPERIMENT_NAME:-musique_demo}
CORPUS_FILE=${CORPUS_FILE:-/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/musique/subset_50/musique_corpus.json}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26106.rc.fas.harvard.edu:8000/v1}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B}
MAX_WORKERS=${MAX_WORKERS:-8}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-offline_indexing/01_chunk_ner_50}

# ---- environment setup (edit as needed) ----
export BASHRCSOURCED=1
source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

python experiment/offline_indexing/01_chunk_and_ner.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --corpus-file "${CORPUS_FILE}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --max-workers "${MAX_WORKERS}" \
  --output-subdir "${OUTPUT_SUBDIR}"
