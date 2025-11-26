#!/bin/bash
#SBATCH --job-name=chunkner
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ---- user-configurable variables ----
EXPERIMENT_NAME=${EXPERIMENT_NAME:-demo_run}
CORPUS_FILE=${CORPUS_FILE:-/n/home00/msong300/cs2821r-project/rag/reproduce/dataset/musique_corpus.json}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8000/v1}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B-Instruct}
MAX_WORKERS=${MAX_WORKERS:-8}

# ---- environment setup (edit as needed) ----
source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

python experiment/offline_indexing/01_chunk_and_ner.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --corpus-file "${CORPUS_FILE}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --max-workers "${MAX_WORKERS}"

