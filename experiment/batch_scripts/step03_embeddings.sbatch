#!/bin/bash
#SBATCH --job-name=embeddings
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

EXPERIMENT_NAME=${EXPERIMENT_NAME:-demo_run}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B-Instruct}
EMBED_MODEL_NAME=${EMBED_MODEL_NAME:-facebook/contriever-msmarco}
EMBED_BATCH_SIZE=${EMBED_BATCH_SIZE:-32}

source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

python experiment/offline_indexing/03_encode_embeddings.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --embedding-model-name "${EMBED_MODEL_NAME}" \
  --embedding-batch-size "${EMBED_BATCH_SIZE}"

