#!/bin/bash
#SBATCH --job-name=graphbuild
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/%x-%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/%x-%j.err

set -eo pipefail

LOG_ROOT=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs
mkdir -p "${LOG_ROOT}"

CACHE_ROOT=/n/netscratch/tambe_lab/Lab/msong300
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-musique_demo}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B}
EMBED_MODEL_NAME=${EMBED_MODEL_NAME:-facebook/contriever-msmarco}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26106.rc.fas.harvard.edu:8000/v1}

export BASHRCSOURCED=1
source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

python experiment/offline_indexing/04_build_graph.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --embedding-model-name "${EMBED_MODEL_NAME}" \
  --llm-base-url "${LLM_BASE_URL}"


