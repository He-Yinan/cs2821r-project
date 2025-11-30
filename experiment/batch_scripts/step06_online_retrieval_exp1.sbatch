#!/bin/bash
#SBATCH --job-name=online_retrieval_exp1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
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
DATASET_PATH=${DATASET_PATH:-/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/musique/subset_50/musique.json}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8000/v1}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B}
EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME:-facebook/contriever-msmarco}
WORKSPACE_SUBDIR=${WORKSPACE_SUBDIR:-hipporag_workspace}
OUTPUT_TAG=${OUTPUT_TAG:-exp1_zero_shot}
MODES=${MODES:-"dpr hipporag mara"}

# Hyperparameters (matching baseline comparison)
RETRIEVAL_TOP_K=${RETRIEVAL_TOP_K:-20}
LINKING_TOP_K=${LINKING_TOP_K:-5}
MAX_QA_STEPS=${MAX_QA_STEPS:-3}
QA_TOP_K=${QA_TOP_K:-5}
DISABLE_RERANK_FILTER=${DISABLE_RERANK_FILTER:-True}
NUM_FACTS_WITHOUT_RERANK=${NUM_FACTS_WITHOUT_RERANK:-10}

# ---- environment setup ----
export BASHRCSOURCED=1
source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

python experiment/online_retrieval/01_zero_shot_eval.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --dataset-path "${DATASET_PATH}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --embedding-model-name "${EMBEDDING_MODEL_NAME}" \
  --workspace-subdir "${WORKSPACE_SUBDIR}" \
  --output-tag "${OUTPUT_TAG}" \
  --retrieval-top-k "${RETRIEVAL_TOP_K}" \
  --linking-top-k "${LINKING_TOP_K}" \
  --max-qa-steps "${MAX_QA_STEPS}" \
  --qa-top-k "${QA_TOP_K}" \
  --disable-rerank-filter "${DISABLE_RERANK_FILTER}" \
  --num-facts-without-rerank "${NUM_FACTS_WITHOUT_RERANK}" \
  --modes ${MODES}
