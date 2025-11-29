#!/bin/bash
#SBATCH --job-name=graphbuild_baseline
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/home13/yinan/cs2821r-project/logs/%x-%j.out
#SBATCH --error=/n/home13/yinan/cs2821r-project/logs/%x-%j.err

set -eo pipefail

LOG_ROOT=/n/home13/yinan/cs2821r-project/logs
mkdir -p "${LOG_ROOT}"

CACHE_ROOT=/n/holylabs/ydu_lab/Lab/yinan
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-musique_demo}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B}
EMBED_MODEL_NAME=${EMBED_MODEL_NAME:-facebook/contriever-msmarco}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8001/v1}

export BASHRCSOURCED=1
source ~/.bashrc
module load python || true
conda activate hipporag
cd /n/home13/yinan/cs2821r-project

python experiment/offline_indexing/04_build_graph_baseline.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --embedding-model-name "${EMBED_MODEL_NAME}" \
  --llm-base-url "${LLM_BASE_URL}"


