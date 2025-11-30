#!/bin/bash
#SBATCH --job-name=chunkner_baseline
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/n/home13/yinan/cs2821r-project/logs/%x-%j.out
#SBATCH --error=/n/home13/yinan/cs2821r-project/logs/%x-%j.err

set -eo pipefail

LOG_ROOT=/n/home13/yinan/cs2821r-project/logs
mkdir -p "${LOG_ROOT}"

# Cache directories on scratch
CACHE_ROOT=/n/holylabs/ydu_lab/Lab/yinan
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"

# ---- user-configurable variables ----
EXPERIMENT_NAME=${EXPERIMENT_NAME:-musique_demo}
CORPUS_FILE=${CORPUS_FILE:-/n/holylabs/ydu_lab/Lab/yinan/28-results/datasets/musique/subset_50/musique_corpus.json}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8001/v1} # TODO
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B}
MAX_WORKERS=${MAX_WORKERS:-8}

# ---- environment setup (edit as needed) ----
export BASHRCSOURCED=1
source ~/.bashrc
module load python || true
conda activate hipporag
cd /n/home13/yinan/cs2821r-project

python experiment/qa.py
