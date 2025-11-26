#!/bin/bash
#SBATCH --job-name=triples
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

EXPERIMENT_NAME=${EXPERIMENT_NAME:-demo_run}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8000/v1}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B-Instruct}
MAX_WORKERS=${MAX_WORKERS:-8}

source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

python experiment/offline_indexing/02_triple_extraction.py \
  --experiment-name "${EXPERIMENT_NAME}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --llm-model-name "${LLM_MODEL_NAME}" \
  --max-workers "${MAX_WORKERS}"

