#!/bin/bash
#SBATCH --job-name=triples
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

CACHE_ROOT=/n/netscratch/tambe_lab/Lab/msong300
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-musique_demo}
# LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26106.rc.fas.harvard.edu:8000/v1}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26106.rc.fas.harvard.edu:8001/v1}
LLM_MODEL_NAME=${LLM_MODEL_NAME:-Qwen/Qwen3-8B}
MAX_WORKERS=${MAX_WORKERS:-8}
MAX_CHUNKS=${MAX_CHUNKS:-}  # Set to a number (e.g., 5) for testing, leave empty for full run
TEST_OUTPUT=${TEST_OUTPUT:-false}  # Set to "true" to save to triples_test.jsonl

export BASHRCSOURCED=1
source ~/.bashrc
conda activate hipporag
cd /n/home00/msong300/cs2821r-project

CMD_ARGS=(
  --experiment-name "${EXPERIMENT_NAME}"
  --llm-base-url "${LLM_BASE_URL}"
  --llm-model-name "${LLM_MODEL_NAME}"
  --max-workers "${MAX_WORKERS}"
)

if [ -n "${MAX_CHUNKS}" ]; then
  CMD_ARGS+=(--max-chunks "${MAX_CHUNKS}")
  echo "TEST MODE: Processing only ${MAX_CHUNKS} chunks"
fi

if [ "${TEST_OUTPUT}" = "true" ]; then
  CMD_ARGS+=(--test-output)
  echo "TEST MODE: Saving to triples_test.jsonl"
fi

python experiment/offline_indexing/02_triple_extraction.py "${CMD_ARGS[@]}"


