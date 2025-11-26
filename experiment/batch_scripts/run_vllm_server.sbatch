#!/bin/bash
#SBATCH --job-name=vllm-qwen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/%x-%j.out
#SBATCH --error=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/%x-%j.err
set -eo pipefail

LOG_ROOT=/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs
mkdir -p "${LOG_ROOT}"

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B-Instruct}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.94}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
DTYPE=${DTYPE:-bfloat16}

export BASHRCSOURCED=1
source ~/.bashrc
module load python || true
conda activate hipporag

CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}"

