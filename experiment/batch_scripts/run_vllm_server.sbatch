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

# Cache directories on scratch
CACHE_ROOT=/n/netscratch/tambe_lab/Lab/msong300
mkdir -p "${CACHE_ROOT}/hf_home" "${CACHE_ROOT}/hf_cache" "${CACHE_ROOT}/vllm_cache" "${CACHE_ROOT}/tmp"

# Set all cache-related environment variables
export HF_HOME="${CACHE_ROOT}/hf_home"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_cache"
export TRANSFORMERS_CACHE="${CACHE_ROOT}/hf_cache"
export HF_DATASETS_CACHE="${CACHE_ROOT}/hf_cache"
export TMPDIR="${CACHE_ROOT}/tmp"
export TEMP="${CACHE_ROOT}/tmp"
export TMP="${CACHE_ROOT}/tmp"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg_cache"
export XDG_DATA_HOME="${CACHE_ROOT}/xdg_data"
mkdir -p "${XDG_CACHE_HOME}" "${XDG_DATA_HOME}"

# Disable vLLM usage stats to avoid disk space issues
# Try multiple ways to disable usage stats
export VLLM_USAGE_STATS_DISABLE=1
export VLLM_DISABLE_USAGE_STATS=1
# If disabling doesn't work, redirect usage stats to scratch
export VLLM_USAGE_STATS_PATH="${CACHE_ROOT}/vllm_usage_stats.json" 2>/dev/null || true

MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-8B}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
GPU_UTIL=${GPU_UTIL:-0.94}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}
DTYPE=${DTYPE:-bfloat16}

export BASHRCSOURCED=1
source ~/.bashrc
module load python || true
conda activate hipporag

# Run vLLM with download directory pointing to scratch
CUDA_VISIBLE_DEVICES=0 vllm serve "${MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --download-dir "${CACHE_ROOT}/vllm_cache"

