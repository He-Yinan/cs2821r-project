#!/bin/bash
# Setup script to configure cache directories for vLLM
# Source this file before running vLLM commands: source setup_vllm_cache.sh

# Cache directories on scratch
CACHE_ROOT=/n/netscratch/tambe_lab/Lab/msong300
mkdir -p "${CACHE_ROOT}/hf_home" "${CACHE_ROOT}/hf_cache" "${CACHE_ROOT}/vllm_cache" "${CACHE_ROOT}/tmp" "${CACHE_ROOT}/xdg_cache" "${CACHE_ROOT}/xdg_data"

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

# Disable vLLM usage stats to avoid disk space issues
export VLLM_USAGE_STATS_DISABLE=1
export VLLM_DISABLE_USAGE_STATS=1
# If disabling doesn't work, redirect usage stats to scratch
export VLLM_USAGE_STATS_PATH="${CACHE_ROOT}/vllm_usage_stats.json" 2>/dev/null || true

echo "Cache directories configured to use: ${CACHE_ROOT}"
echo "Run your vLLM command now, e.g.:"
echo "  CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 12288 --gpu-memory-utilization 0.94 --download-dir ${CACHE_ROOT}/vllm_cache"













