#!/bin/bash
#SBATCH --job-name=mara-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

# Experiment parameters (can be overridden via environment variables)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-musique_demo}
NUM_QUESTIONS=${NUM_QUESTIONS:-50}
LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8000/v1}
LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3-8B}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-facebook/contriever-msmarco}
DATASET_PATH=${DATASET_PATH:-/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/musique/subset_50}

# Project root
PROJECT_ROOT=/n/home00/msong300/cs2821r-project
cd "${PROJECT_ROOT}"

# Activate conda environment
export BASHRCSOURCED=1
source ~/.bashrc 2>/dev/null || true
module load python 2>/dev/null || true

# Initialize conda if not already initialized
if ! command -v conda &> /dev/null; then
    # Try to find conda installation
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/n/sw/conda3/etc/profile.d/conda.sh" ]; then
        source "/n/sw/conda3/etc/profile.d/conda.sh"
    else
        # Try to get conda base from conda info
        CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
        if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
            source "$CONDA_BASE/etc/profile.d/conda.sh"
        fi
    fi
fi

# Activate the environment
conda activate hipporag || {
    echo "Error: Failed to activate 'hipporag' conda environment"
    echo "Please ensure the environment exists: conda env list"
    exit 1
}

# Verify Python is available
python -c "import numpy" 2>/dev/null || {
    echo "Warning: numpy not found in current environment"
    echo "Current Python: $(which python)"
    echo "Python version: $(python --version 2>&1)"
}

echo "=========================================="
echo "MARA-RAG Evaluation (SBATCH)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Number of questions: ${NUM_QUESTIONS}"
echo "LLM URL: ${LLM_BASE_URL}"
echo "Dataset path: ${DATASET_PATH}"
echo "=========================================="
echo ""

# Run the evaluation script
bash experiment/online_retrieval/run_mara_evaluation.sh \
    "${EXPERIMENT_NAME}" \
    "${NUM_QUESTIONS}" \
    musique.json

echo ""
echo "=========================================="
echo "Job completed: ${SLURM_JOB_ID}"
echo "=========================================="

