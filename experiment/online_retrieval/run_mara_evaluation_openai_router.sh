#!/bin/bash
# MARA-RAG evaluation script with OpenAI GPT-4.1 as query router and Qwen as QA agent
# Usage: ./run_mara_evaluation_openai_router.sh [experiment_name] [num_questions] [questions_file] [output_subdir]
#   - experiment_name: Name of experiment (default: musique_demo)
#   - num_questions: Number of questions to evaluate (default: 50)
#   - questions_file: Path to questions file (default: musique.json, will be found in dataset path)
#   - output_subdir: Output subdirectory name (default: mara_results_openai_router)

set -e

EXPERIMENT_NAME=${1:-musique_demo}
NUM_QUESTIONS=${2:-50}
QUESTIONS_FILE_ARG=${3:-}
OUTPUT_SUBDIR_ARG=${4:-}

# LLM Configuration
# Router: OpenAI GPT-4.1 (via OpenAI API)
ROUTER_LLM_MODEL=${ROUTER_LLM_MODEL:-gpt-4.1}
ROUTER_LLM_BASE_URL=${ROUTER_LLM_BASE_URL:-https://api.openai.com/v1}

# OpenAI API Key (can be overridden by environment variable)
export OPENAI_API_KEY=${OPENAI_API_KEY:-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}

# QA: Qwen (via vLLM server)
QA_LLM_BASE_URL=${QA_LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8000/v1}
QA_LLM_MODEL=${QA_LLM_MODEL:-Qwen/Qwen3-8B}

# Other settings
EMBEDDING_MODEL=${EMBEDDING_MODEL:-facebook/contriever-msmarco}
DATASET_PATH=${DATASET_PATH:-/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/musique/subset_50}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Activate conda environment
echo "Activating conda environment..."
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

# Verify Python is available and check if numpy is installed
python -c "import numpy" 2>/dev/null || {
    echo "Warning: numpy not found in current environment"
    echo "Current Python: $(which python)"
    echo "Python version: $(python --version 2>&1)"
}

echo "=========================================="
echo "MARA-RAG End-to-End Evaluation"
echo "  (OpenAI GPT-4.1 Router + Qwen QA)"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Number of questions: $NUM_QUESTIONS"
echo ""
echo "Router LLM: $ROUTER_LLM_MODEL at $ROUTER_LLM_BASE_URL"
echo "QA LLM: $QA_LLM_MODEL at $QA_LLM_BASE_URL"
echo "Dataset path: $DATASET_PATH"
echo ""

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    echo "The API key should have been set at the top of this script."
    exit 1
else
    echo "✓ OpenAI API key configured (length: ${#OPENAI_API_KEY} chars)"
    # Verify it's the expected key (first few chars)
    if [[ "$OPENAI_API_KEY" == sk-proj-* ]]; then
        echo "✓ API key format looks correct"
    else
        echo "Warning: API key format may be incorrect"
    fi
fi

# Check if vLLM server is running (for QA)
echo "Checking vLLM server connection (for QA)..."
HEALTH_URL="${QA_LLM_BASE_URL%/v1}/health"
if ! curl -s "$HEALTH_URL" > /dev/null 2>&1; then
    echo "Warning: vLLM server may not be running at $QA_LLM_BASE_URL"
    echo "Please check the server status"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 1: Preprocess graph (if matrices don't exist)
RESULTS_BASE="/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/experiments"
MATRIX_DIR="$RESULTS_BASE/$EXPERIMENT_NAME/mara_matrices"

# Check if matrices exist and are valid
MATRICES_EXIST=false
if [ -d "$MATRIX_DIR" ] && [ -n "$(ls -A $MATRIX_DIR/*.npz 2>/dev/null)" ] && [ -f "$MATRIX_DIR/graph_metadata.pkl" ]; then
    MATRICES_EXIST=true
fi

if [ "$MATRICES_EXIST" = false ]; then
    echo ""
    echo "Step 1: Preprocessing graph into relation matrices..."
    echo "Note: Using QA LLM ($QA_LLM_MODEL) for graph preprocessing"
    cd "$PROJECT_ROOT"
    python -m experiment.online_retrieval.graph_preprocessing \
        --experiment-name "$EXPERIMENT_NAME" \
        --workspace-subdir hipporag_workspace \
        --output-subdir mara_matrices \
        --llm-model-name "$QA_LLM_MODEL" \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --llm-base-url "$QA_LLM_BASE_URL"
else
    echo ""
    echo "Step 1: Skipping graph preprocessing (matrices already exist in $MATRIX_DIR)"
fi

# Step 2: Run evaluation
echo ""
echo "Step 2: Running MARA-RAG evaluation..."
echo "  Router: $ROUTER_LLM_MODEL (OpenAI)"
echo "  QA: $QA_LLM_MODEL (vLLM)"
cd "$PROJECT_ROOT"

# Find questions file (default to musique.json in dataset path)
QUESTIONS_FILE="${QUESTIONS_FILE_ARG:-musique.json}"

# If just filename provided, try to find it in dataset path
if [ ! -f "$QUESTIONS_FILE" ]; then
    if [ -f "$DATASET_PATH/$QUESTIONS_FILE" ]; then
        QUESTIONS_FILE="$DATASET_PATH/$QUESTIONS_FILE"
    elif [ -f "$DATASET_PATH/musique.json" ]; then
        QUESTIONS_FILE="$DATASET_PATH/musique.json"
    elif [ -f "experiment/dataset/$EXPERIMENT_NAME/questions.jsonl" ]; then
        QUESTIONS_FILE="experiment/dataset/$EXPERIMENT_NAME/questions.jsonl"
    elif [ -f "reproduce/dataset/${EXPERIMENT_NAME}.json" ]; then
        QUESTIONS_FILE="reproduce/dataset/${EXPERIMENT_NAME}.json"
    else
        echo "Error: Could not find questions file"
        echo "Checked:"
        echo "  - $DATASET_PATH/$QUESTIONS_FILE"
        echo "  - $DATASET_PATH/musique.json"
        echo "  - experiment/dataset/$EXPERIMENT_NAME/questions.jsonl"
        echo "  - reproduce/dataset/${EXPERIMENT_NAME}.json"
        exit 1
    fi
fi

echo "Using questions file: $QUESTIONS_FILE"

# Allow overriding output subdir (default: mara_results_openai_router)
# Priority: command-line argument > environment variable > default
OUTPUT_SUBDIR=${OUTPUT_SUBDIR_ARG:-${OUTPUT_SUBDIR:-mara_results_openai_router}}

# Ensure API key is exported and visible to Python
export OPENAI_API_KEY

# Run the experiment (API key will be available to Python via os.getenv)
python -m experiment.online_retrieval.run_mara_experiment \
    --experiment-name "$EXPERIMENT_NAME" \
    --questions-file "$QUESTIONS_FILE" \
    --workspace-subdir hipporag_workspace \
    --matrix-subdir mara_matrices \
    --output-subdir "$OUTPUT_SUBDIR" \
    --llm-model-name "$QA_LLM_MODEL" \
    --embedding-model-name "$EMBEDDING_MODEL" \
    --llm-base-url "$QA_LLM_BASE_URL" \
    --router-llm-model-name "$ROUTER_LLM_MODEL" \
    --router-llm-base-url "$ROUTER_LLM_BASE_URL" \
    --dataset-path "$DATASET_PATH" \
    --num-questions "$NUM_QUESTIONS" \
    --clear-cache

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $RESULTS_BASE/$EXPERIMENT_NAME/$OUTPUT_SUBDIR/"
echo "=========================================="

