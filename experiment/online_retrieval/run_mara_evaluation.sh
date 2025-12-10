#!/bin/bash
# Simple end-to-end script to run MARA-RAG evaluation
# Usage: ./run_mara_evaluation.sh [experiment_name] [num_questions] [questions_file] [output_subdir] [--visualize]
#   - experiment_name: Name of experiment (default: musique_demo)
#   - num_questions: Number of questions to evaluate (default: 50)
#   - questions_file: Path to questions file (default: musique.json, will be found in dataset path)
#   - output_subdir: Output subdirectory name (default: mara_results, can also be set via OUTPUT_SUBDIR env var)
#   - --visualize: Optional flag to generate animated PPR visualization GIFs

set -e

# Parse positional arguments
EXPERIMENT_NAME=${1:-musique_demo}
NUM_QUESTIONS=${2:-50}
QUESTIONS_FILE_ARG=${3:-}
OUTPUT_SUBDIR_ARG=${4:-}

# Check for --visualize, --visualize-exact-match, and --qa-top-k flags in all arguments
VISUALIZE_FLAG=""
VISUALIZE_EXACT_MATCH_FLAG=""
QA_TOP_K=""
i=1
while [ $i -le $# ]; do
    eval "arg=\${$i}"
    if [ "$arg" = "--visualize" ]; then
        VISUALIZE_FLAG="--visualize"
    elif [ "$arg" = "--visualize-exact-match" ]; then
        VISUALIZE_EXACT_MATCH_FLAG="--visualize-exact-match"
    elif [ "$arg" = "--qa-top-k" ] && [ $((i+1)) -le $# ]; then
        next_idx=$((i+1))
        eval "QA_TOP_K=\${$next_idx}"
        i=$((i+1))  # Skip the next argument (the value)
    fi
    i=$((i+1))
done

LLM_BASE_URL=${LLM_BASE_URL:-http://holygpu7c26105.rc.fas.harvard.edu:8000/v1}
LLM_MODEL=${LLM_MODEL:-Qwen/Qwen3-8B}
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

# Check for visualization packages if --visualize flag is present
if [ -n "$VISUALIZE_FLAG" ]; then
    echo "Checking visualization dependencies..."
    MISSING_PACKAGES=()
    
    python -c "import matplotlib" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
    python -c "import matplotlib.animation" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
    python -c "import networkx" 2>/dev/null || MISSING_PACKAGES+=("networkx")
    python -c "from PIL import Image" 2>/dev/null || MISSING_PACKAGES+=("pillow")
    
    if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
        echo "Warning: Missing visualization packages: ${MISSING_PACKAGES[*]}"
        echo "Installing missing packages..."
        
        # Set cache directory for conda and pip
        CACHE_DIR="/n/netscratch/tambe_lab/Lab/msong300/.conda"
        mkdir -p "$CACHE_DIR"
        export CONDA_PKGS_DIRS="$CACHE_DIR"
        export PIP_CACHE_DIR="$CACHE_DIR/pip_cache"
        mkdir -p "$PIP_CACHE_DIR"
        
        echo "Using cache directory: $CACHE_DIR"
        
        # Get the conda environment's pip and python
        PIP_CMD=$(which pip)
        PYTHON_CMD=$(which python)
        echo "Using pip: $PIP_CMD"
        echo "Using python: $PYTHON_CMD"
        
        # Try conda install first (preferred for conda environments)
        if command -v conda &> /dev/null; then
            echo "Attempting to install via conda..."
            # Use conda install with -y flag to avoid prompts and custom cache dir
            if conda install -y matplotlib networkx pillow -c conda-forge 2>&1 | tee /tmp/conda_install.log; then
                echo "✓ Visualization packages installed via conda"
            else
                echo "Conda install failed, trying pip with --user flag..."
                # Try pip with --user flag (installs to ~/.local, which is writable)
                # Use --cache-dir to use the specified cache directory
                if "$PIP_CMD" install --user --cache-dir "$PIP_CACHE_DIR" matplotlib networkx pillow 2>&1; then
                    echo "✓ Visualization packages installed via pip (user install)"
                else
                    echo "Error: Failed to install visualization packages"
                    echo ""
                    echo "Please install manually using one of these methods:"
                    echo "  1. CONDA_PKGS_DIRS=$CACHE_DIR conda install matplotlib networkx pillow -c conda-forge"
                    echo "  2. pip install --user --cache-dir $PIP_CACHE_DIR matplotlib networkx pillow"
                    echo ""
                    echo "Or activate the environment and install:"
                    echo "  conda activate hipporag"
                    echo "  CONDA_PKGS_DIRS=$CACHE_DIR conda install matplotlib networkx pillow -c conda-forge"
                    exit 1
                fi
            fi
        else
            # No conda, try pip with --user flag
            echo "Conda not available, trying pip with --user flag..."
            if "$PIP_CMD" install --user --cache-dir "$PIP_CACHE_DIR" matplotlib networkx pillow 2>&1; then
                echo "✓ Visualization packages installed via pip (user install)"
            else
                echo "Error: Failed to install visualization packages"
                echo "Please install manually: pip install --user --cache-dir $PIP_CACHE_DIR matplotlib networkx pillow"
                exit 1
            fi
        fi
    else
        echo "✓ All visualization packages are available"
    fi
fi

echo "=========================================="
echo "MARA-RAG End-to-End Evaluation"
echo "=========================================="
echo "Experiment: $EXPERIMENT_NAME"
echo "Number of questions: $NUM_QUESTIONS"
echo "LLM URL: $LLM_BASE_URL"
echo "Dataset path: $DATASET_PATH"
echo ""

# Check if vLLM server is running
echo "Checking vLLM server connection..."
# Try health endpoint
HEALTH_URL="${LLM_BASE_URL%/v1}/health"
if ! curl -s "$HEALTH_URL" > /dev/null 2>&1; then
    echo "Warning: vLLM server may not be running at $LLM_BASE_URL"
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
    cd "$PROJECT_ROOT"
    python -m experiment.online_retrieval.graph_preprocessing \
        --experiment-name "$EXPERIMENT_NAME" \
        --workspace-subdir hipporag_workspace \
        --output-subdir mara_matrices \
        --llm-model-name "$LLM_MODEL" \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --llm-base-url "$LLM_BASE_URL"
else
    echo ""
    echo "Step 1: Skipping graph preprocessing (matrices already exist in $MATRIX_DIR)"
fi

# Step 2: Run evaluation
echo ""
echo "Step 2: Running MARA-RAG evaluation..."
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

# Allow overriding output subdir (default: mara_results)
# Priority: command-line argument > environment variable > default
OUTPUT_SUBDIR=${OUTPUT_SUBDIR_ARG:-${OUTPUT_SUBDIR:-mara_results}}

# Build Python command arguments
PYTHON_ARGS=(
    --experiment-name "$EXPERIMENT_NAME"
    --questions-file "$QUESTIONS_FILE"
    --workspace-subdir hipporag_workspace
    --matrix-subdir mara_matrices
    --output-subdir "$OUTPUT_SUBDIR"
    --llm-model-name "$LLM_MODEL"
    --embedding-model-name "$EMBEDDING_MODEL"
    --llm-base-url "$LLM_BASE_URL"
    --dataset-path "$DATASET_PATH"
    --num-questions "$NUM_QUESTIONS"
    --clear-cache
    --verbose
)

# Add --qa-top-k if provided
if [ -n "$QA_TOP_K" ]; then
    PYTHON_ARGS+=(--qa-top-k "$QA_TOP_K")
    echo "QA top-k passages: $QA_TOP_K"
fi

# Add --visualize flag if requested
if [ -n "$VISUALIZE_FLAG" ]; then
    PYTHON_ARGS+=("--visualize")
    echo "Visualization enabled: will generate animated PPR GIFs"
fi

# Add --visualize-exact-match flag if requested
if [ -n "$VISUALIZE_EXACT_MATCH_FLAG" ]; then
    PYTHON_ARGS+=("--visualize-exact-match")
    echo "Visualization mode: only for questions with exact match = 1.0"
fi

# Execute Python command
python -m experiment.online_retrieval.run_mara_experiment "${PYTHON_ARGS[@]}"

echo ""
echo "=========================================="
echo "Evaluation complete!"
echo "Results saved to: $RESULTS_BASE/$EXPERIMENT_NAME/$OUTPUT_SUBDIR/"
echo "=========================================="

