#!/bin/bash
#
# Quick Start Script for MARA-RAG
#
# This script demonstrates the complete MARA-RAG pipeline from preprocessing
# to evaluation.
#
# Prerequisites:
# 1. HippoRAG graph already built using offline indexing scripts (01-05)
# 2. LLM server running (e.g., vLLM at http://localhost:8000/v1)
# 3. Questions file prepared in JSONL format
#

set -e  # Exit on error

# Configuration
EXPERIMENT_NAME="musique_demo"
QUESTIONS_FILE="questions.jsonl"
LLM_BASE_URL="http://localhost:8000/v1"
LLM_MODEL="Qwen/Qwen3-8B-Instruct"
EMBEDDING_MODEL="facebook/contriever-msmarco"
NUM_QUESTIONS=50  # Set to empty string to process all questions

# Directories (relative to experiment folder)
WORKSPACE_DIR="hipporag_workspace"
MATRIX_DIR="mara_matrices"
RESULTS_DIR="mara_results"

echo "========================================="
echo "MARA-RAG Quick Start"
echo "========================================="
echo ""
echo "Experiment: $EXPERIMENT_NAME"
echo "Questions file: $QUESTIONS_FILE"
echo "LLM endpoint: $LLM_BASE_URL"
echo ""

# Step 1: Check prerequisites
echo "Step 1: Checking prerequisites..."
echo ""

WORKSPACE_PATH="experiment/dataset/$EXPERIMENT_NAME/$WORKSPACE_DIR"
if [ ! -d "$WORKSPACE_PATH" ]; then
    echo "ERROR: HippoRAG workspace not found at $WORKSPACE_PATH"
    echo "Please run offline indexing scripts (01-05) first."
    exit 1
fi

QUESTIONS_PATH="experiment/dataset/$EXPERIMENT_NAME/$QUESTIONS_FILE"
if [ ! -f "$QUESTIONS_PATH" ]; then
    echo "ERROR: Questions file not found at $QUESTIONS_PATH"
    echo "Please prepare a questions.jsonl file."
    exit 1
fi

# Check if LLM server is running
if ! curl -s "$LLM_BASE_URL/models" > /dev/null 2>&1; then
    echo "WARNING: Cannot connect to LLM server at $LLM_BASE_URL"
    echo "Please start your LLM server (e.g., vLLM) before running this script."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✓ Prerequisites check complete"
echo ""

# Step 2: Preprocess graph into relation matrices
echo "========================================="
echo "Step 2: Preprocessing graph"
echo "========================================="
echo ""

if [ -d "experiment/dataset/$EXPERIMENT_NAME/$MATRIX_DIR" ]; then
    echo "Relation matrices already exist in $MATRIX_DIR"
    read -p "Rebuild matrices? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "experiment/dataset/$EXPERIMENT_NAME/$MATRIX_DIR"
        BUILD_MATRICES=true
    else
        BUILD_MATRICES=false
    fi
else
    BUILD_MATRICES=true
fi

if [ "$BUILD_MATRICES" = true ]; then
    echo "Building relation-type specific matrices..."
    python experiment/online_retrieval/graph_preprocessing.py \
        --experiment-name "$EXPERIMENT_NAME" \
        --workspace-subdir "$WORKSPACE_DIR" \
        --output-subdir "$MATRIX_DIR" \
        --llm-model-name "$LLM_MODEL" \
        --embedding-model-name "$EMBEDDING_MODEL" \
        --llm-base-url "$LLM_BASE_URL"

    echo ""
    echo "✓ Graph preprocessing complete"
else
    echo "✓ Using existing matrices"
fi
echo ""

# Step 3: Run MARA-RAG experiment
echo "========================================="
echo "Step 3: Running MARA-RAG experiment"
echo "========================================="
echo ""

CMD="python experiment/online_retrieval/run_mara_experiment.py \
    --experiment-name $EXPERIMENT_NAME \
    --questions-file $QUESTIONS_FILE \
    --workspace-subdir $WORKSPACE_DIR \
    --matrix-subdir $MATRIX_DIR \
    --output-subdir $RESULTS_DIR \
    --llm-model-name $LLM_MODEL \
    --embedding-model-name $EMBEDDING_MODEL \
    --llm-base-url $LLM_BASE_URL"

if [ -n "$NUM_QUESTIONS" ]; then
    CMD="$CMD --num-questions $NUM_QUESTIONS"
fi

echo "Running experiment with command:"
echo "$CMD"
echo ""

eval $CMD

echo ""
echo "✓ Experiment complete"
echo ""

# Step 4: Display results
echo "========================================="
echo "Step 4: Results Summary"
echo "========================================="
echo ""

METRICS_FILE="experiment/dataset/$EXPERIMENT_NAME/$RESULTS_DIR/metrics.json"
if [ -f "$METRICS_FILE" ]; then
    echo "Aggregate metrics:"
    cat "$METRICS_FILE"
    echo ""
    echo ""
    echo "Results saved to:"
    echo "  - experiment/dataset/$EXPERIMENT_NAME/$RESULTS_DIR/results.jsonl"
    echo "  - experiment/dataset/$EXPERIMENT_NAME/$RESULTS_DIR/metrics.json"
else
    echo "Metrics file not found at $METRICS_FILE"
fi

echo ""
echo "========================================="
echo "MARA-RAG Quick Start Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  - Review results in experiment/dataset/$EXPERIMENT_NAME/$RESULTS_DIR/"
echo "  - Compare with baseline HippoRAG performance"
echo "  - Analyze relation weight patterns for different query types"
echo ""
