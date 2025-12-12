# cs2821r-project

MARA-RAG: Multi-hop Relation-Aware Retrieval for Question Answering

This project implements MARA-RAG, a relation-aware retrieval system that enhances HippoRAG 2 with dynamic graph weighting based on query characteristics.

## Setup

### Environment

```sh
# Create conda environment
mamba create -n hipporag python=3.10
conda activate hipporag
pip install -r requirements.txt
```

### VLLM Server

Start a VLLM server for LLM inference (required for indexing and retrieval):

```sh
salloc --partition gpu_test --gpus 1 --time 02:00:00 --mem=32G --cpus-per-task=4
module load python
source activate hipporag
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 12288 --gpu-memory-utilization 0.94
```

## Experiment Pipeline

The experiment consists of two phases: **offline indexing** (build knowledge graph) and **online retrieval** (query answering).

### Phase 1: Offline Indexing

Build a knowledge graph from corpus documents. Run these steps sequentially:

#### Step 1: Chunking and NER
Chunk documents and extract named entities.

```sh
python experiment/offline_indexing/01_chunk_and_ner.py \
  --experiment-name musique_demo \
  --corpus-file path/to/corpus.json \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model-name Qwen/Qwen3-8B \
  --output-subdir offline_indexing/01_chunk_ner
```

#### Step 2: Triple Extraction
Extract relation triples from chunks.

```sh
python experiment/offline_indexing/02_triple_extraction.py \
  --experiment-name musique_demo \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model-name Qwen/Qwen3-8B \
  --input-subdir offline_indexing/01_chunk_ner \
  --output-subdir offline_indexing/02_triples
```

#### Step 3: Embedding Encoding
Generate embeddings for chunks and entities.

```sh
python experiment/offline_indexing/03_encode_embeddings.py \
  --experiment-name musique_demo \
  --embedding-model-name facebook/contriever-msmarco \
  --chunk-dir offline_indexing/01_chunk_ner \
  --triple-dir offline_indexing/02_triples \
  --output-subdir offline_indexing/03_embeddings
```

#### Step 4: Graph Construction
Build the HippoRAG knowledge graph.

```sh
python experiment/offline_indexing/04_build_graph.py \
  --experiment-name musique_demo \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model-name Qwen/Qwen3-8B \
  --embedding-model-name facebook/contriever-msmarco \
  --chunk-dir offline_indexing/01_chunk_ner \
  --triple-dir offline_indexing/02_triples \
  --workspace-subdir hipporag_workspace
```

#### Step 5: Graph Cleanup (Optional)
Clean and optimize the graph.

```sh
python experiment/offline_indexing/05_cleanup_graph.py \
  --experiment-name musique_demo \
  --workspace-subdir hipporag_workspace
```

### Phase 2: Online Retrieval

#### Step 1: Graph Preprocessing (MARA-RAG)
Build relation-type matrices from the knowledge graph.

```sh
python experiment/online_retrieval/graph_preprocessing.py \
  --experiment-name musique_demo \
  --workspace-subdir hipporag_workspace \
  --output-subdir mara_matrices
```

This creates separate adjacency matrices for each relation type (hierarchical, temporal, spatial, causality, attribution, synonym, passage).

#### Step 2: Run MARA-RAG Experiment
Run end-to-end evaluation with relation-aware retrieval.

```sh
python experiment/online_retrieval/run_mara_experiment.py \
  --experiment-name musique_demo \
  --questions-file questions.jsonl \
  --workspace-subdir hipporag_workspace \
  --matrix-subdir mara_matrices \
  --output-subdir mara_results \
  --llm-base-url http://localhost:8000/v1 \
  --llm-model-name Qwen/Qwen3-8B
```

**MARA-RAG Algorithm:**
1. **Query Routing**: LLM analyzes query and assigns weights to relation types (hierarchical, temporal, spatial, causality, attribution)
2. **Dynamic Graph Weighting**: Constructs query-specific adjacency matrix as weighted combination of relation-type matrices
3. **Relation-Aware PPR**: Runs Personalized PageRank with the dynamic matrix
4. **Answer Generation**: Uses retrieved passages to generate final answer

## Running with SLURM Batch Scripts

Batch scripts are available in `experiment/batch_scripts/` for running on cluster:

```sh
# Offline indexing
sbatch experiment/batch_scripts/step01_chunk_ner.sbatch
sbatch experiment/batch_scripts/step02_triples.sbatch
sbatch experiment/batch_scripts/step03_embeddings.sbatch
sbatch experiment/batch_scripts/step04_graph.sbatch
sbatch experiment/batch_scripts/step05_cleanup_graph.sbatch

# Online retrieval
sbatch experiment/batch_scripts/step06_online_retrieval_exp1.sbatch
```

Configure variables in each script (e.g., `EXPERIMENT_NAME`, `LLM_BASE_URL`, `CORPUS_FILE`).

## Quick Start Example

Complete workflow for a small dataset:

```sh
# 1. Start VLLM server (in separate terminal)
vllm serve Qwen/Qwen3-8B --dtype bfloat16 --max-model-len 12288

# 2. Offline indexing
python experiment/offline_indexing/01_chunk_and_ner.py \
  --experiment-name demo --corpus-file corpus.json \
  --llm-base-url http://localhost:8000/v1

python experiment/offline_indexing/02_triple_extraction.py \
  --experiment-name demo --llm-base-url http://localhost:8000/v1

python experiment/offline_indexing/03_encode_embeddings.py \
  --experiment-name demo --embedding-model-name facebook/contriever-msmarco

python experiment/offline_indexing/04_build_graph.py \
  --experiment-name demo --llm-base-url http://localhost:8000/v1

# 3. Online retrieval
python experiment/online_retrieval/graph_preprocessing.py --experiment-name demo

python experiment/online_retrieval/run_mara_experiment.py \
  --experiment-name demo --questions-file questions.jsonl \
  --llm-base-url http://localhost:8000/v1
```

## Project Structure

```
experiment/
├── offline_indexing/     # Steps 01-05: Build knowledge graph
├── online_retrieval/      # MARA-RAG retrieval and evaluation
├── batch_scripts/         # SLURM scripts for cluster execution
└── common/                # Shared utilities

rag/                       # HippoRAG submodule
```

## Key Differences: MARA-RAG vs HippoRAG

- **HippoRAG**: Fixed adjacency matrix, hardcoded beta (0.05)
- **MARA-RAG**: Dynamic query-specific matrix, adaptive beta, relation-type weighting

See `experiment/online_retrieval/README.md` for detailed MARA-RAG documentation.