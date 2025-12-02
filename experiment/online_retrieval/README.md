# MARA-RAG: Multi-hop Relation-Aware Retrieval

This directory contains the implementation of MARA-RAG, a relation-aware online retrieval approach that builds on HippoRAG 2.

## Overview

MARA-RAG enhances HippoRAG's Personalized PageRank (PPR) retrieval by:

1. **Relation-Type Classification**: Categorizing edges in the knowledge graph by semantic relation types (hierarchical, temporal, spatial, causality, attribution)
2. **Query Routing**: Using an LLM to analyze each query and assign weights to different relation types
3. **Dynamic Graph Weighting**: Constructing a query-specific adjacency matrix as a weighted combination of relation-type matrices
4. **Adaptive Beta**: Using query-specific beta values to balance entity-level vs. passage-level retrieval

## Architecture

```
┌─────────────────┐
│   Query Input   │
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         │                  │                  │
    ┌────▼────┐       ┌────▼────┐       ┌────▼────┐
    │ Router  │       │  Fact   │       │  Dense  │
    │ (LLM)   │       │Retrieval│       │Retrieval│
    └────┬────┘       └────┬────┘       └────┬────┘
         │                  │                  │
    Weights: {              │                  │
      hierarchical: 0.2,    │                  │
      temporal: 0.3,        │                  │
      spatial: 0.1,         │                  │
      causality: 0.3,       │                  │
      attribution: 0.1,     │                  │
      beta: 0.05            │                  │
    }                       │                  │
         │                  │                  │
         └──────────┬───────┴──────────────────┘
                    │
            ┌───────▼────────┐
            │ Relation-Aware │
            │      PPR       │
            │  (Dynamic A)   │
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │  Retrieved     │
            │  Passages      │
            └────────────────┘
```

## Files

- **`graph_preprocessing.py`**: Builds relation-type specific adjacency matrices from HippoRAG graph
- **`query_router.py`**: LLM-based query analyzer that assigns relation weights and beta value
- **`relation_aware_ppr.py`**: Modified PPR algorithm with dynamic matrix construction
- **`run_mara_experiment.py`**: End-to-end experiment script for evaluation
- **`utils.py`**: Helper utilities for relation type normalization

## Prerequisites

1. **Offline Indexing Completed**: You must have already built a HippoRAG knowledge graph using the offline indexing scripts (`01_chunk_and_ner.py` through `05_cleanup_graph.py`)

2. **LLM Server Running**: You need an OpenAI-compatible LLM endpoint (e.g., vLLM server) for:
   - Query routing
   - Question answering

3. **Dependencies**: All HippoRAG dependencies plus scipy

## Usage

### Step 1: Preprocess Graph into Relation Matrices

```bash
python experiment/online_retrieval/graph_preprocessing.py \
  --experiment-name musique_demo \
  --workspace-subdir hipporag_workspace \
  --output-subdir mara_matrices
```

This will:
- Load the HippoRAG graph from `experiment/dataset/musique_demo/hipporag_workspace/`
- Classify edges by relation type
- Build separate sparse matrices for each relation type
- Save matrices to `experiment/dataset/musique_demo/mara_matrices/`

**Output files:**
- `matrix_hierarchical.npz`: Edges with hierarchical relations
- `matrix_temporal.npz`: Edges with temporal relations
- `matrix_spatial.npz`: Edges with spatial relations
- `matrix_causality.npz`: Edges with causality relations
- `matrix_attribution.npz`: Edges with attribution relations
- `matrix_synonym.npz`: Synonym edges
- `matrix_passage.npz`: Passage-entity edges
- `graph_metadata.pkl`: Metadata (node mappings, passage indices, etc.)

### Step 2: Run MARA-RAG Experiment

```bash
python experiment/online_retrieval/run_mara_experiment.py \
  --experiment-name musique_demo \
  --questions-file questions.jsonl \
  --workspace-subdir hipporag_workspace \
  --matrix-subdir mara_matrices \
  --output-subdir mara_results \
  --llm-base-url http://localhost:8000/v1 \
  --num-questions 50
```

This will:
- Load the HippoRAG graph and relation matrices
- For each question:
  1. Route query to get relation weights
  2. Retrieve facts using HippoRAG's fact retrieval
  3. Run relation-aware PPR with dynamic matrix
  4. Generate answer using retrieved passages
  5. Evaluate against ground truth
- Save results and metrics

**Output files:**
- `results.jsonl`: Per-question results with retrieved docs, answers, F1 scores
- `metrics.json`: Aggregate metrics (mean/median F1, recall@k)

### Step 3: Analyze Results

```python
import json

# Load results
with open('experiment/dataset/musique_demo/mara_results/results.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# Examine a result
result = results[0]
print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"F1 Score: {result['f1_score']}")
print(f"Relation Weights: {result['relation_weights']}")

# Load aggregate metrics
with open('experiment/dataset/musique_demo/mara_results/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Mean F1: {metrics['mean_f1']}")
print(f"Recall@5: {metrics.get('recall@5', 'N/A')}")
```

## Question File Format

The questions file should be a JSONL file with the following format:

```json
{
  "question": "What is the capital of the country where Albert Einstein was born?",
  "answer": ["Berlin"],
  "context": [
    ["Albert Einstein", ["Albert Einstein was born in Ulm, Germany.", "He later moved to Switzerland."]],
    ["Germany", ["Germany is a country in Europe.", "The capital of Germany is Berlin."]]
  ]
}
```

Fields:
- `question` (required): The question text
- `answer` (optional): List of gold answers for F1 evaluation
- `context` (optional): List of [title, [sentences]] for retrieval recall evaluation
- `supporting_facts` (optional): List of [title, sent_id] for ground truth passages

## Configuration Options

### Graph Preprocessing

- `--experiment-name`: Name of experiment folder under `experiment/dataset/`
- `--workspace-subdir`: HippoRAG workspace directory (default: `hipporag_workspace`)
- `--output-subdir`: Output directory for matrices (default: `mara_matrices`)

### Experiment Script

- `--experiment-name`: Name of experiment folder
- `--questions-file`: Path to questions JSONL (relative to experiment folder or absolute)
- `--workspace-subdir`: HippoRAG workspace directory (default: `hipporag_workspace`)
- `--matrix-subdir`: MARA matrices directory (default: `mara_matrices`)
- `--output-subdir`: Output directory (default: `mara_results`)
- `--llm-model-name`: LLM model name (default: `Qwen/Qwen3-8B-Instruct`)
- `--llm-base-url`: OpenAI-compatible LLM endpoint (default: `http://localhost:8000/v1`)
- `--num-questions`: Number of questions to evaluate (default: all)
- `--verbose`: Print debug information

## Relation Type Categories

MARA-RAG classifies edges into the following categories:

| Category | Description | Example Relations |
|----------|-------------|-------------------|
| **Hierarchical** | Is-a, part-of relationships | "is capital of", "subclass of" |
| **Temporal** | Time-related relationships | "founded in", "occurred during" |
| **Spatial** | Location-based relationships | "located in", "adjacent to" |
| **Causality** | Cause-effect relationships | "causes", "results in", "leads to" |
| **Attribution** | Properties and characteristics | "created by", "painted by", "has property" |
| **Synonym** | Semantic equivalence | Automatically detected synonyms |
| **Passage** | Passage-entity connections | Always included with weight 1.0 |

## How MARA-RAG Differs from HippoRAG

### HippoRAG 2 (Original)
- Uses a **fixed adjacency matrix** for all queries
- Uses a **hardcoded beta value (0.05)** for passage node weight
- Treats all relation types equally during graph traversal

### MARA-RAG (Our Approach)
- Uses a **dynamic adjacency matrix** constructed per query:
  ```
  A_dynamic = w_hier * M_hier + w_temp * M_temp + w_spat * M_spat +
              w_caus * M_caus + w_attr * M_attr + M_syn + M_pass
  ```
- Uses a **query-specific beta value** determined by the router
- Weights different relation types based on query characteristics

### Example

**Query:** "What caused World War I?"

**HippoRAG 2:**
- All edges weighted equally
- Beta = 0.05 (hardcoded)

**MARA-RAG:**
- Causality edges: weight = 0.5 (high)
- Temporal edges: weight = 0.3 (medium)
- Other edges: weight = 0.05 each (low)
- Beta = 0.03 (emphasize graph structure over dense retrieval)

## Implementation Notes

1. **Minimal Changes**: MARA-RAG only modifies the PPR component of HippoRAG. All other components (fact retrieval, recognition memory, QA) remain unchanged.

2. **Compatibility**: The approach works with any existing HippoRAG knowledge graph. No changes to offline indexing are required (though we recommend using the updated `04_build_graph.py` that adds relation_type attributes).

3. **Efficiency**: Relation matrices are pre-computed offline. Online retrieval only involves:
   - One LLM call for routing (~100 tokens)
   - Matrix addition to build dynamic adjacency matrix (fast)
   - Standard PPR execution (same as HippoRAG)

4. **Debugging**: Use `--verbose` flag to see:
   - Relation weights for each query
   - Number of facts retrieved
   - Number of seed nodes
   - PPR execution time

## Troubleshooting

### "Workspace directory not found"
- Run offline indexing scripts first (01-05)
- Check that `--workspace-subdir` matches your HippoRAG workspace

### "Failed to load graph"
- Ensure offline indexing completed successfully
- Check that the graph pickle file exists in workspace

### "Could not find valid JSON in LLM response"
- LLM may not be following the prompt correctly
- Try adjusting temperature or using a different model
- Check LLM server logs for errors

### Low F1 scores
- Check that relation matrices were built correctly
- Verify LLM is generating reasonable relation weights
- Compare with baseline HippoRAG performance

## Citation

If you use MARA-RAG in your research, please cite:

```bibtex
@article{hipporag2,
  title={HippoRAG 2: Advanced Multi-hop Reasoning with Knowledge Graphs},
  author={...},
  journal={arXiv preprint arXiv:2502.14802},
  year={2025}
}
```

## License

This code is released under the same license as HippoRAG.
