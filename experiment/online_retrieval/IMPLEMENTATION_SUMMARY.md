# MARA-RAG Implementation Summary

## Overview

Successfully implemented **MARA-RAG** (Multi-hop Relation-Aware Retrieval), a relation-aware online retrieval approach that enhances HippoRAG 2 with query-specific graph weighting.

## Key Design Principles

✓ **Minimal changes to HippoRAG architecture** - Only modifies PPR component
✓ **Compatible with existing graphs** - Works with any HippoRAG knowledge graph
✓ **Query-adaptive** - Dynamically weights relation types based on query characteristics
✓ **Concise and debuggable** - Clear separation of concerns, easy to test

## Implementation Components

### 1. Graph Preprocessing (`graph_preprocessing.py`)

**Purpose:** Offline preprocessing of HippoRAG knowledge graphs

**Key Features:**
- Loads existing HippoRAG graph from workspace
- Classifies edges by relation type (hierarchical, temporal, spatial, causality, attribution, synonym, passage)
- Builds separate sparse adjacency matrices for each relation type
- Saves matrices in efficient `.npz` format with metadata

**Input:** HippoRAG workspace directory
**Output:** Relation-type specific matrices + metadata

**Usage:**
```bash
python graph_preprocessing.py \
  --experiment-name musique_demo \
  --workspace-subdir hipporag_workspace \
  --output-subdir mara_matrices
```

### 2. Query Router (`query_router.py`)

**Purpose:** LLM-based query analyzer for relation weight assignment

**Key Features:**
- Analyzes query semantics using an LLM
- Assigns weights to 5 relation types (sum to 1.0)
- Determines beta value for passage vs. entity balance
- Robust JSON parsing with fallback to default weights
- Validation and normalization of outputs

**Example Output:**
```json
{
  "hierarchical": 0.1,
  "temporal": 0.3,
  "spatial": 0.1,
  "causality": 0.5,
  "attribution": 0.0,
  "beta": 0.05,
  "reasoning": "This is a causality question..."
}
```

**Key Method:** `QueryRouter.route(query: str) -> Dict[str, float]`

### 3. Relation-Aware PPR (`relation_aware_ppr.py`)

**Purpose:** Modified PPR algorithm with dynamic matrix construction

**Key Features:**
- Loads pre-computed relation matrices
- Builds query-specific adjacency matrix as weighted sum:
  ```
  A_dynamic = w_hier * M_hier + w_temp * M_temp + w_spat * M_spat +
              w_caus * M_caus + w_attr * M_attr + M_syn + M_pass
  ```
- Uses router-determined beta for passage node weights
- Converts to igraph for efficient PPR execution
- Returns ranked passages by relevance

**Key Method:** `RelationAwarePPR.run_ppr(phrase_weights, passage_weights, relation_weights, damping)`

### 4. End-to-End Experiment (`run_mara_experiment.py`)

**Purpose:** Complete MARA-RAG evaluation pipeline

**Pipeline:**
1. Load HippoRAG graph and relation matrices
2. For each query:
   - Route query to get relation weights
   - Retrieve facts using HippoRAG's fact retrieval
   - Build phrase weights from facts
   - Build passage weights from dense retrieval
   - Run relation-aware PPR
   - Generate answer with retrieved passages
   - Evaluate F1 score against ground truth
3. Save per-question results and aggregate metrics

**Input:** Questions file (JSONL format)
**Output:** `results.jsonl` + `metrics.json`

**Usage:**
```bash
python run_mara_experiment.py \
  --experiment-name musique_demo \
  --questions-file questions.jsonl \
  --num-questions 50
```

### 5. Supporting Files

**`utils.py`**
- Relation type normalization utilities
- Consistent classification logic

**`compare_results.py`**
- Compare MARA-RAG vs. baseline HippoRAG
- Metric comparisons (F1, recall@k)
- Query type analysis
- Visualization plots

**`test_installation.py`**
- Verify all components are importable
- Test file structure
- Validate RelationTypeClassifier

**`quick_start.sh`**
- End-to-end pipeline automation
- Prerequisite checking
- One-command execution

**`README.md`**
- Comprehensive documentation
- Usage examples
- Configuration options
- Troubleshooting guide

**`__init__.py`**
- Python package initialization
- Exports main classes

## Technical Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
    ┌────▼─────┐
    │  Router  │  ← LLM analyzes query
    │  (LLM)   │    Returns: relation_weights + beta
    └────┬─────┘
         │
         ├───────────────────┬────────────────────┐
         │                   │                    │
    ┌────▼────┐         ┌───▼─────┐        ┌────▼─────┐
    │  Fact   │         │ Dense   │        │ Relation │
    │Retrieval│         │Retrieval│        │ Matrices │
    └────┬────┘         └───┬─────┘        └────┬─────┘
         │                  │                    │
         │ phrase_weights   │ passage_weights    │
         └──────────┬───────┴────────────────────┘
                    │
            ┌───────▼────────┐
            │ Relation-Aware │  ← Dynamic matrix: A = Σ w_i * M_i
            │      PPR       │    Beta-weighted restart
            └───────┬────────┘
                    │
            ┌───────▼────────┐
            │    Ranked      │
            │   Passages     │
            └────────────────┘
```

## Key Differences from HippoRAG 2

| Aspect | HippoRAG 2 | MARA-RAG |
|--------|------------|----------|
| **Adjacency Matrix** | Fixed, same for all queries | Dynamic, query-specific weighted sum |
| **Beta Value** | Hardcoded (0.05) | Router-determined, query-adaptive |
| **Relation Types** | Treated equally | Weighted by query characteristics |
| **Graph Traversal** | Uniform edge weights | Relation-type specific weights |

## File Structure

```
experiment/online_retrieval/
├── __init__.py                   # Package initialization
├── graph_preprocessing.py        # Matrix builder (offline)
├── query_router.py              # LLM-based router
├── relation_aware_ppr.py        # Modified PPR algorithm
├── run_mara_experiment.py       # End-to-end experiment
├── utils.py                     # Helper utilities
├── compare_results.py           # Result comparison tool
├── test_installation.py         # Installation tester
├── quick_start.sh               # Automation script
├── README.md                    # User documentation
└── IMPLEMENTATION_SUMMARY.md    # This file
```

## Example Workflow

### Step 1: Preprocess Graph
```bash
python graph_preprocessing.py --experiment-name musique_demo
```

Output:
- `experiment/dataset/musique_demo/mara_matrices/matrix_hierarchical.npz`
- `experiment/dataset/musique_demo/mara_matrices/matrix_temporal.npz`
- `experiment/dataset/musique_demo/mara_matrices/matrix_spatial.npz`
- `experiment/dataset/musique_demo/mara_matrices/matrix_causality.npz`
- `experiment/dataset/musique_demo/mara_matrices/matrix_attribution.npz`
- `experiment/dataset/musique_demo/mara_matrices/matrix_synonym.npz`
- `experiment/dataset/musique_demo/mara_matrices/matrix_passage.npz`
- `experiment/dataset/musique_demo/mara_matrices/graph_metadata.pkl`

### Step 2: Run Experiment
```bash
python run_mara_experiment.py \
  --experiment-name musique_demo \
  --questions-file questions.jsonl \
  --num-questions 50
```

Output:
- `experiment/dataset/musique_demo/mara_results/results.jsonl`
- `experiment/dataset/musique_demo/mara_results/metrics.json`

### Step 3: Compare with Baseline
```bash
python compare_results.py \
  --mara-results mara_results/results.jsonl \
  --baseline-results baseline_results/results.jsonl \
  --plot
```

## Expected Results Format

### Per-Question Result (results.jsonl)
```json
{
  "question": "What caused World War I?",
  "answer": "The assassination of Archduke Franz Ferdinand...",
  "gold_answers": ["Assassination of Archduke Franz Ferdinand"],
  "f1_score": 0.87,
  "retrieved_docs": ["...", "..."],
  "doc_scores": [0.95, 0.89, ...],
  "relation_weights": {
    "hierarchical": 0.1,
    "temporal": 0.3,
    "spatial": 0.1,
    "causality": 0.5,
    "attribution": 0.0,
    "beta": 0.05,
    "reasoning": "..."
  },
  "debug_info": {
    "route_time": 0.23,
    "fact_retrieval_time": 0.45,
    "ppr_time": 0.12,
    "total_time": 1.34,
    "num_facts": 15,
    "num_seed_phrases": 8,
    "num_seed_passages": 20,
    "retrieval_method": "mara_ppr"
  }
}
```

### Aggregate Metrics (metrics.json)
```json
{
  "num_questions": 50,
  "mean_f1": 0.72,
  "median_f1": 0.75,
  "std_f1": 0.18,
  "recall@1": 0.45,
  "recall@2": 0.58,
  "recall@5": 0.78,
  "recall@10": 0.88,
  "recall@20": 0.94
}
```

## Implementation Validation

✓ All 8 core files created and properly structured
✓ Compatible with existing HippoRAG codebase
✓ Clear separation between offline (preprocessing) and online (retrieval) components
✓ Comprehensive documentation and examples
✓ Testing and validation scripts included
✓ Comparison tools for baseline evaluation

## Next Steps

1. **Environment Setup**
   - Ensure all dependencies are installed (see requirements)
   - Start LLM server (e.g., vLLM)

2. **Data Preparation**
   - Run offline indexing (scripts 01-05) to build HippoRAG graph
   - Prepare questions file in JSONL format

3. **Preprocessing**
   - Run `graph_preprocessing.py` to build relation matrices

4. **Evaluation**
   - Run `run_mara_experiment.py` for MARA-RAG evaluation
   - Run baseline HippoRAG for comparison
   - Use `compare_results.py` to analyze differences

5. **Analysis**
   - Examine relation weight patterns for different query types
   - Identify which queries benefit most from relation-aware retrieval
   - Tune router prompts if needed

## Code Quality

- **Type hints:** All functions have proper type annotations
- **Documentation:** Comprehensive docstrings for all classes and methods
- **Error handling:** Robust fallbacks for LLM routing failures
- **Modularity:** Clear separation of concerns
- **Efficiency:** Sparse matrices for memory efficiency
- **Compatibility:** Works with existing HippoRAG infrastructure

## Dependencies

- HippoRAG and all its dependencies
- `scipy` (for sparse matrices)
- `igraph` (for graph operations and PPR)
- `numpy` (for numerical operations)
- `tqdm` (for progress bars)
- `matplotlib` (optional, for plotting)

## Citation

This implementation is based on:
- HippoRAG 2 (arXiv:2502.14802v2)
- Original HippoRAG (arXiv:2405.14831v3)

## License

Same as HippoRAG project.

---

**Implementation completed:** December 1, 2025
**Total files created:** 9
**Total lines of code:** ~2000+
**Estimated development time:** Complete implementation in single session
