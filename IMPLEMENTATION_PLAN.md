# Multi-Agent Relation-Aware Retrieval Implementation Plan

## 1. Current System Understanding

### 1.1 Current Retrieval Flow
The current HippoRAG2 retrieval process follows this flow:

1. **`retrieve()`** - Main entry point
   - Calls `get_fact_scores()` to compute fact-query similarity
   - Calls `rerank_facts()` to filter/rerank facts using DSPy filter
   - Calls `graph_search_with_fact_entities()` or `dense_passage_retrieval()`

2. **`graph_search_with_fact_entities()`** - Core graph-based retrieval
   - Assigns phrase weights based on selected facts
   - Gets passage scores via DPR
   - Combines phrase and passage weights into `node_weights`
   - Calls `run_ppr()` with uniform edge weights

3. **`run_ppr()`** - Personalized PageRank execution
   - Uses igraph's `personalized_pagerank()` with uniform `weights='weight'`
   - No relation-type awareness in transition probabilities

### 1.2 What's Already Implemented (Offline Indexing)

**Entity-Entity Edges:**
- Stored in `self.fact_edge_meta` with structure:
  ```python
  {
    (node_key, node_2_key): {
      "triples": [
        {
          "chunk_id": str,
          "subject": str,
          "predicate": str,
          "object": str,
          "relation_type": str,  # HIERARCHICAL, TEMPORAL, SPATIAL, CAUSALITY, ATTRIBUTION
          "confidence": float
        }
      ]
    }
  }
  ```
- Edge weights in `self.node_to_node_stats` are confidence-weighted

**Entity-Passage Edges:**
- Stored in `self.passage_entity_edges` with structure:
  ```python
  [
    {
      "passage_id": str,
      "entity_id": str,
      "weight": float,
      "relation_type": str,  # PRIMARY, SECONDARY, PERIPHERAL
      "components": {
        "role_score": float,
        "similarity_score": float,
        "position_score": float
      }
    }
  ]
  ```
- Edge weights in `self.node_to_node_stats` are composite weights

## 2. Proposed Multi-Agent Relation-Aware Retrieval

### 2.1 Architecture Overview

```
Query → Manager Agent → Relation Influence Factors (β)
                      ↓
         Seed Node Initialization (phrase + passage)
                      ↓
         Relation-Aware Weighted PPR
                      ↓
         Ranked Documents
```

### 2.2 Key Components

#### 2.2.1 Manager Agent (Relation Influence Factor Generator)
- **Input**: Query string
- **Output**: Relation influence factors (β) for each relation type
  - Entity-Entity: `β_HIERARCHICAL`, `β_TEMPORAL`, `β_SPATIAL`, `β_CAUSALITY`, `β_ATTRIBUTION`
  - Entity-Passage: `β_PRIMARY`, `β_SECONDARY`, `β_PERIPHERAL`
  - Constraint: `Σ β = 1.0`

#### 2.2.2 Seed Node Initialization
- Keep current logic: phrase nodes from facts + passage nodes from DPR
- Weighted balancing of these nodes (same as current)

#### 2.2.3 Relation-Aware Weighted PPR
- Modify transition matrix to weight edges by relation types
- Edge weight = original_weight × β[relation_type]
- Run PPR with modified edge weights

## 3. Implementation Plan

### 3.1 File Structure

```
rag/src/hipporag/
├── agents/
│   ├── __init__.py
│   ├── manager_agent.py          # Manager agent for relation influence factors
│   └── base_agent.py              # Base agent interface (optional)
├── retrieval/
│   ├── __init__.py
│   ├── relation_aware_ppr.py     # Relation-aware PPR implementation
│   └── graph_engine.py            # Graph utilities for relation-aware traversal
└── prompts/
    └── templates/
        └── relation_influence.py # Prompt template for manager agent

experiment/
├── retrieval/
│   ├── __init__.py
│   ├── experiment_runner.py      # Main experiment runner
│   └── evaluation.py             # Evaluation metrics
└── batch_scripts/
    └── run_relation_aware_retrieval.sbatch
```

### 3.2 Implementation Steps

#### Step 1: Manager Agent Implementation
**File**: `rag/src/hipporag/agents/manager_agent.py`

**Responsibilities:**
- Take query as input
- Use LLM to generate relation influence factors
- Parse and validate output (ensure Σ β = 1.0)
- Return structured dict of β values

**Key Methods:**
```python
class ManagerAgent:
    def __init__(self, hipporag: HippoRAG):
        self.llm = hipporag.llm_model
        self.prompt_manager = hipporag.prompt_template_manager
    
    def get_relation_influence_factors(self, query: str) -> Dict[str, float]:
        """
        Returns:
            {
                "entity_entity": {
                    "HIERARCHICAL": float,
                    "TEMPORAL": float,
                    "SPATIAL": float,
                    "CAUSALITY": float,
                    "ATTRIBUTION": float
                },
                "entity_passage": {
                    "PRIMARY": float,
                    "SECONDARY": float,
                    "PERIPHERAL": float
                }
            }
        """
```

#### Step 2: Relation-Aware PPR Implementation
**File**: `rag/src/hipporag/retrieval/relation_aware_ppr.py`

**Responsibilities:**
- Build relation-type-aware transition matrix
- Apply β weights to edge transitions
- Execute PPR with modified weights

**Key Methods:**
```python
def run_relation_aware_ppr(
    graph: ig.Graph,
    reset_prob: np.ndarray,
    relation_influence_factors: Dict[str, Dict[str, float]],
    fact_edge_meta: Dict,
    passage_entity_edges: List[Dict],
    node_name_to_vertex_idx: Dict[str, int],
    damping: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run PPR with relation-aware edge weights.
    
    Steps:
    1. Get edge metadata (relation types) from graph
    2. Create modified edge weights: original_weight × β[relation_type]
    3. Run PPR with modified weights
    """
```

#### Step 3: Graph Engine for Relation Metadata
**File**: `rag/src/hipporag/retrieval/graph_engine.py`

**Responsibilities:**
- Extract relation types from graph edges
- Map edges to their relation types
- Provide utilities for relation-aware graph operations

**Key Methods:**
```python
def get_edge_relation_types(
    graph: ig.Graph,
    fact_edge_meta: Dict,
    passage_entity_edges: List[Dict],
    node_name_to_vertex_idx: Dict[str, int]
) -> Dict[Tuple[int, int], str]:
    """
    Returns mapping: (source_vertex_idx, target_vertex_idx) -> relation_type
    """
```

#### Step 4: Integration into HippoRAG
**File**: `rag/src/hipporag/HippoRAG.py`

**Modifications:**
1. Add `ManagerAgent` instance in `__init__`
2. Create new method `graph_search_with_relation_aware_ppr()` that:
   - Calls manager agent for β values
   - Uses relation-aware PPR instead of standard PPR
3. Optionally add flag in config to enable/disable relation-aware retrieval

**New Method:**
```python
def graph_search_with_relation_aware_ppr(
    self,
    query: str,
    link_top_k: int,
    query_fact_scores: np.ndarray,
    top_k_facts: List[Tuple],
    top_k_fact_indices: List[str],
    passage_node_weight: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Relation-aware version of graph_search_with_fact_entities.
    """
    # 1. Get relation influence factors from manager agent
    # 2. Initialize seed nodes (same as current)
    # 3. Run relation-aware PPR
    # 4. Return ranked documents
```

#### Step 5: Prompt Template for Manager Agent
**File**: `rag/src/hipporag/prompts/templates/relation_influence.py`

**Template Structure:**
- System prompt explaining the task
- Few-shot examples
- Output format: JSON with relation influence factors

#### Step 6: Configuration Updates
**File**: `rag/src/hipporag/utils/config_utils.py`

**New Config Fields:**
```python
use_relation_aware_retrieval: bool = field(
    default=False,
    metadata={"help": "Enable relation-aware multi-agent retrieval"}
)
manager_agent_temperature: float = field(
    default=0.3,
    metadata={"help": "Temperature for manager agent LLM calls"}
)
```

### 3.3 Implementation Details

#### 3.3.1 Relation Influence Factor Generation

**Prompt Design:**
- Input: Query + context about relation types
- Output: JSON with normalized β values
- Validation: Ensure Σ β = 1.0, all β >= 0

**Example Output:**
```json
{
  "entity_entity": {
    "HIERARCHICAL": 0.2,
    "TEMPORAL": 0.1,
    "SPATIAL": 0.1,
    "CAUSALITY": 0.3,
    "ATTRIBUTION": 0.3
  },
  "entity_passage": {
    "PRIMARY": 0.6,
    "SECONDARY": 0.3,
    "PERIPHERAL": 0.1
  }
}
```

#### 3.3.2 Relation-Aware Edge Weighting

**For Entity-Entity Edges:**
- Look up relation_type from `fact_edge_meta`
- Apply: `new_weight = original_weight × β[relation_type]`

**For Entity-Passage Edges:**
- Look up relation_type from `passage_entity_edges`
- Apply: `new_weight = original_weight × β[relation_type]`

**For Synonymy Edges:**
- Default to ATTRIBUTION or use uniform weight

#### 3.3.3 PPR Execution

**Approach 1: Modify Graph Edge Weights Temporarily**
- Create temporary copy of graph
- Modify edge weights
- Run PPR
- Discard temporary graph

**Approach 2: Custom PPR with Weight Function**
- Use igraph's PPR with custom weight function
- Weight function looks up relation type and applies β

**Recommended: Approach 1** (simpler, more explicit)

### 3.4 Backward Compatibility

- Keep existing `graph_search_with_fact_entities()` unchanged
- Add new `graph_search_with_relation_aware_ppr()` method
- Add config flag to choose between methods
- Default to original method for backward compatibility

## 4. Testing Strategy

### 4.1 Unit Tests
- Manager agent output parsing and validation
- Relation-aware PPR edge weight computation
- Graph engine relation type extraction

### 4.2 Integration Tests
- End-to-end retrieval with relation-aware PPR
- Comparison with baseline retrieval
- Performance benchmarks

## 5. Next Steps (After Core Implementation)

### 5.1 Online-to-Offline Feedback Loop
- Track query performance metrics
- Adjust offline edge weights based on online retrieval results
- Implement feedback mechanism

### 5.2 Experiment Runner
- Scripts for running experiments
- Evaluation metrics
- Comparison with baseline

## 6. Implementation Order

1. ✅ **Step 1**: Manager Agent (relation influence factors)
2. ✅ **Step 2**: Graph Engine (relation metadata extraction)
3. ✅ **Step 3**: Relation-Aware PPR (core algorithm)
4. ✅ **Step 4**: Integration into HippoRAG
5. ✅ **Step 5**: Prompt Template
6. ✅ **Step 6**: Configuration
7. ⏭️ **Step 7**: Testing
8. ⏭️ **Step 8**: Experiment Runner

## 7. Key Design Decisions

1. **Separate Manager Agent Class**: Keeps concerns separated, easier to test
2. **Temporary Graph Modification**: Simpler than custom PPR implementation
3. **Backward Compatibility**: Keep original method, add new one
4. **LLM-based β Generation**: Flexible, can be improved with fine-tuning
5. **Structured Output**: JSON format for reliable parsing

## 8. Potential Challenges & Solutions

1. **Challenge**: LLM output may not sum to 1.0
   - **Solution**: Normalize β values after parsing

2. **Challenge**: Performance overhead of relation-aware PPR
   - **Solution**: Cache relation type mappings, optimize edge weight computation

3. **Challenge**: Missing relation types in graph
   - **Solution**: Default to uniform distribution or ATTRIBUTION

4. **Challenge**: Graph modification overhead
   - **Solution**: Only modify weights, not graph structure; use temporary copy



