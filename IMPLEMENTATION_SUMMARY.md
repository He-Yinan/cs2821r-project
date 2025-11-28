# Relation-Aware Multi-Agent Retrieval - Implementation Summary

## ✅ Implementation Complete

All core components for the relation-aware multi-agent retrieval system have been successfully implemented.

## 📁 Files Created

### 1. Manager Agent
- **`rag/src/hipporag/agents/manager_agent.py`**
  - Generates relation influence factors (β values) from queries
  - Uses LLM to analyze query and determine relevance of each relation type
  - Validates and normalizes β values to sum to 1.0

### 2. Prompt Template
- **`rag/src/hipporag/prompts/templates/relation_influence.py`**
  - Prompt template for manager agent
  - Includes few-shot examples for different query types
  - Structured JSON output format

### 3. Graph Engine
- **`rag/src/hipporag/retrieval/graph_engine.py`**
  - Extracts relation types from graph edges
  - Maps entity-entity edges to relation types (HIERARCHICAL, TEMPORAL, SPATIAL, CAUSALITY, ATTRIBUTION)
  - Maps entity-passage edges to relation types (PRIMARY, SECONDARY, PERIPHERAL)
  - Handles synonymy edges with default relation types

### 4. Relation-Aware PPR
- **`rag/src/hipporag/retrieval/relation_aware_ppr.py`**
  - Implements relation-aware Personalized PageRank
  - Applies β values to edge weights: `new_weight = original_weight × β[relation_type]`
  - Runs PPR with modified edge weights

### 5. Integration
- **`rag/src/hipporag/HippoRAG.py`** (modified)
  - Added `graph_search_with_relation_aware_ppr()` method
  - Integrated manager agent initialization
  - Modified `retrieve()` to use relation-aware retrieval when enabled

### 6. Configuration
- **`rag/src/hipporag/utils/config_utils.py`** (modified)
  - Added `use_relation_aware_retrieval: bool` flag
  - Added `manager_agent_temperature: float` parameter

## 🔧 How to Use

### Enable Relation-Aware Retrieval

```python
from hipporag import HippoRAG
from hipporag.utils.config_utils import BaseConfig

# Create config with relation-aware retrieval enabled
config = BaseConfig()
config.use_relation_aware_retrieval = True
config.manager_agent_temperature = 0.3

# Initialize HippoRAG
hipporag = HippoRAG(global_config=config)

# Index documents (same as before)
hipporag.index(docs)

# Retrieve with relation-aware PPR
results = hipporag.retrieve(queries=["When was Radio City founded?"])
```

### How It Works

1. **Query Analysis**: Manager agent analyzes the query and generates β values for each relation type
2. **Seed Node Initialization**: Same as original - phrase nodes from facts + passage nodes from DPR
3. **Relation-Aware PPR**: 
   - Edge weights are modified: `weight_new = weight_original × β[relation_type]`
   - PPR runs with modified weights
   - Results are ranked by relation-aware scores

## 📊 Relation Types

### Entity-Entity Relations
- **HIERARCHICAL**: is_a, part_of, subclass_of, membership, family relations
- **TEMPORAL**: dates, time-based events, founded_in, started_on
- **SPATIAL**: location, geographic relations, located_in, adjacent_to
- **CAUSALITY**: causes, results_in, leads_to, enables, created_by
- **ATTRIBUTION**: properties, attributes, roles, painted_by, owned_by

### Entity-Passage Relations
- **PRIMARY**: Core entities central to passage's main topic
- **SECONDARY**: Supporting entities providing context
- **PERIPHERAL**: Minor entities mentioned in passing

## 🔄 Backward Compatibility

- Original `graph_search_with_fact_entities()` method remains unchanged
- Relation-aware retrieval is **opt-in** via `use_relation_aware_retrieval` flag
- Default behavior is unchanged (uses original retrieval)

## 🧪 Testing

To test the implementation:

```python
# Test manager agent
from hipporag.agents.manager_agent import ManagerAgent
from hipporag import HippoRAG

hipporag = HippoRAG()
hipporag.global_config.use_relation_aware_retrieval = True
hipporag.manager_agent = ManagerAgent(
    llm=hipporag.llm_model,
    prompt_template_manager=hipporag.prompt_template_manager
)

beta_values = hipporag.manager_agent.get_relation_influence_factors(
    "When was Radio City founded?"
)
print(beta_values)
```

## 📝 Next Steps

1. **Experiment Runner**: Create scripts to run experiments comparing relation-aware vs. standard retrieval
2. **Evaluation**: Implement evaluation metrics and comparison scripts
3. **Online-to-Offline Feedback**: Implement feedback loop to adjust offline weights based on online performance
4. **Performance Optimization**: Cache relation type mappings, optimize edge weight computation

## 🐛 Known Limitations

1. **LLM Dependency**: Manager agent requires LLM calls, which adds latency
2. **Graph Copy Overhead**: Relation-aware PPR creates a copy of the graph (temporary)
3. **Missing Relation Types**: Edges without relation type metadata default to ATTRIBUTION or PERIPHERAL

## 📚 References

- Original HippoRAG2 paper: `2502.14802v2.pdf`
- Implementation plan: `IMPLEMENTATION_PLAN.md`
- ToDos: `ToDos.pdf`


