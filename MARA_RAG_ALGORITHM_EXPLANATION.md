# MARA-RAG: Multi-Agent Relation-Aware Retrieval Algorithm

## Overview

MARA-RAG (Multi-Agent Relation-Aware RAG) is a retrieval system that uses a heterogeneous knowledge graph with relation-aware Personalized PageRank (PPR) to retrieve relevant passages. The algorithm combines dense retrieval, fact-based entity linking, and graph traversal with dynamic relation weighting.

## Complete Algorithm Flow

### Phase 1: Initialization and Query Encoding

#### Step 1.1: Query Embedding Generation
**Location**: `get_query_embeddings()` (Line 1570)

**Process**:
- Encodes the query using two different instructions:
  1. **Query-to-Fact embedding**: Optimized for matching queries to facts (triples)
  2. **Query-to-Passage embedding**: Optimized for matching queries to passages
- Uses the same embedding model but with different instructions to capture different semantic aspects
- Caches embeddings to avoid recomputation

**Design Rationale**: Different embedding spaces help match queries to different types of nodes (entities vs passages) more effectively.

---

### Phase 2: Fact Retrieval and Filtering

#### Step 2.1: Fact-Query Similarity Scoring
**Location**: `get_fact_scores()` (Line 1606)

**Process**:
1. **Input**: Query string
2. **Computation**:
   - Retrieves query embedding (query-to-fact version)
   - Computes cosine similarity between query embedding and all fact embeddings
   - Facts are stored as tuples: `(subject, predicate, object)`
   - Returns normalized similarity scores for all facts

**Formula**:
```
fact_score[i] = cosine_similarity(query_embedding, fact_embedding[i])
```

**Output**: Array of similarity scores, one per fact in the knowledge base

**Design Rationale**: Dense retrieval finds facts semantically similar to the query, providing initial relevance signals.

---

#### Step 2.2: Fact Reranking (Recognition Memory)
**Location**: `rerank_facts()` (Line 1963)

**Process**:
1. **Candidate Selection**:
   - Selects top `linking_top_k` facts by similarity score
   - These are the initial candidate facts

2. **LLM-Based Filtering**:
   - Uses `DSPyFilter` (rerank.py) to filter facts
   - **Input to LLM**:
     - Query
     - List of candidate facts in JSON format
   - **LLM Task**: Select facts that are most relevant for answering the query
   - **Output**: Filtered list of facts (typically 4 or fewer)

3. **Fact Parsing**:
   - Parses facts from embedding store
   - Handles multiple formats:
     - `|||` delimited: `"subject ||| predicate ||| object"`
     - Python tuple: `('subject', 'predicate', 'object')`
     - JSON format
   - Returns validated fact tuples

**Design Rationale**: 
- Initial dense retrieval may return many facts, some irrelevant
- LLM filtering acts as "recognition memory" - selects facts that require reasoning to connect to the query
- Reduces noise and focuses on high-quality facts

**Example**:
```
Query: "When was the company founded?"
Candidate Facts (top 10 by similarity):
  - ("Company", "founded", "2020")
  - ("Company", "located in", "New York")
  - ("Company", "employs", "1000 people")
  ...
  
After LLM Filtering (top 4):
  - ("Company", "founded", "2020")  ✓ Relevant
  - ("Company", "founded by", "John Doe")  ✓ Relevant
```

---

### Phase 3: Seed Node Selection

#### Step 3.1: Entity Seed Node Weighting
**Location**: `graph_search_with_relation_aware_ppr()` (Lines 1912-1964)

**Process**:

1. **Extract Entities from Facts**:
   - For each filtered fact `(subject, predicate, object)`:
     - Extract `subject` and `object` as entity phrases
     - These become candidate seed nodes

2. **Compute Entity Weights**:
   ```
   For each fact f with score s:
     For each entity e in {subject, object}:
       weighted_score = s / num_chunks_containing_e
       entity_weights[e] += weighted_score
       entity_occurrence_count[e] += 1
   
   Final weight = entity_weights[e] / entity_occurrence_count[e]
   ```
   
   **Normalization**: Divides by occurrence count to average scores when an entity appears in multiple facts
   
   **Inverse Document Frequency (IDF)**: Divides by number of chunks containing the entity to downweight common entities

3. **Top-K Selection**:
   - Selects top `link_top_k` entities by weight
   - Creates `phrase_weights` array (one weight per graph node)
   - Only entities that exist in the graph are included

**Design Rationale**:
- Entities from relevant facts are good starting points for graph traversal
- IDF weighting prevents common entities from dominating
- Averaging prevents entities appearing in many facts from getting excessive weight

**Example**:
```
Facts:
  - ("Nissan", "founded", "1933") with score 0.9
  - ("Nissan", "headquartered", "Japan") with score 0.8
  
Entity Weights:
  - "Nissan": (0.9 + 0.8) / 2 = 0.85
  - "1933": 0.9 / 1 = 0.9
  - "Japan": 0.8 / 1 = 0.8
```

---

#### Step 3.2: Passage Seed Node Weighting
**Location**: `graph_search_with_relation_aware_ppr()` (Lines 1973-1983)

**Process**:

1. **Dense Passage Retrieval (DPR)**:
   - Computes query-to-passage similarity using query-to-passage embeddings
   - Ranks all passages by similarity
   - Normalizes scores to [0, 1] range

2. **Weight Assignment**:
   ```
   For each passage p with DPR score s:
     passage_weight[p] = s * passage_node_weight
   ```
   
   Where `passage_node_weight` is a configurable parameter (default: 0.05)

3. **Add to Linking Score Map**:
   - Adds passage text and weight to `linking_score_map`
   - This allows passages to be seed nodes in PPR

**Design Rationale**:
- Passages directly relevant to the query should also be seed nodes
- Lower weight (0.05) compared to entities prevents passages from dominating
- Combines dense retrieval signal with graph-based traversal

---

### Phase 4: Relation-Aware Weighting

#### Step 4.1: Manager Agent - Relation Influence Factors
**Location**: `manager_agent.get_relation_influence_factors()` (manager_agent.py)

**Process**:

1. **Query Analysis**:
   - Manager Agent (LLM) analyzes the query
   - Determines which relation types are most relevant

2. **Beta Value Generation**:
   - Outputs β values for each relation type category:
   
   **Entity-Entity Relations**:
   - `HIERARCHICAL`: For "is-a", "part-of" relationships
   - `TEMPORAL`: For time-based relationships
   - `SPATIAL`: For location-based relationships
   - `CAUSALITY`: For cause-effect relationships
   - `ATTRIBUTION`: For attribute/property relationships
   
   **Entity-Passage Relations**:
   - `PRIMARY`: Strong passage-entity connection
   - `SECONDARY`: Moderate connection
   - `PERIPHERAL`: Weak connection

3. **Normalization**:
   - Values within each category sum to 1.0
   - Ensures proper probability distribution

**Example**:
```
Query: "When was the company founded?"

Manager Agent Output:
{
  "entity_entity": {
    "TEMPORAL": 0.7,      // High - query is about time
    "HIERARCHICAL": 0.1,
    "SPATIAL": 0.05,
    "CAUSALITY": 0.1,
    "ATTRIBUTION": 0.05
  },
  "entity_passage": {
    "PRIMARY": 0.6,
    "SECONDARY": 0.3,
    "PERIPHERAL": 0.1
  }
}
```

**Design Rationale**: 
- Different queries require different relation types
- Temporal queries should prioritize TEMPORAL edges
- Hierarchical queries should prioritize HIERARCHICAL edges
- Makes traversal query-aware

---

### Phase 5: Graph Traversal with Relation-Aware PPR

#### Step 5.1: Edge Weight Modification
**Location**: `run_relation_aware_ppr()` (relation_aware_ppr.py, Lines 80-142)

**Process**:

1. **For Each Edge in Graph**:
   ```
   original_weight = edge['weight']  // From graph construction
   relation_type = edge['relation_type']  // From graph construction
   
   beta = get_beta_for_relation_type(relation_type)
   modified_weight = original_weight * beta
   ```

2. **Beta Lookup**:
   - Entity-entity edges: Lookup in `beta_ee[relation_type]`
   - Entity-passage edges: Lookup in `beta_ep[relation_type]`
   - SYNONYMY edges: Map to ATTRIBUTION beta
   - Unknown types: Use average beta for category

3. **Weight Update**:
   - Creates a copy of the graph
   - Updates all edge weights with modified values
   - Original graph remains unchanged

**Example**:
```
Edge: ("Nissan", "founded", "1933")
  original_weight = 0.9 (confidence from OpenIE)
  relation_type = "TEMPORAL"
  beta = 0.7 (from Manager Agent)
  modified_weight = 0.9 × 0.7 = 0.63

Edge: ("Nissan", "located in", "Japan")
  original_weight = 0.8
  relation_type = "SPATIAL"
  beta = 0.05 (from Manager Agent)
  modified_weight = 0.8 × 0.05 = 0.04
```

**Design Rationale**:
- Original weights represent general importance/confidence
- Beta values represent query-specific relevance
- Multiplication combines both signals
- Temporal edges get boosted for temporal queries, etc.

---

#### Step 5.2: Personalized PageRank Execution
**Location**: `run_relation_aware_ppr()` (Lines 146-153)

**Key Answer: PPR uses BOTH node weights AND edge weights**

**Node Weights (Reset Probabilities)**:
- **Purpose**: Determine which nodes are "seed nodes" - starting points for the random walk
- **Source**: 
  - `phrase_weights`: Entity nodes from facts (e.g., "Nissan", "1933")
  - `passage_weights`: Passage nodes from DPR
  - Combined: `node_weights = phrase_weights + passage_weights`
- **Usage**: Passed as `reset=reset_prob` parameter
- **Meaning**: Probability of "teleporting" (jumping) to that node during random walk
- **Example**: If "Nissan" has high node weight, the walker frequently teleports back to it

**Edge Weights (Transition Probabilities)**:
- **Purpose**: Determine which edges to follow during random walk
- **Source**: 
  - Original edge weight (from graph construction) × Beta value (from Manager Agent)
  - `modified_weight = original_weight × beta(relation_type)`
- **Usage**: Passed as `weights='weight'` parameter
- **Meaning**: Probability of following that edge when at a node
- **Example**: If TEMPORAL edges have high beta, the walker prefers following temporal relations

**Process**:

1. **Reset Probability (Seed Nodes)**:
   - `reset_prob[i] = node_weights[i]` for each node i
   - Normalized so sum = 1.0 (handled by PPR algorithm)
   - Represents probability of "teleporting" to that node

2. **PPR Computation**:
   ```python
   pagerank_scores = graph.personalized_pagerank(
       weights='weight',           # EDGE weights - which edges to follow
       reset=reset_prob,           # NODE weights - which nodes to teleport to
       damping=0.5,                # Teleportation probability
       directed=False              # Undirected graph
   )
   ```

3. **Algorithm**:
   - Random walk on graph with **modified edge weights** (controls transitions)
   - At each step: 
     - **50% chance**: Follow an edge (probability based on **edge weight**)
     - **50% chance**: Teleport to a seed node (probability based on **node weight**)
   - Damping factor (0.5): 50% chance of following edge, 50% chance of teleporting
   - Converges to stationary distribution representing node importance

**Mathematical Formulation**:
```
PR(v) = (1-d) * reset_prob(v) + d * Σ(PR(u) * w(u,v) / Σw(u,*))
```
Where:
- `PR(v)`: PageRank score of node v
- `d`: damping factor (0.5)
- `reset_prob(v)`: **NODE weight** - Seed node probability (teleportation)
- `w(u,v)`: **EDGE weight** - Modified edge weight from u to v (transition)

**How They Work Together**:
1. **Node weights** determine WHERE the walk starts/restarts (seed nodes)
2. **Edge weights** determine HOW the walk traverses the graph (which relations to follow)
3. High node weight on "Nissan" → walker frequently returns to "Nissan"
4. High edge weight on TEMPORAL edges → walker prefers temporal relations
5. Result: Nodes reachable from "Nissan" via TEMPORAL edges get high scores

**Design Rationale**:
- PPR naturally combines seed node importance (node weights) with graph structure (edge weights)
- Modified edge weights bias traversal toward relevant relation types
- Higher scores indicate nodes reachable via relevant relations from seed nodes

---

#### Step 5.3: Passage Score Extraction
**Location**: `run_relation_aware_ppr()` (Lines 168-177)

**Process**:

1. **Extract Passage Scores**:
   - For each passage node, extract its PPR score
   - These represent how "reachable" the passage is from seed nodes via relevant relations

2. **Sorting**:
   - Sort passages by PPR score (descending)
   - Return top passages

**Output**: 
- `sorted_doc_ids`: Indices of passages sorted by relevance
- `sorted_doc_scores`: PPR scores for each passage

**Design Rationale**:
- Passages with high PPR scores are:
  1. Connected to relevant seed entities
  2. Reachable via relation types important for the query
  3. Well-connected in the graph structure

---

## Complete Example Walkthrough

**Query**: "When was Nissan founded?"

### Step 1: Query Encoding
- Query-to-fact embedding: `[0.1, 0.3, ..., 0.5]`
- Query-to-passage embedding: `[0.2, 0.4, ..., 0.6]`

### Step 2: Fact Retrieval
- **Initial Facts** (top 10 by similarity):
  1. ("Nissan", "founded", "1933") - score: 0.95
  2. ("Nissan", "headquartered", "Japan") - score: 0.85
  3. ("Nissan", "manufactures", "cars") - score: 0.75
  ...

- **After LLM Filtering** (top 4):
  1. ("Nissan", "founded", "1933") ✓
  2. ("Nissan", "founded by", "Yoshisuke Aikawa") ✓
  3. ("Nissan", "established", "1933") ✓
  4. ("Nissan Motor Co.", "founded", "1933") ✓

### Step 3: Seed Node Selection
- **Entity Seeds**:
  - "Nissan": weight = 0.9 (from multiple facts)
  - "1933": weight = 0.95 (from founding fact)
  - "Yoshisuke Aikawa": weight = 0.85
  
- **Passage Seeds**:
  - Passage about Nissan history: DPR score = 0.9 → weight = 0.9 × 0.05 = 0.045

### Step 4: Manager Agent
- **Beta Values**:
  - TEMPORAL: 0.7 (high - query is about time)
  - HIERARCHICAL: 0.1
  - SPATIAL: 0.05
  - CAUSALITY: 0.1
  - ATTRIBUTION: 0.05

### Step 5: Graph Traversal
- **Edge Weight Modification**:
  - Edge ("Nissan", "founded", "1933") with TEMPORAL relation:
    - Original: 0.9
    - Modified: 0.9 × 0.7 = 0.63 (boosted)
  
  - Edge ("Nissan", "located in", "Japan") with SPATIAL relation:
    - Original: 0.8
    - Modified: 0.8 × 0.05 = 0.04 (downweighted)

- **PPR Execution**:
  - Starts from seed nodes: "Nissan", "1933", etc.
  - Traverses graph with modified weights
  - Temporal edges are preferred (higher weight)
  - Passages connected via temporal relations get higher scores

- **Final Results**:
  - Top passage: "Nissan was founded in 1933 by Yoshisuke Aikawa..." (score: 0.85)
  - Second: "Nissan Motor Company history..." (score: 0.72)
  - ...

---

## Key Design Decisions

### 1. **Two-Stage Fact Retrieval**
- **Dense Retrieval**: Fast, finds many candidates
- **LLM Filtering**: Slow but accurate, selects best facts
- **Rationale**: Combines speed of dense retrieval with accuracy of LLM reasoning

### 2. **Hybrid Seed Selection**
- **Entity Seeds**: From facts (semantic relevance)
- **Passage Seeds**: From DPR (direct relevance)
- **Rationale**: Multiple signals improve coverage

### 3. **Relation-Aware Weighting**
- **Original Weights**: General importance/confidence
- **Beta Values**: Query-specific relevance
- **Multiplication**: Combines both signals
- **Rationale**: Makes traversal query-aware while preserving graph structure

### 4. **IDF Normalization for Entities**
- Divides entity weight by number of chunks containing it
- **Rationale**: Prevents common entities from dominating

### 5. **Averaging Entity Scores**
- Divides by occurrence count when entity appears in multiple facts
- **Rationale**: Prevents entities in many facts from getting excessive weight

---

## Comparison with Vanilla HippoRAG

| Aspect | Vanilla HippoRAG | MARA-RAG |
|--------|------------------|----------|
| **Edge Weights** | Fixed (from graph construction) | Dynamic (original × beta) |
| **Relation Types** | Ignored | Used to weight edges |
| **Query Awareness** | Limited (only seed nodes) | High (seed nodes + edge weights) |
| **Traversal** | Uniform relation preference | Query-specific relation preference |

---

## Summary

MARA-RAG retrieval is a multi-stage process:

1. **Encode** query in multiple embedding spaces
2. **Retrieve** facts via dense similarity + LLM filtering
3. **Select** seed nodes (entities from facts + passages from DPR)
4. **Weight** edges dynamically based on query-relevant relation types
5. **Traverse** graph with Personalized PageRank using modified weights
6. **Return** top passages ranked by PPR scores

The key innovation is **relation-aware weighting**: edges are dynamically weighted based on how relevant their relation type is for answering the query, making graph traversal query-aware rather than static.


