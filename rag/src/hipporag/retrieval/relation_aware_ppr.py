"""
Relation-Aware Personalized PageRank implementation.
"""

import logging
import numpy as np
import igraph as ig
from typing import Dict, Tuple, Optional, List
import copy

from ..utils.logging_utils import get_logger
from .graph_engine import build_relation_type_mapping, get_relation_type_for_edge

logger = get_logger(__name__)


def run_relation_aware_ppr(
    graph: ig.Graph,
    reset_prob: np.ndarray,
    relation_influence_factors: Dict[str, Dict[str, float]],
    fact_edge_meta: Dict,
    passage_entity_edges: List[Dict],
    node_name_to_vertex_idx: Dict[str, int],
    entity_node_keys: List[str],
    passage_node_keys: List[str],
    damping: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run Personalized PageRank with relation-aware edge weights.
    
    The transition probabilities are weighted by relation types using the
    relation influence factors (β values) from the manager agent.
    
    Args:
        graph: The igraph graph object
        reset_prob: Reset probability distribution for each node
        relation_influence_factors: Beta values from manager agent
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
        fact_edge_meta: Metadata for entity-entity edges
        passage_entity_edges: List of passage-entity edge metadata
        node_name_to_vertex_idx: Mapping from node name to vertex index
        entity_node_keys: List of entity node keys
        passage_node_keys: List of passage node keys
        damping: Damping factor for PPR (default: 0.5)
        
    Returns:
        Tuple of (sorted_doc_ids, sorted_doc_scores) where:
        - sorted_doc_ids: Array of passage node indices sorted by score
        - sorted_doc_scores: Array of PPR scores for sorted passages
    """
    # Validate inputs
    if damping is None:
        damping = 0.5
    
    reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
    
    # Create a copy of the graph to modify edge weights
    graph_copy = graph.copy()
    
    # Get beta values
    beta_ee = relation_influence_factors.get('entity_entity', {})
    beta_ep = relation_influence_factors.get('entity_passage', {})
    
    # Modify edge weights based on relation types
    # Read relation_type directly from graph edge attributes (same way as in step04)
    modified_weights = []
    for edge in graph_copy.es:
        source_idx = edge.source
        target_idx = edge.target
        
        # Get original weight (same way as in step04)
        original_weight = edge['weight'] if 'weight' in edge.attributes() else 1.0
        
        # Get relation type directly from edge attribute (same way as stored in step04)
        # This matches how edges were added in add_new_edges() with relation_type attribute
        if 'relation_type' in edge.attributes():
            relation_type = edge['relation_type']
        else:
            # Fallback: try to infer from metadata if relation_type attribute is missing
            # This should rarely happen if graph was built correctly in step04
            logger.warning(f"Edge {source_idx}->{target_idx} missing relation_type attribute, using fallback")
            if fact_edge_meta is not None and node_name_to_vertex_idx is not None:
                edge_relation_types = build_relation_type_mapping(
                    graph=graph,
                    fact_edge_meta=fact_edge_meta or {},
                    passage_entity_edges=passage_entity_edges or [],
                    node_name_to_vertex_idx=node_name_to_vertex_idx,
                    entity_node_keys=entity_node_keys or [],
                    passage_node_keys=passage_node_keys or []
                )
                relation_type = get_relation_type_for_edge(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    edge_relation_types=edge_relation_types,
                    default='ATTRIBUTION'
                )
            else:
                relation_type = 'ATTRIBUTION'
        
        # Determine beta value based on relation type
        if relation_type in beta_ee:
            # Entity-entity edge
            beta = beta_ee[relation_type]
        elif relation_type in beta_ep:
            # Entity-passage edge
            beta = beta_ep[relation_type]
        elif relation_type == 'SYNONYMY':
            # SYNONYMY is an entity-entity relation type (synonymy edges)
            # Map to ATTRIBUTION or use average of entity-entity betas
            if 'ATTRIBUTION' in beta_ee:
                beta = beta_ee['ATTRIBUTION']
            else:
                beta = np.mean(list(beta_ee.values())) if beta_ee else 1.0
        else:
            # Unknown relation type - use uniform weight
            logger.warning(f"Unknown relation type: {relation_type}, using default beta")
            if relation_type in ['PRIMARY', 'SECONDARY', 'PERIPHERAL']:
                # Likely entity-passage edge, use average
                beta = np.mean(list(beta_ep.values())) if beta_ep else 1.0
            else:
                # Likely entity-entity edge, use average
                beta = np.mean(list(beta_ee.values())) if beta_ee else 1.0
        
        # Apply beta to original weight
        modified_weight = original_weight * beta
        modified_weights.append(modified_weight)
    
    # Update edge weights in graph copy
    graph_copy.es['weight'] = modified_weights
    
    # Run PPR with modified weights
    try:
        pagerank_scores = graph_copy.personalized_pagerank(
            vertices=range(len(node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
    except Exception as e:
        logger.error(f"Error running relation-aware PPR: {e}")
        # Fallback to original PPR
        logger.warning("Falling back to original PPR without relation-aware weights")
        pagerank_scores = graph.personalized_pagerank(
            vertices=range(len(node_name_to_vertex_idx)),
            damping=damping,
            directed=False,
            weights='weight',
            reset=reset_prob,
            implementation='prpack'
        )
    
    # Extract passage node scores
    passage_node_idxs = [node_name_to_vertex_idx[node_key] for node_key in passage_node_keys]
    doc_scores = np.array([pagerank_scores[idx] for idx in passage_node_idxs])
    
    # Sort by score
    sorted_doc_ids = np.argsort(doc_scores)[::-1]
    sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]
    
    # Verify ALL passages are included (matching baseline HippoRAG)
    assert len(doc_scores) == len(passage_node_keys), \
        f"PPR scored {len(doc_scores)} passages but expected {len(passage_node_keys)} (all passages)"
    assert len(sorted_doc_ids) == len(passage_node_keys), \
        f"PPR returned {len(sorted_doc_ids)} passages but expected {len(passage_node_keys)} (all passages)"
    
    # Log PPR traversal information
    logger.info(f"Relation-aware PPR completed:")
    logger.info(f"  - Total passages scored: {len(doc_scores)} (all passages included)")
    logger.info(f"  - Top 10 passage scores: {sorted_doc_scores[:10]}")
    logger.info(f"  - Score range: [{sorted_doc_scores[-1]:.6f}, {sorted_doc_scores[0]:.6f}]")
    
    # Log seed node information
    seed_node_indices = np.where(reset_prob > 0)[0]
    if len(seed_node_indices) > 0:
        seed_scores = reset_prob[seed_node_indices]
        logger.info(f"  - Seed nodes: {len(seed_node_indices)} nodes with non-zero reset probability")
        logger.info(f"  - Seed score range: [{seed_scores.min():.6f}, {seed_scores.max():.6f}]")
        logger.info(f"  - Top 5 seed node scores: {np.sort(seed_scores)[::-1][:5]}")
    
    # Log relation type distribution
    relation_type_counts = {}
    for edge in graph_copy.es:
        if 'relation_type' in edge.attributes():
            rt = edge['relation_type']
            relation_type_counts[rt] = relation_type_counts.get(rt, 0) + 1
    if relation_type_counts:
        logger.info(f"  - Edge relation types: {dict(sorted(relation_type_counts.items(), key=lambda x: x[1], reverse=True)[:10])}")
    
    logger.debug(f"Relation-aware PPR completed. Top 5 scores: {sorted_doc_scores[:5]}")
    
    return sorted_doc_ids, sorted_doc_scores


def apply_relation_weights_to_graph(
    graph: ig.Graph,
    relation_influence_factors: Dict[str, Dict[str, float]],
    fact_edge_meta: Dict = None,
    passage_entity_edges: List[Dict] = None,
    node_name_to_vertex_idx: Dict[str, int] = None,
    entity_node_keys: List[str] = None,
    passage_node_keys: List[str] = None
) -> ig.Graph:
    """
    Apply relation influence factors to graph edge weights.
    
    This creates a modified copy of the graph with relation-aware weights.
    Reads relation_type directly from graph edge attributes (same way as stored in step04).
    
    Args:
        graph: Original graph
        relation_influence_factors: Beta values from manager agent
        fact_edge_meta: Optional metadata for fallback (if relation_type attribute missing)
        passage_entity_edges: Optional metadata for fallback (if relation_type attribute missing)
        node_name_to_vertex_idx: Optional mapping for fallback
        entity_node_keys: Optional entity keys for fallback
        passage_node_keys: Optional passage keys for fallback
        
    Returns:
        Modified graph copy with relation-aware weights
    """
    graph_copy = graph.copy()
    
    beta_ee = relation_influence_factors.get('entity_entity', {})
    beta_ep = relation_influence_factors.get('entity_passage', {})
    
    modified_weights = []
    for edge in graph_copy.es:
        source_idx = edge.source
        target_idx = edge.target
        
        # Get original weight (same way as in step04)
        original_weight = edge['weight'] if 'weight' in edge.attributes() else 1.0
        
        # Get relation type directly from edge attribute (same way as stored in step04)
        if 'relation_type' in edge.attributes():
            relation_type = edge['relation_type']
        else:
            # Fallback: try to infer from metadata if relation_type attribute is missing
            logger.warning(f"Edge {source_idx}->{target_idx} missing relation_type attribute, using fallback")
            if fact_edge_meta is not None and node_name_to_vertex_idx is not None:
                edge_relation_types = build_relation_type_mapping(
                    graph=graph,
                    fact_edge_meta=fact_edge_meta or {},
                    passage_entity_edges=passage_entity_edges or [],
                    node_name_to_vertex_idx=node_name_to_vertex_idx,
                    entity_node_keys=entity_node_keys or [],
                    passage_node_keys=passage_node_keys or []
                )
                relation_type = get_relation_type_for_edge(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    edge_relation_types=edge_relation_types,
                    default='ATTRIBUTION'
                )
            else:
                relation_type = 'ATTRIBUTION'
        
        if relation_type in beta_ee:
            beta = beta_ee[relation_type]
        elif relation_type in beta_ep:
            beta = beta_ep[relation_type]
        else:
            # Default to average or 1.0
            if relation_type in ['PRIMARY', 'SECONDARY', 'PERIPHERAL']:
                beta = np.mean(list(beta_ep.values())) if beta_ep else 1.0
            else:
                beta = np.mean(list(beta_ee.values())) if beta_ee else 1.0
        
        modified_weights.append(original_weight * beta)
    
    graph_copy.es['weight'] = modified_weights
    
    return graph_copy

