"""
Graph Engine for extracting relation metadata and building relation type mappings.
"""

import logging
from typing import Dict, Tuple, List, Optional, Set
import numpy as np
import igraph as ig

from ..utils.misc_utils import compute_mdhash_id
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_edge_relation_types(
    graph: ig.Graph,
    fact_edge_meta: Dict,
    passage_entity_edges: List[Dict],
    node_name_to_vertex_idx: Dict[str, int]
) -> Dict[Tuple[int, int], str]:
    """
    Extract relation types for all edges in the graph.
    
    Args:
        graph: The igraph graph object
        fact_edge_meta: Metadata for entity-entity edges from fact_edge_meta
        passage_entity_edges: List of passage-entity edge metadata
        node_name_to_vertex_idx: Mapping from node name to vertex index
        
    Returns:
        Dictionary mapping (source_vertex_idx, target_vertex_idx) -> relation_type
        For entity-entity edges: HIERARCHICAL, TEMPORAL, SPATIAL, CAUSALITY, ATTRIBUTION
        For entity-passage edges: PRIMARY, SECONDARY, PERIPHERAL
        For synonymy edges: defaults to ATTRIBUTION
    """
    edge_relation_types = {}
    
    # Process entity-entity edges from fact_edge_meta
    for (node_key_1, node_key_2), meta in fact_edge_meta.items():
        if node_key_1 in node_name_to_vertex_idx and node_key_2 in node_name_to_vertex_idx:
            v1_idx = node_name_to_vertex_idx[node_key_1]
            v2_idx = node_name_to_vertex_idx[node_key_2]
            
            # Get the most common relation type from triples
            relation_types = []
            for triple_info in meta.get('triples', []):
                rtype = triple_info.get('relation_type', 'ATTRIBUTION')
                relation_types.append(rtype)
            
            # Use most common relation type, or default to ATTRIBUTION
            if relation_types:
                from collections import Counter
                most_common = Counter(relation_types).most_common(1)[0][0]
                edge_relation_types[(v1_idx, v2_idx)] = most_common
                edge_relation_types[(v2_idx, v1_idx)] = most_common  # Undirected graph
            else:
                edge_relation_types[(v1_idx, v2_idx)] = 'ATTRIBUTION'
                edge_relation_types[(v2_idx, v1_idx)] = 'ATTRIBUTION'
    
    # Process entity-passage edges from passage_entity_edges
    for edge_info in passage_entity_edges:
        passage_id = edge_info.get('passage_id')
        entity_id = edge_info.get('entity_id')
        relation_type = edge_info.get('relation_type', 'PERIPHERAL')
        
        if passage_id in node_name_to_vertex_idx and entity_id in node_name_to_vertex_idx:
            passage_idx = node_name_to_vertex_idx[passage_id]
            entity_idx = node_name_to_vertex_idx[entity_id]
            
            edge_relation_types[(passage_idx, entity_idx)] = relation_type
            edge_relation_types[(entity_idx, passage_idx)] = relation_type  # Undirected graph
    
    return edge_relation_types


def build_relation_type_mapping(
    graph: ig.Graph,
    fact_edge_meta: Dict,
    passage_entity_edges: List[Dict],
    node_name_to_vertex_idx: Dict[str, int],
    entity_node_keys: List[str],
    passage_node_keys: List[str]
) -> Dict[Tuple[int, int], str]:
    """
    Build a comprehensive mapping of edge (vertex_idx, vertex_idx) to relation type.
    
    This function identifies all edges and their relation types:
    - Entity-Entity edges: from fact_edge_meta
    - Entity-Passage edges: from passage_entity_edges
    - Synonymy edges: default to ATTRIBUTION
    
    Args:
        graph: The igraph graph object
        fact_edge_meta: Metadata for entity-entity edges
        passage_entity_edges: List of passage-entity edge metadata
        node_name_to_vertex_idx: Mapping from node name to vertex index
        entity_node_keys: List of entity node keys
        passage_node_keys: List of passage node keys
        
    Returns:
        Dictionary mapping (source_vertex_idx, target_vertex_idx) -> relation_type
    """
    edge_relation_types = get_edge_relation_types(
        graph=graph,
        fact_edge_meta=fact_edge_meta,
        passage_entity_edges=passage_entity_edges,
        node_name_to_vertex_idx=node_name_to_vertex_idx
    )
    
    # For edges not in our metadata, check if they're synonymy edges
    # Synonymy edges are entity-entity edges not in fact_edge_meta
    entity_node_set = set(entity_node_keys)
    passage_node_set = set(passage_node_keys)
    
    # Get all edges from graph
    for edge in graph.es:
        source_idx = edge.source
        target_idx = edge.target
        
        edge_key = (source_idx, target_idx)
        reverse_key = (target_idx, source_idx)
        
        # Skip if already processed
        if edge_key in edge_relation_types or reverse_key in edge_relation_types:
            continue
        
        # Get node names
        source_name = graph.vs[source_idx]['name']
        target_name = graph.vs[target_idx]['name']
        
        # Check if this is a synonymy edge (entity-entity edge not in fact_edge_meta)
        if source_name in entity_node_set and target_name in entity_node_set:
            # Synonymy edge - default to ATTRIBUTION
            edge_relation_types[edge_key] = 'ATTRIBUTION'
            edge_relation_types[reverse_key] = 'ATTRIBUTION'
        elif (source_name in entity_node_set and target_name in passage_node_set) or \
             (source_name in passage_node_set and target_name in entity_node_set):
            # Entity-passage edge not in passage_entity_edges - default to PERIPHERAL
            edge_relation_types[edge_key] = 'PERIPHERAL'
            edge_relation_types[reverse_key] = 'PERIPHERAL'
        else:
            # Unknown edge type - default to ATTRIBUTION
            edge_relation_types[edge_key] = 'ATTRIBUTION'
            edge_relation_types[reverse_key] = 'ATTRIBUTION'
    
    logger.debug(f"Built relation type mapping for {len(edge_relation_types)} edges")
    
    return edge_relation_types


def get_relation_type_for_edge(
    source_idx: int,
    target_idx: int,
    edge_relation_types: Dict[Tuple[int, int], str],
    default: str = 'ATTRIBUTION'
) -> str:
    """
    Get relation type for a specific edge.
    
    Args:
        source_idx: Source vertex index
        target_idx: Target vertex index
        edge_relation_types: Mapping from (source, target) to relation type
        default: Default relation type if not found
        
    Returns:
        Relation type string
    """
    edge_key = (source_idx, target_idx)
    reverse_key = (target_idx, source_idx)
    
    if edge_key in edge_relation_types:
        return edge_relation_types[edge_key]
    elif reverse_key in edge_relation_types:
        return edge_relation_types[reverse_key]
    else:
        return default


