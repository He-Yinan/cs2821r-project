#!/usr/bin/env python3
"""
Utility functions for MARA-RAG.
"""

def normalize_relation_type(relation_type: str) -> str:
    """
    Normalize a relation type string to one of the standard categories.

    This function should match the normalization used in HippoRAG's
    add_fact_edges method to ensure consistency.

    Args:
        relation_type: Raw relation type string

    Returns:
        Normalized relation type (uppercase): HIERARCHICAL, TEMPORAL,
        SPATIAL, CAUSALITY, ATTRIBUTION, or default ATTRIBUTION
    """

    if not relation_type:
        return "ATTRIBUTION"

    rel_lower = str(relation_type).lower()

    # Hierarchical: is-a, part-of relationships
    hierarchical_keywords = [
        'is_a', 'is a', 'isa', 'is-a',
        'part_of', 'part of', 'partof', 'part-of',
        'subclass', 'superclass', 'instance of',
        'type of', 'kind of',
        'capital of', 'is capital', 'capital',
        'member of', 'belongs to',
        'category', 'subcategory'
    ]

    # Temporal: time-related relationships
    temporal_keywords = [
        'occurred', 'happen', 'took place',
        'founded', 'established', 'created',
        'built', 'constructed',
        'born', 'died', 'lived',
        'started', 'began', 'ended', 'finished',
        'during', 'before', 'after',
        'when', 'date', 'year', 'time',
        'in year', 'in century'
    ]

    # Spatial: location-based relationships
    spatial_keywords = [
        'located', 'location', 'place',
        'in country', 'in city', 'in state',
        'adjacent', 'near', 'next to',
        'contain', 'inside', 'within',
        'border', 'surround',
        'where', 'position',
        'north of', 'south of', 'east of', 'west of'
    ]

    # Causality: cause-effect relationships
    causality_keywords = [
        'cause', 'result', 'effect',
        'lead to', 'led to', 'leads to',
        'trigger', 'enable', 'allow',
        'because', 'due to', 'owing to',
        'consequence', 'impact',
        'why', 'reason',
        'produce', 'create', 'generate'
    ]

    # Attribution: properties and characteristics
    attribution_keywords = [
        'property', 'attribute', 'characteristic',
        'feature', 'quality', 'trait',
        'has', 'have', 'possess',
        'made by', 'created by', 'authored by',
        'written by', 'painted by', 'composed by',
        'designed by', 'built by',
        'owned by', 'belong',
        'color', 'size', 'shape', 'material',
        'title', 'name', 'description'
    ]

    # Check each category
    for kw in hierarchical_keywords:
        if kw in rel_lower:
            return "HIERARCHICAL"

    for kw in temporal_keywords:
        if kw in rel_lower:
            return "TEMPORAL"

    for kw in spatial_keywords:
        if kw in rel_lower:
            return "SPATIAL"

    for kw in causality_keywords:
        if kw in rel_lower:
            return "CAUSALITY"

    for kw in attribution_keywords:
        if kw in rel_lower:
            return "ATTRIBUTION"

    # Default to ATTRIBUTION if no match
    return "ATTRIBUTION"
