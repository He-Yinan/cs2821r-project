"""
MARA-RAG: Multi-hop Relation-Aware Retrieval for HippoRAG

This package implements relation-aware online retrieval that enhances
HippoRAG's Personalized PageRank with query-specific graph weighting.

Components:
- graph_preprocessing: Builds relation-type specific adjacency matrices
- query_router: LLM-based query analyzer for relation weight assignment
- relation_aware_ppr: Modified PPR with dynamic graph weighting
- run_mara_experiment: End-to-end experiment runner
"""

import sys
from pathlib import Path

# Add project root to Python path for hipporag imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

__version__ = "1.0.0"
__author__ = "CS2821R Project Team"

from .query_router import QueryRouter
from .relation_aware_ppr import RelationAwarePPR

__all__ = [
    "QueryRouter",
    "RelationAwarePPR",
]
