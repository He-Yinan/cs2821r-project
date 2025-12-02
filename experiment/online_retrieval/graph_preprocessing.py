#!/usr/bin/env python3
"""
Graph Preprocessing for MARA-RAG: Relation-Aware Matrix Construction

This script loads a HippoRAG knowledge graph and builds separate adjacency matrices
for each relation type (HIERARCHICAL, TEMPORAL, SPATIAL, CAUSALITY, ATTRIBUTION,
SYNONYMY, and PASSAGE-ENTITY edges). These matrices are used during online retrieval
to dynamically weight different relation types based on query characteristics.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig


class RelationTypeClassifier:
    """
    Classifies relation types into semantic categories.

    The HippoRAG graph already has relation_type attributes on edges:
    - Fact edges: HIERARCHICAL, TEMPORAL, SPATIAL, CAUSALITY, ATTRIBUTION
    - Synonymy edges: SYNONYMY
    - Passage edges: PRIMARY, SECONDARY, PERIPHERAL
    """

    # Map from relation_type values to our semantic categories
    RELATION_TYPE_MAPPING = {
        # Fact edge relation types (from add_fact_edges)
        'HIERARCHICAL': 'hierarchical',
        'TEMPORAL': 'temporal',
        'SPATIAL': 'spatial',
        'CAUSALITY': 'causality',
        'ATTRIBUTION': 'attribution',

        # Synonymy edges
        'SYNONYMY': 'synonym',

        # Passage-entity edges (treat all as passage type)
        'PRIMARY': 'passage',
        'SECONDARY': 'passage',
        'PERIPHERAL': 'passage',
    }

    @classmethod
    def classify_edge(cls, relation_type: str) -> str:
        """
        Classify an edge's relation_type into one of our semantic categories.

        Args:
            relation_type: The relation_type attribute from the graph edge

        Returns:
            One of: 'hierarchical', 'temporal', 'spatial', 'causality',
                   'attribution', 'synonym', 'passage', or 'other'
        """
        if not relation_type:
            return 'other'

        relation_type = str(relation_type).upper()
        return cls.RELATION_TYPE_MAPPING.get(relation_type, 'other')


def build_relation_matrices(hipporag: HippoRAG, output_dir: Path) -> Dict[str, csr_matrix]:
    """
    Build separate sparse adjacency matrices for each relation type.

    Args:
        hipporag: Loaded HippoRAG instance with graph
        output_dir: Directory to save the matrices

    Returns:
        Dictionary mapping relation type to its sparse adjacency matrix
    """

    print("Building relation-type specific matrices from HippoRAG graph...")

    graph = hipporag.graph
    num_nodes = len(graph.vs)

    # Initialize edge lists for each relation type
    edge_lists = defaultdict(list)  # relation_type -> [(source, target, weight), ...]

    # Get all edges with their attributes
    print(f"Processing {len(graph.es)} edges from graph...")

    for edge in tqdm(graph.es, desc="Classifying edges"):
        source = edge.source
        target = edge.target
        weight = edge['weight'] if 'weight' in edge.attributes() else 1.0
        relation_type = edge['relation_type'] if 'relation_type' in edge.attributes() else None

        # Classify the edge
        rel_category = RelationTypeClassifier.classify_edge(relation_type)

        # Add to the appropriate edge list
        edge_lists[rel_category].append((source, target, weight))

    # Build sparse matrices for each relation type
    matrices = {}

    print("\nBuilding sparse matrices for each relation type:")
    for rel_type, edges in edge_lists.items():
        if len(edges) == 0:
            print(f"  - {rel_type}: 0 edges (skipping)")
            continue

        # Extract source, target, and weights
        sources = [e[0] for e in edges]
        targets = [e[1] for e in edges]
        weights = [e[2] for e in edges]

        # Create sparse matrix
        matrix = csr_matrix(
            (weights, (sources, targets)),
            shape=(num_nodes, num_nodes),
            dtype=np.float32
        )

        matrices[rel_type] = matrix
        print(f"  - {rel_type}: {len(edges)} edges, matrix shape {matrix.shape}, nnz={matrix.nnz}")

    # Save matrices
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as .npz files (sparse matrix format)
    for rel_type, matrix in matrices.items():
        matrix_file = output_dir / f"matrix_{rel_type}.npz"
        save_npz(matrix_file, matrix)
        print(f"Saved {rel_type} matrix to {matrix_file}")

    # Save metadata
    metadata = {
        'num_nodes': num_nodes,
        'num_edges': len(graph.es),
        'relation_types': {k: len(v) for k, v in edge_lists.items()},
        'node_name_to_vertex_idx': hipporag.node_name_to_vertex_idx,
        'passage_node_keys': hipporag.passage_node_keys,
        'passage_node_idxs': hipporag.passage_node_idxs,
    }

    metadata_file = output_dir / "graph_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\nSaved metadata to {metadata_file}")

    return matrices


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess HippoRAG graph into relation-type specific matrices for MARA-RAG"
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment folder name (e.g., 'musique_demo')"
    )
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="HippoRAG workspace directory (relative to experiment folder)"
    )
    parser.add_argument(
        "--output-subdir",
        default="mara_matrices",
        help="Output directory for relation matrices (relative to experiment folder)"
    )
    parser.add_argument(
        "--llm-model-name",
        default="Qwen/Qwen3-8B-Instruct",
        help="LLM model name (for config)"
    )
    parser.add_argument(
        "--embedding-model-name",
        default="facebook/contriever-msmarco",
        help="Embedding model name"
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible LLM endpoint"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # Setup paths
    experiment_dir = PROJECT_ROOT / "experiment" / "dataset" / args.experiment_name
    workspace_dir = experiment_dir / args.workspace_subdir
    output_dir = experiment_dir / args.output_subdir

    if not workspace_dir.exists():
        raise FileNotFoundError(
            f"Workspace directory not found: {workspace_dir}\n"
            f"Please run offline indexing (01-05 scripts) first to build the graph."
        )

    print(f"Loading HippoRAG graph from {workspace_dir}...")

    # Load HippoRAG with existing graph
    config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.llm_base_url,
    )

    hipporag = HippoRAG(global_config=config)

    # Load the graph (this should load from the pickle file in workspace_dir)
    if not hasattr(hipporag, 'graph') or hipporag.graph is None:
        hipporag.load_igraph()

    if hipporag.graph is None:
        raise RuntimeError(
            f"Failed to load graph from {workspace_dir}. "
            f"Please ensure the graph has been built using the offline indexing scripts."
        )

    print(f"Graph loaded: {len(hipporag.graph.vs)} nodes, {len(hipporag.graph.es)} edges")

    # Build relation-type specific matrices
    matrices = build_relation_matrices(hipporag, output_dir)

    print(f"\n✓ Successfully preprocessed graph into {len(matrices)} relation-type matrices")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
