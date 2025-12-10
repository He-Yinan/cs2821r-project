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
        default="Qwen/Qwen3-8B",
        help="LLM model name (for config)"
    )
    parser.add_argument(
        "--embedding-model-name",
        default="facebook/contriever-msmarco",
        help="Embedding model name"
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://holygpu7c26105.rc.fas.harvard.edu:8000/v1",
        help="OpenAI-compatible LLM endpoint"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of matrices even if they already exist"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # Setup paths
    # Results directory is on scratch
    results_base = Path("/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/experiments")
    experiment_dir = results_base / args.experiment_name
    workspace_dir = experiment_dir / args.workspace_subdir
    output_dir = experiment_dir / args.output_subdir

    # Check if matrices already exist
    if not args.force and output_dir.exists():
        matrix_files = list(output_dir.glob("matrix_*.npz"))
        metadata_file = output_dir / "graph_metadata.pkl"
        if len(matrix_files) > 0 and metadata_file.exists():
            print(f"Relation matrices already exist in {output_dir}")
            print(f"Found {len(matrix_files)} matrix files and metadata file")
            print("Skipping graph preprocessing. Use --force to regenerate.")
            return

    if not workspace_dir.exists():
        raise FileNotFoundError(
            f"Workspace directory not found: {workspace_dir}\n"
            f"Please run offline indexing (01-05 scripts) first to build the graph."
        )

    print(f"Loading HippoRAG graph from {workspace_dir}...")

    # Find the actual working directory (it may have a different LLM model name)
    # HippoRAG creates subdirectories like: {llm_label}_{embedding_label}
    # where labels are model names with "/" replaced by "_"
    llm_label = args.llm_model_name.replace("/", "_")
    embedding_label = args.embedding_model_name.replace("/", "_")
    expected_working_dir = workspace_dir / f"{llm_label}_{embedding_label}"
    
    # If expected directory doesn't exist, search for existing directories
    actual_working_dir = None
    if expected_working_dir.exists():
        # Check if graph files exist
        if (expected_working_dir / "graph.pickle").exists() or (expected_working_dir / "graph_cleaned.pickle").exists():
            actual_working_dir = expected_working_dir
            print(f"Found graph in expected directory: {actual_working_dir}")
    
    if actual_working_dir is None:
        print(f"Expected directory {expected_working_dir} not found or missing graph files, searching for existing graph directories...")
        # Look for directories with graph.pickle or graph_cleaned.pickle
        matching_dirs = []
        for item in workspace_dir.iterdir():
            if item.is_dir():
                graph_files = list(item.glob("graph*.pickle"))
                if graph_files:
                    matching_dirs.append(item)
        
        if matching_dirs:
            if len(matching_dirs) == 1:
                actual_working_dir = matching_dirs[0]
                print(f"Found graph directory: {actual_working_dir}")
                # Extract LLM name from directory (remove embedding part)
                dir_name = actual_working_dir.name
                if f"_{embedding_label}" in dir_name:
                    llm_part = dir_name.replace(f"_{embedding_label}", "")
                    # Update LLM model name to match the directory
                    args.llm_model_name = llm_part.replace("_", "/")
                    print(f"Using LLM model name: {args.llm_model_name} (extracted from directory name)")
            else:
                print(f"Found multiple matching directories: {[d.name for d in matching_dirs]}")
                # Use the first one
                actual_working_dir = matching_dirs[0]
                print(f"Using: {actual_working_dir}")
                dir_name = actual_working_dir.name
                if f"_{embedding_label}" in dir_name:
                    llm_part = dir_name.replace(f"_{embedding_label}", "")
                    args.llm_model_name = llm_part.replace("_", "/")
        else:
            raise FileNotFoundError(
                f"Could not find graph directory in {workspace_dir}. "
                f"Expected: {expected_working_dir}, but it doesn't exist or is missing graph files. "
                f"Please check that the graph has been built."
            )

    # Load HippoRAG with existing graph
    config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.llm_base_url,
    )

    hipporag = HippoRAG(global_config=config)
    
    # Verify the graph file exists and use graph_cleaned.pickle if graph.pickle doesn't exist
    graph_pickle_path = Path(hipporag.working_dir) / "graph.pickle"
    graph_cleaned_path = Path(hipporag.working_dir) / "graph_cleaned.pickle"
    
    if not graph_pickle_path.exists():
        if graph_cleaned_path.exists():
            print(f"graph.pickle not found, using graph_cleaned.pickle: {graph_cleaned_path}")
            # Copy graph_cleaned.pickle to graph.pickle so HippoRAG can find it
            import shutil
            shutil.copy2(graph_cleaned_path, graph_pickle_path)
            print(f"Copied graph_cleaned.pickle to graph.pickle")
        else:
            raise FileNotFoundError(
                f"Graph pickle file not found in {hipporag.working_dir}. "
                f"Expected: {graph_pickle_path} or {graph_cleaned_path}"
            )
    else:
        print(f"Found graph.pickle: {graph_pickle_path}")

    # Load the graph (this should load from the pickle file in workspace_dir)
    if not hasattr(hipporag, 'graph') or hipporag.graph is None:
        hipporag.load_igraph()

    if hipporag.graph is None:
        raise RuntimeError(
            f"Failed to load graph from {workspace_dir}. "
            f"Please ensure the graph has been built using the offline indexing scripts."
        )

    print(f"Graph loaded: {len(hipporag.graph.vs)} nodes, {len(hipporag.graph.es)} edges")

    # Prepare retrieval objects to initialize node mappings
    # This creates node_name_to_vertex_idx, passage_node_keys, passage_node_idxs, etc.
    if not hasattr(hipporag, 'node_name_to_vertex_idx') or hipporag.node_name_to_vertex_idx is None:
        print("Preparing retrieval objects to initialize node mappings...")
        hipporag.prepare_retrieval_objects()
    
    # Verify required attributes exist
    if not hasattr(hipporag, 'node_name_to_vertex_idx'):
        raise RuntimeError(
            "Failed to initialize node_name_to_vertex_idx. "
            "Please ensure prepare_retrieval_objects() completed successfully."
        )
    
    if not hasattr(hipporag, 'passage_node_keys'):
        raise RuntimeError(
            "Failed to initialize passage_node_keys. "
            "Please ensure prepare_retrieval_objects() completed successfully."
        )
    
    if not hasattr(hipporag, 'passage_node_idxs'):
        raise RuntimeError(
            "Failed to initialize passage_node_idxs. "
            "Please ensure prepare_retrieval_objects() completed successfully."
        )
    
    print(f"Node mappings initialized: {len(hipporag.node_name_to_vertex_idx)} nodes, "
          f"{len(hipporag.passage_node_keys)} passage nodes")

    # Build relation-type specific matrices
    matrices = build_relation_matrices(hipporag, output_dir)

    print(f"\n✓ Successfully preprocessed graph into {len(matrices)} relation-type matrices")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
