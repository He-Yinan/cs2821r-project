#!/usr/bin/env python3
"""
Relation-Aware Personalized PageRank for MARA-RAG

This module implements a modified PPR algorithm that dynamically weights different
relation types based on query characteristics. Instead of using a fixed adjacency
matrix, it constructs a query-specific matrix as a weighted combination of
relation-type specific matrices.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Tuple

import igraph as ig
import numpy as np
from scipy.sparse import csr_matrix, load_npz


class RelationAwarePPR:
    """
    Personalized PageRank with dynamic relation-type weighting.

    This class loads pre-computed relation-type specific adjacency matrices
    and executes PPR with a query-specific weighted combination of these matrices.
    """

    def __init__(self, matrix_dir: Path):
        """
        Initialize the relation-aware PPR engine.

        Args:
            matrix_dir: Directory containing the relation-type matrices
                       (output from graph_preprocessing.py)
        """
        self.matrix_dir = Path(matrix_dir)

        # Load metadata
        metadata_file = self.matrix_dir / "graph_metadata.pkl"
        with open(metadata_file, 'rb') as f:
            self.metadata = pickle.load(f)

        self.num_nodes = self.metadata['num_nodes']
        self.passage_node_idxs = self.metadata['passage_node_idxs']
        self.passage_node_keys = self.metadata['passage_node_keys']

        # Load relation-type specific matrices
        self.matrices = self._load_matrices()

        print(f"Loaded {len(self.matrices)} relation-type matrices with {self.num_nodes} nodes")

    def _load_matrices(self) -> Dict[str, csr_matrix]:
        """
        Load all relation-type specific adjacency matrices.

        Returns:
            Dictionary mapping relation type to its sparse matrix
        """
        matrices = {}

        # Look for all matrix files
        for matrix_file in self.matrix_dir.glob("matrix_*.npz"):
            # Extract relation type from filename (e.g., "matrix_hierarchical.npz" -> "hierarchical")
            rel_type = matrix_file.stem.replace("matrix_", "")

            # Load sparse matrix
            matrix = load_npz(matrix_file)
            matrices[rel_type] = matrix

            print(f"  Loaded {rel_type}: {matrix.shape}, nnz={matrix.nnz}")

        return matrices

    def build_dynamic_adjacency_matrix(self, weights: Dict[str, float]) -> csr_matrix:
        """
        Build a query-specific adjacency matrix as a weighted sum of relation matrices.

        The dynamic matrix is constructed as:
            A_dynamic = w_hier * M_hier + w_temp * M_temp + w_spat * M_spat +
                       w_caus * M_caus + w_attr * M_attr + M_syn + M_pass

        Where:
        - w_* are the relation weights from the router
        - M_syn (synonym) and M_pass (passage) are always included with weight 1.0

        Args:
            weights: Dictionary with keys 'hierarchical', 'temporal', 'spatial',
                    'causality', 'attribution' (from router)

        Returns:
            Dynamic adjacency matrix (sparse CSR format)
        """

        # Start with zero matrix
        dynamic_matrix = csr_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)

        # Add weighted relation-type matrices
        for rel_type in ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']:
            weight = weights.get(rel_type, 0.0)

            if rel_type in self.matrices and weight > 0:
                dynamic_matrix = dynamic_matrix + weight * self.matrices[rel_type]

        # Always include synonym and passage edges (with weight 1.0)
        if 'synonym' in self.matrices:
            dynamic_matrix = dynamic_matrix + self.matrices['synonym']

        if 'passage' in self.matrices:
            dynamic_matrix = dynamic_matrix + self.matrices['passage']

        # Handle 'other' edges (if any)
        if 'other' in self.matrices:
            # Use average of all relation weights for 'other' edges
            avg_weight = np.mean([weights.get(k, 0.0) for k in
                                 ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']])
            if avg_weight > 0:
                dynamic_matrix = dynamic_matrix + avg_weight * self.matrices['other']

        return dynamic_matrix.tocsr()

    def build_restart_vector(self,
                            phrase_weights: np.ndarray,
                            passage_weights: np.ndarray,
                            beta: float) -> np.ndarray:
        """
        Build the restart probability vector for PPR.

        This follows HippoRAG 2's approach but uses a dynamic beta value:
            restart[i] = phrase_weights[i] + beta * passage_weights[i]

        Args:
            phrase_weights: Weights for phrase/entity nodes (from fact retrieval)
            passage_weights: Weights for passage nodes (from dense retrieval)
            beta: Passage weight multiplier from router (replaces HippoRAG's hardcoded 0.05)

        Returns:
            Normalized restart probability vector
        """

        # Combine phrase and passage weights
        restart = phrase_weights + beta * passage_weights

        # Normalize to probability distribution
        restart_sum = restart.sum()
        if restart_sum > 0:
            restart = restart / restart_sum
        else:
            # Fallback: uniform distribution over all nodes
            restart = np.ones(self.num_nodes) / self.num_nodes

        return restart

    def run_ppr(self,
               phrase_weights: np.ndarray,
               passage_weights: np.ndarray,
               relation_weights: Dict[str, float],
               damping: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run relation-aware Personalized PageRank.

        Args:
            phrase_weights: Node weights for phrase/entity nodes (shape: [num_nodes])
            passage_weights: Node weights for passage nodes (shape: [num_nodes])
            relation_weights: Dictionary with relation type weights and beta value
            damping: PPR damping factor (default: 0.5)

        Returns:
            Tuple of (sorted_doc_ids, sorted_doc_scores):
                - sorted_doc_ids: Passage indices sorted by relevance (descending)
                - sorted_doc_scores: Corresponding PPR scores
        """

        # Extract beta from relation_weights
        beta = relation_weights.get('beta', 0.05)

        # Build dynamic adjacency matrix
        dynamic_matrix = self.build_dynamic_adjacency_matrix(relation_weights)

        # Build restart vector
        restart_vector = self.build_restart_vector(phrase_weights, passage_weights, beta)

        # Convert sparse matrix to igraph
        # Note: igraph's PPR expects an igraph object, so we need to convert
        graph = self._sparse_matrix_to_igraph(dynamic_matrix)

        # Run PPR using igraph
        pagerank_scores = graph.personalized_pagerank(
            vertices=range(self.num_nodes),
            damping=damping,
            directed=False,
            weights='weight',
            reset=restart_vector,
            implementation='prpack'
        )

        # Extract scores for passage nodes only
        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])

        # Sort by score (descending)
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]

        return sorted_doc_ids, sorted_doc_scores

    def _sparse_matrix_to_igraph(self, matrix: csr_matrix) -> ig.Graph:
        """
        Convert a sparse adjacency matrix to an igraph Graph object.

        Args:
            matrix: Sparse adjacency matrix

        Returns:
            igraph Graph with edge weights
        """

        # Convert to COO format for easier edge extraction
        coo = matrix.tocoo()

        # Create edge list
        edges = list(zip(coo.row, coo.col))
        weights = coo.data.tolist()

        # Create igraph
        graph = ig.Graph(n=self.num_nodes, edges=edges, directed=False)
        graph.es['weight'] = weights

        return graph


def test_relation_aware_ppr():
    """
    Test the RelationAwarePPR with example data.
    """
    print("RelationAwarePPR test example:")
    print("\nUsage:")
    print("  ppr = RelationAwarePPR(matrix_dir='path/to/mara_matrices')")
    print("  phrase_weights = np.array([...])  # Node weights from fact retrieval")
    print("  passage_weights = np.array([...])  # Node weights from dense retrieval")
    print("  relation_weights = {")
    print("      'hierarchical': 0.1,")
    print("      'temporal': 0.3,")
    print("      'spatial': 0.2,")
    print("      'causality': 0.3,")
    print("      'attribution': 0.1,")
    print("      'beta': 0.05")
    print("  }")
    print("  sorted_doc_ids, scores = ppr.run_ppr(phrase_weights, passage_weights, relation_weights)")
    print("\nNote: Requires preprocessed matrices from graph_preprocessing.py")


if __name__ == "__main__":
    test_relation_aware_ppr()
