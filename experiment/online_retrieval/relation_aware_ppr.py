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

    def _normalize_rows(self, matrix: csr_matrix) -> csr_matrix:
        """
        Normalize each row of the adjacency matrix so outgoing edges sum to 1.
        
        This makes the matrix a proper stochastic transition matrix for PPR.
        Nodes with no outgoing edges (dangling nodes) get uniform self-loops.
        
        Args:
            matrix: Sparse adjacency matrix (CSR format)
            
        Returns:
            Row-normalized sparse matrix (CSR format)
        """
        # Compute row sums
        row_sums = np.array(matrix.sum(axis=1)).flatten()
        
        # Handle zero-sum rows (dangling nodes) - add self-loops with weight 1
        dangling_mask = row_sums == 0
        if np.any(dangling_mask):
            # Add self-loops for dangling nodes
            dangling_indices = np.where(dangling_mask)[0]
            # Create sparse matrix with self-loops only for dangling nodes
            from scipy.sparse import csr_matrix as csr
            # Create (row, col, data) for self-loops
            self_loop_rows = dangling_indices
            self_loop_cols = dangling_indices
            self_loop_data = np.ones(len(dangling_indices), dtype=np.float32)
            # Create sparse matrix with same shape as original
            self_loops = csr(
                (self_loop_data, (self_loop_rows, self_loop_cols)),
                shape=matrix.shape,
                dtype=np.float32
            )
            matrix = matrix + self_loops
            # Update row sums (now all should be >= 1)
            row_sums = np.array(matrix.sum(axis=1)).flatten()
        
        # Normalize: divide each row by its sum
        # Avoid division by zero (shouldn't happen after handling dangling nodes)
        row_sums_inv = 1.0 / np.maximum(row_sums, 1e-10)
        
        # Create diagonal matrix with inverse row sums
        from scipy.sparse import diags
        diag_matrix = diags(row_sums_inv, format='csr')
        
        # Multiply to normalize rows: normalized = diag(row_sums_inv) @ matrix
        normalized_matrix = diag_matrix @ matrix
        
        return normalized_matrix.tocsr()

    def build_dynamic_adjacency_matrix(self, weights: Dict[str, float], verbose: bool = False) -> csr_matrix:
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

        if verbose:
            print(f"  Building dynamic matrix from relation weights:")
            print(f"    Relation weights: {weights}")

        # Add weighted relation-type matrices
        for rel_type in ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']:
            weight = weights.get(rel_type, 0.0)

            if rel_type in self.matrices and weight > 0:
                dynamic_matrix = dynamic_matrix + weight * self.matrices[rel_type]
                if verbose:
                    print(f"    Added {rel_type} edges: {self.matrices[rel_type].nnz} edges (weight: {weight:.3f})")

        # Include synonym and passage edges with reduced weight (0.1) to emphasize relation weights
        # This prevents synonym/passage edges from dominating the graph structure
        synonym_weight = 0.1
        passage_weight = 0.1
        
        if 'synonym' in self.matrices:
            dynamic_matrix = dynamic_matrix + synonym_weight * self.matrices['synonym']
            if verbose:
                print(f"  Added synonym edges: {self.matrices['synonym'].nnz} edges (weight: {synonym_weight})")

        if 'passage' in self.matrices:
            dynamic_matrix = dynamic_matrix + passage_weight * self.matrices['passage']
            if verbose:
                print(f"  Added passage edges: {self.matrices['passage'].nnz} edges (weight: {passage_weight})")

        # Handle 'other' edges (if any)
        if 'other' in self.matrices:
            # Use average of all relation weights for 'other' edges
            avg_weight = np.mean([weights.get(k, 0.0) for k in
                                 ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']])
            if avg_weight > 0:
                dynamic_matrix = dynamic_matrix + avg_weight * self.matrices['other']

        # CRITICAL: Normalize rows so each node's outgoing edges sum to 1
        # This makes it a proper stochastic transition matrix for PPR
        dynamic_matrix = self._normalize_rows(dynamic_matrix)

        return dynamic_matrix.tocsr()

    def build_restart_vector(self,
                            phrase_weights: np.ndarray,
                            passage_weights: np.ndarray,
                            beta: float,
                            verbose: bool = False) -> np.ndarray:
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
        # IMPORTANT: Prioritize query-similar nodes by giving more weight to high-similarity passages
        # Original HippoRAG used beta=0.05, but we boost top passages for better convergence
        # 
        # Note: phrase_weights are already weighted by fact scores and document frequency
        # passage_weights are normalized DPR scores (0-1 range), already sorted by query similarity
        # We scale passage_weights by beta, but also boost top passages more
        # 
        # Boost top passages (most query-similar) with additional weight
        # This ensures query-similar passages get higher restart probability
        passage_weights_boosted = passage_weights.copy()
        if passage_weights_boosted.sum() > 0:
            # Find top passages (already sorted by similarity in passage_weights)
            top_passage_indices = np.argsort(passage_weights_boosted)[::-1][:min(10, (passage_weights_boosted > 0).sum())]
            if len(top_passage_indices) > 0:
                # Boost top passages by 2x to emphasize query similarity
                passage_weights_boosted[top_passage_indices] *= 2.0
        
        restart = phrase_weights + beta * passage_weights_boosted
        
        # Debug: Check if phrase weights are too small
        phrase_sum = phrase_weights.sum()
        passage_sum = (beta * passage_weights_boosted).sum()
        if verbose and phrase_sum > 0:
            print(f"  Restart vector composition:")
            print(f"    - Phrase weight sum: {phrase_sum:.6f} ({phrase_sum/(phrase_sum+passage_sum)*100:.1f}%)")
            print(f"    - Passage weight sum: {passage_sum:.6f} ({passage_sum/(phrase_sum+passage_sum)*100:.1f}%)")
            print(f"    - Ratio phrase/passage: {phrase_sum/passage_sum:.3f}" if passage_sum > 0 else "    - No passage weights")

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
               damping: float = 0.15,  # Reduced from 0.5 to 0.15 for better convergence
               verbose: bool = False,
               max_iterations: int = 50,
               tolerance: float = 1e-6,
               return_iterations: bool = False):
        """
        Run relation-aware Personalized PageRank with iterative updates.

        Args:
            phrase_weights: Node weights for phrase/entity nodes (shape: [num_nodes])
            passage_weights: Node weights for passage nodes (shape: [num_nodes])
            relation_weights: Dictionary with relation type weights and beta value
            damping: PPR damping factor (default: 0.5, corresponds to α in the equation)
            verbose: Whether to print detailed progress logs
            max_iterations: Maximum number of PPR iterations
            tolerance: Convergence tolerance

        Returns:
            Tuple of (sorted_doc_ids, sorted_doc_scores):
                - sorted_doc_ids: Passage indices sorted by relevance (descending)
                - sorted_doc_scores: Corresponding PPR scores
        """

        # Extract beta from relation_weights
        beta = relation_weights.get('beta', 0.05)
        alpha = damping  # α = damping factor (teleportation probability)

        if verbose:
            print(f"\n{'='*80}")
            print(f"=== Relation-Aware PPR Algorithm ===")
            print(f"{'='*80}")
            print(f"Parameters:")
            print(f"  - Damping factor (α): {alpha}")
            print(f"  - Beta (passage weight): {beta}")
            print(f"  - Max iterations: {max_iterations}")
            print(f"  - Convergence tolerance: {tolerance}")
            print(f"  - Total nodes: {self.num_nodes}")
            print(f"  - Passage nodes: {len(self.passage_node_idxs)}")

        # Build dynamic adjacency matrix
        if verbose:
            print(f"\n[Step 1] Building dynamic adjacency matrix...")
        dynamic_matrix = self.build_dynamic_adjacency_matrix(relation_weights, verbose=verbose)
        
        if verbose:
            # Verify normalization
            row_sums = np.array(dynamic_matrix.sum(axis=1)).flatten()
            print(f"  Matrix shape: {dynamic_matrix.shape}")
            print(f"  Non-zero entries: {dynamic_matrix.nnz}")
            print(f"  Row sum statistics:")
            print(f"    - Min: {row_sums.min():.6f}, Max: {row_sums.max():.6f}, Mean: {row_sums.mean():.6f}")
            print(f"    - Rows not summing to 1.0: {(np.abs(row_sums - 1.0) > 1e-5).sum()}")
            if (np.abs(row_sums - 1.0) > 1e-5).sum() > 0:
                print(f"    ⚠️  WARNING: Some rows don't sum to 1.0! Matrix may not be properly normalized.")

        # Build restart vector
        if verbose:
            print(f"\n[Step 2] Building restart vector (p_reset)...")
        restart_vector = self.build_restart_vector(phrase_weights, passage_weights, beta, verbose=verbose)
        
        if verbose:
            phrase_support = (phrase_weights > 0).sum()
            passage_support = (passage_weights > 0).sum()
            restart_support = (restart_vector > 0).sum()
            print(f"  Phrase nodes with non-zero weight: {phrase_support}")
            print(f"  Passage nodes with non-zero weight: {passage_support}")
            print(f"  Total nodes in restart vector: {restart_support}")
            print(f"  Restart vector sum: {restart_vector.sum():.6f} (should be 1.0)")
            # Show top seed nodes
            top_restart_indices = np.argsort(restart_vector)[::-1][:10]
            print(f"  Top 10 seed nodes (by restart probability):")
            for i, idx in enumerate(top_restart_indices, 1):
                node_type = "passage" if idx in self.passage_node_idxs else "entity"
                print(f"    {i:2d}. Node {idx:5d} ({node_type}): {restart_vector[idx]:.6f}")

        # Initialize PPR vector: v^(0) = p_reset
        # Ensure restart vector sums to 1.0
        restart_sum = restart_vector.sum()
        if abs(restart_sum - 1.0) > 1e-6:
            if verbose:
                print(f"  ⚠️  WARNING: Restart vector sum = {restart_sum:.6f}, normalizing to 1.0")
            restart_vector = restart_vector / restart_sum if restart_sum > 0 else np.ones(self.num_nodes) / self.num_nodes
        
        v = restart_vector.copy()
        
        # Verify initial sum
        if verbose:
            initial_sum = v.sum()
            if abs(initial_sum - 1.0) > 1e-6:
                print(f"  ⚠️  WARNING: Initial v sum = {initial_sum:.6f}, normalizing to 1.0")
                v = v / initial_sum if initial_sum > 0 else restart_vector.copy()
        
        # Store iterations for visualization
        ppr_iterations = [v.copy()] if (verbose or return_iterations) else []
        
        if verbose:
            print(f"\n[Step 3] Running PPR iterations...")
            print(f"  Initial state: {np.sum(v > 0)} nodes with non-zero probability")
            print(f"  Initial probability sum: {v.sum():.6f} (should be 1.0)")
        
        # Iterative PPR: v^(t+1) = (1-α) M_dynamic v^(t) + α p_reset
        for iteration in range(max_iterations):
            v_old = v.copy()
            
            # v^(t+1) = (1-α) M_dynamic v^(t) + α p_reset
            # First term: propagate through graph
            # If matrix is row-normalized and v_old sums to 1.0, v_propagated should also sum to 1.0
            v_propagated = dynamic_matrix @ v_old
            
            # Verify propagated sum (should equal v_old.sum() if matrix is properly normalized)
            propagated_sum = v_propagated.sum()
            if verbose and iteration < 3:
                print(f"    Propagated sum check: {propagated_sum:.6f} (should equal v_old.sum() = {v_old.sum():.6f})")
            
            # Second term: restart with probability α
            v = (1 - alpha) * v_propagated + alpha * restart_vector
            
            # CRITICAL: Normalize to ensure probability conservation
            # Due to numerical errors, the sum might drift slightly from 1.0
            prob_sum = v.sum()
            if abs(prob_sum - 1.0) > 1e-6:
                # Renormalize to fix numerical errors
                v = v / prob_sum if prob_sum > 0 else restart_vector.copy()
                if verbose:
                    print(f"  ⚠️  WARNING: Probability sum = {prob_sum:.6f}, renormalized to {v.sum():.6f}")
            
            # Store iteration for visualization
            if verbose or return_iterations:
                ppr_iterations.append(v.copy())
            
            # Check convergence
            diff = np.linalg.norm(v - v_old)
            
            # Limit verbose output to max 10 iterations
            # If more than 10 iterations, log every Nth iteration to show max 10
            max_logged_iterations = 10
            if iteration < max_logged_iterations:
                should_log = True
            else:
                # Calculate step size to show approximately max_logged_iterations
                log_step = max(1, iteration // max_logged_iterations)
                should_log = (iteration % log_step == 0) or (diff < tolerance)
            
            if verbose and should_log:
                top_nodes = np.argsort(v)[::-1][:5]
                top_probs = v[top_nodes]
                prob_sum = v.sum()
                print(f"  Iteration {iteration:3d}: diff={diff:.2e}, prob_sum={prob_sum:.6f}, top nodes: {top_nodes.tolist()}")
                print(f"    Top node probabilities: {top_probs.tolist()}")
                if iteration < 3:
                    # Show propagation details for first few iterations
                    top_propagated = np.argsort(v_propagated)[::-1][:3]
                    print(f"    Top propagated nodes: {top_propagated.tolist()}")
                    print(f"    Top restart nodes: {np.argsort(restart_vector)[::-1][:3].tolist()}")
                    print(f"    Propagated sum: {v_propagated.sum():.6f}, Restart sum: {restart_vector.sum():.6f}")
            
            if diff < tolerance:
                if verbose:
                    print(f"  ✓ Converged after {iteration + 1} iterations (diff={diff:.2e} < {tolerance})")
                break
        
        if verbose:
            if iteration == max_iterations - 1:
                print(f"  ⚠️  Reached max iterations ({max_iterations}) without convergence (final diff={diff:.2e})")
            print(f"  Final state: {np.sum(v > 0)} nodes with non-zero probability")

        # Extract scores for passage nodes only
        doc_scores = np.array([v[idx] for idx in self.passage_node_idxs])
        
        # Normalize passage scores so they sum to 1.0 (probability distribution over passages only)
        # This makes the scores more interpretable - they represent the probability of each passage
        # being relevant, given that we're only considering passages
        passage_score_sum = doc_scores.sum()
        if passage_score_sum > 0:
            doc_scores = doc_scores / passage_score_sum
        else:
            # Fallback: uniform distribution if all scores are zero
            doc_scores = np.ones(len(doc_scores)) / len(doc_scores)
            if verbose:
                print(f"  ⚠️  WARNING: All passage scores are zero, using uniform distribution")

        # Sort by score (descending)
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids]
        
        # Limit to top 50 passages (as requested)
        # HippoRAG typically uses top-k passages by PPR score
        max_retrieved = 50
        if len(sorted_doc_ids) > max_retrieved:
            sorted_doc_ids = sorted_doc_ids[:max_retrieved]
            sorted_doc_scores = sorted_doc_scores[:max_retrieved]

        if verbose:
            print(f"\n[Step 4] Final retrieval results:")
            print(f"  Total passages retrieved: {len(sorted_doc_ids)}")
            print(f"  Score statistics (normalized over {len(self.passage_node_idxs)} passages):")
            print(f"    - Max score: {sorted_doc_scores.max():.6f}")
            print(f"    - Min score: {sorted_doc_scores.min():.6f}")
            print(f"    - Mean score: {sorted_doc_scores.mean():.6f}")
            print(f"    - Median score: {np.median(sorted_doc_scores):.6f}")
            print(f"    - Score sum: {sorted_doc_scores.sum():.6f} (should be 1.0)")
            print(f"  Note: Scores are normalized to sum to 1.0 across passage nodes only,")
            print(f"        making them interpretable as probabilities for passage relevance.")
            print(f"  Top 20 passages (by PPR score):")
            for i, (doc_idx, score) in enumerate(zip(sorted_doc_ids[:20], sorted_doc_scores[:20]), 1):
                passage_key = self.passage_node_keys[doc_idx] if doc_idx < len(self.passage_node_keys) else f"idx_{doc_idx}"
                print(f"    {i:2d}. Passage {doc_idx:5d} (key: {passage_key[:50]}...): score={score:.6f} ({score*100:.2f}%)")
            print(f"{'='*80}\n")

        # Return iterations if requested (for visualization)
        if return_iterations:
            return sorted_doc_ids, sorted_doc_scores, ppr_iterations
        else:
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
