#!/usr/bin/env python3
"""
MARA-RAG End-to-End Experiment Script

This script runs the complete MARA-RAG (Multi-hop Relation-Aware Retrieval)
pipeline on a dataset with pre-built knowledge graphs.

Pipeline:
1. Load HippoRAG graph and relation matrices
2. For each query:
   a. Route query to get relation weights and beta
   b. Retrieve facts using HippoRAG's fact retrieval
   c. Run relation-aware PPR with dynamic matrix
   d. Generate answer using retrieved passages
   e. Evaluate against ground truth
3. Save results and compute metrics
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from hipporag.evaluation.qa_eval import QAF1Score, QAExactMatch

# Import our MARA-RAG components (use relative imports)
sys.path.insert(0, str(Path(__file__).parent))
from query_router import QueryRouter
from relation_aware_ppr import RelationAwarePPR


class MARAExperiment:
    """
    Main experiment runner for MARA-RAG.
    """

    def __init__(self,
                 hipporag: HippoRAG,
                 router: QueryRouter,
                 ppr_engine: RelationAwarePPR,
                 config: BaseConfig,
                 qa_top_k: int = 5):
        """
        Initialize MARA experiment.

        Args:
            hipporag: HippoRAG instance with loaded graph
            router: QueryRouter for relation weight assignment
            qa_top_k: Number of top passages to use for QA (default: 5)
            ppr_engine: RelationAwarePPR for dynamic PPR
            config: HippoRAG configuration
        """
        self.hipporag = hipporag
        self.router = router
        self.ppr_engine = ppr_engine
        self.config = config
        self.qa_top_k = qa_top_k

        # Prepare HippoRAG for retrieval
        if not self.hipporag.ready_to_retrieve:
            self.hipporag.prepare_retrieval_objects()

    def retrieve_single_query(self,
                             query: str,
                             verbose: bool = False) -> Tuple[List[str], List[float], Dict]:
        """
        Retrieve documents for a single query using MARA-RAG.

        Args:
            query: Query string
            verbose: Whether to print debug information

        Returns:
            Tuple of (retrieved_docs, doc_scores, debug_info)
        """

        debug_info = {'query': query}
        start_time = time.time()

        # Step 1: Route query to get relation weights
        route_start = time.time()
        try:
            relation_weights = self.router.route(query)
            debug_info['route_time'] = time.time() - route_start
            debug_info['route_success'] = True
        except Exception as e:
            # Fallback to default weights if routing fails
            # This ensures retrieval can still proceed even if routing fails
            msg = f"Query routing failed ({e}), using default weights. Retrieval will continue."
            warnings.warn(msg)
            relation_weights = self.router._get_default_weights()
            debug_info['route_time'] = time.time() - route_start
            debug_info['route_success'] = False
            debug_info['route_error'] = str(e)
            debug_info['route_note'] = msg
        
        debug_info['relation_weights'] = relation_weights
        
        # Validate that relation_weights has all required keys
        required_keys = ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution', 'beta']
        for key in required_keys:
            if key not in relation_weights:
                msg = (
                    f"Missing required key '{key}' in relation_weights, using default. "
                    "Retrieval will continue with fallback weights."
                )
                warnings.warn(msg)
                relation_weights = self.router._get_default_weights()
                debug_info['relation_weights'] = relation_weights
                debug_info['route_success'] = False
                debug_info.setdefault('route_error', f"Missing key '{key}' in relation_weights")
                debug_info['route_note'] = msg
                break

        if verbose:
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"Relation weights:")
            print(f"  - hierarchical: {relation_weights.get('hierarchical', 0):.3f}")
            print(f"  - temporal: {relation_weights.get('temporal', 0):.3f}")
            print(f"  - spatial: {relation_weights.get('spatial', 0):.3f}")
            print(f"  - causality: {relation_weights.get('causality', 0):.3f}")
            print(f"  - attribution: {relation_weights.get('attribution', 0):.3f}")
            print(f"  - beta: {relation_weights.get('beta', 0):.3f}")
            print(f"Route success: {debug_info.get('route_success', False)}")

        # Step 2: Fact retrieval (using HippoRAG's existing method)
        fact_start = time.time()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"=== Fact Retrieval Debug ===")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Total facts in graph: {len(self.hipporag.fact_node_keys)}")
            print(f"Fact embeddings shape: {self.hipporag.fact_embeddings.shape if hasattr(self.hipporag, 'fact_embeddings') and len(self.hipporag.fact_embeddings) > 0 else 'N/A'}")
        
        query_fact_scores = self.hipporag.get_fact_scores(query)
        
        if verbose:
            print(f"\nFact scores after embedding comparison:")
            print(f"  - Fact scores shape: {query_fact_scores.shape}")
            print(f"  - Non-zero fact scores: {(query_fact_scores > 0).sum()}")
            print(f"  - Max score: {query_fact_scores.max() if query_fact_scores.shape[0] > 0 else 0.0:.4f}")
            print(f"  - Mean score: {query_fact_scores.mean() if query_fact_scores.shape[0] > 0 else 0.0:.4f}")
            print(f"  - Min score: {query_fact_scores.min() if query_fact_scores.shape[0] > 0 else 0.0:.4f}")
            
            if query_fact_scores.shape[0] > 0:
                top_scores_indices = np.argsort(query_fact_scores)[::-1][:20]  # Show top 20
                print(f"\nTop 20 facts by embedding similarity:")
                for rank, idx in enumerate(top_scores_indices, 1):
                    score = query_fact_scores[idx]
                    if score > 0 or rank <= 10:  # Always show top 10, even if score is 0
                        fact_key = self.hipporag.fact_node_keys[idx] if idx < len(self.hipporag.fact_node_keys) else f"idx_{idx}"
                        # Try to get fact content for display
                        try:
                            fact_row = self.hipporag.fact_embedding_store.get_row(fact_key)
                            fact_content = str(fact_row.get('content', 'N/A'))[:100]
                        except:
                            fact_content = 'N/A'
                        print(f"  {rank:2d}. Fact idx={idx:4d}, key={fact_key[:40]}..., score={score:.4f}")
                        print(f"      Content: {fact_content}")
        
        top_k_fact_indices, top_k_facts, rerank_log = self.hipporag.rerank_facts(
            query, query_fact_scores
        )
        debug_info['fact_retrieval_time'] = time.time() - fact_start
        debug_info['num_facts'] = len(top_k_facts)
        debug_info['sample_facts'] = top_k_facts[:3] if len(top_k_facts) > 0 else []  # Store sample for debugging
        debug_info['top_fact_scores'] = query_fact_scores[top_k_fact_indices[:10]].tolist() if len(top_k_fact_indices) > 0 else []
        debug_info['rerank_log'] = rerank_log

        if verbose:
            print(f"\n{'='*80}")
            print(f"After Reranking (Recognition Memory):")
            print(f"{'='*80}")
            print(f"Retrieved {len(top_k_facts)} facts after recognition memory and reranking")
            if len(top_k_facts) > 0:
                print(f"\nTop facts with scores:")
                for i, (fact, idx) in enumerate(zip(top_k_facts[:10], top_k_fact_indices[:10]), 1):
                    score = query_fact_scores[idx] if idx < len(query_fact_scores) else 0.0
                    print(f"  {i:2d}. {fact} (idx={idx}, score={score:.4f})")
            else:
                print("\n⚠️  WARNING: No facts retrieved! This will fall back to DPR.")
                print(f"  - Total facts in graph: {len(self.hipporag.fact_node_keys)}")
                print(f"  - Non-zero fact scores before rerank: {(query_fact_scores > 0).sum()}")
                print(f"  - Max fact score: {query_fact_scores.max() if query_fact_scores.shape[0] > 0 else 0.0:.4f}")
                print(f"  - Mean fact score: {query_fact_scores.mean() if query_fact_scores.shape[0] > 0 else 0.0:.4f}")
                if 'facts_before_rerank' in rerank_log:
                    print(f"  - Facts before rerank: {len(rerank_log.get('facts_before_rerank', []))}")
                if 'facts_after_rerank' in rerank_log:
                    print(f"  - Facts after rerank: {len(rerank_log.get('facts_after_rerank', []))}")
            print(f"{'='*80}\n")

        # Step 3: If no facts found, fall back to dense retrieval
        if len(top_k_facts) == 0:
            if verbose:
                print("No facts found, falling back to DPR")

            sorted_doc_ids, sorted_doc_scores = self.hipporag.dense_passage_retrieval(query)
            debug_info['retrieval_method'] = 'dpr_fallback'
            # For DPR, sorted_doc_ids are indices into passage_node_keys
            # We'll extract passage keys in Step 7

        else:
            # Step 4: Build phrase weights from facts (and record seed phrases)
            if verbose:
                print(f"\n{'='*80}")
                print(f"Building phrase weights from {len(top_k_facts)} facts:")
                print(f"{'='*80}")
                for i, fact in enumerate(top_k_facts[:10], 1):
                    print(f"  {i}. {fact}")
            
            phrase_weights, seed_phrases, seed_phrase_to_key = self._build_phrase_weights(
                top_k_facts, top_k_fact_indices, query_fact_scores, verbose=verbose
            )
            debug_info['num_seed_phrases'] = int((phrase_weights > 0).sum())
            debug_info['seed_phrases'] = seed_phrases
            debug_info['seed_phrase_to_key'] = seed_phrase_to_key  # Store mapping for visualization

            if verbose:
                print(f"\n{'='*80}")
                print(f"Seed phrases ({len(seed_phrases)} total):")
                print(f"{'='*80}")
                for i, phrase in enumerate(seed_phrases[:30], 1):  # Show first 30
                    print(f"  {i}. {phrase}")
                if len(seed_phrases) > 30:
                    print(f"  ... and {len(seed_phrases) - 30} more")
                print(f"Number of nodes with non-zero phrase weights: {debug_info['num_seed_phrases']}")
                print(f"{'='*80}\n")

            # Step 5: Build passage weights from dense retrieval (top 50 by similarity)
            passage_weights = self._build_passage_weights(query, verbose=verbose)
            debug_info['num_seed_passages'] = int((passage_weights > 0).sum())
            debug_info['passage_weights'] = passage_weights  # Store for visualization

            # Step 6: Run relation-aware PPR
            ppr_start = time.time()
            # Return iterations if visualization is requested
            return_iterations = getattr(self, '_visualize_ppr', False)
            ppr_result = self.ppr_engine.run_ppr(
                phrase_weights=phrase_weights,
                passage_weights=passage_weights,
                relation_weights=relation_weights,
                damping=self.config.damping,
                verbose=verbose,
                return_iterations=return_iterations
            )
            if return_iterations:
                sorted_doc_ids, sorted_doc_scores, ppr_iterations = ppr_result
                debug_info['ppr_iterations'] = ppr_iterations
            else:
                sorted_doc_ids, sorted_doc_scores = ppr_result
            debug_info['ppr_time'] = time.time() - ppr_start
            debug_info['retrieval_method'] = 'mara_ppr'

        # Step 7: Get top-k documents (chunks) and their passage keys
        # IMPORTANT: sorted_doc_ids are already sorted by score (descending)
        # So we maintain that order when extracting passage keys
        num_to_retrieve = getattr(self.config, 'retrieval_top_k', 20)
        top_docs = []
        top_scores = []
        top_passage_keys = []  # Store passage keys in score-sorted order
        
        # Determine which passage_node_keys to use based on retrieval method
        if debug_info.get('retrieval_method') == 'dpr_fallback':
            # For DPR, use HippoRAG's passage_node_keys
            passage_keys_source = self.hipporag.passage_node_keys
        else:
            # For MARA-PPR, use PPR engine's passage_node_keys
            passage_keys_source = self.ppr_engine.passage_node_keys
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"=== Final Retrieval Results ===")
            print(f"{'='*80}")
            print(f"Retrieval method: {debug_info.get('retrieval_method', 'unknown')}")
            print(f"Extracting top {num_to_retrieve} passages...")
        
        # Extract in score-sorted order (sorted_doc_ids is already sorted descending)
        for i, idx in enumerate(sorted_doc_ids[:num_to_retrieve]):
            try:
                passage_key = passage_keys_source[idx]
                doc_row = self.hipporag.chunk_embedding_store.get_row(passage_key)
                top_docs.append(doc_row["content"])
                top_scores.append(float(sorted_doc_scores[i]))
                top_passage_keys.append(passage_key)  # Already in score-sorted order
                
                if verbose and i < 10:  # Show top 10
                    doc_preview = doc_row["content"][:100].replace('\n', ' ')
                    print(f"  {i+1:2d}. Passage {idx:5d} (key: {passage_key[:40]}...): score={sorted_doc_scores[i]:.6f}")
                    print(f"      Content: {doc_preview}...")
            except (KeyError, IndexError) as e:
                if verbose:
                    print(f"Warning: Could not retrieve document for index {idx}: {e}")
                continue

        debug_info['total_time'] = time.time() - start_time
        debug_info['retrieved_passage_keys'] = top_passage_keys  # Already sorted by score
        debug_info['retrieved_passage_scores'] = top_scores  # Store scores for debugging

        if verbose:
            print(f"\nRetrieved {len(top_docs)} passages total")
            print(f"Score range: {min(top_scores):.6f} - {max(top_scores):.6f}")
            print(f"{'='*80}\n")

        return top_docs, top_scores, debug_info

    def _build_phrase_weights(self,
                             top_k_facts: List[Tuple],
                             top_k_fact_indices: List[int],
                             query_fact_scores: np.ndarray,
                             verbose: bool = False) -> Tuple[np.ndarray, List[str]]:
        """
        Build phrase node weights from retrieved facts.

        Instead of matching entity strings, we use the fact keys to look up
        the actual entities in the graph that are connected by these facts.

        Args:
            top_k_facts: List of (subject, predicate, object) triples
            top_k_fact_indices: Indices of these facts in query_fact_scores
            query_fact_scores: Scores for all facts

        Returns:
            Tuple of (phrase_weights, seed_phrases):
                - phrase_weights: Array of phrase weights (shape: [num_nodes])
                - seed_phrases: List of unique seed phrases (strings) that mapped to graph nodes
        """

        from hipporag.utils.misc_utils import compute_mdhash_id, text_processing

        phrase_weights = np.zeros(self.ppr_engine.num_nodes)
        number_of_occurs = np.zeros(self.ppr_engine.num_nodes)
        seed_phrase_set: set[str] = set()
        seed_phrase_to_key: Dict[str, str] = {}  # Map processed phrase to entity key
        
        # Debug: track what's happening
        matched_entities = []
        unmatched_entities = []

        # Get fact keys from indices
        fact_keys = []
        for rank, fact_idx in enumerate(top_k_fact_indices):
            if fact_idx < len(self.hipporag.fact_node_keys):
                fact_keys.append(self.hipporag.fact_node_keys[fact_idx])
            else:
                fact_keys.append(None)

        for rank, f in enumerate(top_k_facts):
            # Validate fact structure
            if not isinstance(f, (tuple, list)) or len(f) < 3:
                continue
            
            # Get fact score
            fact_idx = top_k_fact_indices[rank] if rank < len(top_k_fact_indices) else rank
            if query_fact_scores.ndim > 0 and fact_idx < len(query_fact_scores):
                fact_score = query_fact_scores[fact_idx]
            else:
                fact_score = 1.0  # Default score if not available

            # Extract subject and object from fact
            subject_str = str(f[0]).strip()
            object_str = str(f[2]).strip()
            
            if not subject_str or not object_str:
                continue
            
            # IMPORTANT: When building the graph, entity keys are computed from RAW strings
            # (see HippoRAG.add_fact_edges line 957-958: compute_mdhash_id(content=subj, ...))
            # So we must use the raw strings here too, NOT text_processing!
            # The graph stores: entity-<md5("Mississippi River")>
            # Not: entity-<md5("mississippi river")> (after text_processing)
            
            # Compute entity keys using RAW strings (same as graph building)
            subject_key = compute_mdhash_id(content=subject_str, prefix="entity-")
            object_key = compute_mdhash_id(content=object_str, prefix="entity-")
            
            # Keep processed versions for display/logging
            subject_processed = text_processing(subject_str)
            object_processed = text_processing(object_str)
            
            # Look up entity nodes in the graph using raw string keys
            subject_node_id = self.hipporag.node_name_to_vertex_idx.get(subject_key, None)
            object_node_id = self.hipporag.node_name_to_vertex_idx.get(object_key, None)
            
            # Fallback: Use fact_edge_meta to find connected entities if direct lookup fails
            # fact_edge_meta maps (entity1_key, entity2_key) -> {triples: [...]}
            # We can search it to find entity keys that match our fact
            if (subject_node_id is None or object_node_id is None):
                subject_lower = subject_str.lower().strip()
                object_lower = object_str.lower().strip()
                
                # First, try fact_edge_meta if available
                if hasattr(self.hipporag, 'fact_edge_meta') and self.hipporag.fact_edge_meta:
                    for (e1_key, e2_key), meta in self.hipporag.fact_edge_meta.items():
                        for triple_info in meta.get('triples', []):
                            triple_subj = str(triple_info.get('subject', '')).strip().lower()
                            triple_obj = str(triple_info.get('object', '')).strip().lower()
                            
                            # Check if this triple matches our fact (forward or reverse)
                            if (triple_subj == subject_lower and triple_obj == object_lower):
                                # Found matching fact - use the entity keys from the edge
                                if subject_node_id is None:
                                    subject_node_id = self.hipporag.node_name_to_vertex_idx.get(e1_key, None)
                                    if subject_node_id is not None:
                                        subject_key = e1_key
                                        subject_processed = triple_info.get('subject', subject_str)
                                if object_node_id is None:
                                    object_node_id = self.hipporag.node_name_to_vertex_idx.get(e2_key, None)
                                    if object_node_id is not None:
                                        object_key = e2_key
                                        object_processed = triple_info.get('object', object_str)
                                break
                            elif (triple_obj == subject_lower and triple_subj == object_lower):
                                # Reverse direction
                                if subject_node_id is None:
                                    subject_node_id = self.hipporag.node_name_to_vertex_idx.get(e2_key, None)
                                    if subject_node_id is not None:
                                        subject_key = e2_key
                                        subject_processed = triple_info.get('object', subject_str)
                                if object_node_id is None:
                                    object_node_id = self.hipporag.node_name_to_vertex_idx.get(e1_key, None)
                                    if object_node_id is not None:
                                        object_key = e1_key
                                        object_processed = triple_info.get('subject', object_str)
                                break
                
                # If still not found, try searching ALL graph nodes directly
                # This is a fallback - iterate through all nodes to find matches
                if (subject_node_id is None or object_node_id is None) and hasattr(self.hipporag, 'graph'):
                    graph = self.hipporag.graph
                    if graph is not None and 'name' in graph.vs.attributes():
                        # Search all nodes for matching entity names
                        for node_idx, node_name in enumerate(graph.vs['name']):
                            if not node_name or not node_name.startswith('entity-'):
                                continue
                            
                            # Get the entity content (remove 'entity-' prefix and decode if needed)
                            # Actually, node_name IS the key (like 'entity-6ca4a48f5be2caa16ecb8fbad41942a8')
                            # So we need to check if this key corresponds to our entity
                            
                            # Try matching by computing the key from our entity string
                            if subject_node_id is None:
                                # Check if this node's key matches our subject key
                                if node_name == subject_key:
                                    subject_node_id = node_idx
                                    # Try to get the original entity string from the graph
                                    # (we might need to look it up from entity_embedding_store)
                                    if hasattr(self.hipporag, 'entity_embedding_store'):
                                        try:
                                            entity_row = self.hipporag.entity_embedding_store.get_row(node_name)
                                            if entity_row and 'content' in entity_row:
                                                subject_processed = str(entity_row['content'])
                                        except:
                                            pass
                            
                            if object_node_id is None:
                                # Check if this node's key matches our object key
                                if node_name == object_key:
                                    object_node_id = node_idx
                                    # Try to get the original entity string from the graph
                                    if hasattr(self.hipporag, 'entity_embedding_store'):
                                        try:
                                            entity_row = self.hipporag.entity_embedding_store.get_row(node_name)
                                            if entity_row and 'content' in entity_row:
                                                object_processed = str(entity_row['content'])
                                        except:
                                            pass
                            
                            # If we found both, we can break early
                            if subject_node_id is not None and object_node_id is not None:
                                break
            
            # Add weights for matched entities
            for entity_node_id, entity_key, entity_processed in [
                (subject_node_id, subject_key, subject_processed),
                (object_node_id, object_key, object_processed)
            ]:
                if entity_node_id is not None:
                    weighted_fact_score = fact_score

                    # Weight by document frequency (specificity)
                    if hasattr(self.hipporag, 'ent_node_to_chunk_ids') and self.hipporag.ent_node_to_chunk_ids:
                        chunk_count = len(self.hipporag.ent_node_to_chunk_ids.get(entity_key, set()))
                        if chunk_count > 0:
                            weighted_fact_score /= chunk_count

                    # Accumulate weights (will be averaged later)
                    phrase_weights[entity_node_id] += weighted_fact_score
                    number_of_occurs[entity_node_id] += 1
                    # Store the processed phrase (this is what was matched in the graph)
                    seed_phrase_set.add(entity_processed)
                    # Store mapping from processed phrase to entity key for visualization
                    seed_phrase_to_key[entity_processed] = entity_key
                    matched_entities.append((entity_processed, entity_key, weighted_fact_score))
                else:
                    unmatched_entities.append((entity_processed if 'entity_processed' in locals() else str(f[0] if entity_node_id == subject_node_id else f[2]), 
                                              entity_key))

        # Debug: print matching statistics
        if verbose or len(matched_entities) == 0:
            print(f"\nEntity Matching Statistics:")
            print(f"  Total facts processed: {len(top_k_facts)}")
            print(f"  Total entities extracted: {len(matched_entities) + len(unmatched_entities)}")
            print(f"  Matched entities: {len(matched_entities)}")
            print(f"  Unmatched entities: {len(unmatched_entities)}")
            
            if len(matched_entities) > 0:
                print(f"\n  Matched entities (first 10):")
                for phrase, key, score in matched_entities[:10]:
                    print(f"    - '{phrase}' (key: {key[:40]}..., score: {score:.4f})")
            
            if len(unmatched_entities) > 0:
                print(f"\n  Unmatched entities (first 10):")
                for phrase, key in unmatched_entities[:10]:
                    print(f"    - '{phrase}' -> computed key: {key[:50]}...")
                    # Check if this key exists in the graph
                    if hasattr(self.hipporag, 'node_name_to_vertex_idx'):
                        if key in self.hipporag.node_name_to_vertex_idx:
                            print(f"      ✓ Key EXISTS in graph! (node_id: {self.hipporag.node_name_to_vertex_idx[key]})")
                        else:
                            print(f"      ✗ Key NOT in graph")
                            # Check if there are any entity nodes at all
                            entity_keys = [k for k in self.hipporag.node_name_to_vertex_idx.keys() if k.startswith('entity-')]
                            if entity_keys:
                                print(f"      (Graph has {len(entity_keys)} entity nodes, e.g., {entity_keys[0][:50]}...)")
                            else:
                                print(f"      (Graph has NO entity nodes starting with 'entity-')")
        
        if len(matched_entities) == 0:
            print(f"\n⚠️  ERROR: No entities matched in graph! This will cause retrieval to fail.")
            print(f"  Check if graph has entity nodes and if text_processing matches graph normalization.")
            # Additional debugging: show sample entity keys from graph
            if hasattr(self.hipporag, 'node_name_to_vertex_idx'):
                entity_keys = [k for k in self.hipporag.node_name_to_vertex_idx.keys() if k.startswith('entity-')]
                print(f"  Graph has {len(entity_keys)} entity nodes")
                if entity_keys and len(unmatched_entities) > 0:
                    print(f"  Sample computed key: {unmatched_entities[0][1][:50]}...")
                    print(f"  Sample graph entity key: {entity_keys[0][:50]}...")
                    # Try to see if we can find a match by checking the entity content
                    if hasattr(self.hipporag, 'entity_embedding_store') and unmatched_entities:
                        try:
                            sample_key = unmatched_entities[0][1]
                            # Try to get entity content from embedding store
                            entity_row = self.hipporag.entity_embedding_store.get_row(sample_key)
                            if entity_row:
                                print(f"  Entity store has key '{sample_key[:50]}...' with content: {str(entity_row.get('content', 'N/A'))[:100]}")
                        except Exception as e:
                            print(f"  Could not check entity store: {e}")

        # Clean up any inf/nan values from division
        phrase_weights = np.where(
            (np.isnan(phrase_weights) | np.isinf(phrase_weights)), 
            0, 
            phrase_weights
        )

        # Average scores for phrases that appear multiple times
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            phrase_weights = np.where(number_of_occurs > 0, phrase_weights / number_of_occurs, phrase_weights)
        
        # Scale phrase weights to make them more prominent in restart vector
        # Since fact scores are normalized (0-1), we scale them up to compete with passage weights
        # This ensures entities have reasonable restart probability
        if phrase_weights.max() > 0:
            # Scale so max phrase weight is comparable to max passage weight (after beta scaling)
            # If max passage weight is ~1.0 and beta is 0.05-0.1, max scaled passage weight is ~0.1
            # So we want max phrase weight to be around 0.1-0.2 to have similar influence
            phrase_weight_scale = 0.15 / phrase_weights.max() if phrase_weights.max() > 0 else 1.0
            phrase_weights = phrase_weights * phrase_weight_scale
            
            if verbose:
                print(f"  Scaled phrase weights by {phrase_weight_scale:.3f} to improve restart probability")
                print(f"  Phrase weight range after scaling: {phrase_weights[phrase_weights > 0].min():.6f} - {phrase_weights[phrase_weights > 0].max():.6f}")

        # Convert set to sorted list for stable output
        seed_phrases = sorted(seed_phrase_set)

        return phrase_weights, seed_phrases, seed_phrase_to_key

    def _build_passage_weights(self, query: str, verbose: bool = False) -> np.ndarray:
        """
        Build passage node weights from dense retrieval.
        
        Selects top 50 passages ranked by query-passage similarity (DPR scores).

        Args:
            query: Query string
            verbose: Whether to log passage selection details

        Returns:
            Array of passage weights (shape: [num_nodes])
        """

        from hipporag.utils.misc_utils import min_max_normalize

        passage_weights = np.zeros(self.ppr_engine.num_nodes)

        # Get dense passage retrieval scores (already sorted by similarity, descending)
        # IMPORTANT: Prioritize passages that are more similar to the query
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.hipporag.dense_passage_retrieval(query)
        normalized_dpr_scores = min_max_normalize(dpr_sorted_doc_scores)

        # Assign weights to passage nodes
        # IMPORTANT: Limit to top 20 passages for initial seed (most query-similar)
        # This reduces initial nodes and focuses on most relevant passages
        top_k_passages = min(20, len(dpr_sorted_doc_ids))  # Changed from 50 to 20
        
        if verbose:
            print(f"\n[Passage Weight Building]")
            print(f"  Total passages available: {len(dpr_sorted_doc_ids)}")
            print(f"  Selecting top {top_k_passages} passages by query-passage similarity")
            print(f"  DPR score range: {dpr_sorted_doc_scores.min():.4f} - {dpr_sorted_doc_scores.max():.4f}")
            print(f"  Top 10 passages by similarity:")
            for i in range(min(10, top_k_passages)):
                doc_id = dpr_sorted_doc_ids[i]
                score = dpr_sorted_doc_scores[i]
                norm_score = normalized_dpr_scores[i]
                passage_key = self.ppr_engine.passage_node_keys[doc_id] if doc_id < len(self.ppr_engine.passage_node_keys) else f"idx_{doc_id}"
                print(f"    {i+1:2d}. Passage {doc_id:5d} (key: {passage_key[:40]}...): DPR={score:.4f}, normalized={norm_score:.4f}")
        
        for i, dpr_doc_id in enumerate(dpr_sorted_doc_ids[:top_k_passages].tolist()):
            passage_node_key = self.ppr_engine.passage_node_keys[dpr_doc_id]
            passage_dpr_score = normalized_dpr_scores[i]
            passage_node_id = self.hipporag.node_name_to_vertex_idx.get(passage_node_key, None)
            if passage_node_id is not None:
                # Use normalized DPR score directly (no extra scaling)
                passage_weights[passage_node_id] = passage_dpr_score

        if verbose:
            non_zero_count = (passage_weights > 0).sum()
            print(f"  Assigned weights to {non_zero_count} passage nodes")
            print(f"  Passage weight range: {passage_weights[passage_weights > 0].min():.6f} - {passage_weights[passage_weights > 0].max():.6f}")

        return passage_weights

    def run_qa(self, query: str, retrieved_docs: List[str], verbose: bool = False) -> Tuple[str, Dict]:
        """
        Generate answer for query using retrieved documents.
        
        Only uses top 5 passages as context for QA.

        Args:
            query: Query string
            retrieved_docs: List of retrieved document texts (already sorted by score)
            verbose: Whether to log QA input/output

        Returns:
            Tuple of (answer, metadata)
        """

        # Use top K passages for QA (configurable via --qa-top-k argument)
        top_k_for_qa = getattr(self, 'qa_top_k', 5)  # Default to 5 if not set
        qa_docs = retrieved_docs[:top_k_for_qa]
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"=== QA Agent ===")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Using top {len(qa_docs)} passages (out of {len(retrieved_docs)} retrieved):")
            for i, doc in enumerate(qa_docs, 1):
                doc_preview = doc[:200].replace('\n', ' ') if doc else ""
                print(f"  {i}. {doc_preview}...")
            print(f"{'='*80}")

        # Use HippoRAG's QA method
        from hipporag.utils.misc_utils import QuerySolution

        query_solution = QuerySolution(question=query, docs=qa_docs)
        results, messages, metadata = self.hipporag.qa([query_solution])
        
        # Extract answer from QuerySolution (answer is set by qa method)
        answer = results[0].answer if hasattr(results[0], 'answer') and results[0].answer else ""
        meta = metadata[0] if metadata and len(metadata) > 0 else {}
        
        if verbose:
            print(f"\nQA Output:")
            print(f"  Predicted Answer: {answer}")
            if messages and len(messages) > 0:
                print(f"  Messages exchanged: {len(messages)}")
                # Show last message if available
                if isinstance(messages[0], list) and len(messages[0]) > 0:
                    last_msg = messages[0][-1]
                    if isinstance(last_msg, dict) and 'content' in last_msg:
                        print(f"  Last message preview: {str(last_msg['content'])[:200]}...")
            print(f"{'='*80}\n")

        return answer, meta


def load_questions(questions_file: Path) -> List[Dict]:
    """
    Load questions from JSONL or JSON file.

    Expected format:
    JSONL: One question per line
    {
        "question": "...",
        "answer": [...],  # List of gold answers
        "supporting_facts": [[title1, sent_id1], ...],  # Optional
        "context": [[title, [sent1, sent2, ...]], ...]  # Optional
    }
    
    JSON: Array of questions
    [
        {"question": "...", "answer": [...], ...},
        ...
    ]

    Args:
        questions_file: Path to questions JSONL or JSON file

    Returns:
        List of question dictionaries
    """

    questions = []
    questions_file = Path(questions_file)
    
    # Check file extension
    if questions_file.suffix == '.json':
        # Load as JSON array
        with open(questions_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                questions = data
            elif isinstance(data, dict) and 'questions' in data:
                questions = data['questions']
            else:
                # Single question object
                questions = [data]
    else:
        # Load as JSONL (one question per line)
        with open(questions_file, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
    
    return questions


def load_corpus(corpus_file: Path) -> Dict[str, Dict]:
    """
    Load the MuSiQue corpus file.
    
    Expected formats:
    1. {title: {sentences: [...], ...}}
    2. [{title: "...", sentences: [...]}, ...]
    3. {title: [sent1, sent2, ...]}  (direct sentence list)

    Args:
        corpus_file: Path to corpus JSON file
        
    Returns:
        Dictionary mapping title to document structure with 'sentences' key
    """
    corpus = {}
    
    if not corpus_file.exists():
        return corpus
    
    with open(corpus_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
        # Handle different corpus formats
        if isinstance(data, dict):
            for title, doc_data in data.items():
                if isinstance(doc_data, dict):
                    # Format: {title: {sentences: [...], ...}}
                    if 'sentences' in doc_data:
                        corpus[title] = {'sentences': doc_data['sentences']}
                    elif 'paragraphs' in doc_data:
                        # Convert paragraphs to sentences
                        sentences = []
                        for para in doc_data['paragraphs']:
                            # Split by sentence boundaries
                            para_sents = para.replace('! ', '. ').replace('? ', '. ').split('. ')
                            sentences.extend([s.strip() + '.' for s in para_sents if s.strip()])
                        corpus[title] = {'sentences': sentences}
                elif isinstance(doc_data, list):
                    # Format: {title: [sent1, sent2, ...]}
                    corpus[title] = {'sentences': doc_data}
        elif isinstance(data, list):
            # Format: [{title: "...", sentences: [...]}, ...]
            for doc in data:
                title = doc.get('title', '')
                if not title:
                    continue
                if 'sentences' in doc:
                    corpus[title] = {'sentences': doc['sentences']}
                elif 'paragraphs' in doc:
                    # Convert paragraphs to sentences
                    sentences = []
                    for para in doc['paragraphs']:
                        para_sents = para.replace('! ', '. ').replace('? ', '. ').split('. ')
                        sentences.extend([s.strip() + '.' for s in para_sents if s.strip()])
                    corpus[title] = {'sentences': sentences}
    
    return corpus


def build_chunk_to_doc_mapping(corpus: Dict[str, Dict], 
                                questions: List[Dict], 
                                hipporag,
                                dataset_path: Path) -> Dict[str, Tuple[str, int]]:
    """
    Build a mapping from chunk keys (passage keys) to (title, sent_id) document identifiers.
    
    This maps retrieved chunks back to their original document IDs by:
    1. Loading the full corpus to get all documents
    2. Matching chunk content to corpus sentences
    3. Mapping chunk keys to (title, sent_id) pairs
    
    Args:
        corpus: Dictionary mapping title to document structure with sentences
        questions: List of question dictionaries (for fallback matching)
        hipporag: HippoRAG instance with chunk_embedding_store
        dataset_path: Path to dataset directory
        
    Returns:
        Dictionary mapping chunk_key to (title, sent_id) tuples
    """
    chunk_to_doc = {}
    
    # Get all chunks from the embedding store
    all_chunks = hipporag.chunk_embedding_store.get_all_id_to_rows()
    print(f"Total chunks in embedding store: {len(all_chunks)}")
    
    # First, try to build mapping from corpus
    if corpus:
        print(f"Building mapping from corpus with {len(corpus)} documents...")
        matched_from_corpus = 0
        
        for title, doc_data in corpus.items():
            sentences = doc_data.get('sentences', [])
            for sent_id, sentence in enumerate(sentences):
                sentence_normalized = sentence.strip()
                # Normalize: remove extra whitespace, lowercase for comparison
                sentence_normalized_clean = ' '.join(sentence_normalized.lower().split())
                
                # Try to find matching chunk by content
                for chunk_key, chunk_row in all_chunks.items():
                    if chunk_key in chunk_to_doc:
                        continue  # Already mapped
                    
                    chunk_content = chunk_row['content'].strip()
                    # Remove title prefix if present
                    chunk_without_title = chunk_content
                    if chunk_content.startswith(title):
                        chunk_without_title = chunk_content[len(title):].strip()
                    
                    # Normalize for comparison
                    chunk_normalized = ' '.join(chunk_content.lower().split())
                    chunk_without_title_normalized = ' '.join(chunk_without_title.lower().split())
                    
                    # Match if chunk content matches sentence (various formats)
                    if (chunk_normalized == sentence_normalized_clean or
                        chunk_without_title_normalized == sentence_normalized_clean or
                        chunk_normalized.endswith(sentence_normalized_clean) or
                        sentence_normalized_clean in chunk_normalized or
                        chunk_without_title_normalized.endswith(sentence_normalized_clean)):
                        chunk_to_doc[chunk_key] = (title, sent_id)
                        matched_from_corpus += 1
                        break  # One chunk per sentence (first match)
        
        print(f"Matched {matched_from_corpus} chunks from corpus")
    
    # Fallback: also try matching from question contexts
    # This helps catch any chunks that weren't in the corpus file
    matched_from_questions = 0
    for q in questions:
        if 'context' not in q:
            continue
        
        # MuSiQue format: context is [[title, [sent1, sent2, ...]], ...]
        for title, sentences in q['context']:
            for sent_id, sentence in enumerate(sentences):
                sentence_normalized = sentence.strip()
                sentence_normalized_clean = ' '.join(sentence_normalized.lower().split())
                
                # Skip if already mapped
                # Try to find matching chunk by content
                for chunk_key, chunk_row in all_chunks.items():
                    if chunk_key in chunk_to_doc:
                        continue  # Already mapped
                    
                    chunk_content = chunk_row['content'].strip()
                    # Remove title prefix if present
                    chunk_without_title = chunk_content
                    if chunk_content.startswith(title):
                        chunk_without_title = chunk_content[len(title):].strip()
                    
                    # Normalize for comparison
                    chunk_normalized = ' '.join(chunk_content.lower().split())
                    chunk_without_title_normalized = ' '.join(chunk_without_title.lower().split())
                    
                    # Match if chunk content matches sentence (various formats)
                    if (chunk_normalized == sentence_normalized_clean or
                        chunk_without_title_normalized == sentence_normalized_clean or
                        chunk_normalized.endswith(sentence_normalized_clean) or
                        sentence_normalized_clean in chunk_normalized or
                        chunk_without_title_normalized.endswith(sentence_normalized_clean)):
                        chunk_to_doc[chunk_key] = (title, sent_id)
                        matched_from_questions += 1
    
    if matched_from_questions > 0:
        print(f"Matched {matched_from_questions} additional chunks from question contexts")
    
    print(f"Total mapped chunks: {len(chunk_to_doc)}")
    return chunk_to_doc


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, strip whitespace)."""
    return text.lower().strip()


def is_chunk_in_passage(chunk_text: str, gold_passage_text: str) -> bool:
    """
    Check if a retrieved chunk is contained in a gold passage.
    
    Retrieved chunks often have a title prefix (first line), so we:
    1. Remove the title (first line) from the chunk
    2. Check if the remaining text is in the gold passage
    3. Also check if gold passage is in chunk (reverse direction)
    """
    chunk_normalized = normalize_text(chunk_text)
    passage_normalized = normalize_text(gold_passage_text)
    
    # Remove title prefix (first line) from chunk if present
    # Retrieved docs often have format: "Title\nText content..."
    if '\n' in chunk_normalized:
        # Get text after first newline (remove title)
        chunk_without_title = '\n'.join(chunk_normalized.split('\n')[1:]).strip()
        if chunk_without_title:
            chunk_normalized = chunk_without_title
    
    # Check both directions:
    # 1. If chunk (without title) is in gold passage
    # 2. If gold passage is in chunk (in case chunk is longer)
    return (chunk_normalized in passage_normalized) or (passage_normalized in chunk_normalized)


def load_gold_passages(musique_path: Path) -> Dict[str, List[str]]:
    """
    Load musique.json and return a dict mapping question -> list of gold passage texts.
    
    Gold passages are paragraphs where is_supporting=True.
    """
    gold_passages = {}
    
    if not musique_path.exists():
        print(f"Warning: musique.json not found at {musique_path}")
        return gold_passages
    
    with open(musique_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        question = entry['question']
        # Get all paragraphs where is_supporting=True
        supporting_paragraphs = [
            para['paragraph_text'] 
            for para in entry.get('paragraphs', [])
            if para.get('is_supporting', False)
        ]
        gold_passages[question] = supporting_paragraphs
    
    return gold_passages


def compute_recall_at_k(retrieved_docs: List[str], 
                        doc_scores: List[float], 
                        gold_passages: List[str], 
                        k: int) -> float:
    """
    Compute Recall@k.
    
    Args:
        retrieved_docs: List of retrieved document chunks (text content)
        doc_scores: List of scores for retrieved documents
        gold_passages: List of gold passage texts
        k: Number of top documents to consider
    
    Returns:
        recall@k value (0.0 to 1.0)
    """
    if not gold_passages:
        return 0.0
    
    # Sort by scores (descending)
    sorted_pairs = sorted(
        zip(retrieved_docs, doc_scores),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Take top k
    top_k_docs = [doc for doc, score in sorted_pairs[:k]]
    
    # Check which gold passages have at least one retrieved chunk
    gold_passages_with_retrieval = set()
    for chunk in top_k_docs:
        for i, gold_passage in enumerate(gold_passages):
            if is_chunk_in_passage(chunk, gold_passage):
                gold_passages_with_retrieval.add(i)
    
    # Recall = number of gold passages that have at least one retrieved chunk / total gold passages
    recall = len(gold_passages_with_retrieval) / len(gold_passages) if gold_passages else 0.0
    return recall


def compute_retrieval_metrics(results: List[Dict],
                               questions: List[Dict],
                               musique_path: Path,
                               k_list: List[int] = [1, 2, 5, 10, 12, 20]) -> Dict:
    """
    Compute retrieval recall@k metrics using text matching.
    
    Logic:
    1. Load gold passages from musique.json (paragraphs where is_supporting=True)
    2. For each question, get retrieved docs and scores from results
    3. Sort retrieved docs by scores (descending)
    4. Check if each retrieved chunk is contained in any gold passage
    5. Compute Recall@k: number of gold passages with at least one retrieved chunk / total gold passages

    Args:
        results: List of result dictionaries with 'question', 'retrieved_docs', 'doc_scores'
        questions: List of question dictionaries
        musique_path: Path to musique.json file
        k_list: List of k values for recall@k

    Returns:
        Dictionary of metrics
    """
    # Load gold passages from musique.json
    print(f"Loading gold passages from {musique_path}...")
    gold_passages_dict = load_gold_passages(musique_path)
    print(f"Loaded gold passages for {len(gold_passages_dict)} questions")
    
    # Create a mapping from question text to result
    results_by_question = {r['question']: r for r in results}
    
    # Compute metrics for each question
    recall_at_k_dict = {k: [] for k in k_list}
    matched_questions = 0
    
    for question_data in questions:
        question = question_data['question']
        
        # Skip if no result for this question
        if question not in results_by_question:
            continue
        
        # Skip if no gold passages for this question
        if question not in gold_passages_dict:
            continue
        
        matched_questions += 1
        result = results_by_question[question]
        retrieved_docs = result.get('retrieved_docs', [])
        doc_scores = result.get('doc_scores', [])
        gold_passage_list = gold_passages_dict[question]
        
        # Ensure lengths match
        if len(retrieved_docs) != len(doc_scores):
            min_len = min(len(retrieved_docs), len(doc_scores))
            retrieved_docs = retrieved_docs[:min_len]
            doc_scores = doc_scores[:min_len]
        
        # Compute recall@k for different k values
    for k in k_list:
            recall = compute_recall_at_k(retrieved_docs, doc_scores, gold_passage_list, k)
            recall_at_k_dict[k].append(recall)
    
    print(f"\nMatched {matched_questions} questions with gold passages")
    
    # Compute average metrics
    metrics = {}
    for k in k_list:
        recalls = recall_at_k_dict[k]
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
        metrics[f'recall@{k}'] = float(avg_recall)
        print(f"  Recall@{k}: {avg_recall:.4f} (computed over {len(recalls)} questions)")
    
    metrics['num_questions'] = matched_questions

    return metrics


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MARA-RAG experiment")

    parser.add_argument(
        "--experiment-name",
        required=True,
        help="Experiment folder name"
    )
    parser.add_argument(
        "--questions-file",
        required=True,
        help="Path to questions JSONL file (relative to experiment folder or absolute)"
    )
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="HippoRAG workspace directory"
    )
    parser.add_argument(
        "--matrix-subdir",
        default="mara_matrices",
        help="MARA matrices directory (output from graph_preprocessing.py)"
    )
    parser.add_argument(
        "--output-subdir",
        default="mara_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--llm-model-name",
        default="Qwen/Qwen3-8B",
        help="LLM model name"
    )
    parser.add_argument(
        "--embedding-model-name",
        default="facebook/contriever-msmarco",
        help="Embedding model name"
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://holygpu7c26105.rc.fas.harvard.edu:8000/v1",
        help="OpenAI-compatible LLM endpoint for QA"
    )
    parser.add_argument(
        "--router-llm-model-name",
        default=None,
        help="LLM model name for query router (default: same as --llm-model-name)"
    )
    parser.add_argument(
        "--router-llm-base-url",
        default=None,
        help="OpenAI-compatible LLM endpoint for query router (default: same as --llm-base-url)"
    )
    parser.add_argument(
        "--dataset-path",
        default="/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/musique/subset_50",
        help="Base path to dataset folder containing musique.json, musique_corpus.json, etc."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear LLM cache before running (ensures fresh LLM calls)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate PPR graph visualization for each query"
    )
    parser.add_argument(
        "--visualize-exact-match",
        action="store_true",
        help="Only generate visualizations for questions with exact match = 1.0 (requires --visualize)"
    )
    parser.add_argument(
        "--qa-top-k",
        type=int,
        default=5,
        help="Number of top passages to use as context for QA (default: 5)"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # Setup paths
    # Results directory is on scratch
    results_base = Path("/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/experiments")
    experiment_dir = results_base / args.experiment_name
    workspace_dir = experiment_dir / args.workspace_subdir
    matrix_dir = experiment_dir / args.matrix_subdir
    output_dir = experiment_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    import logging
    import sys
    from datetime import datetime
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting MARA-RAG experiment")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Output directory: {output_dir}")
    
    # Store logger for use throughout the script
    import logging
    logger = logging.getLogger(__name__)
    
    # Also check local experiment directory for questions file
    local_experiment_dir = PROJECT_ROOT / "experiment" / "dataset" / args.experiment_name

    # Define dataset_path early (used later for corpus loading)
    dataset_path = Path(args.dataset_path)

    # Load questions
    questions_file = Path(args.questions_file)
    if not questions_file.is_absolute():
        # First try dataset path
        if (dataset_path / questions_file).exists():
            questions_file = dataset_path / questions_file
        # Then try local experiment dir
        elif (local_experiment_dir / questions_file).exists():
            questions_file = local_experiment_dir / questions_file
        # Then try results dir
        elif (experiment_dir / questions_file).exists():
            questions_file = experiment_dir / questions_file
        else:
            # Default: try dataset path with common filenames
            if (dataset_path / "musique.json").exists():
                questions_file = dataset_path / "musique.json"
            else:
                questions_file = local_experiment_dir / questions_file  # Fallback

    logger.info(f"Loading questions from {questions_file}...")
    print(f"Loading questions from {questions_file}...")
    if not questions_file.exists():
        error_msg = f"Questions file not found: {questions_file}\nPlease specify --questions-file or ensure the file exists in --dataset-path"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    questions = load_questions(questions_file)
    logger.info(f"Loaded {len(questions)} questions")

    if args.num_questions:
        questions = questions[:args.num_questions]

    print(f"Loaded {len(questions)} questions")

    # Clear cache if requested
    if args.clear_cache:
        cache_dir = workspace_dir / "llm_cache"
        if cache_dir.exists():
            print(f"\nClearing LLM cache from {cache_dir}...")
            import shutil
            for cache_file in cache_dir.glob("*.sqlite"):
                print(f"  Removing cache file: {cache_file.name}")
                cache_file.unlink()
            for lock_file in cache_dir.glob("*.lock"):
                lock_file.unlink()
            print("  Cache cleared!")
        else:
            print(f"\nNo cache directory found at {cache_dir}, nothing to clear.")

    # Load HippoRAG
    print(f"\nLoading HippoRAG from {workspace_dir}...")
    # Use hyperparameters consistent with the original HippoRAG paper where applicable
    config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.llm_base_url,
        # Retrieval hyperparameters
        retrieval_top_k=200,
        linking_top_k=5,
        damping=0.15,  # Reduced from 0.5 to 0.15 for better convergence to relevant passages
        # QA hyperparameters
        max_qa_steps=3,
        qa_top_k=5,
    )

    hipporag = HippoRAG(global_config=config)

    # Load graph
    if not hasattr(hipporag, 'graph') or hipporag.graph is None:
        hipporag.load_igraph()

    print(f"Graph loaded: {len(hipporag.graph.vs)} nodes, {len(hipporag.graph.es)} edges")

    # Initialize router with separate LLM if specified
    print("\nInitializing query router...")
    if args.router_llm_model_name or args.router_llm_base_url:
        # Create separate LLM instance for router
        from hipporag.llm import _get_llm_class
        router_llm_config = BaseConfig(
            save_dir=str(workspace_dir),
            llm_name=args.router_llm_model_name or args.llm_model_name,
            llm_base_url=args.router_llm_base_url or args.llm_base_url,
        )
        router_llm = _get_llm_class(router_llm_config)
        print(f"Router LLM: {args.router_llm_model_name or args.llm_model_name} at {args.router_llm_base_url or args.llm_base_url}")
    else:
        # Use same LLM as QA
        router_llm = hipporag.llm_model
        print(f"Router LLM: Using same LLM as QA ({args.llm_model_name})")
    
    router = QueryRouter(llm=router_llm)

    # Load relation-aware PPR engine
    print(f"Loading relation matrices from {matrix_dir}...")
    ppr_engine = RelationAwarePPR(matrix_dir=matrix_dir)

    # Create experiment runner
    experiment = MARAExperiment(
        hipporag=hipporag,
        router=router,
        ppr_engine=ppr_engine,
        config=config,
        qa_top_k=args.qa_top_k
    )
    
    # Enable PPR iteration capture if visualization is requested
    if args.visualize:
        experiment._visualize_ppr = True
        print(f"Visualization enabled: experiment._visualize_ppr = {experiment._visualize_ppr}")
    
    # Enable PPR iteration capture if visualization is requested
    if args.visualize:
        experiment._visualize_ppr = True

    # Run retrieval + QA on all questions
    print(f"\nRunning MARA-RAG on {len(questions)} questions...")
    results = []

    for q_data in tqdm(questions, desc="Evaluating queries", unit="query", total=len(questions)):
        query = q_data['question']
        
        # Get gold answer if available (for logging)
        gold_answers = q_data.get('answer', [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]
        
        # Log gold answer for manual comparison
        if args.verbose:
            print(f"\n{'='*80}")
            print(f"=== Gold Answer ===")
            print(f"{'='*80}")
            print(f"Gold answers: {gold_answers}")
            print(f"{'='*80}\n")

        # Retrieve documents
        retrieved_docs, doc_scores, debug_info = experiment.retrieve_single_query(
            query, verbose=args.verbose
        )

        # Store retrieved passage keys for document-level recall computation
        retrieved_passage_keys = debug_info.get('retrieved_passage_keys', [])
        
        # Create animated visualization if requested
        # Only visualize if PPR was actually used (not DPR fallback)
        # If --visualize-exact-match is set, only visualize when exact match = 1.0
        retrieval_method = debug_info.get('retrieval_method', 'unknown')
        should_visualize = args.visualize and len(retrieved_passage_keys) > 0 and retrieval_method == 'mara_ppr'
        
        # Check exact match if we need to conditionally visualize
        if should_visualize and hasattr(args, 'visualize_exact_match') and args.visualize_exact_match:
            # We need to compute EM first - it will be computed later, so we'll check it after QA
            # For now, we'll set a flag and check after QA
            should_visualize = False  # Will be set to True after QA if EM = 1.0
        
        if should_visualize:
            try:
                print(f"\n{'='*80}")
                print(f"Creating animated PPR visualization...")
                print(f"{'='*80}")
                
                from experiment.online_retrieval.visualize_ppr_animated import create_animated_ppr_visualization
                from experiment.online_retrieval.visualize_ppr import load_gold_passages
                
                # Get PPR iterations if available
                ppr_iterations = debug_info.get('ppr_iterations', [])
                
                print(f"Debug: args.visualize = {args.visualize}")
                print(f"Debug: retrieved_passage_keys count = {len(retrieved_passage_keys)}")
                print(f"Debug: ppr_iterations count = {len(ppr_iterations)}")
                
                if len(ppr_iterations) == 0:
                    print("ERROR: No PPR iterations available for visualization!")
                    print("This might happen if:")
                    print("  1. The retrieval method was 'dpr_fallback' (not using PPR)")
                    print("  2. The return_iterations flag was not set correctly")
                    print("  3. The PPR algorithm didn't run")
                else:
                    # Get gold passages for this question
                    dataset_path = Path(args.dataset_path)
                    musique_file = dataset_path / "musique.json"
                    gold_passages = load_gold_passages(musique_file, query)
                    
                    # Map gold passages to passage keys (simplified - match by content)
                    gold_passage_keys = set()
                    for gold_passage in gold_passages:
                        # Try to find matching passage key by content
                        for passage_key in ppr_engine.passage_node_keys:
                            try:
                                passage_row = hipporag.chunk_embedding_store.get_row(passage_key)
                                passage_content = passage_row.get('content', '')
                                # Check if gold passage is contained in chunk content
                                if gold_passage.lower().strip() in passage_content.lower():
                                    gold_passage_keys.add(passage_key)
                                    break
                            except:
                                continue
                    
                    # Get seed information
                    seed_phrases = debug_info.get('seed_phrases', [])
                    seed_phrase_to_key = debug_info.get('seed_phrase_to_key', {})  # Map from processed phrase to entity key
                    
                    # Extract seed passages from passage_weights (top passages with non-zero weights)
                    seed_passages = []
                    passage_weights = debug_info.get('passage_weights', None)
                    if passage_weights is not None:
                        # passage_weights is a numpy array indexed by node_idx (not passage_idx)
                        # We need to map node_idx to passage_node_idxs to get passage keys
                        if isinstance(passage_weights, np.ndarray):
                            # Get node indices with non-zero weights
                            non_zero_node_indices = np.where(passage_weights > 0)[0]
                            if len(non_zero_node_indices) > 0:
                                # Sort by weight (descending)
                                sorted_node_indices = non_zero_node_indices[np.argsort(passage_weights[non_zero_node_indices])[::-1]]
                                # Get top 50
                                top_node_indices = sorted_node_indices[:50]
                                
                                # Map node indices to passage keys
                                # passage_node_idxs maps passage_idx -> node_idx
                                # We need to reverse this: node_idx -> passage_idx -> passage_key
                                node_idx_to_passage_idx = {node_idx: passage_idx 
                                                          for passage_idx, node_idx in enumerate(ppr_engine.passage_node_idxs)}
                                
                                for node_idx in top_node_indices:
                                    if node_idx in node_idx_to_passage_idx:
                                        passage_idx = node_idx_to_passage_idx[node_idx]
                                        if passage_idx < len(ppr_engine.passage_node_keys):
                                            passage_key = ppr_engine.passage_node_keys[passage_idx]
                                            seed_passages.append(passage_key)
                        print(f"  Extracted {len(seed_passages)} seed passages from passage_weights")
                    
                    # Create visualization directory
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    viz_path = viz_dir / f"ppr_animated_query_{len(results):03d}.gif"
                    
                    print(f"  PPR iterations captured: {len(ppr_iterations)}")
                    print(f"  Gold passages found: {len(gold_passage_keys)}")
                    print(f"  Seed entities: {len(seed_phrases)}")
                    print(f"  Retrieved passages: {len(retrieved_passage_keys)}")
                    
                    # Convert doc_scores to list if it's a numpy array, otherwise use as-is
                    if isinstance(doc_scores, np.ndarray):
                        scores_list = doc_scores[:20].tolist() if len(doc_scores) >= 20 else doc_scores.tolist()
                    else:
                        # Already a list, just slice it
                        scores_list = doc_scores[:20] if len(doc_scores) >= 20 else doc_scores
                    
                    create_animated_ppr_visualization(
                        graph=hipporag.graph,
                        passage_node_keys=ppr_engine.passage_node_keys,
                        passage_node_idxs=ppr_engine.passage_node_idxs,
                        gold_passage_keys=gold_passage_keys,
                        retrieved_passage_keys=retrieved_passage_keys[:20],
                        retrieved_scores=scores_list,
                        relation_weights=debug_info.get('relation_weights', {}),
                        seed_entities=seed_phrases,
                        seed_passages=seed_passages,
                        ppr_iterations=ppr_iterations,
                        output_path=viz_path,
                        max_nodes=200,
                        fps=2,
                        node_name_to_vertex_idx=hipporag.node_name_to_vertex_idx,  # Pass mapping for better entity matching
                        seed_phrase_to_key=seed_phrase_to_key  # Pass mapping from processed phrase to entity key
                    )
                    
                    print(f"✓ Animated visualization saved to: {viz_path}")
                    print(f"{'='*80}\n")
            except Exception as e:
                print(f"ERROR: Could not create visualization: {e}")
                import traceback
                traceback.print_exc()
                print(f"{'='*80}\n")

        # Generate answer (only using top 5 passages)
        answer, qa_metadata = experiment.run_qa(query, retrieved_docs, verbose=args.verbose)
        
        # Display gold answer after QA response for comparison
        if args.verbose:
            print(f"\n{'='*80}")
            print(f"=== Answer Comparison ===")
            print(f"{'='*80}")
            print(f"Predicted Answer: {answer}")
            print(f"Gold Answers: {gold_answers}")
            print(f"{'='*80}\n")

        # Compute F1 and Exact Match scores
        from hipporag.utils.eval_utils import normalize_answer
        from collections import Counter
        
        def compute_f1(gold: str, predicted: str) -> float:
            """Compute F1 score between gold and predicted answers."""
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())
            
            if num_same == 0:
                return 0.0
            
            precision = 1.0 * num_same / len(predicted_tokens) if len(predicted_tokens) > 0 else 0.0
            recall = 1.0 * num_same / len(gold_tokens) if len(gold_tokens) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            return f1
        
        f1_scores = [compute_f1(gold_ans, answer) for gold_ans in gold_answers]
        max_f1 = max(f1_scores) if f1_scores else 0.0

        # Exact Match (EM): 1 if any gold answer matches predicted after normalization
        norm_pred = normalize_answer(answer) if answer else ""
        em = 0.0
        for gold_ans in gold_answers:
            if normalize_answer(gold_ans) == norm_pred and norm_pred != "":
                em = 1.0
                break

        # Store result
        result = {
            'question': query,
            'answer': answer,
            'gold_answers': gold_answers,
            'f1_score': max_f1,
            'em_score': em,
            'retrieved_docs': retrieved_docs,
            'retrieved_passage_keys': retrieved_passage_keys,  # Store for document mapping
            'doc_scores': doc_scores,
            'relation_weights': debug_info['relation_weights'],
            # Seed phrases used to initialize graph search (if available)
            'seed_phrases': debug_info.get('seed_phrases', []),
            'debug_info': debug_info,
        }

        results.append(result)
        
        # Create visualization AFTER computing EM score if --visualize-exact-match is set
        # Only visualize questions with exact match = 1.0
        # Note: argparse converts --visualize-exact-match to args.visualize_exact_match
        if (hasattr(args, 'visualize_exact_match') and getattr(args, 'visualize_exact_match', False) and 
            em == 1.0 and len(retrieved_passage_keys) > 0 and 
            debug_info.get('retrieval_method', 'unknown') == 'mara_ppr'):
            try:
                print(f"\n{'='*80}")
                print(f"Creating animated PPR visualization (Exact Match = 1.0)...")
                print(f"{'='*80}")
                
                from experiment.online_retrieval.visualize_ppr_animated import create_animated_ppr_visualization
                from experiment.online_retrieval.visualize_ppr import load_gold_passages
                
                # Get PPR iterations if available
                ppr_iterations = debug_info.get('ppr_iterations', [])
                
                if len(ppr_iterations) == 0:
                    print("ERROR: No PPR iterations available for visualization!")
                else:
                    # Get gold passages for this question
                    dataset_path = Path(args.dataset_path)
                    musique_file = dataset_path / "musique.json"
                    gold_passages = load_gold_passages(musique_file, query)
                    
                    # Map gold passages to passage keys
                    gold_passage_keys = set()
                    for gold_passage in gold_passages:
                        for passage_key in ppr_engine.passage_node_keys:
                            try:
                                passage_row = hipporag.chunk_embedding_store.get_row(passage_key)
                                passage_content = passage_row.get('content', '')
                                if gold_passage.lower().strip() in passage_content.lower():
                                    gold_passage_keys.add(passage_key)
                                    break
                            except:
                                continue
                    
                    # Get seed information
                    seed_phrases = debug_info.get('seed_phrases', [])
                    seed_phrase_to_key = debug_info.get('seed_phrase_to_key', {})
                    
                    # Extract seed passages
                    seed_passages = []
                    passage_weights = debug_info.get('passage_weights', None)
                    if passage_weights is not None and isinstance(passage_weights, np.ndarray):
                        non_zero_node_indices = np.where(passage_weights > 0)[0]
                        if len(non_zero_node_indices) > 0:
                            sorted_node_indices = non_zero_node_indices[np.argsort(passage_weights[non_zero_node_indices])[::-1]]
                            top_node_indices = sorted_node_indices[:50]
                            node_idx_to_passage_idx = {node_idx: passage_idx 
                                                      for passage_idx, node_idx in enumerate(ppr_engine.passage_node_idxs)}
                            for node_idx in top_node_indices:
                                if node_idx in node_idx_to_passage_idx:
                                    passage_idx = node_idx_to_passage_idx[node_idx]
                                    if passage_idx < len(ppr_engine.passage_node_keys):
                                        passage_key = ppr_engine.passage_node_keys[passage_idx]
                                        seed_passages.append(passage_key)
                    
                    # Create visualization directory
                    viz_dir = output_dir / "visualizations"
                    viz_dir.mkdir(exist_ok=True)
                    viz_path = viz_dir / f"ppr_animated_query_{len(results):03d}.gif"
                    
                    # Convert doc_scores to list if needed
                    if isinstance(doc_scores, np.ndarray):
                        scores_list = doc_scores[:20].tolist() if len(doc_scores) >= 20 else doc_scores.tolist()
                    else:
                        scores_list = doc_scores[:20] if len(doc_scores) >= 20 else doc_scores
                    
                    create_animated_ppr_visualization(
                        graph=hipporag.graph,
                        passage_node_keys=ppr_engine.passage_node_keys,
                        passage_node_idxs=ppr_engine.passage_node_idxs,
                        gold_passage_keys=gold_passage_keys,
                        retrieved_passage_keys=retrieved_passage_keys[:20],
                        retrieved_scores=scores_list,
                        relation_weights=debug_info.get('relation_weights', {}),
                        seed_entities=seed_phrases,
                        seed_passages=seed_passages,
                        ppr_iterations=ppr_iterations,
                        output_path=viz_path,
                        max_nodes=200,
                        fps=2,
                        node_name_to_vertex_idx=hipporag.node_name_to_vertex_idx,
                        seed_phrase_to_key=seed_phrase_to_key
                    )
                    print(f"✓ Visualization saved for exact match question: {viz_path}")
            except Exception as e:
                print(f"ERROR creating visualization: {e}")
                import traceback
                traceback.print_exc()

    # Compute aggregate QA metrics
    f1_scores = [r['f1_score'] for r in results]
    em_scores = [r['em_score'] for r in results]
    aggregate_metrics = {
        'num_questions': len(results),
        'mean_f1': float(np.mean(f1_scores)),
        'median_f1': float(np.median(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
        'mean_em': float(np.mean(em_scores)),
    }

    # Compute retrieval recall if gold docs are available
    print("\nComputing retrieval metrics...")
    
    # Find musique.json file (contains gold passages with is_supporting=True)
    dataset_path = Path(args.dataset_path)
    musique_file = dataset_path / "musique.json"
    if not musique_file.exists():
        # Try alternative locations and filenames
        alt_paths = [
            dataset_path / "questions.json",
            dataset_path / "questions.jsonl",
            local_experiment_dir / "musique.json",
            experiment_dir / "musique.json",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                musique_file = alt_path
                break
    
    # Compute retrieval metrics using text matching approach
    if musique_file.exists():
        retrieval_metrics = compute_retrieval_metrics(
            results=results,
            questions=questions,
            musique_path=musique_file,
            k_list=[1, 2, 5, 10, 12, 20]
        )
        aggregate_metrics.update(retrieval_metrics)
        print(f"Retrieval metrics computed successfully")
    else:
        print(f"Warning: musique.json not found. Tried:")
        print(f"  - {dataset_path / 'musique.json'}")
        print(f"  - {dataset_path / 'questions.json'}")
        print("Skipping retrieval metrics")
        # Still add placeholder metrics so the structure is consistent
        for k in [1, 2, 5, 10, 12, 20]:
            aggregate_metrics[f'recall@{k}'] = 0.0

    # Print summary
    print("\n" + "="*80)
    print("MARA-RAG Experiment Results")
    print("="*80)
    print(f"Dataset: {args.experiment_name}")
    print(f"Questions: {len(results)}")
    print(f"\nAggregate Metrics:")
    for metric, value in aggregate_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Helper function to convert numpy arrays to lists for JSON serialization
    def convert_numpy_to_list(obj):
        """Recursively convert numpy arrays and other non-serializable types to JSON-compatible types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_list(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_to_list(item) for item in obj]
        else:
            return obj
    
    # Save results
    results_file = output_dir / "results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            # Convert numpy arrays to lists before JSON serialization
            result_serializable = convert_numpy_to_list(result)
            f.write(json.dumps(result_serializable) + '\n')
    print(f"\nSaved per-question results to: {results_file}")

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"Saved aggregate metrics to: {metrics_file}")
    logger.info(f"Saved aggregate metrics to: {metrics_file}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Experiment completed successfully!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"{'='*80}")
    print("\n✓ Experiment completed successfully!")
    print(f"Log file saved to: {log_file}")


if __name__ == "__main__":
    main()
