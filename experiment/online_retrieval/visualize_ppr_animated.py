#!/usr/bin/env python3
"""
Animated visualization tool for Relation-Aware PPR algorithm.

This script creates an animated visualization showing:
1. How PPR explores the graph over iterations
2. Probability propagation through the graph
3. Gold passages highlighted
4. Different relation types with different colors
5. Node visit order and score evolution
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import networkx as nx
from collections import defaultdict


def create_animated_ppr_visualization(
    graph,
    passage_node_keys: List[str],
    passage_node_idxs: List[int],
    gold_passage_keys: Set[str],
    retrieved_passage_keys: List[str],
    retrieved_scores: List[float],
    relation_weights: Dict[str, float],
    seed_entities: List[str],
    seed_passages: List[str],
    ppr_iterations: List[np.ndarray],  # List of probability vectors for each iteration
    output_path: Path,
    max_nodes: int = 200,
    fps: int = 2,
    node_name_to_vertex_idx: Dict[str, int] = None,  # Optional: mapping from node name to vertex index
    seed_phrase_to_key: Dict[str, str] = None  # Optional: mapping from processed phrase to entity key
):
    """
    Create an animated visualization of the PPR graph exploration.
    
    Args:
        graph: igraph graph object
        passage_node_keys: List of passage node keys
        passage_node_idxs: List of passage node indices
        gold_passage_keys: Set of gold passage keys
        retrieved_passage_keys: List of retrieved passage keys (sorted by score)
        retrieved_scores: List of scores for retrieved passages
        relation_weights: Dictionary of relation type weights
        seed_entities: List of seed entity names
        seed_passages: List of seed passage keys
        ppr_iterations: List of probability vectors (one per iteration)
        output_path: Path to save animation (should be .gif or .mp4)
        max_nodes: Maximum number of nodes to visualize
        fps: Frames per second for animation
    """
    
    # Color scheme for relation types - more distinguishable colors
    relation_colors = {
        'hierarchical': '#FF0000',  # Bright Red
        'temporal': '#00FF00',      # Bright Green
        'spatial': '#0000FF',       # Bright Blue
        'causality': '#FF00FF',     # Magenta
        'attribution': '#00FFFF',   # Cyan
        'synonym': '#FFA500',       # Orange
        'passage': '#800080',       # Purple
        'other': '#808080'          # Gray
    }
    
    # Get node names
    node_names = graph.vs['name'] if 'name' in graph.vs.attributes() else [f"node_{i}" for i in range(len(graph.vs))]
    
    # Build mapping of passage keys to indices
    passage_key_to_idx = {key: idx for idx, key in enumerate(passage_node_keys)}
    
    # Identify important nodes
    seed_entity_indices = set()
    seed_passage_indices = set()
    gold_passage_indices = set()
    retrieved_passage_indices = set()
    
    # Find seed entity nodes
    # Try to match entity names to graph nodes
    # Entities in the graph are stored with keys like "entity-<hash>"
    # We'll try to match by computing the hash (most reliable method)
    from hipporag.utils.misc_utils import compute_mdhash_id
    
    print(f"DEBUG: Looking for {len(seed_entities)} seed entities in graph with {len(node_names)} nodes")
    print(f"DEBUG: Seed entity names: {seed_entities[:20]}")  # Show first 20
    
    matched_entities = []
    unmatched_entities = []
    
    for entity_name in seed_entities:
        if not entity_name or not isinstance(entity_name, str):
            continue
        entity_name_clean = entity_name.strip()
        if not entity_name_clean:
            continue
        
        matched = False
        matched_node_idx = None
        
        # Method 1: Use seed_phrase_to_key mapping if available (most reliable - uses exact keys from fact matching)
        if seed_phrase_to_key is not None and entity_name_clean in seed_phrase_to_key:
            entity_key = seed_phrase_to_key[entity_name_clean]
            if node_name_to_vertex_idx is not None and entity_key in node_name_to_vertex_idx:
                node_idx = node_name_to_vertex_idx[entity_key]
                if node_idx < len(node_names):
                    seed_entity_indices.add(node_idx)
                    matched = True
                    matched_node_idx = node_idx
                    print(f"  ✓ Matched '{entity_name_clean}' via seed_phrase_to_key -> key: {entity_key[:40]}... -> node {node_idx}")
        
        # Method 2: Try direct lookup in node_name_to_vertex_idx if available (fastest and most reliable)
        if not matched and node_name_to_vertex_idx is not None:
            try:
                entity_key = compute_mdhash_id(content=entity_name_clean, prefix="entity-")
                if entity_key in node_name_to_vertex_idx:
                    node_idx = node_name_to_vertex_idx[entity_key]
                    if node_idx < len(node_names):
                        seed_entity_indices.add(node_idx)
                        matched = True
                        matched_node_idx = node_idx
            except Exception as e:
                pass
        
        # Method 2: Try computing hash to match entity key (if Method 1 didn't work)
        if not matched:
            try:
                entity_key = compute_mdhash_id(content=entity_name_clean, prefix="entity-")
                # Search for this key in node names
                for i, name in enumerate(node_names):
                    if name == entity_key:
                        seed_entity_indices.add(i)
                        matched = True
                        matched_node_idx = i
                        break
            except Exception as e:
                pass
        
        # Method 3: Try substring matching on entity- nodes
        if not matched:
            entity_name_lower = entity_name_clean.lower()
            # Try exact match first
            for i, name in enumerate(node_names):
                if name.startswith('entity-'):
                    name_lower = name.lower()
                    # Check if entity name appears in node name (more flexible matching)
                    # Try: exact match, contains, or ends with
                    if (entity_name_lower == name_lower or 
                        entity_name_lower in name_lower or 
                        name_lower.endswith(entity_name_lower) or
                        name_lower.startswith(f"entity-{entity_name_lower}")):
                        seed_entity_indices.add(i)
                        matched = True
                        matched_node_idx = i
                        break
        
        # Method 4: Try matching with text_processing normalization
        if not matched:
            try:
                from hipporag.utils.misc_utils import text_processing
                entity_normalized = text_processing(entity_name_clean)
                entity_key_normalized = compute_mdhash_id(content=entity_normalized, prefix="entity-")
                for i, name in enumerate(node_names):
                    if name == entity_key_normalized:
                        seed_entity_indices.add(i)
                        matched = True
                        matched_node_idx = i
                        break
            except:
                pass
        
        if matched:
            matched_entities.append((entity_name_clean, matched_node_idx))
        else:
            unmatched_entities.append(entity_name_clean)
    
    print(f"DEBUG: Found {len(seed_entity_indices)} seed entity nodes")
    print(f"DEBUG: Matched entities ({len(matched_entities)}): {matched_entities[:10]}")
    if unmatched_entities:
        print(f"DEBUG: Unmatched entities ({len(unmatched_entities)}): {unmatched_entities[:10]}")
        print(f"DEBUG: Trying to find unmatched entities in node names...")
        # Try one more time with more aggressive matching
        for entity_name in unmatched_entities:
            entity_name_lower = entity_name.lower()
            # Search all entity nodes
            for i, name in enumerate(node_names):
                if name.startswith('entity-'):
                    name_lower = name.lower()
                    # Very flexible matching: check if any word from entity appears
                    entity_words = set(entity_name_lower.split())
                    name_words = set(name.replace('entity-', '').replace('-', ' ').split())
                    if entity_words and name_words and (entity_words & name_words):
                        seed_entity_indices.add(i)
                        print(f"  Matched '{entity_name}' to node {i} ({name}) via word overlap")
                        break
    
    # Find seed passage nodes
    # seed_passages should be passage keys (like "chunk-...")
    print(f"DEBUG: Looking for {len(seed_passages)} seed passages")
    
    for passage_key in seed_passages:
        if not passage_key:
            continue
        matched = False
        # Try direct key lookup
        if passage_key in passage_key_to_idx:
            idx = passage_key_to_idx[passage_key]
            if idx < len(passage_node_idxs):
                node_idx = passage_node_idxs[idx]
                seed_passage_indices.add(node_idx)
                matched = True
        else:
            # Try matching by searching node names
            for i, name in enumerate(node_names):
                if name == passage_key:
                    seed_passage_indices.add(i)
                    matched = True
                    break
                if passage_key in name and name.startswith('chunk-'):
                    seed_passage_indices.add(i)
                    matched = True
                    break
    
    print(f"DEBUG: Found {len(seed_passage_indices)} seed passage nodes")
    
    # Find gold passage nodes
    print(f"DEBUG: Looking for {len(gold_passage_keys)} gold passages")
    for passage_key in gold_passage_keys:
        if passage_key in passage_key_to_idx:
            idx = passage_key_to_idx[passage_key]
            if idx < len(passage_node_idxs):
                node_idx = passage_node_idxs[idx]
                gold_passage_indices.add(node_idx)
                print(f"  ✓ Found gold passage: {passage_key[:40]}... -> node {node_idx}")
        else:
            print(f"  ✗ Gold passage key not found in passage_key_to_idx: {passage_key[:40]}...")
            # Try to find it by searching node names
            for i, name in enumerate(node_names):
                if name == passage_key or (passage_key in name and name.startswith('chunk-')):
                    gold_passage_indices.add(i)
                    print(f"  ✓ Found gold passage via node name search: {passage_key[:40]}... -> node {i}")
                    break
    
    print(f"DEBUG: Found {len(gold_passage_indices)} gold passage nodes")
    
    # Find retrieved passage nodes (will be updated per iteration based on top probabilities)
    # Don't add them here - we'll compute top passages per iteration
    retrieved_passage_indices = set()  # Will be computed dynamically per iteration
    
    # Build relevant nodes set (seed, gold, and their neighbors)
    # IMPORTANT: Ensure ALL seed entities and passages are included
    # Start with ALL seed nodes (entities + passages) and gold nodes
    priority_nodes = seed_entity_indices | seed_passage_indices | gold_passage_indices
    relevant_nodes = priority_nodes.copy()
    
    # Add neighbors of priority nodes (1-hop) to show connections
    for node_idx in list(priority_nodes):
        if node_idx < len(graph.vs):
            neighbors = graph.neighbors(node_idx)
            relevant_nodes.update(neighbors)
    
    # Debug: Print seed node counts
    print(f"Visualization: Found {len(seed_entity_indices)} seed entities, {len(seed_passage_indices)} seed passages")
    print(f"  Total priority nodes (seed + gold): {len(priority_nodes)}")
    print(f"  Total relevant nodes (with neighbors): {len(relevant_nodes)}")
    if len(seed_entity_indices) > 0:
        print(f"  Seed entity indices (first 20): {sorted(list(seed_entity_indices))[:20]}")
    if len(seed_passage_indices) > 0:
        print(f"  Seed passage indices (first 20): {sorted(list(seed_passage_indices))[:20]}")
    
    # IMPORTANT: Always include ALL seed entities, passages, and gold passages, even if we exceed max_nodes
    # Increase max_nodes if needed to include all priority nodes
    required_nodes = len(seed_entity_indices) + len(seed_passage_indices) + len(gold_passage_indices)
    effective_max_nodes = max(max_nodes, required_nodes + 500)  # Add larger buffer for neighbors
    
    # Ensure ALL gold passages are in relevant_nodes (they might not have neighbors)
    relevant_nodes.update(gold_passage_indices)
    
    # Limit to effective_max_nodes, but ALWAYS keep ALL priority nodes (seed + gold)
    if len(relevant_nodes) > effective_max_nodes:
        other_nodes = relevant_nodes - priority_nodes
        # Keep only enough other nodes to fill remaining space
        keep_others = list(other_nodes)[:effective_max_nodes - len(priority_nodes)]
        relevant_nodes = priority_nodes | set(keep_others)
        print(f"  Limited to {len(relevant_nodes)} nodes (kept all {len(priority_nodes)} priority nodes including {len(gold_passage_indices)} gold)")
    else:
        print(f"  Using all {len(relevant_nodes)} relevant nodes (within limit, including {len(gold_passage_indices)} gold passages)")
    
    # Create NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes
    node_positions = {}
    for node_idx in relevant_nodes:
        node_name = node_names[node_idx] if node_idx < len(node_names) else f"node_{node_idx}"
        G.add_node(node_idx, name=node_name)
    
    # CRITICAL: Verify ALL gold passages are in G and add them if missing
    missing_gold = gold_passage_indices - set(G.nodes())
    if missing_gold:
        print(f"WARNING: {len(missing_gold)} gold passage nodes not in G, adding them...")
        for node_idx in missing_gold:
            if node_idx < len(node_names):
                node_name = node_names[node_idx] if node_idx < len(node_names) else f"node_{node_idx}"
                G.add_node(node_idx, name=node_name)
                # Add edges if they exist in the original graph
                if node_idx < len(graph.vs):
                    neighbors = graph.neighbors(node_idx)
                    for neighbor_idx in neighbors:
                        if neighbor_idx in G.nodes():
                            G.add_edge(node_idx, neighbor_idx)
    
    # Final verification
    gold_in_G = gold_passage_indices & set(G.nodes())
    print(f"DEBUG: Gold passages in G: {len(gold_in_G)}/{len(gold_passage_indices)}")
    if len(gold_in_G) < len(gold_passage_indices):
        print(f"WARNING: {len(gold_passage_indices) - len(gold_in_G)} gold passages still missing from G!")
        missing_final = gold_passage_indices - gold_in_G
        for node_idx in missing_final:
            print(f"  Missing gold node: {node_idx} (name: {node_names[node_idx] if node_idx < len(node_names) else 'N/A'})")
    
    # Add edges
    edge_colors_map = {}
    for edge in graph.es:
        source = edge.source
        target = edge.target
        
        if source in relevant_nodes and target in relevant_nodes:
            G.add_edge(source, target)
            
            # Get relation type and color
            # In igraph, edge attributes are accessed using dictionary syntax: edge['attribute_name']
            try:
                rel_type = edge['relation_type']
            except (KeyError, TypeError, AttributeError):
                rel_type = 'other'
            
            if isinstance(rel_type, str):
                rel_type = rel_type.upper()
                if 'HIERARCHICAL' in rel_type:
                    color = relation_colors['hierarchical']
                elif 'TEMPORAL' in rel_type:
                    color = relation_colors['temporal']
                elif 'SPATIAL' in rel_type:
                    color = relation_colors['spatial']
                elif 'CAUSAL' in rel_type:
                    color = relation_colors['causality']
                elif 'ATTRIBUTION' in rel_type:
                    color = relation_colors['attribution']
                elif 'SYNONYM' in rel_type:
                    color = relation_colors['synonym']
                elif 'PASSAGE' in rel_type or 'PRIMARY' in rel_type or 'SECONDARY' in rel_type:
                    color = relation_colors['passage']
                else:
                    color = relation_colors['other']
            else:
                color = relation_colors['other']
            
            edge_colors_map[(source, target)] = color
    
    # Create concentric circular layout: entities in inner ring, passages in outer ring
    print(f"Creating concentric circular layout for {len(G.nodes())} nodes...")
    
    # Separate nodes into entities and passages
    entity_nodes = [n for n in G.nodes() if n < len(node_names) and node_names[n].startswith('entity-')]
    passage_nodes = [n for n in G.nodes() if n not in entity_nodes]
    
    print(f"  Entity nodes: {len(entity_nodes)}, Passage nodes: {len(passage_nodes)}")
    
    # Create positions dictionary
    pos = {}
    
    # Inner ring for entities (smaller radius)
    inner_radius = 0.3
    outer_radius = 1.0
    if len(entity_nodes) > 0:
        inner_positions = nx.circular_layout(entity_nodes, scale=inner_radius, center=(0, 0))
        pos.update(inner_positions)
    
    # Outer ring for passages (larger radius)
    if len(passage_nodes) > 0:
        outer_positions = nx.circular_layout(passage_nodes, scale=outer_radius, center=(0, 0))
        pos.update(outer_positions)
    
    # Refine with spring layout to create force-directed layout while maintaining general structure
    # This will balance the graph while keeping entities more central and passages more peripheral
    pos = nx.spring_layout(G, pos=pos, k=1.5, iterations=100, seed=42)
    
    print(f"Layout created successfully (inner radius: {inner_radius if len(entity_nodes) > 0 else 'N/A'}, outer radius: {outer_radius if len(passage_nodes) > 0 else 'N/A'})")
    
    # Track edge visit counts across iterations (for relative visualization)
    edge_visit_counts = {}  # (source, target) -> cumulative visit count
    
    # Base node size (all nodes start at this size)
    base_node_size = 100
    
    # Track cumulative probability for nodes (for cumulative size scaling)
    cumulative_node_probs = {}  # node_idx -> cumulative probability sum across iterations
    
    # Store base layout for dynamic adjustments
    base_pos = pos.copy()
    
    # Create figure with space for legend
    fig, ax = plt.subplots(figsize=(22, 16))
    
    def render_iteration(iteration, ax_to_use):
        """Render a specific iteration on the given axes"""
        ax_to_use.clear()
        
        if iteration >= len(ppr_iterations):
            iteration = len(ppr_iterations) - 1
        
        prob_vec = ppr_iterations[iteration]
        prev_prob_vec = ppr_iterations[iteration - 1] if iteration > 0 else None
        
        # Determine which passages to show as "retrieved" (blue)
        # In final iteration, use actual retrieved_passage_keys
        # In other iterations, show top passages by current probability
        current_retrieved_passage_indices = set()
        is_final_iteration = (iteration == len(ppr_iterations) - 1)
        
        if is_final_iteration:
            # Final iteration: use actual retrieved passages
            print(f"DEBUG: Final iteration - looking for {len(retrieved_passage_keys)} retrieved passages")
            for passage_key in retrieved_passage_keys:
                if passage_key in passage_key_to_idx:
                    passage_idx = passage_key_to_idx[passage_key]
                    if passage_idx < len(passage_node_idxs):
                        node_idx = passage_node_idxs[passage_idx]
                        current_retrieved_passage_indices.add(node_idx)
                        print(f"  ✓ Found retrieved passage: {passage_key[:40]}... -> node {node_idx}")
                    else:
                        print(f"  ✗ Passage idx {passage_idx} out of range for passage_node_idxs (len={len(passage_node_idxs)})")
                else:
                    print(f"  ✗ Passage key not found in passage_key_to_idx: {passage_key[:40]}...")
            print(f"DEBUG: Final iteration - found {len(current_retrieved_passage_indices)} retrieved passage nodes")
        elif iteration > 0:
            # Other iterations: show top 5 passages by current probability
            top_n_passages_per_iter = 5
            passage_probs = []
            for idx in passage_node_idxs:
                if idx < len(prob_vec):
                    passage_probs.append((idx, prob_vec[idx]))
            
            # Sort by probability and take top N
            passage_probs.sort(key=lambda x: x[1], reverse=True)
            top_passage_indices = [idx for idx, prob in passage_probs[:top_n_passages_per_iter]]
            current_retrieved_passage_indices = set(top_passage_indices)
        
        # Calculate cumulative probability for nodes (for cumulative size scaling)
        # Track cumulative probability sum across all iterations to make changes more obvious
        for node_idx in relevant_nodes:
            if node_idx < len(prob_vec):
                current_prob = prob_vec[node_idx]
                if node_idx not in cumulative_node_probs:
                    cumulative_node_probs[node_idx] = 0.0
                # Add current probability to cumulative sum (with iteration-based scaling)
                # Later iterations get higher weight to make changes more obvious
                iteration_weight = 1.0 + (iteration * 0.2)  # Weight increases with iteration
                cumulative_node_probs[node_idx] += current_prob * iteration_weight
        
        # Calculate edge visit counts (cumulative across iterations)
        # Track how many times each edge has been traversed
        edge_flows = {}
        for (source, target) in G.edges():
            if source < len(prob_vec) and target < len(prob_vec):
                # Estimate flow as source probability
                source_prob = prob_vec[source]
                edge_key = (source, target)
                edge_flows[edge_key] = source_prob
                
                # Track cumulative visits (simplified: if flow > threshold, count as visit)
                if edge_key not in edge_visit_counts:
                    edge_visit_counts[edge_key] = 0
                if source_prob > 1e-6:  # Threshold for "visited"
                    edge_visit_counts[edge_key] += 1
            else:
                edge_flows[(source, target)] = 0.0
        
        # Normalize edge visit counts for color intensity (relative to max visits)
        max_visits = max(edge_visit_counts.values()) if edge_visit_counts else 1.0
        visit_norm = Normalize(vmin=0, vmax=max_visits) if max_visits > 0 else Normalize(vmin=0, vmax=1.0)
        
        # Normalize cumulative probabilities for node size (always increasing)
        max_cumulative = max(cumulative_node_probs.values()) if cumulative_node_probs else 1.0
        cumulative_norm = Normalize(vmin=0, vmax=max_cumulative) if max_cumulative > 0 else Normalize(vmin=0, vmax=1.0)
        
        # Create dynamic layout: adjust positions based on probability
        # High-probability nodes move toward center, low-probability nodes stay at edges
        dynamic_pos = {}
        center = (0.0, 0.0)
        for node_idx in relevant_nodes:
            base_x, base_y = base_pos.get(node_idx, (0.0, 0.0))
            if node_idx < len(prob_vec):
                prob = prob_vec[node_idx]
                # Normalize probability (0 to max_prob maps to 0 to 1)
                max_prob = prob_vec.max() if len(prob_vec) > 0 else 1.0
                prob_factor = prob / max_prob if max_prob > 0 else 0.0
                
                # Move high-probability nodes toward center
                # prob_factor: 0 (edge) to 1 (center)
                # Use exponential scaling to make high-prob nodes move more dramatically
                center_attraction = prob_factor ** 1.5  # Exponential scaling
                
                # Interpolate between base position and center
                new_x = base_x * (1.0 - center_attraction * 0.6) + center[0] * (center_attraction * 0.6)
                new_y = base_y * (1.0 - center_attraction * 0.6) + center[1] * (center_attraction * 0.6)
                dynamic_pos[node_idx] = (new_x, new_y)
            else:
                dynamic_pos[node_idx] = (base_x, base_y)
        
        # Use dynamic positions for this iteration (update pos variable)
        pos = dynamic_pos
        
        # Draw edges with colors based on relation type and visit count (relative)
        # Start with very light colors, darken only if visited multiple times
        edge_colors = []
        edge_widths = []
        edge_alphas = []
        for (source, target) in G.edges():
            base_color = edge_colors_map.get((source, target), relation_colors['other'])
            edge_key = (source, target)
            visit_count = edge_visit_counts.get(edge_key, 0)
            
            # Convert hex color to RGB
            import matplotlib.colors as mcolors
            rgb = mcolors.hex2color(base_color)
            
            # Change saturation only (not brightness/width) - keep stroke width constant
            # Start with very light saturation, increase based on visit count
            # Use cumulative scaling to make changes more obvious in later iterations
            base_saturation = 0.1 + 0.9 * visit_norm(visit_count) if max_visits > 0 else 0.1
            # Apply cumulative scaling: later iterations get more saturated more quickly (more exaggerated)
            cumulative_scale = 1.0 + (iteration * 0.4)  # Increased from 0.2 to 0.4 for more exaggeration
            saturation_factor = min(1.0, base_saturation * cumulative_scale)
            
            # Adjust saturation: mix base color with white based on saturation factor
            # Lower saturation = more white (lighter), higher saturation = more color (darker)
            # saturation_factor: 0.15 (very light) to 1.0 (full color)
            saturated_rgb = tuple(rgb[i] * saturation_factor + (1.0 - saturation_factor) * 0.9 for i in range(3))
            edge_colors.append(saturated_rgb)
            
            # Keep width constant (don't change stroke width)
            edge_widths.append(0.3)  # Fixed width
            # Keep alpha constant too (or make it slightly more opaque with visits)
            edge_alphas.append(0.3 + 0.4 * visit_norm(visit_count) if max_visits > 0 else 0.3)
        
        # Draw edges using dynamic positions
        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=edge_alphas,
            ax=ax_to_use,
            arrows=False
        )
        
        # Debug: verify retrieved passages are in relevant_nodes and G BEFORE drawing
        if is_final_iteration and current_retrieved_passage_indices:
            missing_from_relevant = current_retrieved_passage_indices - relevant_nodes
            if missing_from_relevant:
                print(f"WARNING: {len(missing_from_relevant)} retrieved passage nodes not in relevant_nodes, adding them...")
                relevant_nodes.update(missing_from_relevant)
            
            # Ensure all retrieved passages are in G (they might not be if they weren't in the initial relevant_nodes)
            missing_from_G = [idx for idx in current_retrieved_passage_indices if idx not in G.nodes()]
            if missing_from_G:
                print(f"WARNING: {len(missing_from_G)} retrieved passage nodes not in G, adding them...")
                for node_idx in missing_from_G:
                    if node_idx < len(node_names):
                        node_name = node_names[node_idx] if node_idx < len(node_names) else f"node_{node_idx}"
                        G.add_node(node_idx, name=node_name)
                        # Add edges if they exist in the original graph
                        if node_idx < len(graph.vs):
                            neighbors = graph.neighbors(node_idx)
                            for neighbor_idx in neighbors:
                                if neighbor_idx in G.nodes():
                                    G.add_edge(node_idx, neighbor_idx)
        
        # Draw nodes with fixed colors for node types, size based on probability
        # Use different shapes: triangles for retrieved passages, circles for others
        node_colors = []
        node_sizes = []
        node_shapes = []  # Track which nodes should be triangles
        
        for node_idx in relevant_nodes:
            # Fixed color based on node type (no probability blending)
            # CRITICAL: Check retrieved passages FIRST before other passage types
            if node_idx in current_retrieved_passage_indices:
                node_color = '#0000FF'  # Blue for retrieved passages
                node_shape = '^'  # Triangle (upward pointing)
            elif node_idx in seed_entity_indices:
                node_color = '#FF0000'  # Red for seed entities
                node_shape = 'o'  # Circle
            elif node_idx in seed_passage_indices:
                node_color = '#FFA500'  # Orange for seed passages
                node_shape = 'o'  # Circle
            elif node_idx in gold_passage_indices:
                node_color = '#00FF00'  # Green for gold passages
                node_shape = 'o'  # Circle
            elif node_idx < len(node_names) and node_names[node_idx].startswith('entity-'):
                node_color = '#CCCCCC'  # Gray for other entities
                node_shape = 'o'  # Circle
            else:
                node_color = '#D3D3D3'  # Light gray for other passages
                node_shape = 'o'  # Circle
            
            node_colors.append(node_color)
            node_shapes.append(node_shape)
            
            # Size based on CUMULATIVE probability (always increasing, never shrinking)
            # Use cumulative probability sum to make size changes more obvious
            if node_idx in cumulative_node_probs:
                cumulative_prob = cumulative_node_probs[node_idx]
                # Normalize cumulative probability (0 to max_cumulative maps to 0 to 1)
                normalized_cumulative = cumulative_norm(cumulative_prob) if max_cumulative > 0 else 0
                # Apply cumulative scaling factor to make changes more obvious in later iterations (more exaggerated)
                cumulative_scale = 1.0 + (iteration * 0.3)  # Increased from 0.15 to 0.3 for more exaggeration
                # Size grows from base_node_size to 5x base size (increased from 3x) based on cumulative probability
                # Use more aggressive exponential scaling
                size_scale = 1.0 + 4.0 * (normalized_cumulative ** 1.5) * cumulative_scale  # Increased from 2.0 and 1.2
                node_sizes.append(base_node_size * size_scale)
            else:
                # Node not in cumulative tracking yet, use base size
                node_sizes.append(base_node_size)
        
        # Draw nodes with different shapes
        # Separate nodes by shape (triangles vs circles) since matplotlib doesn't support mixed shapes in one call
        circle_nodes = []
        triangle_nodes = []
        circle_colors = []
        triangle_colors = []
        circle_sizes = []
        triangle_sizes = []
        circle_positions = {}
        triangle_positions = {}
        
        for i, node_idx in enumerate(relevant_nodes):
            if node_shapes[i] == '^':  # Triangle
                triangle_nodes.append(node_idx)
                triangle_colors.append(node_colors[i])
                triangle_sizes.append(node_sizes[i])
                triangle_positions[node_idx] = pos[node_idx]
            else:  # Circle
                circle_nodes.append(node_idx)
                circle_colors.append(node_colors[i])
                circle_sizes.append(node_sizes[i])
                circle_positions[node_idx] = pos[node_idx]
        
        # Draw circle nodes
        if circle_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=circle_nodes,
                node_color=circle_colors,
                node_size=circle_sizes,
                node_shape='o',
                alpha=0.8,
                ax=ax_to_use
            )
        
        # Draw triangle nodes
        if triangle_nodes:
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=triangle_nodes,
                node_color=triangle_colors,
                node_size=triangle_sizes,
                node_shape='^',
                alpha=0.8,
                ax=ax_to_use
            )
        
        # Debug: verify retrieved passages are in relevant_nodes and G
        if is_final_iteration and current_retrieved_passage_indices:
            missing_from_relevant = current_retrieved_passage_indices - relevant_nodes
            if missing_from_relevant:
                print(f"WARNING: {len(missing_from_relevant)} retrieved passage nodes not in relevant_nodes, adding them...")
                relevant_nodes.update(missing_from_relevant)
            
            # Ensure all retrieved passages are in G (they might not be if they weren't in the initial relevant_nodes)
            missing_from_G = [idx for idx in current_retrieved_passage_indices if idx not in G.nodes()]
            if missing_from_G:
                print(f"WARNING: {len(missing_from_G)} retrieved passage nodes not in G, adding them...")
                for node_idx in missing_from_G:
                    if node_idx < len(node_names):
                        node_name = node_names[node_idx] if node_idx < len(node_names) else f"node_{node_idx}"
                        G.add_node(node_idx, name=node_name)
                        # Add edges if they exist in the original graph
                        if node_idx < len(graph.vs):
                            neighbors = graph.neighbors(node_idx)
                            for neighbor_idx in neighbors:
                                if neighbor_idx in G.nodes():
                                    G.add_edge(node_idx, neighbor_idx)
        
        # Draw labels for important nodes
        important_nodes = seed_entity_indices | seed_passage_indices | gold_passage_indices | current_retrieved_passage_indices
        labels = {}
        for node_idx in important_nodes:
            if node_idx < len(node_names):
                name = node_names[node_idx]
                if node_idx in seed_entity_indices:
                    label = f"SE:{name[:15]}"
                elif node_idx in seed_passage_indices:
                    label = f"SP:{name[:15]}"
                elif node_idx in gold_passage_indices:
                    label = f"GOLD:{name[:15]}"
                elif node_idx in current_retrieved_passage_indices:
                    # In final iteration, label as "RET" (retrieved), otherwise "TOP"
                    if is_final_iteration:
                        label = f"RET:{name[:15]}"
                    else:
                        label = f"TOP:{name[:15]}"
                else:
                    label = name[:20]
                labels[node_idx] = label
        
        
        # Draw labels using dynamic positions
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=8,
            font_weight='bold',
            ax=ax_to_use
        )
        
        # Title with iteration info
        total_prob = prob_vec.sum()
        active_nodes = (prob_vec > 1e-6).sum()
        top_node_idx = np.argmax(prob_vec)
        top_prob = prob_vec[top_node_idx]
        
        ax_to_use.set_title(
            f'PPR Iteration {iteration} | '
            f'Total Prob: {total_prob:.4f} | '
            f'Active Nodes: {active_nodes} | '
            f'Top Node Prob: {top_prob:.6f}',
            fontsize=14,
            fontweight='bold'
        )
        
        # Add legend
        from matplotlib.patches import Patch, Circle
        from matplotlib.lines import Line2D
        
        legend_elements = [
            # Node type legend
            Patch(facecolor='#FF0000', label='Seed Entity (SE)', edgecolor='black', linewidth=1),
            Patch(facecolor='#FFA500', label='Seed Passage (SP)', edgecolor='black', linewidth=1),
            Patch(facecolor='#00FF00', label='Gold Passage (GOLD)', edgecolor='black', linewidth=1),
            Patch(facecolor='#0000FF', label='Retrieved Passage (RET)', edgecolor='black', linewidth=1),
            Patch(facecolor='#CCCCCC', label='Other Entity', edgecolor='black', linewidth=1),
            Patch(facecolor='#D3D3D3', label='Other Passage', edgecolor='black', linewidth=1),
            # Size legend (show small and large circles)
            Circle((0, 0), 0.5, facecolor='gray', label='Node Size = Probability', edgecolor='black'),
            # Edge type legend
            Line2D([0], [0], color=relation_colors['hierarchical'], lw=2, label='Hierarchical'),
            Line2D([0], [0], color=relation_colors['temporal'], lw=2, label='Temporal'),
            Line2D([0], [0], color=relation_colors['spatial'], lw=2, label='Spatial'),
            Line2D([0], [0], color=relation_colors['causality'], lw=2, label='Causality'),
            Line2D([0], [0], color=relation_colors['attribution'], lw=2, label='Attribution'),
            Line2D([0], [0], color=relation_colors['synonym'], lw=2, label='Synonym'),
            Line2D([0], [0], color=relation_colors['passage'], lw=2, label='Passage'),
            Line2D([0], [0], color=relation_colors['other'], lw=2, label='Other'),
            # Edge intensity legend
            Line2D([0], [0], color='black', lw=0.5, alpha=0.3, label='Edge Darkness = Flow'),
        ]
        
        # Place legend closer to the plot (moved from 1.02 to 0.98)
        ax_to_use.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.98, 1), 
                 fontsize=9, framealpha=0.9, title='Legend', title_fontsize=10)
        
        ax_to_use.axis('off')
    
    def animate(frame):
        """Animation function that maps frame to iteration with faster speed"""
        # Map frame to iteration - faster animation (fewer frames per iteration)
        frames_so_far = 0
        iteration = 0
        for iter_idx in range(len(ppr_iterations)):
            if iter_idx == 0:
                frames_for_this = 3  # 3 frames for iteration 0 (was 10)
            elif iter_idx < 3:
                frames_for_this = 2  # 2 frames for early iterations (was 5)
            elif iter_idx < 5:
                frames_for_this = 1  # 1 frame for mid iterations (was 3)
            else:
                frames_for_this = 1  # 1 frame for later iterations (unchanged)
            
            if frame < frames_so_far + frames_for_this:
                iteration = iter_idx
                break
            frames_so_far += frames_for_this
        
        iteration = min(iteration, len(ppr_iterations) - 1)
        # CRITICAL: Call render_iteration to actually update the plot
        # This will clear ax and redraw everything for this iteration
        render_iteration(iteration, ax)
        # Return list of artists for blitting (but we use blit=False, so this is just for compatibility)
        return []
    
    # Create animation with variable speed (slow at start, faster later)
    # Map frame to iteration with variable frames per iteration
    def get_iteration_for_frame(frame):
        """Map frame number to iteration, with more frames for early iterations"""
        frames_so_far = 0
        for iter_idx in range(len(ppr_iterations)):
            if iter_idx == 0:
                frames_for_this = 10  # 10 frames for iteration 0
            elif iter_idx < 3:
                frames_for_this = 5  # 5 frames for early iterations
            elif iter_idx < 5:
                frames_for_this = 3  # 3 frames for mid iterations
            else:
                frames_for_this = 1  # 1 frame for later iterations
            
            if frame < frames_so_far + frames_for_this:
                return iter_idx
            frames_so_far += frames_for_this
        return len(ppr_iterations) - 1
    
    # Calculate total frames (faster animation - fewer frames)
    total_frames = sum(
        3 if i == 0 else (2 if i < 3 else (1 if i < 5 else 1))
        for i in range(len(ppr_iterations))
    ) + 2
    
    # Variable interval - faster overall
    def get_interval_for_frame(frame):
        iteration = get_iteration_for_frame(frame)
        if iteration == 0:
            return 500  # 0.5 seconds for iteration 0 (was 1000)
        elif iteration < 3:
            return 400  # 0.4 seconds for early (was 800)
        elif iteration < 5:
            return 300  # 0.3 seconds for mid (was 600)
        else:
            return 200  # 0.2 seconds for later (was 400)
    
    # Create wrapper that uses variable speed
    def animate_with_speed(frame):
        iteration = get_iteration_for_frame(frame)
        # Update frame mapping in animate function
        frames_per_iteration = 3
        return animate(iteration * frames_per_iteration + min(iteration, frames_per_iteration - 1))
    
    # Create animation - use blit=False to ensure full redraw each frame
    # Use faster interval (500ms instead of 1000ms)
    print(f"Creating animation with {total_frames} frames for {len(ppr_iterations)} iterations...")
    anim = animation.FuncAnimation(
        fig, animate, frames=total_frames, interval=500, repeat=True, blit=False
    )
    
    # Save as GIF
    output_path = Path(output_path)
    if output_path.suffix.lower() != '.gif':
        output_path = output_path.with_suffix('.gif')
    
    print(f"Saving animated GIF to {output_path}...")
    try:
        # Use pillow writer for GIF with proper fps
        # Calculate actual fps based on interval (500ms = 2 fps)
        actual_fps = 2.0  # 2 frames per second (500ms interval)
        Writer = animation.writers['pillow']
        writer = Writer(fps=actual_fps, metadata=dict(artist='MARA-RAG'), bitrate=1800)
        anim.save(str(output_path), writer=writer)
        print(f"✓ Animation saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving animation with pillow writer: {e}")
        print("Trying alternative method...")
        try:
            # Try with explicit fps parameter
            anim.save(str(output_path), writer='pillow', fps=2.0)
            print(f"✓ Animation saved successfully to: {output_path}")
        except Exception as e2:
            print(f"ERROR: Failed to save animation: {e2}")
            import traceback
            traceback.print_exc()
            raise
    
    # Save static images of first and final iteration
    # IMPORTANT: Use the SAME graph G and layout pos to ensure consistency with GIF
    print(f"Saving static images...")
    static_output_dir = output_path.parent
    
    # First iteration - use same render_iteration function to ensure consistency
    # Reset edge visit counts for iteration 0 to show initial state
    edge_visit_counts_backup = edge_visit_counts.copy()
    edge_visit_counts.clear()
    
    fig_static, ax_static = plt.subplots(figsize=(22, 16))
    render_iteration(0, ax_static)  # Same function as animation uses
    first_iter_path = static_output_dir / f"{output_path.stem}_iteration_0.png"
    plt.savefig(first_iter_path, dpi=150, bbox_inches='tight')
    print(f"✓ First iteration saved to: {first_iter_path}")
    plt.close(fig_static)
    
    # Restore visit counts for final iteration
    edge_visit_counts.update(edge_visit_counts_backup)
    
    # Final iteration - use same render_iteration function
    # This will show retrieved passages in blue (is_final_iteration=True)
    fig_static, ax_static = plt.subplots(figsize=(22, 16))
    final_iteration = len(ppr_iterations) - 1
    render_iteration(final_iteration, ax_static)  # Same function as animation uses
    # Verify that retrieved passages are shown
    print(f"  Final iteration: showing {len(retrieved_passage_keys)} retrieved passages in blue")
    final_iter_path = static_output_dir / f"{output_path.stem}_iteration_final.png"
    plt.savefig(final_iter_path, dpi=150, bbox_inches='tight')
    print(f"✓ Final iteration saved to: {final_iter_path}")
    plt.close(fig_static)
    
    plt.close(fig)


if __name__ == "__main__":
    print("Animated PPR Visualization Tool")
    print("This script should be called from run_mara_experiment.py with PPR iteration data")

