#!/usr/bin/env python3
"""
Visualization tool for Relation-Aware PPR algorithm.

This script creates visualizations showing:
1. How PPR explores the graph
2. Gold passages highlighted
3. Different relation types with different colors
4. PPR propagation paths
5. Node visit order and scores
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Set
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from collections import defaultdict


def load_gold_passages(musique_path: Path, question: str) -> List[str]:
    """Load gold passages for a specific question."""
    if not musique_path.exists():
        return []
    
    with open(musique_path, 'r') as f:
        data = json.load(f)
    
    for entry in data:
        if entry['question'] == question:
            return [
                para['paragraph_text'] 
                for para in entry.get('paragraphs', [])
                if para.get('is_supporting', False)
            ]
    return []


def create_ppr_visualization(
    graph,
    passage_node_keys: List[str],
    passage_node_idxs: List[int],
    gold_passage_keys: Set[str],
    retrieved_passage_keys: List[str],
    retrieved_scores: List[float],
    relation_weights: Dict[str, float],
    seed_entities: List[str],
    seed_passages: List[str],
    output_path: Path,
    max_nodes: int = 200
):
    """
    Create a visualization of the PPR graph exploration.
    
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
        output_path: Path to save visualization
        max_nodes: Maximum number of nodes to visualize
    """
    
    # Create NetworkX graph for visualization
    G = nx.Graph()
    
    # Color scheme for relation types
    relation_colors = {
        'hierarchical': '#FF6B6B',  # Red
        'temporal': '#4ECDC4',      # Teal
        'spatial': '#45B7D1',       # Blue
        'causality': '#FFA07A',     # Light Salmon
        'attribution': '#98D8C8',   # Mint
        'synonym': '#DDA0DD',       # Plum
        'passage': '#F0E68C',       # Khaki
        'other': '#D3D3D3'          # Light Gray
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
    for entity_name in seed_entities:
        for i, name in enumerate(node_names):
            if name.startswith('entity-') and entity_name.lower() in name.lower():
                seed_entity_indices.add(i)
                break
    
    # Find seed passage nodes
    for passage_key in seed_passages:
        if passage_key in passage_key_to_idx:
            idx = passage_key_to_idx[passage_key]
            node_idx = passage_node_idxs[idx]
            seed_passage_indices.add(node_idx)
    
    # Find gold passage nodes
    for passage_key in gold_passage_keys:
        if passage_key in passage_key_to_idx:
            idx = passage_key_to_idx[passage_key]
            node_idx = passage_node_idxs[idx]
            gold_passage_indices.add(node_idx)
    
    # Find retrieved passage nodes (top 20)
    for passage_key in retrieved_passage_keys[:20]:
        if passage_key in passage_key_to_idx:
            idx = passage_key_to_idx[passage_key]
            node_idx = passage_node_idxs[idx]
            retrieved_passage_indices.add(node_idx)
    
    # Add nodes and edges to NetworkX graph
    # Only add nodes that are relevant (seed, gold, retrieved, or connected to them)
    relevant_nodes = seed_entity_indices | seed_passage_indices | gold_passage_indices | retrieved_passage_indices
    
    # Add neighbors of relevant nodes (1-hop)
    for node_idx in list(relevant_nodes):
        neighbors = graph.neighbors(node_idx)
        relevant_nodes.update(neighbors)
    
    # Limit to max_nodes
    if len(relevant_nodes) > max_nodes:
        # Prioritize: seed > gold > retrieved > others
        priority_nodes = seed_entity_indices | seed_passage_indices | gold_passage_indices | retrieved_passage_indices
        other_nodes = relevant_nodes - priority_nodes
        keep_others = list(other_nodes)[:max_nodes - len(priority_nodes)]
        relevant_nodes = priority_nodes | set(keep_others)
    
    # Add nodes
    node_colors = []
    node_sizes = []
    node_labels = {}
    
    for node_idx in relevant_nodes:
        node_name = node_names[node_idx] if node_idx < len(node_names) else f"node_{node_idx}"
        G.add_node(node_idx, name=node_name)
        
        # Determine node color and size
        if node_idx in seed_entity_indices:
            color = '#FF0000'  # Red for seed entities
            size = 500
            label = f"SE:{node_name[:20]}"
        elif node_idx in seed_passage_indices:
            color = '#FFA500'  # Orange for seed passages
            size = 400
            label = f"SP:{node_name[:20]}"
        elif node_idx in gold_passage_indices:
            color = '#00FF00'  # Green for gold passages
            size = 600
            label = f"GOLD:{node_name[:20]}"
        elif node_idx in retrieved_passage_indices:
            color = '#0000FF'  # Blue for retrieved passages
            size = 300
            label = f"RET:{node_name[:20]}"
        elif node_name.startswith('entity-'):
            color = '#CCCCCC'  # Gray for other entities
            size = 100
            label = f"E:{node_name[:15]}"
        else:
            color = '#EEEEEE'  # Light gray for other passages
            size = 50
            label = f"P:{node_name[:15]}"
        
        node_colors.append(color)
        node_sizes.append(size)
        node_labels[node_idx] = label
    
    # Add edges with relation type colors
    edge_colors = []
    edge_widths = []
    
    for edge in graph.es:
        source = edge.source
        target = edge.target
        
        if source in relevant_nodes and target in relevant_nodes:
            G.add_edge(source, target)
            
            # Get relation type
            rel_type = edge.get('relation_type', 'other')
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
            
            edge_colors.append(color)
            
            # Edge width based on relation weight
            weight = edge.get('weight', 1.0)
            edge_widths.append(max(0.5, weight * 2))
    
    # Create visualization
    plt.figure(figsize=(20, 16))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.6,
        arrows=False
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8
    )
    
    # Draw labels (only for important nodes)
    important_nodes = seed_entity_indices | seed_passage_indices | gold_passage_indices | retrieved_passage_indices
    important_labels = {n: node_labels[n] for n in important_nodes if n in node_labels}
    nx.draw_networkx_labels(
        G, pos,
        labels=important_labels,
        font_size=8,
        font_weight='bold'
    )
    
    # Create legend
    legend_elements = [
        mpatches.Patch(color='#FF0000', label='Seed Entity'),
        mpatches.Patch(color='#FFA500', label='Seed Passage'),
        mpatches.Patch(color='#00FF00', label='Gold Passage'),
        mpatches.Patch(color='#0000FF', label='Retrieved Passage'),
        mpatches.Patch(color='#CCCCCC', label='Other Entity'),
        mpatches.Patch(color='#EEEEEE', label='Other Passage'),
    ]
    
    # Add relation type legend
    for rel_type, color in relation_colors.items():
        if rel_type in relation_weights or rel_type in ['synonym', 'passage']:
            weight = relation_weights.get(rel_type, 0.1 if rel_type in ['synonym', 'passage'] else 0.0)
            legend_elements.append(
                mpatches.Patch(color=color, label=f'{rel_type} (w={weight:.2f})')
            )
    
    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.title(
        f'PPR Graph Exploration\n'
        f'Seed Entities: {len(seed_entity_indices)}, '
        f'Seed Passages: {len(seed_passage_indices)}, '
        f'Gold Passages: {len(gold_passage_indices)}, '
        f'Retrieved: {len(retrieved_passage_indices)}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def create_ppr_path_visualization(
    graph,
    passage_node_keys: List[str],
    passage_node_idxs: List[int],
    gold_passage_keys: Set[str],
    retrieved_passage_keys: List[str],
    retrieved_scores: List[float],
    seed_entities: List[str],
    output_path: Path
):
    """
    Create a simplified path visualization showing how PPR propagates from seeds to gold passages.
    """
    
    # Build mapping
    passage_key_to_idx = {key: idx for idx, key in enumerate(passage_node_keys)}
    
    # Find seed entity nodes
    node_names = graph.vs['name'] if 'name' in graph.vs.attributes() else [f"node_{i}" for i in range(len(graph.vs))]
    seed_entity_indices = set()
    for entity_name in seed_entities:
        for i, name in enumerate(node_names):
            if name.startswith('entity-') and entity_name.lower() in name.lower():
                seed_entity_indices.add(i)
                break
    
    # Find gold passage nodes
    gold_passage_indices = set()
    for passage_key in gold_passage_keys:
        if passage_key in passage_key_to_idx:
            idx = passage_key_to_idx[passage_key]
            node_idx = passage_node_idxs[idx]
            gold_passage_indices.add(node_idx)
    
    # Find paths from seed entities to gold passages
    paths = []
    for seed_idx in list(seed_entity_indices)[:5]:  # Limit to 5 seeds
        for gold_idx in list(gold_passage_indices)[:3]:  # Limit to 3 gold passages
            try:
                path = graph.get_shortest_paths(seed_idx, gold_idx, mode='all')
                if path and len(path[0]) > 0:
                    paths.append(path[0])
            except:
                pass
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Plot paths
    for i, path in enumerate(paths[:10]):  # Limit to 10 paths
        if len(path) < 2:
            continue
        
        # Create simple 2D layout
        x_coords = np.linspace(0, 10, len(path))
        y_coords = np.array([i * 0.5] * len(path))
        
        # Color based on path length
        color = plt.cm.viridis(i / max(len(paths), 1))
        
        ax.plot(x_coords, y_coords, 'o-', color=color, linewidth=2, markersize=8, alpha=0.7)
        
        # Label start and end
        if len(path) > 0:
            ax.text(x_coords[0], y_coords[0], f'S{i}', fontsize=8, ha='right')
        if len(path) > 1:
            ax.text(x_coords[-1], y_coords[-1], f'G{i}', fontsize=8, ha='left')
    
    ax.set_title('PPR Propagation Paths: Seed Entities → Gold Passages', fontsize=14, fontweight='bold')
    ax.set_xlabel('Path Steps', fontsize=12)
    ax.set_ylabel('Path Index', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Path visualization saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    print("PPR Visualization Tool")
    print("This script should be called from run_mara_experiment.py with visualization data")





