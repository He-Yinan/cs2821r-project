#!/usr/bin/env python3
"""
Quick script to visualize the generated HippoRAG graph.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import defaultdict

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # Add project root for experiment module
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))  # Add rag/src for hipporag module

from experiment.common.io_utils import build_experiment_dir, load_json, write_json, SCRATCH_ROOT


def load_graph(graph_path: Path):
    """Load graph from pickle file."""
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    return ig.Graph.Read_Pickle(str(graph_path))


def print_graph_stats(graph: ig.Graph):
    """Print basic graph statistics."""
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)
    print(f"Number of nodes: {graph.vcount()}")
    print(f"Number of edges: {graph.ecount()}")
    print(f"Is directed: {graph.is_directed()}")
    
    if "name" in graph.vs.attribute_names():
        # Count node types
        entity_nodes = sum(1 for v in graph.vs if v["name"].startswith("entity-"))
        chunk_nodes = sum(1 for v in graph.vs if not v["name"].startswith("entity-"))
        print(f"  - Entity nodes: {entity_nodes}")
        print(f"  - Chunk/Passage nodes: {chunk_nodes}")
    
    if "weight" in graph.es.attribute_names():
        weights = graph.es["weight"]
        print(f"\nEdge weights:")
        print(f"  - Min: {min(weights):.4f}")
        print(f"  - Max: {max(weights):.4f}")
        print(f"  - Mean: {np.mean(weights):.4f}")
        print(f"  - Median: {np.median(weights):.4f}")
    
    # Degree statistics
    degrees = graph.degree()
    print(f"\nNode degrees:")
    print(f"  - Min: {min(degrees)}")
    print(f"  - Max: {max(degrees)}")
    print(f"  - Mean: {np.mean(degrees):.2f}")
    print(f"  - Median: {np.median(degrees):.2f}")
    
    print("=" * 60 + "\n")


def create_hierarchical_layout(graph: ig.Graph):
    """
    Create a hierarchical layout with entity nodes in the center and chunk/passage nodes on the outside.
    """
    # Separate nodes by type
    entity_indices = []
    chunk_indices = []
    
    if "name" in graph.vs.attribute_names():
        for i, v in enumerate(graph.vs):
            if v["name"].startswith("entity-"):
                entity_indices.append(i)
            else:
                chunk_indices.append(i)
    else:
        # If no name attribute, use all nodes as entities
        entity_indices = list(range(graph.vcount()))
    
    # Create layout coordinates
    layout = graph.layout("fr")  # Start with a standard layout
    
    # Get center of mass
    coords = np.array(layout.coords)
    center = coords.mean(axis=0)
    
    # Calculate distances from center
    distances = np.linalg.norm(coords - center, axis=1)
    max_dist = distances.max() if distances.max() > 0 else 1.0
    
    # Place entity nodes in center (within 60% of max distance)
    for i in entity_indices:
        if distances[i] > max_dist * 0.6:
            # Scale down to inner circle
            direction = (coords[i] - center) / (distances[i] + 1e-6)
            coords[i] = center + direction * max_dist * 0.6 * np.random.uniform(0.3, 0.6)
    
    # Place chunk nodes on the outside (beyond 70% of max distance)
    for i in chunk_indices:
        if distances[i] < max_dist * 0.7:
            # Scale up to outer ring
            direction = (coords[i] - center) / (distances[i] + 1e-6)
            coords[i] = center + direction * max_dist * np.random.uniform(0.8, 1.0)
    
    # Convert back to layout
    layout = ig.Layout(coords.tolist())
    return layout


def create_hierarchical_force_layout(graph: ig.Graph, iterations: int = 100):
    """
    Create a hierarchical force-directed layout with entity nodes in center and passage nodes on outside.
    Uses force-directed algorithm with constraints to maintain structure.
    """
    entity_indices = []
    chunk_indices = []
    
    if "name" in graph.vs.attribute_names():
        for i, v in enumerate(graph.vs):
            if v["name"].startswith("entity-"):
                entity_indices.append(i)
            else:
                chunk_indices.append(i)
    else:
        entity_indices = list(range(graph.vcount()))
    
    # Start with a force-directed layout
    print("  Computing initial force-directed layout...")
    try:
        layout = graph.layout("fr", niter=iterations, start_temp=10.0, grid=False)
    except:
        layout = graph.layout("kk", maxiter=iterations)
    
    coords = np.array(layout.coords)
    center = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center, axis=1)
    max_dist = distances.max() if distances.max() > 0 else 1.0
    
    # Separate entities and chunks more clearly
    entity_coords = coords[entity_indices]
    chunk_coords = coords[chunk_indices] if chunk_indices else np.array([])
    
    # Scale entity nodes to inner region (0-50% of max distance)
    if len(entity_coords) > 0:
        entity_distances = np.linalg.norm(entity_coords - center, axis=1)
        entity_max_dist = entity_distances.max() if entity_distances.max() > 0 else 1.0
        for i, idx in enumerate(entity_indices):
            if entity_max_dist > 0:
                direction = (entity_coords[i] - center) / (entity_distances[i] + 1e-6)
                # Scale to inner 50% of space
                new_radius = (entity_distances[i] / entity_max_dist) * max_dist * 0.5
                coords[idx] = center + direction * new_radius
    
    # Scale chunk nodes to outer region (60-100% of max distance)
    if len(chunk_coords) > 0:
        chunk_distances = np.linalg.norm(chunk_coords - center, axis=1)
        chunk_min_dist = chunk_distances.min() if chunk_distances.min() > 0 else 0
        chunk_max_dist = chunk_distances.max() if chunk_distances.max() > 0 else 1.0
        for i, idx in enumerate(chunk_indices):
            if chunk_max_dist > chunk_min_dist:
                direction = (chunk_coords[i] - center) / (chunk_distances[i] + 1e-6)
                # Map to outer 40% of space (60-100%)
                normalized = (chunk_distances[i] - chunk_min_dist) / (chunk_max_dist - chunk_min_dist + 1e-6)
                new_radius = max_dist * (0.6 + normalized * 0.4)
                coords[idx] = center + direction * new_radius
            else:
                # If all chunks are at same distance, place them in outer ring
                direction = (chunk_coords[i] - center) / (chunk_distances[i] + 1e-6)
                coords[idx] = center + direction * max_dist * 0.8
    
    return ig.Layout(coords.tolist())


def visualize_graph_unified(graph: ig.Graph, output_path: Path, max_nodes: int = 500, min_edge_weight: float = None):
    """
    Create unified visualization with all edges, passages outside, entities inside.
    
    Args:
        graph: The graph to visualize
        output_path: Where to save the image
        max_nodes: Maximum number of nodes to show (None or 0 = show all nodes)
        min_edge_weight: Optional minimum edge weight threshold (only show edges above this)
    """
    # Handle "show all nodes" case
    if max_nodes is None or max_nodes <= 0:
        max_nodes = graph.vcount()
        print(f"Creating unified visualization (showing ALL {max_nodes} nodes)...")
    else:
        print(f"Creating unified visualization (showing up to {max_nodes} nodes)...")
    
    if min_edge_weight is not None:
        print(f"  Filtering edges: only showing edges with weight >= {min_edge_weight}")
    
    # If graph is too large, sample a subgraph
    if max_nodes < graph.vcount():
        print(f"Graph has {graph.vcount()} nodes, sampling {max_nodes} nodes...")
        degrees = graph.degree()
        top_nodes = sorted(range(graph.vcount()), key=lambda i: degrees[i], reverse=True)[:max_nodes]
        subgraph = graph.subgraph(top_nodes)
        print(f"Created subgraph with {subgraph.vcount()} nodes, {subgraph.ecount()} edges")
    else:
        subgraph = graph
        print(f"Showing all {subgraph.vcount()} nodes, {subgraph.ecount()} edges")
    
    # Create hierarchical force-directed layout
    print("Computing hierarchical force-directed layout (entities inside, passages outside)...")
    layout = create_hierarchical_force_layout(subgraph, iterations=150)
    
    # Separate nodes by type
    entity_indices = []
    chunk_indices = []
    node_names = []
    
    if "name" in subgraph.vs.attribute_names():
        for i, v in enumerate(subgraph.vs):
            node_names.append(v["name"])
            if v["name"].startswith("entity-"):
                entity_indices.append(i)
            else:
                chunk_indices.append(i)
    else:
        node_names = [f"node_{i}" for i in range(subgraph.vcount())]
        entity_indices = list(range(subgraph.vcount()))
    
    # Color nodes by type
    colors = []
    for i in range(subgraph.vcount()):
        if i in entity_indices:
            colors.append("#FF6B6B")  # Red for entities
        else:
            colors.append("#4ECDC4")  # Teal for chunks
    
    # Size nodes by degree
    degrees = subgraph.degree()
    node_sizes = [max(20, min(150, d * 5 + 30)) for d in degrees]
    
    # Get edge weights and relation types
    # Edge colors in unified graph represent RELATION TYPES:
    # - Each relation type has a distinct color
    # - Colors are different from node colors (entity=red, passage=teal)
    edge_widths = []
    edge_colors = []
    
    if "weight" in subgraph.es.attribute_names() and "relation_type" in subgraph.es.attribute_names():
        weights = subgraph.es["weight"]
        relation_types = subgraph.es["relation_type"]
        
        min_weight = min(weights) if weights else 0.1
        max_weight = max(weights) if weights else 1.0
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        
        # Color mapping for relation types (distinct from node colors: entity=#FF6B6B, passage=#4ECDC4)
        color_map = {
            "HIERARCHICAL": "#E74C3C",  # Dark red (different from entity red)
            "TEMPORAL": "#3498DB",      # Blue (different from passage teal)
            "SPATIAL": "#2ECC71",       # Green
            "CAUSALITY": "#9B59B6",     # Purple
            "ATTRIBUTION": "#F39C12",   # Orange
            "SYNONYMY": "#FFD93D",      # Yellow
            "PRIMARY": "#E67E22",       # Dark orange
            "SECONDARY": "#16A085",     # Dark teal (different from passage teal)
            "PERIPHERAL": "#95A5A6",    # Gray
        }
        
        for w, rt in zip(weights, relation_types):
            # Normalize edge width (0.5 to 4.0) based on weight
            if weight_range > 0:
                normalized = (w - min_weight) / weight_range
                edge_widths.append(0.5 + normalized * 3.5)
            else:
                edge_widths.append(2.0)
            
            # Color by relation type - each relation type has a unique color
            edge_colors.append(color_map.get(str(rt), "#CCCCCC"))
    else:
        edge_widths = [1.0] * subgraph.ecount()
        edge_colors = ["#CCCCCC"] * subgraph.ecount()
    
    # Plot
    fig, ax = plt.subplots(figsize=(20, 20))
    
    # Plot edges with adaptive opacity and optional filtering
    edges_to_plot = []
    if min_edge_weight is not None and "weight" in subgraph.es.attribute_names():
        weights = subgraph.es["weight"]
        for edge in subgraph.es:
            if weights[edge.index] >= min_edge_weight:
                edges_to_plot.append(edge)
        print(f"  Showing {len(edges_to_plot)} of {subgraph.ecount()} edges (weight >= {min_edge_weight})")
    else:
        edges_to_plot = list(subgraph.es)
    
    # Plot edges with varying opacity based on weight
    for edge in edges_to_plot:
        source = edge.source
        target = edge.target
        x_coords = [layout[source][0], layout[target][0]]
        y_coords = [layout[source][1], layout[target][1]]
        
        edge_idx = edge.index
        # Adaptive alpha: higher weight = more visible
        # This helps reduce visual clutter while keeping important edges visible
        base_alpha = 0.12
        if edge_widths[edge_idx] > 3.0:
            edge_alpha = min(0.5, base_alpha + 0.3)
        elif edge_widths[edge_idx] > 2.0:
            edge_alpha = min(0.35, base_alpha + 0.2)
        elif edge_widths[edge_idx] > 1.0:
            edge_alpha = base_alpha + 0.1
        else:
            edge_alpha = base_alpha
        
        ax.plot(x_coords, y_coords, 
                color=edge_colors[edge_idx], 
                linewidth=edge_widths[edge_idx],
                alpha=edge_alpha, 
                zorder=1,
                solid_capstyle='round')
    
    # Plot nodes
    for i in range(subgraph.vcount()):
        ax.scatter(layout[i][0], layout[i][1], 
                  s=node_sizes[i], c=colors[i], 
                  edgecolors='black', linewidths=1.0,
                  alpha=0.9, zorder=2)
    
    # Add legend with node types and edge color explanation
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', label='Entity nodes (inner)'),
        mpatches.Patch(facecolor='#4ECDC4', label='Passage nodes (outer)'),
        mpatches.Circle((0, 0), 1, facecolor='none', edgecolor='black', linewidth=1, label='Node size = Degree'),
    ]
    
    # Add edge color legend if relation types are available
    if "relation_type" in subgraph.es.attribute_names():
        # Count relation types for legend
        from collections import Counter
        relation_types = subgraph.es["relation_type"]
        rt_counts = Counter(relation_types)
        
        # Add most common relation types to legend (top 5)
        edge_legend_items = []
        for rt, count in rt_counts.most_common(5):
            if rt in color_map:
                edge_legend_items.append(
                    mpatches.Patch(facecolor=color_map[rt], label=f'{rt} ({count})', alpha=0.7)
                )
        
        if edge_legend_items:
            legend_elements.extend(edge_legend_items)
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=1)
    
    title = f"HippoRAG Graph - Unified View\n({subgraph.vcount()} nodes, {subgraph.ecount()} edges)\n"
    title += "Edge color = Relation type | Edge thickness = Edge weight | Lower opacity = Lower weight"
    ax.set_title(title, fontsize=14, pad=20)
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Unified visualization saved to: {output_path}")
    plt.close()


def visualize_graph_by_relation_type(graph: ig.Graph, relation_type: str, output_path: Path, max_nodes: int = 500):
    """
    Visualize graph filtered by a specific relation type.
    
    Args:
        graph: The graph to visualize
        relation_type: The relation type to filter by
        output_path: Where to save the image
        max_nodes: Maximum number of nodes to show (None or 0 = show all nodes)
    """
    """
    Create visualization for a specific relation type only.
    """
    print(f"Creating visualization for {relation_type} relations...")
    
    # Filter edges by relation type
    if "relation_type" not in graph.es.attribute_names():
        print(f"Warning: No relation_type attribute found. Skipping {relation_type} plot.")
        return
    
    # Get edges of this relation type
    matching_edges = []
    matching_edge_indices = []
    for i, edge in enumerate(graph.es):
        edge_rt = edge["relation_type"] if "relation_type" in edge.attributes() else ""
        if str(edge_rt) == relation_type:
            matching_edges.append(edge)
            matching_edge_indices.append(i)
    
    if len(matching_edges) == 0:
        print(f"No edges found for relation type: {relation_type}")
        return
    
    # Get all nodes connected by these edges
    connected_nodes = set()
    for edge in matching_edges:
        connected_nodes.add(edge.source)
        connected_nodes.add(edge.target)
    
    # Create subgraph with only these nodes and edges
    node_list = sorted(list(connected_nodes))
    # Handle "show all nodes" case
    if max_nodes is None or max_nodes <= 0:
        max_nodes = len(node_list)
    
    if len(node_list) > max_nodes:
        # Sample by degree
        degrees = graph.degree()
        node_list = sorted(node_list, key=lambda i: degrees[i], reverse=True)[:max_nodes]
        print(f"  Sampling {max_nodes} nodes from {len(connected_nodes)} connected nodes...")
    else:
        print(f"  Showing all {len(node_list)} connected nodes...")
        connected_nodes = set(node_list)
    
    # Filter edges to only those between selected nodes
    filtered_edges = [e for e in matching_edges 
                     if e.source in connected_nodes and e.target in connected_nodes]
    
    if len(filtered_edges) == 0:
        print(f"No edges after filtering for {relation_type}")
        return
    
    # Create subgraph
    subgraph = graph.subgraph(node_list)
    
    # Filter edges in subgraph to only show the relation type
    # Note: igraph subgraph keeps all edges, so we'll filter during plotting
    
    # Create layout - use same hierarchical layout as unified graph
    layout = create_hierarchical_force_layout(subgraph, iterations=150)
    
    # Separate nodes
    entity_indices = []
    chunk_indices = []
    if "name" in subgraph.vs.attribute_names():
        for i, v in enumerate(subgraph.vs):
            if v["name"].startswith("entity-"):
                entity_indices.append(i)
            else:
                chunk_indices.append(i)
    else:
        entity_indices = list(range(subgraph.vcount()))
    
    # Colors
    colors = []
    for i in range(subgraph.vcount()):
        if i in entity_indices:
            colors.append("#FF6B6B")
        else:
            colors.append("#4ECDC4")
    
    # Sizes
    degrees = subgraph.degree()
    node_sizes = [max(20, min(150, d * 5 + 30)) for d in degrees]
    
    # Use a distinct color for the specific relation type being visualized
    # (different from entity=#FF6B6B and passage=#4ECDC4)
    relation_color_map = {
        "HIERARCHICAL": "#E74C3C",  # Dark red
        "TEMPORAL": "#3498DB",      # Blue
        "SPATIAL": "#2ECC71",       # Green
        "CAUSALITY": "#9B59B6",     # Purple
        "ATTRIBUTION": "#F39C12",   # Orange
    }
    
    # Get the color for this specific relation type
    edge_color = relation_color_map.get(relation_type, "#34495E")  # Dark gray as fallback
    
    # Edge properties
    edge_widths = []
    edge_colors = []
    
    # Get weights for matching edges
    if "weight" in subgraph.es.attribute_names():
        weights = subgraph.es["weight"]
        min_weight = min(weights) if weights else 0.1
        max_weight = max(weights) if weights else 1.0
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        
        for edge in subgraph.es:
            # Only show edges of the target relation type
            edge_rt = edge["relation_type"] if "relation_type" in edge.attributes() else ""
            if str(edge_rt) == relation_type:
                w = edge["weight"] if "weight" in edge.attributes() else 1.0
                if weight_range > 0:
                    normalized = (w - min_weight) / weight_range
                    edge_widths.append(0.5 + normalized * 4.0)
                else:
                    edge_widths.append(2.0)
                edge_colors.append(edge_color)
            else:
                edge_widths.append(0)  # Hide this edge
                edge_colors.append("none")
    else:
        for edge in subgraph.es:
            edge_rt = edge["relation_type"] if "relation_type" in edge.attributes() else ""
            if str(edge_rt) == relation_type:
                edge_widths.append(2.0)
                edge_colors.append(edge_color)
            else:
                edge_widths.append(0)
                edge_colors.append("none")
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Plot edges - only show edges of the target relation type
    for edge in subgraph.es:
        edge_rt = edge["relation_type"] if "relation_type" in edge.attributes() else ""
        if str(edge_rt) != relation_type:
            continue
        source = edge.source
        target = edge.target
        x_coords = [layout[source][0], layout[target][0]]
        y_coords = [layout[source][1], layout[target][1]]
        
        edge_idx = edge.index
        if edge_widths[edge_idx] > 0:
            # Use higher alpha for relation-specific plots since we're only showing one type
            ax.plot(x_coords, y_coords, 
                   color=edge_colors[edge_idx], 
                   linewidth=edge_widths[edge_idx],
                   alpha=0.6, 
                   zorder=1,
                   solid_capstyle='round')
    
    # Plot nodes with better layering
    entity_x = [layout[i][0] for i in entity_indices]
    entity_y = [layout[i][1] for i in entity_indices]
    entity_sizes = [node_sizes[i] for i in entity_indices]
    entity_colors_list = [colors[i] for i in entity_indices]
    
    chunk_x = [layout[i][0] for i in chunk_indices]
    chunk_y = [layout[i][1] for i in chunk_indices]
    chunk_sizes = [node_sizes[i] for i in chunk_indices]
    chunk_colors_list = [colors[i] for i in chunk_indices]
    
    # Plot passage nodes first (lower zorder)
    if chunk_x:
        ax.scatter(chunk_x, chunk_y, 
                  s=chunk_sizes, c=chunk_colors_list, 
                  edgecolors='black', linewidths=0.8,
                  alpha=0.95, zorder=2)
    
    # Plot entity nodes on top
    if entity_x:
        ax.scatter(entity_x, entity_y, 
                  s=entity_sizes, c=entity_colors_list, 
                  edgecolors='black', linewidths=1.0,
                  alpha=1.0, zorder=3)
    
    # Legend - use distinct colors
    legend_elements = [
        mpatches.Patch(facecolor='#FF6B6B', label='Entity nodes (inner)'),
        mpatches.Patch(facecolor='#4ECDC4', label='Passage nodes (outer)'),
        mpatches.Patch(facecolor=edge_color, label=f'{relation_type} edges', alpha=0.7),
        mpatches.Circle((0, 0), 1, facecolor='none', edgecolor='black', linewidth=1, label='Node size = Degree'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    ax.set_title(f"HippoRAG Graph - {relation_type} Relations Only\n"
                 f"({subgraph.vcount()} nodes, {len(filtered_edges)} edges)\n"
                 f"Edge thickness = Edge weight", fontsize=14, pad=20)
    ax.axis('off')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"{relation_type} visualization saved to: {output_path}")
    plt.close()


def visualize_graph_enhanced(graph: ig.Graph, output_dir: Path, max_nodes: int = 500, min_edge_weight: float = None):
    """
    Create unified visualization and separate plots for each relation type.
    
    Args:
        graph: The graph to visualize
        output_dir: Directory to save visualizations
        max_nodes: Maximum number of nodes to show
        min_edge_weight: Optional minimum edge weight threshold for unified view
    """
    # Create unified plot (full view)
    visualize_graph_unified(graph, output_dir / "graph_unified.png", max_nodes=max_nodes, min_edge_weight=min_edge_weight)
    
    # Create simplified unified plot (only high-weight edges) if no threshold specified
    if min_edge_weight is None:
        # Auto-determine threshold: use median weight
        if "weight" in graph.es.attribute_names():
            weights = graph.es["weight"]
            median_weight = np.median(weights)
            print(f"\nCreating simplified view with edges above median weight ({median_weight:.3f})...")
            visualize_graph_unified(graph, output_dir / "graph_unified_simplified.png", 
                                  max_nodes=max_nodes, min_edge_weight=median_weight)
    
    # Create separate plots for each relation type
    relation_types = ["HIERARCHICAL", "TEMPORAL", "SPATIAL", "CAUSALITY", "ATTRIBUTION"]
    for rt in relation_types:
        visualize_graph_by_relation_type(
            graph, 
            rt, 
            output_dir / f"graph_{rt.lower()}.png", 
            max_nodes=max_nodes
        )


def generate_node_edge_samples(graph: ig.Graph, log_path: Path):
    """Generate sample output file with example nodes and edges."""
    print(f"Generating sample node/edge log to: {log_path}")
    
    degrees = graph.degree()
    node_names = [v["name"] if "name" in v.attributes() else f"node_{i}" for i, v in enumerate(graph.vs)]
    
    # Find nodes with 0 degree
    zero_degree_nodes = [(i, node_names[i]) for i, d in enumerate(degrees) if d == 0]
    
    # Find nodes with max degree
    max_degree = max(degrees) if degrees else 0
    max_degree_nodes = [(i, node_names[i], degrees[i]) for i, d in enumerate(degrees) if d == max_degree]
    
    # Sample edges with different weights
    edge_samples = []
    if graph.ecount() > 0:
        if "weight" in graph.es.attribute_names() and "relation_type" in graph.es.attribute_names():
            weights = graph.es["weight"]
            relation_types = graph.es["relation_type"]
            # Get edges with highest weights
            edge_data = [(i, w, rt) for i, (w, rt) in enumerate(zip(weights, relation_types))]
            edge_data.sort(key=lambda x: x[1], reverse=True)
            
            for edge_idx, weight, relation_type in edge_data[:10]:  # Top 10 edges
                edge = graph.es[edge_idx]
                source_name = node_names[edge.source]
                target_name = node_names[edge.target]
                edge_samples.append({
                    "source": source_name,
                    "target": target_name,
                    "weight": float(weight),
                    "relation_type": str(relation_type) if relation_type else "N/A"
                })
        elif "weight" in graph.es.attribute_names():
            weights = graph.es["weight"]
            edge_weight_pairs = [(i, w) for i, w in enumerate(weights)]
            edge_weight_pairs.sort(key=lambda x: x[1], reverse=True)
            
            for edge_idx, weight in edge_weight_pairs[:10]:
                edge = graph.es[edge_idx]
                source_name = node_names[edge.source]
                target_name = node_names[edge.target]
                edge_samples.append({
                    "source": source_name,
                    "target": target_name,
                    "weight": float(weight),
                    "relation_type": "N/A"
                })
    
    # Write log file
    with log_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("HIPPORAG GRAPH SAMPLE ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("NODE SIZE EXPLANATION:\n")
        f.write("-" * 80 + "\n")
        f.write("The size of entity nodes in the visualization represents their DEGREE,\n")
        f.write("which is the number of connections (edges) each node has.\n")
        f.write("Larger nodes = more connections = more central/important in the graph.\n")
        f.write("Smaller nodes = fewer connections = more peripheral/isolated.\n\n")
        
        f.write("NODES WITH 0 DEGREE (Isolated nodes - no connections):\n")
        f.write("-" * 80 + "\n")
        if zero_degree_nodes:
            for idx, (i, name) in enumerate(zero_degree_nodes[:5], 1):
                node_type = "Entity" if name.startswith("entity-") else "Chunk/Passage"
                f.write(f"{idx}. {name} ({node_type})\n")
            if len(zero_degree_nodes) > 5:
                f.write(f"\n... and {len(zero_degree_nodes) - 5} more nodes with 0 degree\n")
        else:
            f.write("No nodes with 0 degree found.\n")
        f.write(f"\nTotal nodes with 0 degree: {len(zero_degree_nodes)}\n\n")
        
        f.write("NODES WITH MAXIMUM DEGREE (Most connected nodes):\n")
        f.write("-" * 80 + "\n")
        if max_degree_nodes:
            f.write(f"Maximum degree: {max_degree}\n")
            f.write(f"Number of nodes with max degree: {len(max_degree_nodes)}\n\n")
            for idx, (i, name, deg) in enumerate(max_degree_nodes[:10], 1):
                node_type = "Entity" if name.startswith("entity-") else "Chunk/Passage"
                f.write(f"{idx}. {name} ({node_type}) - Degree: {deg}\n")
        else:
            f.write("No nodes found.\n")
        f.write("\n")
        
        f.write("SAMPLE EDGES (Top 10 by weight):\n")
        f.write("-" * 80 + "\n")
        if edge_samples:
            for idx, edge in enumerate(edge_samples, 1):
                f.write(f"{idx}. {edge['source']} -> {edge['target']}\n")
                f.write(f"   Weight: {edge['weight']:.4f}, Relation Type: {edge['relation_type']}\n\n")
        else:
            f.write("No edge samples available.\n")
        
        # Count edges by relation type
        f.write("\nEDGE COUNTS BY RELATION TYPE:\n")
        f.write("-" * 80 + "\n")
        if "relation_type" in graph.es.attribute_names():
            relation_counts = defaultdict(int)
            for edge in graph.es:
                rt = str(edge["relation_type"]) if "relation_type" in edge.attributes() else "N/A"
                relation_counts[rt] += 1
            for rt, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{rt}: {count} edges\n")
        else:
            f.write("Relation type information not available.\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"Sample log saved to: {log_path}")


def visualize_degree_distribution(graph: ig.Graph, output_path: Path):
    """Plot degree distribution."""
    degrees = graph.degree()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    ax1.hist(degrees, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Degree Distribution (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log-log plot
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    ax2.loglog(unique_degrees, counts, 'o', markersize=4)
    ax2.set_xlabel('Degree (log scale)')
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Degree Distribution (Log-Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Degree distribution plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize HippoRAG graph")
    parser.add_argument(
        "--experiment-name",
        default="musique_demo",
        help="Experiment name",
    )
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="Workspace subdirectory",
    )
    parser.add_argument(
        "--llm-model-name",
        default="Qwen/Qwen3-8B",
        help="LLM model name (for constructing path)",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="facebook/contriever-msmarco",
        help="Embedding model name (for constructing path)",
    )
    parser.add_argument(
        "--graph-path",
        type=Path,
        default=None,
        help="Direct path to graph.pickle file (overrides other args)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for visualizations (default: experiment directory)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=500,
        help="Maximum nodes to show in visualization (default: 500, use 0 or negative to show all nodes)",
    )
    parser.add_argument(
        "--min-edge-weight",
        type=float,
        default=None,
        help="Minimum edge weight threshold for unified view (default: None, shows all edges)",
    )
    
    args = parser.parse_args()
    
    # Determine graph path
    if args.graph_path:
        graph_path = Path(args.graph_path)
    else:
        workspace_dir = build_experiment_dir(args.experiment_name, args.workspace_subdir)
        llm_label = args.llm_model_name.replace("/", "_")
        embed_label = args.embedding_model_name.replace("/", "_")
        working_dir = workspace_dir / f"{llm_label}_{embed_label}"
        graph_path = working_dir / "graph.pickle"
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = build_experiment_dir(args.experiment_name, "visualizations")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load graph
    print(f"Loading graph from: {graph_path}")
    graph = load_graph(graph_path)
    
    # Print statistics
    print_graph_stats(graph)
    
    # Generate sample node/edge log
    log_path = output_dir / "graph_samples.log"
    generate_node_edge_samples(graph, log_path)
    
    # Create visualizations
    visualize_graph_enhanced(graph, output_dir, max_nodes=args.max_nodes, min_edge_weight=args.min_edge_weight)
    visualize_degree_distribution(graph, output_dir / "degree_distribution.png")
    
    print(f"\nAll visualizations and logs saved to: {output_dir}")


if __name__ == "__main__":
    main()

