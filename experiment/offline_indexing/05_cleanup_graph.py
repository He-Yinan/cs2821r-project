#!/usr/bin/env python3
"""
Step 5: Clean up graph by removing nodes with zero degree (isolated nodes).
These nodes indicate parsing issues or nodes that weren't properly connected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add project root and HippoRAG src to Python path so we can import experiment + hipporag modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

import igraph as ig

from experiment.common.io_utils import build_experiment_dir


def parse_args(argv: sys.argv | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean up HippoRAG graph by removing zero-degree nodes.")
    parser.add_argument("--experiment-name", required=True, help="Experiment folder name")
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-8B", help="LLM name label")
    parser.add_argument("--embedding-model-name", default="facebook/contriever-msmarco", help="Embedding model name")
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="Location (under experiment) for HippoRAG save_dir",
    )
    parser.add_argument(
        "--input-subdir",
        default="offline_indexing/04_graph_50",
        help="Input subdirectory containing the graph to clean",
    )
    parser.add_argument(
        "--output-subdir",
        default="offline_indexing/05_graph_cleaned_50",
        help="Output subdirectory for cleaned graph",
    )
    return parser.parse_args(argv)


def analyze_zero_degree_nodes(graph: ig.Graph) -> Dict:
    """
    Analyze zero-degree nodes and categorize them.
    A node is considered isolated if:
    1. It has degree 0 (no edges at all), OR
    2. It only has self-loops (edges to itself) - these are effectively isolated
    """
    degrees = graph.degree()
    node_names = [v["name"] if "name" in v.attributes() else f"node_{i}" for i, v in enumerate(graph.vs)]
    
    # Find nodes with zero degree
    zero_degree_indices = []
    
    for i, degree in enumerate(degrees):
        if degree == 0:
            # Truly isolated - no edges at all
            zero_degree_indices.append(i)
        else:
            # Check if node only has self-loops (effectively isolated)
            # Get all neighbors
            neighbors = graph.neighbors(i, mode="all")
            # Filter out self (if undirected, self-loops appear as the node itself in neighbors)
            unique_neighbors = set(neighbors)
            unique_neighbors.discard(i)  # Remove self if present
            
            # If no unique neighbors (only self-loops), consider it isolated
            if len(unique_neighbors) == 0:
                zero_degree_indices.append(i)
    
    entity_nodes = []
    passage_nodes = []
    
    for idx in zero_degree_indices:
        node_name = node_names[idx]
        node_info = {
            "index": idx,
            "name": node_name,
            "degree": degrees[idx],
        }
        
        # Add additional attributes if available
        if "content" in graph.vs[idx].attributes():
            content = str(graph.vs[idx]["content"])
            node_info["content"] = content[:100] if len(content) > 100 else content
        
        # Check incident edges for debugging
        incident_edges = graph.incident(idx, mode="all")
        node_info["num_incident_edges"] = len(incident_edges)
        
        if node_name.startswith("entity-"):
            entity_nodes.append(node_info)
        else:
            passage_nodes.append(node_info)
    
    return {
        "total_zero_degree": len(zero_degree_indices),
        "entity_nodes": entity_nodes,
        "passage_nodes": passage_nodes,
        "zero_degree_indices": zero_degree_indices,
    }


def remove_zero_degree_nodes(graph: ig.Graph, zero_degree_indices: List[int]) -> ig.Graph:
    """
    Remove nodes with zero degree from the graph.
    Note: This modifies the graph in-place, so make sure to pass a copy if needed.
    """
    if not zero_degree_indices:
        return graph
    
    # Sort indices in descending order to avoid index shifting issues
    sorted_indices = sorted(zero_degree_indices, reverse=True)
    
    # Verify indices are valid
    max_idx = graph.vcount() - 1
    invalid_indices = [idx for idx in sorted_indices if idx < 0 or idx > max_idx]
    if invalid_indices:
        raise ValueError(f"Invalid vertex indices: {invalid_indices}")
    
    # Delete vertices (igraph handles edge deletion automatically)
    # For undirected graphs, this removes all edges incident to these vertices
    # For directed graphs, this removes both incoming and outgoing edges
    graph.delete_vertices(sorted_indices)
    
    return graph


def write_cleanup_log(log_path: Path, analysis: Dict, stats_before: Dict, stats_after: Dict):
    """Write detailed log of cleanup operation."""
    with log_path.open("w") as f:
        f.write("=" * 80 + "\n")
        f.write("HIPPORAG GRAPH CLEANUP LOG\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total nodes removed: {analysis['total_zero_degree']}\n")
        f.write(f"  - Entity nodes removed: {len(analysis['entity_nodes'])}\n")
        f.write(f"  - Passage nodes removed: {len(analysis['passage_nodes'])}\n\n")
        
        f.write("GRAPH STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write("Before cleanup:\n")
        f.write(f"  - Total nodes: {stats_before['num_nodes']}\n")
        f.write(f"  - Total edges: {stats_before['num_edges']}\n")
        f.write(f"  - Entity nodes: {stats_before['num_entities']}\n")
        f.write(f"  - Passage nodes: {stats_before['num_passages']}\n")
        f.write(f"  - Zero-degree nodes: {stats_before['num_zero_degree']}\n\n")
        
        f.write("After cleanup:\n")
        f.write(f"  - Total nodes: {stats_after['num_nodes']}\n")
        f.write(f"  - Total edges: {stats_after['num_edges']}\n")
        f.write(f"  - Entity nodes: {stats_after['num_entities']}\n")
        f.write(f"  - Passage nodes: {stats_after['num_passages']}\n")
        f.write(f"  - Zero-degree nodes: {stats_after['num_zero_degree']}\n\n")
        
        f.write("REMOVED ENTITY NODES\n")
        f.write("-" * 80 + "\n")
        if analysis['entity_nodes']:
            for idx, node in enumerate(analysis['entity_nodes'], 1):
                f.write(f"{idx}. {node['name']}\n")
                f.write(f"   Index: {node['index']}, Degree: {node.get('degree', 'N/A')}, "
                       f"Incident edges: {node.get('num_incident_edges', 'N/A')}\n")
                if 'content' in node:
                    f.write(f"   Content: {node['content']}\n")
                f.write("\n")
        else:
            f.write("No entity nodes removed.\n\n")
        
        f.write("REMOVED PASSAGE NODES\n")
        f.write("-" * 80 + "\n")
        if analysis['passage_nodes']:
            for idx, node in enumerate(analysis['passage_nodes'], 1):
                f.write(f"{idx}. {node['name']}\n")
                f.write(f"   Index: {node['index']}, Degree: {node.get('degree', 'N/A')}, "
                       f"Incident edges: {node.get('num_incident_edges', 'N/A')}\n")
                if 'content' in node:
                    f.write(f"   Content: {node['content']}\n")
                f.write("\n")
        else:
            f.write("No passage nodes removed.\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("Cleanup complete.\n")
        f.write("=" * 80 + "\n")


def get_graph_stats(graph: ig.Graph) -> Dict:
    """Get statistics about the graph."""
    node_names = [v["name"] if "name" in v.attributes() else f"node_{i}" for i, v in enumerate(graph.vs)]
    degrees = graph.degree()
    
    entity_count = sum(1 for name in node_names if name.startswith("entity-"))
    passage_count = sum(1 for name in node_names if not name.startswith("entity-"))
    zero_degree_count = sum(1 for d in degrees if d == 0)
    
    return {
        "num_nodes": graph.vcount(),
        "num_edges": graph.ecount(),
        "num_entities": entity_count,
        "num_passages": passage_count,
        "num_zero_degree": zero_degree_count,
    }


def main():
    args = parse_args()
    
    # Build paths
    workspace_dir = build_experiment_dir(args.experiment_name, args.workspace_subdir)
    llm_label = args.llm_model_name.replace("/", "_")
    embed_label = args.embedding_model_name.replace("/", "_")
    working_dir = workspace_dir / f"{llm_label}_{embed_label}"
    
    input_graph_path = working_dir / "graph.pickle"
    output_graph_path = working_dir / "graph_cleaned.pickle"
    
    # Create output directory for logs
    output_dir = build_experiment_dir(args.experiment_name, args.output_subdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "cleanup_log.txt"
    
    # Load graph
    print(f"Loading graph from: {input_graph_path}")
    if not input_graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {input_graph_path}")
    
    graph = ig.Graph.Read_Pickle(str(input_graph_path))
    
    # Print graph properties for debugging
    print(f"  Graph loaded: {graph.vcount()} nodes, {graph.ecount()} edges")
    print(f"  Is directed: {graph.is_directed()}")
    if "name" in graph.vs.attribute_names():
        print(f"  Has 'name' attribute: Yes")
    else:
        print(f"  Has 'name' attribute: No")
    
    # Get statistics before cleanup
    print("\nAnalyzing graph before cleanup...")
    stats_before = get_graph_stats(graph)
    print(f"  Total nodes: {stats_before['num_nodes']}")
    print(f"  Total edges: {stats_before['num_edges']}")
    print(f"  Entity nodes: {stats_before['num_entities']}")
    print(f"  Passage nodes: {stats_before['num_passages']}")
    print(f"  Zero-degree nodes: {stats_before['num_zero_degree']}")
    
    # Analyze zero-degree nodes
    print("\nAnalyzing zero-degree nodes...")
    analysis = analyze_zero_degree_nodes(graph)
    print(f"  Found {analysis['total_zero_degree']} zero-degree nodes")
    print(f"    - Entity nodes: {len(analysis['entity_nodes'])}")
    print(f"    - Passage nodes: {len(analysis['passage_nodes'])}")
    
    if analysis['total_zero_degree'] == 0:
        print("\nNo zero-degree nodes found. Graph is already clean!")
        # Still save the graph and create a log
        graph.write_pickle(str(output_graph_path))
        with log_path.open("w") as f:
            f.write("=" * 80 + "\n")
            f.write("HIPPORAG GRAPH CLEANUP LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("No zero-degree nodes found. Graph is already clean.\n")
            f.write("No cleanup was performed.\n\n")
            f.write("GRAPH STATISTICS\n")
            f.write("-" * 80 + "\n")
            for key, value in stats_before.items():
                f.write(f"  {key}: {value}\n")
            f.write("=" * 80 + "\n")
        print(f"Graph saved to: {output_graph_path}")
        print(f"Log saved to: {log_path}")
        return
    
    # Remove zero-degree nodes
    print("\nRemoving zero-degree nodes...")
    print(f"  Removing {len(analysis['zero_degree_indices'])} nodes...")
    
    # Create a copy to avoid modifying the original graph object
    cleaned_graph = graph.copy()
    cleaned_graph = remove_zero_degree_nodes(cleaned_graph, analysis['zero_degree_indices'])
    
    # Verify removal
    print(f"  Verification: Removed {graph.vcount() - cleaned_graph.vcount()} nodes")
    if graph.vcount() - cleaned_graph.vcount() != len(analysis['zero_degree_indices']):
        print(f"  WARNING: Expected to remove {len(analysis['zero_degree_indices'])} nodes, "
              f"but only removed {graph.vcount() - cleaned_graph.vcount()}")
    
    # Get statistics after cleanup
    stats_after = get_graph_stats(cleaned_graph)
    print(f"  After cleanup:")
    print(f"    Total nodes: {stats_after['num_nodes']}")
    print(f"    Total edges: {stats_after['num_edges']}")
    print(f"    Entity nodes: {stats_after['num_entities']}")
    print(f"    Passage nodes: {stats_after['num_passages']}")
    print(f"    Zero-degree nodes: {stats_after['num_zero_degree']}")
    
    # Double-check: verify no zero-degree nodes remain
    if stats_after['num_zero_degree'] > 0:
        print(f"\n  WARNING: Still found {stats_after['num_zero_degree']} zero-degree nodes after cleanup!")
        print(f"  This might indicate an issue with the cleanup process.")
    
    # Save cleaned graph
    print(f"\nSaving cleaned graph to: {output_graph_path}")
    cleaned_graph.write_pickle(str(output_graph_path))
    
    # Also update the original graph.pickle if it exists
    backup_path = working_dir / "graph.pickle.backup"
    if (working_dir / "graph.pickle").exists():
        print(f"Creating backup of original graph: {backup_path}")
        import shutil
        shutil.copy2(working_dir / "graph.pickle", backup_path)
        print(f"Updating original graph.pickle with cleaned version...")
        shutil.copy2(output_graph_path, working_dir / "graph.pickle")
    
    # Write cleanup log
    print(f"Writing cleanup log to: {log_path}")
    write_cleanup_log(log_path, analysis, stats_before, stats_after)
    
    print("\n" + "=" * 80)
    print("Graph cleanup complete!")
    print("=" * 80)
    print(f"Removed {analysis['total_zero_degree']} zero-degree nodes")
    print(f"Cleaned graph saved to: {output_graph_path}")
    print(f"Original graph backed up to: {backup_path}")
    print(f"Cleanup log saved to: {log_path}")


if __name__ == "__main__":
    main()


