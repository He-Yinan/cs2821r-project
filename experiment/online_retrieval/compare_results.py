#!/usr/bin/env python3
"""
Compare MARA-RAG results with baseline HippoRAG.

This script loads results from both MARA-RAG and baseline HippoRAG experiments
and compares their performance across various metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt


def load_results(results_file: Path) -> tuple[List[Dict], Dict]:
    """
    Load results from a JSONL file.

    Args:
        results_file: Path to results.jsonl

    Returns:
        Tuple of (per_question_results, aggregate_metrics)
    """

    results = []
    with open(results_file, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    # Load aggregate metrics if available
    metrics_file = results_file.parent / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    else:
        # Compute metrics from results
        f1_scores = [r['f1_score'] for r in results]
        metrics = {
            'num_questions': len(results),
            'mean_f1': float(np.mean(f1_scores)),
            'median_f1': float(np.median(f1_scores)),
            'std_f1': float(np.std(f1_scores)),
        }

    return results, metrics


def compare_metrics(mara_metrics: Dict, baseline_metrics: Dict) -> None:
    """
    Print a comparison of metrics between MARA-RAG and baseline.

    Args:
        mara_metrics: MARA-RAG metrics
        baseline_metrics: Baseline HippoRAG metrics
    """

    print("\n" + "="*80)
    print("Metric Comparison: MARA-RAG vs. Baseline HippoRAG")
    print("="*80)

    # Define metrics to compare
    metrics_to_compare = [
        ('mean_f1', 'Mean F1 Score', True),
        ('median_f1', 'Median F1 Score', True),
        ('std_f1', 'Std Dev F1', False),
        ('recall@1', 'Recall@1', True),
        ('recall@2', 'Recall@2', True),
        ('recall@5', 'Recall@5', True),
        ('recall@10', 'Recall@10', True),
        ('recall@20', 'Recall@20', True),
    ]

    print(f"\n{'Metric':<25} {'MARA-RAG':>12} {'Baseline':>12} {'Diff':>10} {'%Change':>10}")
    print("-"*80)

    for metric_key, metric_name, higher_is_better in metrics_to_compare:
        if metric_key in mara_metrics and metric_key in baseline_metrics:
            mara_val = mara_metrics[metric_key]
            baseline_val = baseline_metrics[metric_key]
            diff = mara_val - baseline_val
            pct_change = (diff / baseline_val * 100) if baseline_val != 0 else 0

            # Add color coding (if terminal supports it)
            if higher_is_better:
                symbol = "↑" if diff > 0 else "↓" if diff < 0 else "="
            else:
                symbol = "↓" if diff > 0 else "↑" if diff < 0 else "="

            print(f"{metric_name:<25} {mara_val:>12.4f} {baseline_val:>12.4f} "
                  f"{symbol} {diff:>8.4f} {pct_change:>9.2f}%")

    print("-"*80)


def analyze_query_types(mara_results: List[Dict]) -> None:
    """
    Analyze performance by query type based on relation weights.

    Args:
        mara_results: MARA-RAG per-question results
    """

    print("\n" + "="*80)
    print("Performance by Query Type (MARA-RAG)")
    print("="*80)

    # Group queries by dominant relation type
    query_groups = {
        'hierarchical': [],
        'temporal': [],
        'spatial': [],
        'causality': [],
        'attribution': [],
        'mixed': []
    }

    for result in mara_results:
        weights = result.get('relation_weights', {})

        # Find dominant relation type
        relation_weights = {k: v for k, v in weights.items()
                          if k in ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']}

        if not relation_weights:
            query_groups['mixed'].append(result)
            continue

        max_weight = max(relation_weights.values())
        dominant_types = [k for k, v in relation_weights.items() if v == max_weight]

        if len(dominant_types) == 1 and max_weight >= 0.4:
            # Clear dominant type
            query_groups[dominant_types[0]].append(result)
        else:
            # Mixed or no clear dominant type
            query_groups['mixed'].append(result)

    # Compute average F1 for each group
    print(f"\n{'Query Type':<20} {'Count':>8} {'Avg F1':>10} {'Median F1':>12}")
    print("-"*80)

    for query_type, results in query_groups.items():
        if len(results) > 0:
            f1_scores = [r['f1_score'] for r in results]
            avg_f1 = np.mean(f1_scores)
            median_f1 = np.median(f1_scores)
            print(f"{query_type.capitalize():<20} {len(results):>8} {avg_f1:>10.4f} {median_f1:>12.4f}")

    print("-"*80)


def plot_f1_distribution(mara_results: List[Dict],
                        baseline_results: List[Dict],
                        output_file: Path | None = None) -> None:
    """
    Plot F1 score distribution comparison.

    Args:
        mara_results: MARA-RAG results
        baseline_results: Baseline results
        output_file: Optional path to save plot
    """

    mara_f1 = [r['f1_score'] for r in mara_results]
    baseline_f1 = [r['f1_score'] for r in baseline_results]

    plt.figure(figsize=(10, 6))

    plt.hist(baseline_f1, bins=20, alpha=0.5, label='Baseline HippoRAG', color='blue')
    plt.hist(mara_f1, bins=20, alpha=0.5, label='MARA-RAG', color='green')

    plt.xlabel('F1 Score')
    plt.ylabel('Number of Questions')
    plt.title('F1 Score Distribution: MARA-RAG vs. Baseline HippoRAG')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nSaved F1 distribution plot to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Compare MARA-RAG with baseline HippoRAG")

    parser.add_argument(
        "--mara-results",
        required=True,
        help="Path to MARA-RAG results.jsonl"
    )
    parser.add_argument(
        "--baseline-results",
        required=True,
        help="Path to baseline HippoRAG results.jsonl"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save plots (default: same as MARA results)"
    )

    args = parser.parse_args()

    # Load results
    print("Loading MARA-RAG results...")
    mara_results, mara_metrics = load_results(Path(args.mara_results))

    print("Loading baseline HippoRAG results...")
    baseline_results, baseline_metrics = load_results(Path(args.baseline_results))

    # Compare metrics
    compare_metrics(mara_metrics, baseline_metrics)

    # Analyze query types (MARA-RAG only, since it has relation weights)
    analyze_query_types(mara_results)

    # Generate plots if requested
    if args.plot:
        output_dir = args.output_dir or Path(args.mara_results).parent

        plot_f1_distribution(
            mara_results,
            baseline_results,
            output_file=output_dir / "f1_distribution_comparison.png"
        )

    print("\n" + "="*80)
    print("Comparison Complete!")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
