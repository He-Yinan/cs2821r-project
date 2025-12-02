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
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.eval_utils import normalize_answer, compute_f1
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from hipporag.evaluation.qa_eval import QAF1Score, QAExactMatch

# Import our MARA-RAG components
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
                 config: BaseConfig):
        """
        Initialize MARA experiment.

        Args:
            hipporag: HippoRAG instance with loaded graph
            router: QueryRouter for relation weight assignment
            ppr_engine: RelationAwarePPR for dynamic PPR
            config: HippoRAG configuration
        """
        self.hipporag = hipporag
        self.router = router
        self.ppr_engine = ppr_engine
        self.config = config

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
        relation_weights = self.router.route(query)
        debug_info['relation_weights'] = relation_weights
        debug_info['route_time'] = time.time() - route_start

        if verbose:
            print(f"\n=== Query: {query}")
            print(f"Relation weights: {relation_weights}")

        # Step 2: Fact retrieval (using HippoRAG's existing method)
        fact_start = time.time()
        query_fact_scores = self.hipporag.get_fact_scores(query)
        top_k_fact_indices, top_k_facts, rerank_log = self.hipporag.rerank_facts(
            query, query_fact_scores
        )
        debug_info['fact_retrieval_time'] = time.time() - fact_start
        debug_info['num_facts'] = len(top_k_facts)

        if verbose:
            print(f"Retrieved {len(top_k_facts)} facts after recognition memory")

        # Step 3: If no facts found, fall back to dense retrieval
        if len(top_k_facts) == 0:
            if verbose:
                print("No facts found, falling back to DPR")

            sorted_doc_ids, sorted_doc_scores = self.hipporag.dense_passage_retrieval(query)
            debug_info['retrieval_method'] = 'dpr_fallback'

        else:
            # Step 4: Build phrase weights from facts
            phrase_weights = self._build_phrase_weights(
                top_k_facts, top_k_fact_indices, query_fact_scores
            )
            debug_info['num_seed_phrases'] = int((phrase_weights > 0).sum())

            # Step 5: Build passage weights from dense retrieval
            passage_weights = self._build_passage_weights(query)
            debug_info['num_seed_passages'] = int((passage_weights > 0).sum())

            # Step 6: Run relation-aware PPR
            ppr_start = time.time()
            sorted_doc_ids, sorted_doc_scores = self.ppr_engine.run_ppr(
                phrase_weights=phrase_weights,
                passage_weights=passage_weights,
                relation_weights=relation_weights,
                damping=self.config.damping
            )
            debug_info['ppr_time'] = time.time() - ppr_start
            debug_info['retrieval_method'] = 'mara_ppr'

        # Step 7: Get top-k documents
        num_to_retrieve = self.config.retrieval_top_k
        top_docs = [
            self.hipporag.chunk_embedding_store.get_row(self.ppr_engine.passage_node_keys[idx])["content"]
            for idx in sorted_doc_ids[:num_to_retrieve]
        ]
        top_scores = sorted_doc_scores[:num_to_retrieve].tolist()

        debug_info['total_time'] = time.time() - start_time

        return top_docs, top_scores, debug_info

    def _build_phrase_weights(self,
                             top_k_facts: List[Tuple],
                             top_k_fact_indices: List[str],
                             query_fact_scores: np.ndarray) -> np.ndarray:
        """
        Build phrase node weights from retrieved facts.

        This mirrors HippoRAG's graph_search_with_fact_entities method.

        Args:
            top_k_facts: List of (subject, predicate, object) triples
            top_k_fact_indices: Indices of these facts
            query_fact_scores: Scores for all facts

        Returns:
            Array of phrase weights (shape: [num_nodes])
        """

        from hipporag.utils.misc_utils import compute_mdhash_id

        phrase_weights = np.zeros(self.ppr_engine.num_nodes)
        number_of_occurs = np.zeros(self.ppr_engine.num_nodes)

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]
            ] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                phrase_id = self.hipporag.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    # Weight by document frequency (specificity)
                    if len(self.hipporag.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.hipporag.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

        # Average scores for phrases that appear multiple times
        phrase_weights = np.where(number_of_occurs > 0, phrase_weights / number_of_occurs, phrase_weights)

        return phrase_weights

    def _build_passage_weights(self, query: str) -> np.ndarray:
        """
        Build passage node weights from dense retrieval.

        Args:
            query: Query string

        Returns:
            Array of passage weights (shape: [num_nodes])
        """

        from hipporag.utils.misc_utils import min_max_normalize

        passage_weights = np.zeros(self.ppr_engine.num_nodes)

        # Get dense passage retrieval scores
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.hipporag.dense_passage_retrieval(query)
        normalized_dpr_scores = min_max_normalize(dpr_sorted_doc_scores)

        # Assign weights to passage nodes
        for i, dpr_doc_id in enumerate(dpr_sorted_doc_ids.tolist()):
            passage_node_key = self.ppr_engine.passage_node_keys[dpr_doc_id]
            passage_dpr_score = normalized_dpr_scores[i]
            passage_node_id = self.hipporag.node_name_to_vertex_idx[passage_node_key]
            passage_weights[passage_node_id] = passage_dpr_score

        return passage_weights

    def run_qa(self, query: str, retrieved_docs: List[str]) -> Tuple[str, Dict]:
        """
        Generate answer for query using retrieved documents.

        Args:
            query: Query string
            retrieved_docs: List of retrieved document texts

        Returns:
            Tuple of (answer, metadata)
        """

        # Use HippoRAG's QA method
        from hipporag.utils.typing import QuerySolution

        query_solution = QuerySolution(question=query, docs=retrieved_docs)
        results, messages, metadata = self.hipporag.qa([query_solution])

        return results[0].answer, metadata[0]


def load_questions(questions_file: Path) -> List[Dict]:
    """
    Load questions from JSONL file.

    Expected format:
    {
        "question": "...",
        "answer": [...],  # List of gold answers
        "supporting_facts": [[title1, sent_id1], ...],  # Optional
        "context": [[title, [sent1, sent2, ...]], ...]  # Optional
    }

    Args:
        questions_file: Path to questions JSONL file

    Returns:
        List of question dictionaries
    """

    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def compute_retrieval_metrics(gold_docs: List[List[str]],
                              retrieved_docs: List[List[str]],
                              k_list: List[int]) -> Dict:
    """
    Compute retrieval recall@k metrics.

    Args:
        gold_docs: List of lists of gold document texts
        retrieved_docs: List of lists of retrieved document texts
        k_list: List of k values for recall@k

    Returns:
        Dictionary of metrics
    """

    metrics = {}

    for k in k_list:
        recalls = []
        for gold, retrieved in zip(gold_docs, retrieved_docs):
            # Count how many gold docs are in top-k retrieved
            gold_set = set(gold)
            retrieved_set = set(retrieved[:k])
            recall = len(gold_set & retrieved_set) / len(gold_set) if len(gold_set) > 0 else 0.0
            recalls.append(recall)

        metrics[f'recall@{k}'] = float(np.mean(recalls))

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
        default="Qwen/Qwen3-8B-Instruct",
        help="LLM model name"
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
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to evaluate (default: all)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information"
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)

    # Setup paths
    experiment_dir = PROJECT_ROOT / "experiment" / "dataset" / args.experiment_name
    workspace_dir = experiment_dir / args.workspace_subdir
    matrix_dir = experiment_dir / args.matrix_subdir
    output_dir = experiment_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load questions
    questions_file = Path(args.questions_file)
    if not questions_file.is_absolute():
        questions_file = experiment_dir / questions_file

    print(f"Loading questions from {questions_file}...")
    questions = load_questions(questions_file)

    if args.num_questions:
        questions = questions[:args.num_questions]

    print(f"Loaded {len(questions)} questions")

    # Load HippoRAG
    print(f"\nLoading HippoRAG from {workspace_dir}...")
    config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.llm_base_url,
    )

    hipporag = HippoRAG(global_config=config)

    # Load graph
    if not hasattr(hipporag, 'graph') or hipporag.graph is None:
        hipporag.load_igraph()

    print(f"Graph loaded: {len(hipporag.graph.vs)} nodes, {len(hipporag.graph.es)} edges")

    # Initialize router
    print("\nInitializing query router...")
    router = QueryRouter(llm=hipporag.llm)

    # Load relation-aware PPR engine
    print(f"Loading relation matrices from {matrix_dir}...")
    ppr_engine = RelationAwarePPR(matrix_dir=matrix_dir)

    # Create experiment runner
    experiment = MARAExperiment(
        hipporag=hipporag,
        router=router,
        ppr_engine=ppr_engine,
        config=config
    )

    # Run retrieval + QA on all questions
    print(f"\nRunning MARA-RAG on {len(questions)} questions...")
    results = []

    for q_data in tqdm(questions, desc="Processing questions"):
        query = q_data['question']

        # Retrieve documents
        retrieved_docs, doc_scores, debug_info = experiment.retrieve_single_query(
            query, verbose=args.verbose
        )

        # Generate answer
        answer, qa_metadata = experiment.run_qa(query, retrieved_docs)

        # Get gold answer if available
        gold_answers = q_data.get('answer', [])
        if isinstance(gold_answers, str):
            gold_answers = [gold_answers]

        # Compute F1 score
        f1_scores = [compute_f1(answer, gold_ans) for gold_ans in gold_answers]
        max_f1 = max(f1_scores) if f1_scores else 0.0

        # Store result
        result = {
            'question': query,
            'answer': answer,
            'gold_answers': gold_answers,
            'f1_score': max_f1,
            'retrieved_docs': retrieved_docs,
            'doc_scores': doc_scores,
            'relation_weights': debug_info['relation_weights'],
            'debug_info': debug_info
        }

        results.append(result)

    # Compute aggregate metrics
    f1_scores = [r['f1_score'] for r in results]
    aggregate_metrics = {
        'num_questions': len(results),
        'mean_f1': float(np.mean(f1_scores)),
        'median_f1': float(np.median(f1_scores)),
        'std_f1': float(np.std(f1_scores)),
    }

    # Compute retrieval recall if gold docs are available
    if 'context' in questions[0] or 'supporting_facts' in questions[0]:
        print("\nComputing retrieval metrics...")
        # Extract gold docs from context
        gold_docs_list = []
        for q in questions:
            if 'context' in q:
                # MuSiQue format: context is [[title, [sent1, sent2, ...]], ...]
                gold_docs = [sent for title, sents in q['context'] for sent in sents]
            else:
                gold_docs = []
            gold_docs_list.append(gold_docs)

        retrieved_docs_list = [r['retrieved_docs'] for r in results]

        retrieval_metrics = compute_retrieval_metrics(
            gold_docs_list, retrieved_docs_list, k_list=[1, 2, 5, 10, 20]
        )
        aggregate_metrics.update(retrieval_metrics)

    # Print summary
    print("\n" + "="*80)
    print("MARA-RAG Experiment Results")
    print("="*80)
    print(f"Dataset: {args.experiment_name}")
    print(f"Questions: {len(results)}")
    print(f"\nAggregate Metrics:")
    for metric, value in aggregate_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Save results
    results_file = output_dir / "results.jsonl"
    with open(results_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    print(f"\nSaved per-question results to: {results_file}")

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"Saved aggregate metrics to: {metrics_file}")

    print("\n✓ Experiment completed successfully!")


if __name__ == "__main__":
    main()
