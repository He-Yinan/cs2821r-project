#!/usr/bin/env python3
"""
Experiment 1: Zero-Shot Performance (Static Evaluation)

Compares three retrieval modes on a 50-question Musique subset:
  - DPR only (Standard RAG baseline)
  - HippoRAG2 vanilla (graph PPR, relation-blind)
  - MARA-RAG (relation-aware multi-agent PPR)

Metrics:
  - Recall@5 (passage retrieval)
  - F1 (QA)

Outputs (per mode under scratch experiments/<experiment-name>/online_retrieval/exp1_zero_shot/<mode>/):
  - metrics.json          : overall retrieval + QA metrics
  - per_query_logs.jsonl  : detailed logs per question (especially rich for MARA)
  - predictions.jsonl     : flat QA predictions + retrieved passages
"""

from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig
from hipporag.evaluation.retrieval_eval import RetrievalRecall
from hipporag.evaluation.qa_eval import QAExactMatch, QAF1Score

from experiment.common.io_utils import build_experiment_dir, load_json, write_json, write_jsonl


def load_musique_subset(dataset_path: Path):
    """Load 50-question Musique subset and return (questions, gold_docs, gold_answers).

    - questions: list of question strings
    - gold_docs: list[list[str]] of supporting passages (full paragraph_text)
    - gold_answers: list[list[str]] of accepted answers (here single canonical answer)
    """
    data = load_json(dataset_path)
    questions: List[str] = []
    gold_docs: List[List[str]] = []
    gold_answers: List[List[str]] = []

    for sample in data:
        q = sample.get("question", "").strip()
        if not q:
            continue

        # Supporting paragraphs are those with is_supporting == True
        # Store both text and a normalized version for matching
        supporting_paras = []
        supporting_para_texts = []  # Store normalized texts for matching
        for para in sample.get("paragraphs", []):
            if para.get("is_supporting"):
                text = para.get("paragraph_text") or para.get("text") or ""
                if text:
                    supporting_paras.append(str(text))
                    # Normalize for matching (remove extra whitespace)
                    import re
                    normalized = re.sub(r'\s+', ' ', text.strip())
                    supporting_para_texts.append(normalized)

        # Fallback: if no supporting flag, use all paragraphs
        if not supporting_paras:
            for para in sample.get("paragraphs", []):
                text = para.get("paragraph_text") or para.get("text") or ""
                if text:
                    supporting_paras.append(str(text))
                    import re
                    normalized = re.sub(r'\s+', ' ', text.strip())
                    supporting_para_texts.append(normalized)

        ans = sample.get("answer")
        if isinstance(ans, str):
            answers = [ans]
        elif isinstance(ans, list):
            answers = [str(a) for a in ans]
        else:
            answers = []

        questions.append(q)
        gold_docs.append(supporting_paras)
        gold_answers.append(answers)

    return questions, gold_docs, gold_answers


def build_hipporag(
    experiment_name: str,
    llm_model_name: str,
    embedding_model_name: str,
    llm_base_url: str,
    workspace_subdir: str = "hipporag_workspace",
    use_relation_aware: bool = False,
    retrieval_top_k: int = 20,
    linking_top_k: int = 5,
    max_qa_steps: int = 3,
    qa_top_k: int = 5,
    disable_rerank_filter: bool = True,
    num_facts_without_rerank: int = 10,
) -> HippoRAG:
    """Instantiate HippoRAG bound to an existing offline workspace.

    This assumes that offline_indexing steps have already created the workspace
    directory (embeddings, openie results, and graph).
    """
    workspace_dir = build_experiment_dir(experiment_name, workspace_subdir)
    config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=llm_model_name,
        embedding_model_name=embedding_model_name,
        llm_base_url=llm_base_url,
        openie_mode="online",
        force_index_from_scratch=False,
    )
    # Enable/disable relation-aware retrieval
    config.use_relation_aware_retrieval = use_relation_aware
    # Set hyperparameters from arguments
    config.linking_top_k = linking_top_k
    config.retrieval_top_k = retrieval_top_k
    config.qa_top_k = qa_top_k
    config.max_qa_steps = max_qa_steps
    config.disable_rerank_filter = disable_rerank_filter
    config.num_facts_without_rerank = num_facts_without_rerank
    # Increase seed passages for better coverage (was 50)
    config.max_seed_passages = 100
    # CRITICAL FIX: Increase passage node weight to preserve DPR signal
    # Default is 0.05 which heavily downweights direct relevance
    # Increasing to 0.3 balances DPR signal with graph-based PPR
    config.passage_node_weight = 0.3
    # HYBRID SCORING: Combine DPR + PPR scores to preserve direct relevance
    # 0.5 = 50% DPR + 50% PPR (balanced)
    # Higher values (0.6-0.7) favor DPR more, lower values (0.3-0.4) favor PPR more
    config.dpr_ppr_hybrid_alpha = 0.5

    hippo = HippoRAG(global_config=config)

    # Ensure retrieval objects (embeddings + graph) are ready
    if not hippo.ready_to_retrieve:
        try:
            hippo.prepare_retrieval_objects()
        except Exception as e:
            import traceback
            print(f"Error preparing retrieval objects: {e}")
            traceback.print_exc()
            raise

    # Verify graph is loaded
    if hippo.graph is None or hippo.graph.vcount() == 0:
        raise RuntimeError(f"Graph not properly loaded from workspace: {workspace_dir}")
    
    # Verify graph has name attribute
    if "name" not in hippo.graph.vs.attribute_names():
        raise RuntimeError("Graph missing 'name' attribute. Graph may need to be rebuilt.")

    return hippo


def run_mode_dpr(
    hippo: HippoRAG,
    questions: List[str],
    gold_docs: List[List[str]],
    gold_answers: List[List[str]],
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Run DPR-only baseline using rag_qa_dpr."""
    qa_solutions, _, _, retrieval_metrics, qa_metrics = hippo.rag_qa_dpr(
        queries=questions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )

    per_query = []
    for q, sol in zip(questions, qa_solutions):
        per_query.append(
            {
                "question": q,
                "answer": sol.answer,
                "docs": sol.docs,
                "doc_scores": sol.doc_scores.tolist() if sol.doc_scores is not None else None,
                "gold_answers": sol.gold_answers,
                "gold_docs": sol.gold_docs if hasattr(sol, "gold_docs") else None,
            }
        )

    return retrieval_metrics, qa_metrics, per_query


def run_mode_hipporag(
    hippo: HippoRAG,
    questions: List[str],
    gold_docs: List[List[str]],
    gold_answers: List[List[str]],
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Run vanilla HippoRAG2 (relation-blind PPR)."""
    # Make sure relation-aware is disabled
    hippo.global_config.use_relation_aware_retrieval = False
    hippo.manager_agent = None

    qa_solutions, _, _, retrieval_metrics, qa_metrics = hippo.rag_qa(
        queries=questions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )

    per_query = []
    for q, sol in zip(questions, qa_solutions):
        per_query.append(
            {
                "question": q,
                "answer": sol.answer,
                "docs": sol.docs,
                "doc_scores": sol.doc_scores.tolist() if sol.doc_scores is not None else None,
                "gold_answers": sol.gold_answers,
                "gold_docs": sol.gold_docs if hasattr(sol, "gold_docs") else None,
            }
        )

    return retrieval_metrics, qa_metrics, per_query


def run_mode_mara(
    hippo: HippoRAG,
    questions: List[str],
    gold_docs: List[List[str]],
    gold_answers: List[List[str]],
) -> Tuple[Dict[str, float], Dict[str, float], List[Dict[str, Any]]]:
    """Run MARA-RAG (relation-aware PPR + Manager Agent) with rich per-query logs."""
    hippo.global_config.use_relation_aware_retrieval = True

    # Ensure ManagerAgent exists
    if getattr(hippo, "manager_agent", None) is None:
        from hipporag.agents.manager_agent import ManagerAgent

        hippo.manager_agent = ManagerAgent(
            llm=hippo.llm_model,
            prompt_template_manager=hippo.prompt_template_manager,
        )

    # Ensure retrieval objects are ready
    if not hippo.ready_to_retrieve:
        hippo.prepare_retrieval_objects()

    qa_solutions, _, _, retrieval_metrics, qa_metrics = hippo.rag_qa(
        queries=questions,
        gold_docs=gold_docs,
        gold_answers=gold_answers,
    )

    # Build detailed per-query logs
    detailed_logs: List[Dict[str, Any]] = []

    if not hippo.ready_to_retrieve:
        hippo.prepare_retrieval_objects()

    # Pre-encode all queries for retrieval-side logging
    hippo.get_query_embeddings(questions)

    from hipporag.utils.misc_utils import compute_mdhash_id, min_max_normalize
    import numpy as np

    for idx, (q, sol, gold_d, gold_a) in enumerate(zip(questions, qa_solutions, gold_docs, gold_answers)):
        try:
            # 1) Manager agent beta values
            beta_values = hippo.manager_agent.get_relation_influence_factors(q)

            # 2) Fact scores + reranking
            query_fact_scores = hippo.get_fact_scores(q)
            top_k_fact_indices, top_k_facts, rerank_log = hippo.rerank_facts(q, query_fact_scores)
            
            # Log facts being used
            facts_summary = {
                "num_facts": len(top_k_facts),
                "facts": [{"subject": str(f[0]), "predicate": str(f[1]), "object": str(f[2])} 
                         for f in top_k_facts[:5]]  # Log first 5 facts
            }

            # 3) Seed node computation (mirror graph_search_with_relation_aware_ppr)
            linking_score_map: Dict[str, float] = {}
            phrase_scores: Dict[str, List[float]] = {}
            
            # Get graph size safely
            if "name" in hippo.graph.vs.attribute_names():
                graph_size = len(hippo.graph.vs["name"])
            else:
                graph_size = hippo.graph.vcount()
            
            phrase_weights = np.zeros(graph_size)
            passage_weights = np.zeros(graph_size)
            number_of_occurs = np.zeros(graph_size)

            phrases_and_ids = set()

            # Process facts if available
            if len(top_k_facts) > 0 and len(query_fact_scores) > 0:
                for rank, f in enumerate(top_k_facts):
                    if not isinstance(f, (list, tuple)) or len(f) < 3:
                        continue
                    subject_phrase = str(f[0]).lower()
                    object_phrase = str(f[2]).lower()
                    
                    # Get fact score - top_k_fact_indices are array indices
                    if rank < len(top_k_fact_indices) and top_k_fact_indices[rank] < len(query_fact_scores):
                        fact_score = query_fact_scores[top_k_fact_indices[rank]]
                    else:
                        fact_score = 0.0

                    for phrase in [subject_phrase, object_phrase]:
                        phrase_key = compute_mdhash_id(content=phrase, prefix="entity-")
                        phrase_id = hippo.node_name_to_vertex_idx.get(phrase_key, None)

                        if phrase_id is not None and phrase_id < graph_size:
                            weighted_fact_score = fact_score
                            if len(hippo.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                                weighted_fact_score /= len(hippo.ent_node_to_chunk_ids[phrase_key])

                            phrase_weights[phrase_id] += weighted_fact_score
                            number_of_occurs[phrase_id] += 1

                        phrases_and_ids.add((phrase, phrase_id))

                # Avoid division by zero
                nonzero_mask = number_of_occurs > 0
                if np.any(nonzero_mask):
                    phrase_weights[nonzero_mask] /= number_of_occurs[nonzero_mask]

                for phrase, phrase_id in phrases_and_ids:
                    if phrase_id is None or phrase_id >= graph_size:
                        continue
                    if phrase not in phrase_scores:
                        phrase_scores[phrase] = []
                    phrase_scores[phrase].append(phrase_weights[phrase_id])

                for phrase, scores in phrase_scores.items():
                    linking_score_map[phrase] = float(np.mean(scores))

                if hippo.global_config.linking_top_k and len(linking_score_map) > 0:
                    phrase_weights, linking_score_map = hippo.get_top_k_weights(
                        hippo.global_config.linking_top_k,
                        phrase_weights,
                        linking_score_map,
                    )

            # DPR passage scores
            dpr_sorted_doc_ids, dpr_sorted_doc_scores = hippo.dense_passage_retrieval(q)
            normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

            # Limit to top N passages (default: 20)
            max_passages = getattr(hippo.global_config, 'max_seed_passages', 20)
            num_passages_to_use = min(max_passages, len(dpr_sorted_doc_ids))
            
            for i, dpr_sorted_doc_id in enumerate(dpr_sorted_doc_ids[:num_passages_to_use].tolist()):
                if dpr_sorted_doc_id >= len(hippo.passage_node_keys):
                    continue
                passage_node_key = hippo.passage_node_keys[dpr_sorted_doc_id]
                passage_dpr_score = normalized_dpr_sorted_scores[i]
                passage_node_id = hippo.node_name_to_vertex_idx.get(passage_node_key, None)
                if passage_node_id is not None and passage_node_id < graph_size:
                    passage_weights[passage_node_id] = (
                        passage_dpr_score * hippo.global_config.passage_node_weight
                    )

            node_weights = phrase_weights + passage_weights

            # Seed entities: Extract ALL entities from facts, weighted by occurrence frequency
            # This ensures we log all entities mentioned in facts, not just top K after filtering
            seed_entity_nodes = []
            seed_passage_nodes = []
            
            # Get all fact entities (if available from relation-aware retrieval)
            all_fact_entities = getattr(hippo, '_all_fact_entities_from_current_query', {})
            
            if all_fact_entities:
                # Use fact entities directly, weighted by occurrence frequency
                for phrase_key, entity_info in all_fact_entities.items():
                    phrase_id = entity_info['phrase_id']
                    if phrase_id is not None and phrase_id < graph_size:
                        # Weight = average score * occurrence count (to reflect frequency)
                        avg_score = (entity_info['total_score'] / entity_info['occurrence_count'] 
                                   if entity_info['occurrence_count'] > 0 else 0.0)
                        # Weight by occurrence frequency
                        weight = avg_score * entity_info['occurrence_count']
                        seed_entity_nodes.append((phrase_key, weight, entity_info['phrase'], entity_info['occurrence_count']))
            else:
                # Fallback: use node_weights (for non-relation-aware mode or if fact entities not available)
                for node_key, vidx in hippo.node_name_to_vertex_idx.items():
                    if vidx >= graph_size or node_weights[vidx] <= 0:
                        continue
                    if node_key.startswith("entity-"):
                        seed_entity_nodes.append((node_key, float(node_weights[vidx]), None, None))
            
            # Also get passage nodes
            for node_key, vidx in hippo.node_name_to_vertex_idx.items():
                if vidx >= graph_size or node_weights[vidx] <= 0:
                    continue
                if not node_key.startswith("entity-"):
                    seed_passage_nodes.append((node_key, float(node_weights[vidx])))

            # Sort seeds by weight
            seed_entity_nodes.sort(key=lambda x: x[1], reverse=True)
            seed_passage_nodes.sort(key=lambda x: x[1], reverse=True)

            # Also get from entity embedding store as fallback
            entity_rows = hippo.entity_embedding_store.get_all_id_to_rows()
            passage_rows = hippo.chunk_embedding_store.get_all_id_to_rows()

            seed_entities_readable = []
            for item in seed_entity_nodes:
                if len(item) == 4:
                    # From fact entities: (node_key, weight, phrase, occurrence_count)
                    nid, w, phrase_name, occ_count = item
                    entity_name = phrase_name if phrase_name else entity_rows.get(nid, {}).get("content", "")
                else:
                    # Fallback format: (node_key, weight)
                    nid, w = item
                    entity_name = entity_rows.get(nid, {}).get("content", "")
                    # Try to find from facts
                    if not entity_name:
                        for f in top_k_facts:
                            if isinstance(f, (list, tuple)) and len(f) >= 3:
                                for phrase in [str(f[0]).lower(), str(f[2]).lower()]:
                                    test_key = compute_mdhash_id(content=phrase, prefix="entity-")
                                    if test_key == nid:
                                        entity_name = phrase
                                        break
                                if entity_name:
                                    break
                
                seed_entities_readable.append({
                    "node_id": nid,
                    "weight": w,
                    "content": entity_name if entity_name else nid,
                    "occurrence_count": item[3] if len(item) == 4 else None,  # Include occurrence count if available
                })

            seed_passages_readable = [
                {
                    "node_id": nid,
                    "weight": w,
                    "content": passage_rows.get(nid, {}).get("content", ""),
                }
                for nid, w in seed_passage_nodes[:20]
            ]
        except Exception as e:
            import traceback
            print(f"Error processing query {idx}: {e}")
            traceback.print_exc()
            # Fallback values
            beta_values = {"entity_entity": {}, "entity_passage": {}}
            rerank_log = {"facts_before_rerank": [], "facts_after_rerank": []}
            facts_summary = {"num_facts": 0, "facts": []}
            seed_entities_readable = []
            seed_passages_readable = []

        # Build log record (always, even if there was an error)
        log_record: Dict[str, Any] = {
            "index": idx,
            "question": q,
            "gold_docs": gold_d,
            "gold_answers": gold_a,
            "beta_values": beta_values,
            "top_facts_before_rerank": rerank_log.get("facts_before_rerank", []),
            "top_facts_after_rerank": rerank_log.get("facts_after_rerank", []),
            "facts_summary": facts_summary if 'facts_summary' in locals() else {},
            "seed_entities": seed_entities_readable,
            "seed_passages": seed_passages_readable,
            "retrieved_docs": sol.docs if hasattr(sol, 'docs') else [],
            "retrieved_doc_scores": sol.doc_scores.tolist()
            if hasattr(sol, 'doc_scores') and sol.doc_scores is not None
            else None,
            "predicted_answer": sol.answer if hasattr(sol, 'answer') else "",
        }

        detailed_logs.append(log_record)

    return retrieval_metrics, qa_metrics, detailed_logs


def main(argv: List[str] = None) -> None:
    parser = argparse.ArgumentParser(description="Experiment 1: Zero-Shot Relation-Aware Retrieval")
    parser.add_argument("--experiment-name", default="musique_demo", help="Experiment folder name")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/musique/subset_50/musique.json"),
        help="Path to 50-question Musique subset JSON",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://holygpu7c26106.rc.fas.harvard.edu:8000/v1",
        help="OpenAI-compatible endpoint for Qwen",
    )
    parser.add_argument(
        "--llm-model-name",
        default="Qwen/Qwen3-8B",
        help="LLM model name label (must match offline indexing)",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="facebook/contriever-msmarco",
        help="Embedding model name (must match offline indexing)",
    )
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="Workspace subdirectory used by offline indexing",
    )
    parser.add_argument(
        "--output-tag",
        default="exp1_zero_shot",
        help="Tag for output subdirectory under online_retrieval/",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["dpr", "hipporag", "mara"],
        choices=["dpr", "hipporag", "mara"],
        help="Which modes to run (default: all)",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=20,
        help="Number of top passages to retrieve (default: 20)",
    )
    parser.add_argument(
        "--linking-top-k",
        type=int,
        default=5,
        help="Number of top entities to use for linking (default: 5)",
    )
    parser.add_argument(
        "--max-qa-steps",
        type=int,
        default=3,
        help="Maximum number of QA steps (default: 3)",
    )
    parser.add_argument(
        "--qa-top-k",
        type=int,
        default=5,
        help="Number of top passages to use for QA (default: 5)",
    )
    parser.add_argument(
        "--disable-rerank-filter",
        type=lambda x: str(x).lower() in ['true', '1', 'yes', 'on'],
        default=True,
        nargs='?',
        const=True,
        help="Disable rerank filter and use top N facts directly (default: True). Pass False to enable rerank filter.",
    )
    parser.add_argument(
        "--num-facts-without-rerank",
        type=int,
        default=10,
        help="Number of facts to use when rerank filter is disabled (default: 10)",
    )

    args = parser.parse_args(argv)

    # Load dataset
    questions, gold_docs, gold_answers = load_musique_subset(args.dataset_path)
    if not questions:
        raise RuntimeError(f"No questions loaded from {args.dataset_path}")

    # Shared HippoRAG instance (reuse embeddings + graph)
    hippo = build_hipporag(
        experiment_name=args.experiment_name,
        llm_model_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.llm_base_url,
        workspace_subdir=args.workspace_subdir,
        use_relation_aware=False,  # We'll toggle per mode
        retrieval_top_k=args.retrieval_top_k,
        linking_top_k=args.linking_top_k,
        max_qa_steps=args.max_qa_steps,
        qa_top_k=args.qa_top_k,
        disable_rerank_filter=args.disable_rerank_filter,
        num_facts_without_rerank=args.num_facts_without_rerank,
    )

    # Base output directory
    base_out_dir = build_experiment_dir(
        args.experiment_name, "online_retrieval", args.output_tag
    )

    # Run selected modes
    for mode in args.modes:
        print(f"\n=== Running mode: {mode} ===")
        mode_dir = base_out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        if mode == "dpr":
            ret_metrics, qa_metrics, per_query = run_mode_dpr(
                hippo, questions, gold_docs, gold_answers
            )
            logs = per_query
        elif mode == "hipporag":
            ret_metrics, qa_metrics, per_query = run_mode_hipporag(
                hippo, questions, gold_docs, gold_answers
            )
            logs = per_query
        else:  # mara
            ret_metrics, qa_metrics, logs = run_mode_mara(
                hippo, questions, gold_docs, gold_answers
            )

        # Extract Recall@5 and F1
        recall5 = ret_metrics.get("Recall@5", None)
        f1 = qa_metrics.get("F1", None)

        metrics_payload = {
            "mode": mode,
            "retrieval_metrics": ret_metrics,
            "qa_metrics": qa_metrics,
            "Recall@5": recall5,
            "F1": f1,
            "num_questions": len(questions),
        }

        write_json(mode_dir / "metrics.json", metrics_payload)
        write_jsonl(mode_dir / "per_query_logs.jsonl", logs)

        # Also write a compact predictions file for easy inspection
        predictions = [
            {
                "question": log.get("question"),
                "predicted_answer": log.get("predicted_answer")
                or log.get("answer"),
                "gold_answers": log.get("gold_answers"),
                "retrieved_docs": log.get("retrieved_docs") or log.get("docs"),
            }
            for log in logs
        ]
        write_jsonl(mode_dir / "predictions.jsonl", predictions)

        print(f"Mode {mode}: Recall@5={recall5}, F1={f1}")
        print(f"  Metrics written to: {mode_dir / 'metrics.json'}")
        print(f"  Per-query logs written to: {mode_dir / 'per_query_logs.jsonl'}")
        print(f"  Predictions written to: {mode_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
