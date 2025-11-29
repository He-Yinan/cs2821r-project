#!/usr/bin/env python3
"""
Step 4: load embeddings + OpenIE outputs to build the HippoRAG graph artifacts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "HippoRAG" / "src"))

from hipporag.HippoRAG import HippoRAG
from hipporag.utils.config_utils import BaseConfig

from experiment.common.io_utils_baseline import build_experiment_dir, read_jsonl, write_json


def build_openie_info(chunks: List[dict], ner_map: Dict[str, dict], triple_map: Dict[str, dict]) -> List[dict]:
    records = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        records.append(
            {
                "idx": chunk_id,
                "passage": chunk["content"],
                "extracted_entities": ner_map.get(chunk_id, {}).get("entities", []),
                "extracted_triples": triple_map.get(chunk_id, {}).get("triples", []),
            }
        )
    return records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HippoRAG graph from cached artifacts.")
    parser.add_argument("--experiment-name", required=True, help="Experiment folder name")
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-8B-Instruct", help="LLM name label")
    parser.add_argument("--embedding-model-name", default="facebook/contriever-msmarco", help="Embedding model name")
    parser.add_argument("--llm-base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint")
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="Location (under experiment) for HippoRAG save_dir",
    )
    parser.add_argument(
        "--chunk-dir",
        default="offline_indexing/01_chunk_ner",
        help="Directory holding chunks/ner outputs",
    )
    parser.add_argument(
        "--triple-dir",
        default="offline_indexing/02_triples",
        help="Directory holding triple outputs",
    )
    parser.add_argument(
        "--output-subdir",
        default="offline_indexing/04_graph",
        help="Directory to store build manifest",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    chunk_dir = build_experiment_dir(args.experiment_name, args.chunk_dir)
    triple_dir = build_experiment_dir(args.experiment_name, args.triple_dir)
    output_dir = build_experiment_dir(args.experiment_name, args.output_subdir)
    workspace_dir = build_experiment_dir(args.experiment_name, args.workspace_subdir)

    chunks = read_jsonl(chunk_dir / "chunks.jsonl")
    ner_map = {rec["chunk_id"]: rec for rec in read_jsonl(chunk_dir / "ner.jsonl")}
    triple_map = {rec["chunk_id"]: rec for rec in read_jsonl(triple_dir / "triples.jsonl")}

    config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        llm_base_url=args.llm_base_url,
        openie_mode="online",
    )
    hipporag = HippoRAG(global_config=config)

    all_openie_info = build_openie_info(chunks, ner_map, triple_map)
    hipporag.save_openie_results(all_openie_info)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    chunk_triples = [triple_map.get(cid, {}).get("triples", []) for cid in chunk_ids]
    chunk_entities = [ner_map.get(cid, {}).get("entities", []) for cid in chunk_ids]

    hipporag.node_to_node_stats = {}
    hipporag.ent_node_to_chunk_ids = {}

    hipporag.add_fact_edges(chunk_ids, chunk_triples)
    num_new_chunks = hipporag.add_passage_edges(chunk_ids, chunk_entities)

    if num_new_chunks > 0:
        hipporag.add_synonymy_edges()
        hipporag.augment_graph()
        hipporag.save_igraph()

    write_json(
        output_dir / "manifest.json",
        {
            "workspace_dir": str(workspace_dir),
            "graph_path": str(getattr(hipporag, "_graph_pickle_filename", "")),
            "num_chunks": len(chunk_ids),
            "num_entities": len(hipporag.entity_embedding_store.hash_ids),
            "num_facts": len(hipporag.fact_embedding_store.hash_ids),
            "num_new_chunks": num_new_chunks,
        },
    )
    print(f"Graph updated. Saved manifest to {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()

