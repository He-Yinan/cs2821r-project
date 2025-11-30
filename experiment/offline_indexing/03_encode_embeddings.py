#!/usr/bin/env python3
"""
Step 3: encode chunks, entities, and facts into embedding stores.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.embedding_model import _get_embedding_model_class
from hipporag.embedding_store import EmbeddingStore
from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import flatten_facts

from experiment.common.io_utils import build_experiment_dir, read_jsonl, write_json


def dedup_preserve(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def triples_to_strings(triples: List[tuple[str, str, str]]) -> List[str]:
    return [f"{subj} ||| {pred} ||| {obj}" for subj, pred, obj in triples]


def triples_to_strings_with_metadata(chunk_triples: List[List[dict]]) -> List[str]:
    """
    Convert triples to strings including relation_type and confidence.
    Format: "subject ||| predicate ||| object ||| relation_type ||| confidence"
    """
    fact_strings = []
    for triples in chunk_triples:
        for t in triples:
            if isinstance(t, dict):
                subj = str(t.get("subject", "")).strip()
                pred = str(t.get("predicate", "")).strip()
                obj = str(t.get("object", "")).strip()
                rel_type = str(t.get("relation_type", "ATTRIBUTION")).strip()
                conf = t.get("confidence", 1.0)
                try:
                    conf = float(conf)
                except (TypeError, ValueError):
                    conf = 1.0
                
                if subj and obj:
                    fact_strings.append(f"{subj} ||| {pred} ||| {obj} ||| {rel_type} ||| {conf:.3f}")
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                # Fallback for old format (no relation_type/confidence)
                subj, pred, obj = str(t[0]), str(t[1]), str(t[2])
                fact_strings.append(f"{subj} ||| {pred} ||| {obj} ||| ATTRIBUTION ||| 1.000")
    
    # Deduplicate while preserving order
    seen = set()
    unique_strings = []
    for s in fact_strings:
        if s not in seen:
            seen.add(s)
            unique_strings.append(s)
    
    return unique_strings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode embeddings for chunks/entities/facts.")
    parser.add_argument("--experiment-name", required=True, help="Experiment folder name")
    parser.add_argument(
        "--chunk-dir",
        default="offline_indexing/01_chunk_ner",
        help="Sub-directory holding chunks/ner inputs",
    )
    parser.add_argument(
        "--triple-dir",
        default="offline_indexing/02_triples",
        help="Sub-directory holding triple outputs",
    )
    parser.add_argument(
        "--output-subdir",
        default="offline_indexing/03_embeddings",
        help="Sub-directory for storing embedding manifests",
    )
    parser.add_argument(
        "--workspace-subdir",
        default="hipporag_workspace",
        help="Root folder (under experiment) where HippoRAG working dirs live",
    )
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-8B-Instruct", help="LLM name (for workspace label)")
    parser.add_argument("--embedding-model-name", default="facebook/contriever-msmarco", help="Embedding model to use")
    parser.add_argument("--embedding-batch-size", type=int, default=8, help="Batch size for encoding")
    parser.add_argument("--embedding-base-url", default=None, help="Optional OpenAI-compatible endpoint for embeddings")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    chunk_dir = build_experiment_dir(args.experiment_name, args.chunk_dir)
    triple_dir = build_experiment_dir(args.experiment_name, args.triple_dir)
    output_dir = build_experiment_dir(args.experiment_name, args.output_subdir)
    workspace_dir = build_experiment_dir(args.experiment_name, args.workspace_subdir)

    chunks = read_jsonl(chunk_dir / "chunks.jsonl")
    ner_records = read_jsonl(chunk_dir / "ner.jsonl")
    triples = read_jsonl(triple_dir / "triples.jsonl")

    chunk_texts = [c["content"] for c in chunks]
    entity_texts = dedup_preserve(
        entity for record in ner_records for entity in record.get("entities", [])
    )
    chunk_triples = [record.get("triples", []) for record in triples]
    
    # Option 1: Include relation_type and confidence in fact embeddings
    # This makes fact embeddings richer and can help with relation-aware retrieval
    fact_texts = triples_to_strings_with_metadata(chunk_triples)
    
    # Option 2: Original approach (only subject/predicate/object)
    # Uncomment below and comment above to use original format
    # fact_tuples = flatten_facts(chunk_triples)
    # fact_texts = triples_to_strings(fact_tuples)

    llm_label = args.llm_model_name.replace("/", "_")
    embedding_label = args.embedding_model_name.replace("/", "_")
    working_dir = workspace_dir / f"{llm_label}_{embedding_label}"

    embedding_config = BaseConfig(
        save_dir=str(workspace_dir),
        llm_name=args.llm_model_name,
        embedding_model_name=args.embedding_model_name,
        embedding_batch_size=args.embedding_batch_size,
        embedding_base_url=args.embedding_base_url,
    )
    embedding_cls = _get_embedding_model_class(args.embedding_model_name)
    embedding_model = embedding_cls(global_config=embedding_config, embedding_model_name=args.embedding_model_name)

    chunk_store = EmbeddingStore(
        embedding_model, str(working_dir / "chunk_embeddings"), args.embedding_batch_size, "chunk"
    )
    entity_store = EmbeddingStore(
        embedding_model, str(working_dir / "entity_embeddings"), args.embedding_batch_size, "entity"
    )
    fact_store = EmbeddingStore(
        embedding_model, str(working_dir / "fact_embeddings"), args.embedding_batch_size, "fact"
    )

    chunk_store.insert_strings(chunk_texts)
    entity_store.insert_strings(entity_texts)
    fact_store.insert_strings(fact_texts)

    write_json(
        output_dir / "manifest.json",
        {
            "workspace_dir": str(working_dir),
            "chunk_store": str(chunk_store.filename),
            "entity_store": str(entity_store.filename),
            "fact_store": str(fact_store.filename),
            "num_chunks": len(chunk_texts),
            "num_entities": len(entity_texts),
            "num_facts": len(fact_texts),
        },
    )
    print(f"Embedding stores saved under {output_dir}")


if __name__ == "__main__":
    main()

