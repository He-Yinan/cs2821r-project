#!/usr/bin/env python3
"""
Step 2: reuse saved chunks + NER outputs to run triple extraction only.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
from pathlib import Path
from typing import Sequence, Dict

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.utils.config_utils import BaseConfig
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.information_extraction.openie_openai import OpenIE
from hipporag.utils.misc_utils import TripleRawOutput

from experiment.common.io_utils import build_experiment_dir, read_jsonl, write_jsonl, write_json


def init_openie(llm_name: str, llm_base_url: str, cache_root: Path) -> OpenIE:
    config = BaseConfig(
        save_dir=str(cache_root),
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        openie_mode="online",
    )
    llm = CacheOpenAI.from_experiment_config(config)
    return OpenIE(llm_model=llm)


def triple_worker(openie: OpenIE, chunk: dict, ner_info: dict) -> dict:
    triple_output: TripleRawOutput = openie.triple_extraction(
        chunk_key=chunk["chunk_id"],
        passage=chunk["content"],
        named_entities=ner_info.get("entities", []),
    )
    return {
        "chunk_id": triple_output.chunk_id,
        "triples": triple_output.triples,
        "raw_response": triple_output.response,
        "metadata": triple_output.metadata,
    }


def run_triples(openie: OpenIE, chunks: Dict[str, dict], ner_map: Dict[str, dict], max_workers: int):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for chunk_id, chunk in chunks.items():
            if chunk_id not in ner_map:
                continue
            futures[executor.submit(triple_worker, openie, chunk, ner_map[chunk_id])] = chunk_id
        for future in tqdm(as_completed(futures), total=len(futures), desc="Triples"):
            yield future.result()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Triple extraction step.")
    parser.add_argument("--experiment-name", required=True, help="Experiment folder name")
    parser.add_argument("--chunks-path", type=Path, default=None, help="Override path to chunks.jsonl")
    parser.add_argument("--ner-path", type=Path, default=None, help="Override path to ner.jsonl")
    parser.add_argument("--llm-base-url", required=True, help="OpenAI-compatible endpoint")
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-8B-Instruct", help="LLM model identifier")
    parser.add_argument("--max-workers", type=int, default=4, help="Threads for triple extraction")
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Limit number of chunks to process (for testing). If None, process all chunks.",
    )
    parser.add_argument(
        "--test-output",
        action="store_true",
        help="If set, save output to triples_test.jsonl instead of triples.jsonl",
    )
    parser.add_argument(
        "--input-subdir",
        default="offline_indexing/01_chunk_ner",
        help="Where to read chunks/ner if explicit paths not provided",
    )
    parser.add_argument(
        "--output-subdir",
        default="offline_indexing/02_triples",
        help="Where to store triple outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_dir = build_experiment_dir(args.experiment_name, args.input_subdir)
    output_dir = build_experiment_dir(args.experiment_name, args.output_subdir)
    cache_dir = build_experiment_dir(args.experiment_name, "llm_cache")

    chunks_path = args.chunks_path or (input_dir / "chunks.jsonl")
    ner_path = args.ner_path or (input_dir / "ner.jsonl")
    triples_path = output_dir / "triples.jsonl"
    manifest_path = output_dir / "manifest.json"

    chunk_records = read_jsonl(chunks_path)
    chunks = {rec["chunk_id"]: rec for rec in chunk_records}
    ner_records = read_jsonl(ner_path)
    ner_map = {rec["chunk_id"]: rec for rec in ner_records}

    # Limit chunks for testing if requested
    if args.max_chunks is not None and args.max_chunks > 0:
        chunk_ids = list(chunks.keys())[:args.max_chunks]
        chunks = {cid: chunks[cid] for cid in chunk_ids if cid in chunks}
        print(f"TEST MODE: Processing only {len(chunks)} chunks (limited from {len(chunk_records)})")

    # Determine output filename
    if args.test_output:
        triples_path = output_dir / "triples_test.jsonl"
        manifest_path = output_dir / "manifest_test.json"
        print(f"TEST MODE: Saving to {triples_path}")
    else:
        triples_path = output_dir / "triples.jsonl"
        manifest_path = output_dir / "manifest.json"

    openie = init_openie(args.llm_model_name, args.llm_base_url, cache_dir)
    triple_records = list(run_triples(openie, chunks, ner_map, args.max_workers))
    write_jsonl(triples_path, triple_records)

    manifest = {
        "chunks_path": str(chunks_path),
        "ner_path": str(ner_path),
        "triples_path": str(triples_path),
        "num_chunks": len(chunks),
        "num_triples": len(triple_records),
        "max_chunks_limit": args.max_chunks,
        "test_mode": args.test_output,
    }
    write_json(manifest_path, manifest)
    print(f"Triple outputs saved to {triples_path}")
    print(f"Total triples extracted: {sum(len(rec.get('triples', [])) for rec in triple_records)}")


if __name__ == "__main__":
    main()

