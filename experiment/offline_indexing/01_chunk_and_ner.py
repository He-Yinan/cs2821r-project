#!/usr/bin/env python3
"""
Step 1 of the modular offline indexing pipeline:
  - Load a (sub)set of corpus documents.
  - Generate deterministic chunk IDs for each passage.
  - Run NER with the OpenIE module (using the vLLM-served OpenAI-compatible endpoint).
  - Persist chunk metadata and NER outputs to the shared scratch directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.utils.config_utils import BaseConfig
from hipporag.utils.misc_utils import compute_mdhash_id, NerRawOutput
from hipporag.llm.openai_gpt import CacheOpenAI
from hipporag.information_extraction.openie_openai import OpenIE

from experiment.common.io_utils import (
    SCRATCH_ROOT,
    build_experiment_dir,
    load_json,
    write_json,
    write_jsonl,
)


def format_passage(doc: dict) -> str:
    title = str(doc.get("title", "") or "").strip()
    text = str(doc.get("text", "") or doc.get("paragraph_text", "") or "").strip()
    if title and text:
        return f"{title}\n{text}"
    return title or text


def chunk_corpus(corpus: list[dict]) -> list[dict]:
    chunks = []
    for idx, doc in enumerate(corpus):
        content = format_passage(doc)
        if not content:
            continue
        chunk_id = compute_mdhash_id(content, prefix="chunk-")
        chunks.append(
            {
                "chunk_id": chunk_id,
                "content": content,
                "source_index": idx,
                "title": doc.get("title"),
            }
        )
    return chunks


def ner_worker(openie: OpenIE, chunk: dict) -> dict:
    ner_output: NerRawOutput = openie.ner(chunk_key=chunk["chunk_id"], passage=chunk["content"])
    return {
        "chunk_id": ner_output.chunk_id,
        "entities": ner_output.unique_entities,
        "raw_response": ner_output.response,
        "metadata": ner_output.metadata,
    }


def run_ner(openie: OpenIE, chunks: list[dict], max_workers: int) -> Iterable[dict]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(ner_worker, openie, chunk): chunk for chunk in chunks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="NER"):
            yield future.result()


def wait_for_vllm_server(base_url: str, max_wait_seconds: int = 300, check_interval: int = 5) -> None:
    """
    Wait for the vLLM server to be ready by checking the /v1/models endpoint.
    
    Args:
        base_url: Base URL of the vLLM server (e.g., http://host:8000/v1)
        max_wait_seconds: Maximum time to wait in seconds (default: 5 minutes)
        check_interval: Seconds between health checks (default: 5)
    
    Raises:
        ConnectionError: If server is not ready after max_wait_seconds
    """
    models_url = f"{base_url.rstrip('/v1')}/v1/models"
    start_time = time.time()
    
    print(f"Waiting for vLLM server at {base_url}...")
    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.get(models_url, timeout=5)
            if response.status_code == 200:
                models = response.json()
                print(f"✓ vLLM server is ready! Available models: {[m.get('id') for m in models.get('data', [])]}")
                return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            elapsed = int(time.time() - start_time)
            print(f"  Server not ready yet (waited {elapsed}s)... retrying in {check_interval}s")
            time.sleep(check_interval)
        except Exception as e:
            print(f"  Unexpected error checking server: {e}")
            time.sleep(check_interval)
    
    raise ConnectionError(
        f"vLLM server at {base_url} did not become ready after {max_wait_seconds} seconds. "
        f"Please check:\n"
        f"  1. Is the vLLM server job running? (squeue -u $USER)\n"
        f"  2. Is the server listening on the correct host/port?\n"
        f"  3. Check server logs: /n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/logs/vllm-qwen-*.out"
    )


def init_openie(llm_name: str, llm_base_url: str, cache_root: Path) -> OpenIE:
    config = BaseConfig(
        save_dir=str(cache_root),
        llm_name=llm_name,
        llm_base_url=llm_base_url,
        openie_mode="online",
    )
    llm = CacheOpenAI.from_experiment_config(config)
    return OpenIE(llm_model=llm)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk corpus and run NER (step 1).")
    parser.add_argument("--experiment-name", required=True, help="Name for the scratch experiment folder")
    parser.add_argument("--corpus-file", required=True, type=Path, help="Path to <dataset>_corpus.json subset")
    parser.add_argument("--llm-base-url", required=True, help="OpenAI-compatible endpoint for the vLLM server")
    parser.add_argument("--llm-model-name", default="Qwen/Qwen3-8B-Instruct", help="LLM model identifier")
    parser.add_argument("--max-workers", type=int, default=4, help="Threads for concurrent NER calls")
    parser.add_argument(
        "--output-subdir",
        default="offline_indexing/01_chunk_ner",
        help="Relative subdirectory (under experiment) for outputs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    experiment_root = build_experiment_dir(args.experiment_name)
    output_dir = build_experiment_dir(args.experiment_name, args.output_subdir)
    cache_dir = build_experiment_dir(args.experiment_name, "llm_cache")

    corpus = load_json(args.corpus_file)
    chunks = chunk_corpus(corpus)

    chunks_path = output_dir / "chunks.jsonl"
    ner_path = output_dir / "ner.jsonl"
    manifest_path = output_dir / "manifest.json"

    write_jsonl(chunks_path, chunks)

    # Wait for vLLM server to be ready before starting NER
    wait_for_vllm_server(args.llm_base_url)
    
    openie = init_openie(args.llm_model_name, args.llm_base_url, cache_dir)
    ner_records = list(run_ner(openie, chunks, max_workers=args.max_workers))
    write_jsonl(ner_path, ner_records)

    manifest = {
        "corpus_file": str(args.corpus_file),
        "chunks_path": str(chunks_path),
        "ner_path": str(ner_path),
        "num_chunks": len(chunks),
        "scratch_root": str(SCRATCH_ROOT),
    }
    write_json(manifest_path, manifest)
    print(f"Chunks saved to {chunks_path}")
    print(f"NER outputs saved to {ner_path}")
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()

