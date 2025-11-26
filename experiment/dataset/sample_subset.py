#!/usr/bin/env python3
"""
Utility to subsample the reproduced QA datasets together with their corpora.

Given a dataset name (e.g. musique, hotpotqa, 2wikimultihopqa), the script:
  1. Loads both <name>.json and <name>_corpus.json from the source directory.
  2. Randomly samples N indices without replacement (controlled by --seed).
  3. Writes subset copies of both files to the scratch results directory.

Outputs are stored under:
  /n/netscratch/tambe_lab/Lab/msong300/cs2821r-results/datasets/<name>/
with suffixes *_subset.json and *_subset_corpus.json to match the original format.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Sequence, Any

SCRATCH_ROOT = Path("/n/netscratch/tambe_lab/Lab/msong300/cs2821r-results")
DEFAULT_SOURCE_ROOT = Path("rag/reproduce/dataset")


def load_json_list(path: Path) -> List:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in {path}, found {type(data)}")
        return data


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def choose_indices(pop_size: int, sample_size: int, seed: int | None) -> List[int]:
    if sample_size > pop_size:
        raise ValueError(f"Cannot sample {sample_size} items from population {pop_size}")
    rng = random.Random(seed)
    return sorted(rng.sample(range(pop_size), sample_size))


def build_musique_corpus(subset_samples: List[dict]) -> List[dict]:
    corpus = []
    for sample in subset_samples:
        for paragraph in sample.get("paragraphs", []):
            title = paragraph.get("title") or paragraph.get("paragraph_title") or ""
            text = paragraph.get("paragraph_text") or paragraph.get("text") or ""
            if not text:
                continue
            corpus.append({"title": str(title), "text": str(text)})
    return corpus


def build_hotpotqa_corpus(subset_samples: List[dict]) -> List[dict]:
    corpus = []
    idx_counter = 0
    for sample in subset_samples:
        for context in sample.get("context", []):
            if not isinstance(context, list) or len(context) < 2:
                continue
            title = context[0]
            sentences = context[1]
            if isinstance(sentences, list):
                text = "".join(sentences)
            else:
                text = str(sentences)
            corpus.append({"idx": idx_counter, "title": str(title), "text": text})
            idx_counter += 1
    return corpus


def build_2wikimultihopqa_corpus(subset_samples: List[dict]) -> List[dict]:
    corpus = []
    for sample in subset_samples:
        for context in sample.get("context", []):
            if not isinstance(context, list) or len(context) < 2:
                continue
            title = context[0]
            paragraphs = context[1]
            if isinstance(paragraphs, list):
                text = "".join(paragraphs)
            else:
                text = str(paragraphs)
            corpus.append({"title": str(title), "text": text})
    return corpus


def build_corpus_entries(
    dataset_name: str,
    subset_samples: List[dict],
    indices: List[int],
    source_root: Path,
) -> List[dict]:
    name = dataset_name.lower()
    if name == "musique":
        return build_musique_corpus(subset_samples)
    if name.startswith("hotpotqa"):
        return build_hotpotqa_corpus(subset_samples)
    if name in {"2wikimultihopqa", "2wiki"} or name.startswith("2wikimultihopqa"):
        return build_2wikimultihopqa_corpus(subset_samples)

    corpus_path = source_root / f"{dataset_name}_corpus.json"
    if corpus_path.exists():
        full_corpus = load_json_list(corpus_path)
        return [full_corpus[i] for i in indices]

    raise FileNotFoundError(f"No corpus available for dataset {dataset_name}")


def subset_dataset(
    dataset_name: str,
    sample_size: int,
    seed: int | None = None,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    output_root: Path = SCRATCH_ROOT / "datasets",
    output_tag: str | None = None,
) -> dict:
    dataset_path = source_root / f"{dataset_name}.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    samples = load_json_list(dataset_path)
    indices = choose_indices(len(samples), sample_size, seed)
    subset_samples = [samples[i] for i in indices]
    subset_corpus = build_corpus_entries(dataset_name, subset_samples, indices, source_root)

    tag = output_tag or f"subset_{sample_size}"
    out_dir = output_root / dataset_name / tag
    dataset_out = out_dir / f"{dataset_name}.json"
    corpus_out = out_dir / f"{dataset_name}_corpus.json"
    index_out = out_dir / "indices.json"

    write_json(dataset_out, subset_samples)
    write_json(corpus_out, subset_corpus)
    write_json(index_out, {"indices": indices, "seed": seed, "sample_size": sample_size})

    return {
        "dataset_path": str(dataset_out),
        "corpus_path": str(corpus_out),
        "indices_path": str(index_out),
        "num_samples": len(indices),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subsample HippoRAG datasets with matching corpora.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., musique, hotpotqa, 2wikimultihopqa)")
    parser.add_argument("--num-samples", type=int, required=True, help="Number of QA pairs to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT, help="Folder containing full datasets")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SCRATCH_ROOT / "datasets",
        help="Scratch directory to store subset outputs",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional tag appended to the output folder name (default: subset_<N>)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    info = subset_dataset(
        dataset_name=args.dataset,
        sample_size=args.num_samples,
        seed=args.seed,
        source_root=args.source_root,
        output_root=args.output_root,
        output_tag=args.tag,
    )
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

