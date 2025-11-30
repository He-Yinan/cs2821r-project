from argparse import ArgumentTypeError
from dataclasses import dataclass
from hashlib import md5
from typing import Dict, Any, List, Tuple, Literal, Union, Optional
import numpy as np
import re
import logging

from .typing import Triple
from .llm_utils import filter_invalid_triples

logger = logging.getLogger(__name__)


@dataclass
class NerRawOutput:
    chunk_id: str
    response: str
    unique_entities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TripleRawOutput:
    chunk_id: str
    response: str
    # NOTE: triples 现在可以是 typed dict（subject/predicate/object/relation_type/confidence）
    triples: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class LinkingOutput:
    score: np.ndarray
    type: Literal['node', 'dpr']


@dataclass
class QuerySolution:
    question: str
    docs: List[str]
    doc_scores: np.ndarray = None
    doc_ids: Optional[List[str]] = None  # Chunk IDs for the retrieved documents
    answer: str = None
    gold_answers: List[str] = None
    gold_docs: Optional[List[str]] = None

    def to_dict(self):
        return {
            "question": self.question,
            "answer": self.answer,
            "gold_answers": self.gold_answers,
            "docs": self.docs[:5],
            "doc_scores": [round(v, 4) for v in self.doc_scores.tolist()[:5]] if self.doc_scores is not None else None,
            "gold_docs": self.gold_docs,
        }


def _normalize_relation_type(rtype: str) -> str:
    """
    归一化 relation_type 到固定的 5 类：
    HIERARCHICAL / TEMPORAL / SPATIAL / CAUSALITY / ATTRIBUTION
    """
    if not rtype:
        return "ATTRIBUTION"
    rtype = str(rtype).strip().upper()

    mapping = {
        "HIERARCHICAL": "HIERARCHICAL",
        "TEMPORAL": "TEMPORAL",
        "SPATIAL": "SPATIAL",
        "CAUSAL": "CAUSALITY",
        "CAUSALITY": "CAUSALITY",
        "ATTRIBUTION": "ATTRIBUTION",
    }
    return mapping.get(rtype, "ATTRIBUTION")


def text_processing(text):
    if isinstance(text, list):
        return [text_processing(t) for t in text]
    if not isinstance(text, str):
        text = str(text)
    return re.sub('[^A-Za-z0-9 ]', ' ', text.lower()).strip()


def reformat_openie_results(corpus_openie_results) -> (Dict[str, NerRawOutput], Dict[str, TripleRawOutput]):
    """
    把离线/缓存的 OpenIE 结果整理成：
      - ner_output_dict: {chunk_id -> NerRawOutput}
      - triple_output_dict: {chunk_id -> TripleRawOutput (typed triples)}
    兼容两种 triple 格式：
      1) 旧格式: ["subj", "pred", "obj"]
      2) 新格式: {"subject", "predicate", "object", "relation_type", "confidence"}
    """

    ner_output_dict = {
        chunk_item['idx']: NerRawOutput(
            chunk_id=chunk_item['idx'],
            response=None,
            metadata={},
            unique_entities=list(np.unique(chunk_item.get('extracted_entities', [])))
        )
        for chunk_item in corpus_openie_results
    }

    triple_output_dict: Dict[str, TripleRawOutput] = {}

    for chunk_item in corpus_openie_results:
        chunk_id = chunk_item['idx']
        raw_triples = chunk_item.get('extracted_triples', [])

        typed_triples: List[Dict[str, Any]] = []
        for t in raw_triples:
            # 新格式：字典
            if isinstance(t, dict):
                subj = str(t.get("subject", "")).strip()
                pred = str(t.get("predicate", "")).strip()
                obj = str(t.get("object", "")).strip()

                if not subj or not obj:
                    continue

                rtype = _normalize_relation_type(t.get("relation_type", ""))
                try:
                    conf = float(t.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                conf = max(0.0, min(1.0, conf))

            # 旧格式：列表/元组 ["subj", "pred", "obj", ...]
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                subj = str(t[0]).strip()
                pred = str(t[1]).strip()
                obj = str(t[2]).strip()
                if not subj or not obj:
                    continue
                # 没有类型信息时退化为 ATTRIBUTION + 0.5
                rtype = "ATTRIBUTION"
                conf = 0.5
            else:
                # 无法解析的 triple，丢弃
                logger.warning(f"Invalid triple format in corpus_openie_results[{chunk_id}]: {t}")
                continue

            typed_triples.append(
                {
                    "subject": subj,
                    "predicate": pred,
                    "object": obj,
                    "relation_type": rtype,
                    "confidence": conf,
                }
            )

        # 可选的统一过滤逻辑（注意需要 filter_invalid_triples 支持 dict）
        typed_triples = filter_invalid_triples(triples=typed_triples)

        triple_output_dict[chunk_id] = TripleRawOutput(
            chunk_id=chunk_id,
            response=None,
            metadata={},
            triples=typed_triples
        )

    return ner_output_dict, triple_output_dict


def extract_entity_nodes(chunk_triples: List[List[Triple]]) -> (List[str], List[List[str]]):
    """
    从每个 chunk 的 triples 中抽取实体节点列表。
    兼容：
      - dict triple: {"subject": ..., "object": ...}
      - list/tuple triple: [subj, pred, obj]
    """
    chunk_triple_entities: List[List[str]] = []  # a list of lists of unique entities from each chunk's triples

    for triples in chunk_triples:
        triple_entities = set()
        for t in triples:
            subj = None
            obj = None

            if isinstance(t, dict):
                subj = t.get("subject")
                obj = t.get("object")
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                subj = t[0]
                obj = t[2]
            else:
                logger.warning(f"During graph construction, invalid triple is found: {t}")
                continue

            if subj is not None and obj is not None:
                triple_entities.update([str(subj), str(obj)])

        chunk_triple_entities.append(list(triple_entities))

    graph_nodes = list(np.unique([ent for ents in chunk_triple_entities for ent in ents]))
    return graph_nodes, chunk_triple_entities


def flatten_facts(chunk_triples: List[Triple]) -> List[Triple]:
    """
    将所有 chunk 的 triples 展平成去重后的 (subject, predicate, object) 三元组列表，
    用于 fact embedding 等场景。

    注意：
    - 即使 triple 是 dict，也只取 subject/predicate/object 三个字段。
    - 为了保持一致性，所有 facts 都存储为 tuple 格式，但 relation_type 信息
      已经保存在 graph edges 中，可以通过 graph 查询获取。
    """
    graph_triples = []  # a list of unique relation triple (in tuple) from all chunks
    for triples in chunk_triples:
        for t in triples:
            if isinstance(t, dict):
                subj = t.get("subject")
                pred = t.get("predicate")
                obj = t.get("object")
                if subj is None or obj is None:
                    continue
                # Store as tuple for fact embedding
                # Note: relation_type is preserved in graph edges, not in fact strings
                tup = (str(subj), str(pred), str(obj))
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                # Plain triple format
                tup = tuple(str(t[i]) for i in range(min(3, len(t))))
            else:
                continue
            graph_triples.append(tup)

    graph_triples = list(set(graph_triples))
    return graph_triples


def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val

    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x

    return (x - min_val) / range_val


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """
    Compute the MD5 hash of the given content string and optionally prepend a prefix.

    Args:
        content (str): The input string to be hashed.
        prefix (str, optional): A string to prepend to the resulting hash. Defaults to an empty string.

    Returns:
        str: A string consisting of the prefix followed by the hexadecimal representation of the MD5 hash.
    """
    return prefix + md5(content.encode()).hexdigest()


def all_values_of_same_length(data: dict) -> bool:
    """
    Return True if all values in 'data' have the same length or data is an empty dict,
    otherwise return False.
    """
    # Get an iterator over the dictionary's values
    value_iter = iter(data.values())

    # Get the length of the first sequence (handle empty dict case safely)
    try:
        first_length = len(next(value_iter))
    except StopIteration:
        # If the dictionary is empty, treat it as all having "the same length"
        return True

    # Check that every remaining sequence has this same length
    return all(len(seq) == first_length for seq in value_iter)


def string_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )
