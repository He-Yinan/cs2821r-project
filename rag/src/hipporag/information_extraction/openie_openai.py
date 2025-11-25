import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response: str) -> List[str]:
    """
    从 LLM 输出中解析出 {"named_entities": [...]} 里的实体列表。
    """
    pattern = r'\{[^{}]*"named_entities"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    return eval(match.group())["named_entities"]


def _extract_triples_from_response(real_response: str) -> List[Dict[str, Any]]:
    """
    从 LLM 输出中解析出 {"triples": [...]} 里的三元组列表。
    """
    pattern = r'\{[^{}]*"triples"\s*:\s*\[[^\]]*\][^{}]*\}'
    match = re.search(pattern, real_response, re.DOTALL)
    if match is None:
        # If pattern doesn't match, return an empty list
        return []
    return eval(match.group())["triples"]


def _normalize_relation_type(rtype: str) -> str:
    """
    将 LLM 输出的 relation_type 归一化到固定的 5 类：

    - HIERARCHICAL
    - TEMPORAL
    - SPATIAL
    - CAUSALITY
    - ATTRIBUTION
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


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(
            role_mapping={"system": "system", "user": "user", "assistant": "assistant"}
        )
        self.llm_model = llm_model

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        """
        调用 LLM 做命名实体识别，返回 NerRawOutput。
        """
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(
            name='ner',
            passage=passage
        )
        raw_response = ""
        metadata: Dict[str, Any] = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            extracted_entities = _extract_ner_from_response(real_response)
            # 去重但保留顺序
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # 对异常做 logging，并把错误信息写到 metadata
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                unique_entities=[],
                metadata=metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(self, chunk_key: str, passage: str, named_entities: List[str]) -> TripleRawOutput:
        """
        调用 LLM 抽取带类型 + 置信度的三元组。

        每个 triple 的标准格式为：

        {
            "subject": str,
            "predicate": str,
            "object": str,
            "relation_type": str,  # HIERARCHICAL / TEMPORAL / SPATIAL / CAUSALITY / ATTRIBUTION
            "confidence": float    # in [0.0, 1.0]
        }
        """

        # PREPROCESSING：注意这里使用的是新的 typed 模板
        messages = self.prompt_template_manager.render(
            name='triple_extraction_typed',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata: Dict[str, Any] = {}
        try:
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            extracted_triples = _extract_triples_from_response(real_response)

            # 统一规范为 typed triples
            typed_triples: List[Dict[str, Any]] = []
            for t in extracted_triples:
                subj = str(t.get("subject", "")).strip()
                pred = str(t.get("predicate", "")).strip()
                obj = str(t.get("object", "")).strip()

                # 主语或宾语缺失就丢弃
                if not subj or not obj:
                    continue

                rtype = _normalize_relation_type(t.get("relation_type", ""))

                try:
                    conf = float(t.get("confidence", 0.5))
                except (TypeError, ValueError):
                    conf = 0.5
                # clamp 到 [0, 1]
                conf = max(0.0, min(1.0, conf))

                typed_triples.append(
                    {
                        "subject": subj,
                        "predicate": pred,
                        "object": obj,
                        "relation_type": rtype,
                        "confidence": conf,
                    }
                )

            # 仍沿用原来的过滤逻辑，只是 triple 里多了两个字段
            triplets = filter_invalid_triples(triples=typed_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        """
        单个 chunk 的 OpenIE：先 NER 再 triple_extraction。
        """
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(
            chunk_key=chunk_key,
            passage=passage,
            named_entities=ner_output.unique_entities
        )
        return {"ner": ner_output, "triplets": triple_output}

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        # ---------- NER ----------
        ner_results_list: List[NerRawOutput] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        with ThreadPoolExecutor() as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        # ---------- Triple Extraction ----------
        triple_results_list: List[TripleRawOutput] = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor() as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(
                    self.triple_extraction,
                    ner_result.chunk_id,
                    chunk_passages[ner_result.chunk_id],
                    ner_result.unique_entities
                ): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}
        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        return ner_results_dict, triple_results_dict
