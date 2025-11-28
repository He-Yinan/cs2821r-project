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
    支持从包含推理文本的响应中提取 JSON。
    处理双编码JSON（JSON字符串中包含转义的JSON）。
    """
    import json
    
    # Debug: Log response characteristics
    has_newline_escape = '\\n' in real_response
    has_triples = 'triples' in real_response
    has_lt = '<' in real_response
    starts_with_brace = real_response.strip().startswith('{')
    logger.debug(f"Extraction input length: {len(real_response)} chars")
    logger.debug(f"Contains 'triples': {has_triples}")
    logger.debug(f"Contains '\\n': {has_newline_escape}")
    logger.debug(f"Contains '<': {has_lt}")
    logger.debug(f"Starts with '{{': {starts_with_brace}")
    logger.debug(f"First 200 chars: {real_response[:200]}")
    logger.debug(f"Last 200 chars: {real_response[-200:]}")
    
    # Strategy 0: Handle escaped JSON format (with literal \n characters)
    # Look for pattern like: {\n  "triples": [\n    ...\n  ]\n}
    # First, try to unescape the entire response if it contains escaped characters
    if '\\n' in real_response and '"triples"' in real_response:
        logger.debug("Strategy 0: Attempting escaped JSON extraction")
        # Try unescaping the entire response and parsing
        unescaped_response = real_response.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t').replace('\\r', '\r')
        # Try to find JSON in the unescaped version
        json_start = unescaped_response.find('{"triples"')
        if json_start == -1:
            json_start = unescaped_response.find('{\n  "triples"')
        if json_start != -1:
            # Find the matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            for i in range(json_start, len(unescaped_response)):
                char = unescaped_response[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = unescaped_response[json_start:i+1]
                            logger.debug(f"Strategy 0: Found JSON object, length: {len(json_str)}")
                            try:
                                parsed = json.loads(json_str)
                                if "triples" in parsed and isinstance(parsed["triples"], list):
                                    logger.debug(f"Strategy 0: Successfully extracted {len(parsed['triples'])} triples")
                                    return parsed["triples"]
                                else:
                                    logger.debug(f"Strategy 0: JSON parsed but no 'triples' key or not a list")
                            except json.JSONDecodeError as e:
                                logger.debug(f"Strategy 0: JSON decode error: {e}")
                                logger.debug(f"Strategy 0: JSON string (first 500 chars): {json_str[:500]}")
                            break
        
        # Also try to decode the entire response as a JSON string if it looks like one
        if real_response.strip().startswith('"') and real_response.strip().endswith('"'):
            try:
                decoded = json.loads(real_response)
                if isinstance(decoded, str):
                    # The decoded string might be JSON itself
                    inner_parsed = json.loads(decoded)
                    if "triples" in inner_parsed and isinstance(inner_parsed["triples"], list):
                        return inner_parsed["triples"]
            except (json.JSONDecodeError, ValueError):
                pass
    
    # Strategy 1: Try to find JSON block between ```json and ``` or just ```
    logger.debug("Strategy 1: Trying code block extraction")
    code_block_pattern = r'```(?:json)?\s*(\{.*?"triples".*?\})\s*```'
    code_match = re.search(code_block_pattern, real_response, re.DOTALL)
    if code_match:
        try:
            parsed = json.loads(code_match.group(1))
            if "triples" in parsed and isinstance(parsed["triples"], list):
                logger.debug(f"Strategy 1: Successfully extracted {len(parsed['triples'])} triples")
                return parsed["triples"]
        except json.JSONDecodeError as e:
            logger.debug(f"Strategy 1: JSON decode error: {e}")
    else:
        logger.debug("Strategy 1: No code block found")
    
    # Strategy 2: Remove common reasoning tags and extract JSON
    logger.debug("Strategy 2: Removing reasoning tags")
    cleaned = real_response
    # Remove reasoning tags - handle various formats
    before_len = len(cleaned)
    cleaned = re.sub(r'<[^>]*reasoning[^>]*>.*?</[^>]*reasoning[^>]*>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    logger.debug(f"Strategy 2: After cleaning, length: {len(cleaned)} (was {before_len})")
    logger.debug(f"Strategy 2: Cleaned first 300 chars: {cleaned[:300]}")
    
    # If cleaned response contains escaped JSON (with \n, \"), try to unescape it
    if '\\n' in cleaned and '"triples"' in cleaned:
        # Unescape the cleaned response
        unescaped_cleaned = cleaned.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t').replace('\\r', '\r')
        # Try to find JSON starting from "triples"
        json_start = unescaped_cleaned.find('{"triples"')
        if json_start == -1:
            json_start = unescaped_cleaned.find('{\n  "triples"')
        if json_start != -1:
            logger.debug(f"Strategy 2: Found 'triples' at position {json_start} in unescaped cleaned response")
            # Find matching closing brace
            brace_count = 0
            in_string = False
            escape_next = False
            for i in range(json_start, len(unescaped_cleaned)):
                char = unescaped_cleaned[i]
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\':
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = unescaped_cleaned[json_start:i+1]
                            logger.debug(f"Strategy 2: Found JSON object, length: {len(json_str)}")
                            try:
                                parsed = json.loads(json_str)
                                if "triples" in parsed and isinstance(parsed["triples"], list):
                                    logger.debug(f"Strategy 2: Successfully extracted {len(parsed['triples'])} triples")
                                    return parsed["triples"]
                                else:
                                    logger.debug(f"Strategy 2: JSON parsed but no 'triples' key or not a list")
                            except json.JSONDecodeError as e:
                                logger.debug(f"Strategy 2: JSON decode error: {e}")
                                logger.debug(f"Strategy 2: JSON string (first 500 chars): {json_str[:500]}")
                            break
        else:
            logger.debug("Strategy 2: Could not find 'triples' in unescaped cleaned response")
    
    # Strategy 3: Find JSON object by balanced braces - look for any { followed by "triples"
    logger.debug("Strategy 3: Trying balanced brace matching")
    # Find all potential start positions
    start_idx = -1
    search_text = cleaned
    attempts = 0
    while True:
        # Look for {"triples" pattern (with various whitespace)
        match = re.search(r'\{\s*"triples"\s*:', search_text)
        if not match:
            logger.debug(f"Strategy 3: No 'triples' pattern found after {attempts} attempts")
            break
        start_idx = match.start()
        attempts += 1
        logger.debug(f"Strategy 3: Found 'triples' at position {start_idx} (attempt {attempts})")
        
        # Find the matching closing brace using balanced brace counting
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(search_text)):
            char = search_text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = search_text[start_idx:i+1]
                        try:
                            parsed = json.loads(json_str)
                            if "triples" in parsed and isinstance(parsed["triples"], list):
                                logger.debug(f"Strategy 3: Successfully extracted {len(parsed['triples'])} triples")
                                return parsed["triples"]
                        except json.JSONDecodeError as e:
                            logger.debug(f"Strategy 3: JSON decode error at attempt {attempts}: {e}")
                            logger.debug(f"Strategy 3: Extracted JSON string length: {len(json_str)}, first 300 chars: {json_str[:300]}")
                            # Continue searching from after this position
                            search_text = search_text[i+1:]
                            start_idx = -1
                            break
        if start_idx == -1:
            break
        # If we got here without finding a match, try next occurrence
        search_text = search_text[start_idx + 1:]
    
    # Strategy 4: Try to find JSON at the end of the cleaned text
    # Often LLM puts JSON at the end after reasoning
    lines = cleaned.split('\n')
    json_lines = []
    in_json = False
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        if line.startswith('{') or in_json:
            json_lines.insert(0, line)
            in_json = True
            if line.endswith('}') and line.count('{') <= line.count('}'):
                # Might be complete JSON
                json_str = '\n'.join(json_lines)
                try:
                    parsed = json.loads(json_str)
                    if "triples" in parsed and isinstance(parsed["triples"], list):
                        return parsed["triples"]
                except json.JSONDecodeError:
                    pass
    
    # Strategy 5: Last resort - try to find any JSON-like structure with "triples"
    # Use a more permissive regex that handles nested structures
    json_match = re.search(r'\{\s*"triples"\s*:\s*\[.*?\]\s*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if "triples" in parsed and isinstance(parsed["triples"], list):
                return parsed["triples"]
        except json.JSONDecodeError:
            pass
    
    # Strategy 6: Final fallback - try to extract JSON from the entire cleaned response
    # Sometimes the JSON is the only valid JSON in the response
    if cleaned.strip().startswith('{'):
        try:
            parsed = json.loads(cleaned)
            if "triples" in parsed and isinstance(parsed["triples"], list):
                return parsed["triples"]
        except json.JSONDecodeError:
            pass
    
    # Strategy 7: Handle case where JSON is stored as a string with escaped characters
    # Look for the specific pattern: {\n  "triples": [\n    ...\n  ]\n}
    # This handles the case where JSON has literal \n and \" characters
    logger.debug("Strategy 7: Trying escaped character handling")
    if '\\n' in cleaned or '\\"' in cleaned:
        # First, try to unescape the entire cleaned string and parse
        unescaped_cleaned = cleaned.replace('\\n', '\n').replace('\\"', '"').replace('\\t', '\t').replace('\\r', '\r')
        try:
            parsed = json.loads(unescaped_cleaned)
            if "triples" in parsed and isinstance(parsed["triples"], list):
                return parsed["triples"]
        except json.JSONDecodeError:
            pass
        
        # If that fails, try to find the triples array directly using regex
        # Pattern: "triples": [ ... ] where ... contains the list items (with escaped chars)
        # Handle both escaped and unescaped versions
        for text_to_search in [cleaned, unescaped_cleaned]:
            triples_array_pattern = r'"triples"\s*:\s*\[(.*?)\]'
            array_match = re.search(triples_array_pattern, text_to_search, re.DOTALL)
            if array_match:
                array_content = array_match.group(1)
                # Look for 5-element lists: ["subject", "predicate", "object", "type", confidence]
                # Handle both escaped quotes \" and regular quotes "
                triple_pattern = r'\[(?:\\")?([^",\]]+)(?:\\")?\s*,\s*(?:\\")?([^",\]]+)(?:\\")?\s*,\s*(?:\\")?([^",\]]+)(?:\\")?\s*,\s*(?:\\")?([^",\]]+)(?:\\")?\s*,\s*([0-9.]+)\]'
                triple_matches = re.findall(triple_pattern, array_content)
                if triple_matches:
                    triples = []
                    for match in triple_matches:
                        subj, pred, obj, rtype, conf = match
                        # Clean up any remaining escape sequences
                        subj = subj.replace('\\"', '"').replace('\\n', ' ').strip()
                        pred = pred.replace('\\"', '"').replace('\\n', ' ').strip()
                        obj = obj.replace('\\"', '"').replace('\\n', ' ').strip()
                        rtype = rtype.replace('\\"', '"').strip()
                        try:
                            conf_float = float(conf)
                            triples.append([subj, pred, obj, rtype, conf_float])
                        except ValueError:
                            triples.append([subj, pred, obj, rtype, 0.5])
                    if triples:
                        return triples
    
    # If all else fails, return empty list
    logger.warning(f"All extraction strategies failed. Response length: {len(real_response)}")
    logger.warning(f"Response contains 'triples': {'triples' in real_response}")
    if '"triples"' in real_response:
        # Try to find where "triples" appears
        triples_pos = real_response.find('"triples"')
        logger.warning(f"'triples' found at position {triples_pos}")
        logger.warning(f"Context around 'triples' (500 chars): {real_response[max(0, triples_pos-100):min(len(real_response), triples_pos+400)]}")
        return []


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

        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )

        raw_response = ""
        metadata: Dict[str, Any] = {}
        try:
            # LLM INFERENCE with higher max_tokens for complete JSON output
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                max_completion_tokens=4096,  # Increased from default to handle longer outputs
            )
            metadata['cache_hit'] = cache_hit
            if metadata.get('finish_reason') == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response

            # Preprocess: Remove any reasoning tags that might have been included despite instructions
            # This helps ensure extraction works even if LLM doesn't follow instructions perfectly
            preprocessed_response = re.sub(
                r'<[^>]*(?:think|reasoning|redacted)[^>]*>.*?</[^>]*(?:think|reasoning|redacted)[^>]*>',
                '',
                real_response,
                flags=re.DOTALL | re.IGNORECASE
            )
            preprocessed_response = preprocessed_response.strip()

            logger.debug(f"Calling extraction for chunk {chunk_key}")
            extracted_triples = _extract_triples_from_response(preprocessed_response)
            
            # Fallback extraction if initial extraction failed
            if not extracted_triples and '"triples"' in real_response:
                logger.warning(f"Initial extraction failed for chunk {chunk_key}, attempting fallback")
                fallback_triples = _extract_triples_from_response(real_response)
                if fallback_triples:
                    logger.info(f"Fallback extraction succeeded for chunk {chunk_key}: found {len(fallback_triples)} triples")
                    extracted_triples = fallback_triples

            # 统一规范为 typed triples
            typed_triples: List[Dict[str, Any]] = []
            for t in extracted_triples:
                # Handle both dict and list/tuple formats
                if isinstance(t, (list, tuple)):
                    if len(t) >= 5:
                        # Full format: [subject, predicate, object, relation_type, confidence]
                        subj = str(t[0]).strip()
                        pred = str(t[1]).strip()
                        obj = str(t[2]).strip()
                        rtype = _normalize_relation_type(str(t[3]).strip() if len(t) > 3 else "ATTRIBUTION")
                        try:
                            conf = float(t[4]) if len(t) > 4 else 0.5
                        except (TypeError, ValueError):
                            conf = 0.5
                    elif len(t) >= 3:
                        # Minimal format: [subject, predicate, object]
                        subj = str(t[0]).strip()
                        pred = str(t[1]).strip()
                        obj = str(t[2]).strip()
                        rtype = "ATTRIBUTION"  # Default for list format
                        conf = 0.5  # Default confidence
                    elif len(t) == 2:
                        # Incomplete triple: [subject, predicate] - skip silently
                        continue
                    else:
                        # Invalid length - skip silently
                        continue
                elif isinstance(t, dict):
                    # Dict format: {"subject": ..., "predicate": ..., "object": ..., ...}
                    subj = str(t.get("subject", "")).strip()
                    pred = str(t.get("predicate", "")).strip()
                    obj = str(t.get("object", "")).strip()
                    rtype = _normalize_relation_type(t.get("relation_type", ""))
                    try:
                        conf = float(t.get("confidence", 0.5))
                    except (TypeError, ValueError):
                        conf = 0.5
                else:
                    # Unknown format - skip silently (was causing too many warnings)
                    continue

                # 主语或宾语缺失就丢弃
                if not subj or not obj:
                    continue
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
            # Convert typed triples (dicts) to list format for filtering
            list_format_triples = [[t["subject"], t["predicate"], t["object"]] for t in typed_triples]
            filtered_list_triples = filter_invalid_triples(triples=list_format_triples)
            
            # Convert back to typed format, preserving relation_type and confidence
            # Create a mapping from (subject, predicate, object) to the full typed triple
            typed_triples_map = {
                (t["subject"], t["predicate"], t["object"]): t 
                for t in typed_triples
            }
            triplets = [
                typed_triples_map[(subj, pred, obj)]
                for subj, pred, obj in filtered_list_triples
                if (subj, pred, obj) in typed_triples_map
            ]
            logger.info(f"Chunk {chunk_key}: {len(extracted_triples)} extracted -> {len(typed_triples)} typed -> {len(triplets)} valid triples")

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
