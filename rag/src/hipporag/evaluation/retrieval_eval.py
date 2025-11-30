from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import re


from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig




logger = get_logger(__name__)



class RetrievalRecall(BaseMetric):
    
    metric_name: str = "retrieval_recall"
    
    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)
        
    
    def calculate_metric_scores(self, gold_docs: List[List[str]], retrieved_docs: List[List[str]], k_list: List[int] = [1, 5, 10, 20], 
                                retrieved_doc_ids: Optional[List[List[str]]] = None) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates Recall@k for each example and pools results for all queries.
        
        Matching strategy:
        - If retrieved_doc_ids is provided, matches by checking if chunk content is contained in gold paragraphs
        - Otherwise, falls back to exact string matching

        Args:
            gold_docs (List[List[str]]): List of lists containing the ground truth (relevant documents/paragraphs) for each query.
            retrieved_docs (List[List[str]]): List of lists containing the retrieved documents (chunks) for each query.
            k_list (List[int]): List of k values to calculate Recall@k for.
            retrieved_doc_ids (Optional[List[List[str]]]): Optional list of chunk IDs for retrieved documents.

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A pooled dictionary with the averaged Recall@k across all examples.
                - A list of dictionaries with Recall@k for each example.
        """
        k_list = sorted(set(k_list))
        
        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in k_list}
        
        # Debug: log first example to diagnose matching issues
        debug_first_example = True
        
        for example_idx, (example_gold_docs, example_retrieved_docs) in enumerate(zip(gold_docs, retrieved_docs)):
            if len(example_retrieved_docs) < k_list[-1]:
                logger.warning(f"Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({k_list[-1]})")
            
            example_eval_result = {f"Recall@{k}": 0.0 for k in k_list}
            
            # Normalize gold docs (strip whitespace, convert to set for faster lookup)
            gold_docs_normalized = [doc.strip() for doc in example_gold_docs if doc.strip()]
            gold_docs_set = set(gold_docs_normalized)
            
            # Compute Recall@k for each k
            for k in k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                
                # Match retrieved chunks to gold paragraphs
                # Strategy: A chunk matches a gold paragraph if:
                # 1. Exact match (after normalization)
                # 2. Chunk is contained in paragraph (substring match)
                # 3. Paragraph is contained in chunk (chunk might be longer due to context)
                matched_gold_docs = set()
                
                for retrieved_chunk in top_k_docs:
                    retrieved_chunk_normalized = retrieved_chunk.strip()
                    if not retrieved_chunk_normalized:
                        continue
                    
                    # Normalize: remove extra whitespace, convert to single spaces
                    retrieved_chunk_normalized = re.sub(r'\s+', ' ', retrieved_chunk_normalized)
                    
                    # Extract paragraph text from chunk (handle title prefix)
                    # Chunks may have format "Title\nParagraph text" or just "Paragraph text"
                    chunk_lines = retrieved_chunk_normalized.split('\n', 1)
                    if len(chunk_lines) > 1:
                        # Has title prefix, extract paragraph text (second part)
                        chunk_paragraph_text = chunk_lines[1].strip()
                    else:
                        # No title, use full chunk
                        chunk_paragraph_text = retrieved_chunk_normalized
                    
                    # Normalize paragraph text
                    chunk_paragraph_normalized = re.sub(r'\s+', ' ', chunk_paragraph_text)
                    chunk_paragraph_lower = chunk_paragraph_normalized.lower()
                    
                    # Check if chunk exactly matches any gold doc
                    matched = False
                    for gold_doc in gold_docs_normalized:
                        gold_doc_normalized = re.sub(r'\s+', ' ', gold_doc.strip())
                        gold_doc_lower = gold_doc_normalized.lower()
                        
                        # Exact match (case-insensitive for robustness)
                        # Match against paragraph text (without title) for chunks with titles
                        if chunk_paragraph_lower == gold_doc_lower:
                            matched_gold_docs.add(gold_doc)
                            matched = True
                            break
                    
                    if not matched:
                        # Check substring matches (paragraph text in gold or gold in paragraph text)
                        # Use a minimum length threshold to avoid false matches
                        min_chunk_length = 30  # Minimum chunk length to consider
                        
                        for gold_doc in gold_docs_normalized:
                            gold_doc_normalized = re.sub(r'\s+', ' ', gold_doc.strip())
                            gold_doc_lower = gold_doc_normalized.lower()
                            
                            if len(chunk_paragraph_lower) >= min_chunk_length:
                                # Check if paragraph text (from chunk) is contained in gold paragraph
                                # This handles the case where chunk has title prefix
                                if chunk_paragraph_lower in gold_doc_lower:
                                    matched_gold_docs.add(gold_doc)
                                    matched = True
                                    break
                                # Check if gold paragraph is contained in chunk paragraph text
                                elif gold_doc_lower in chunk_paragraph_lower:
                                    matched_gold_docs.add(gold_doc)
                                    matched = True
                                    break
                                # Check word overlap (at least 80% of chunk paragraph words in gold)
                                elif len(chunk_paragraph_lower) > 0:
                                    chunk_words = set(chunk_paragraph_lower.split())
                                    gold_words = set(gold_doc_lower.split())
                                    if len(chunk_words) > 0:
                                        overlap_ratio = len(chunk_words & gold_words) / len(chunk_words)
                                        if overlap_ratio >= 0.8:  # 80% word overlap
                                            matched_gold_docs.add(gold_doc)
                                            matched = True
                                            break
                
                # Compute recall: how many gold paragraphs were matched
                if gold_docs_normalized:  # Avoid division by zero
                    example_eval_result[f"Recall@{k}"] = len(matched_gold_docs) / len(gold_docs_normalized)
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0
                
                # Debug logging for first example
                if debug_first_example and example_idx == 0 and k == 5:
                    logger.info(f"=== Retrieval Metrics Debug (Example 0, Recall@5) ===")
                    logger.info(f"  Gold docs count: {len(gold_docs_normalized)}")
                    logger.info(f"  Matched gold docs: {len(matched_gold_docs)}")
                    logger.info(f"  Retrieved chunks count: {len(top_k_docs)}")
                    logger.info(f"  Recall@5: {example_eval_result['Recall@5']:.4f}")
                    if len(matched_gold_docs) == 0 and len(top_k_docs) > 0:
                        logger.warning(f"  ⚠️  No matches found! Debugging...")
                        logger.info(f"  First retrieved chunk (first 300 chars):")
                        logger.info(f"    '{top_k_docs[0][:300] if top_k_docs else 'N/A'}'")
                        logger.info(f"  First gold doc (first 300 chars):")
                        logger.info(f"    '{gold_docs_normalized[0][:300] if gold_docs_normalized else 'N/A'}'")
                        # Check if there's any substring match
                        if top_k_docs and gold_docs_normalized:
                            chunk_full = top_k_docs[0]
                            chunk_norm = re.sub(r'\s+', ' ', chunk_full.strip())
                            # Extract paragraph text (handle title prefix)
                            chunk_lines = chunk_norm.split('\n', 1)
                            if len(chunk_lines) > 1:
                                chunk_para = chunk_lines[1].strip()
                                logger.info(f"  Chunk has title prefix: '{chunk_lines[0][:50]}...'")
                            else:
                                chunk_para = chunk_norm
                            chunk_para_lower = chunk_para.lower()
                            para_norm = re.sub(r'\s+', ' ', gold_docs_normalized[0].strip().lower())
                            logger.info(f"  Chunk paragraph text length: {len(chunk_para)}")
                            logger.info(f"  Gold paragraph length: {len(para_norm)}")
                            logger.info(f"  Paragraph text in gold? {chunk_para_lower in para_norm}")
                            logger.info(f"  Gold in paragraph text? {para_norm in chunk_para_lower}")
                            # Check word overlap
                            chunk_words = set(chunk_para_lower.split())
                            gold_words = set(para_norm.split())
                            if len(chunk_words) > 0:
                                overlap = len(chunk_words & gold_words)
                                overlap_ratio = overlap / len(chunk_words)
                                logger.info(f"  Word overlap: {overlap}/{len(chunk_words)} = {overlap_ratio:.2%}")
                    debug_first_example = False
            
            # Append example results
            example_eval_results.append(example_eval_result)
            
            # Accumulate pooled results
            for k in k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # Average pooled results over all examples
        num_examples = len(gold_docs)
        for k in k_list:
            pooled_eval_results[f"Recall@{k}"] /= num_examples

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results, example_eval_results