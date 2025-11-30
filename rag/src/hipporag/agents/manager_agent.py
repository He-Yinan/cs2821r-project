"""
Manager Agent for generating relation influence factors.
"""

import json
import logging
from typing import Dict, Any, Optional
import numpy as np

from ..llm.base import BaseLLM
from ..prompts.prompt_template_manager import PromptTemplateManager
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class ManagerAgent:
    """
    Manager agent that generates relation influence factors (β values) for relation-aware retrieval.
    
    Given a query, it determines how relevant each relation type is for answering the query.
    """
    
    def __init__(self, llm: BaseLLM, prompt_template_manager: PromptTemplateManager):
        """
        Initialize the Manager Agent.
        
        Args:
            llm: The LLM model to use for generating relation influence factors
            prompt_template_manager: The prompt template manager for rendering prompts
        """
        self.llm = llm
        self.prompt_template_manager = prompt_template_manager
        
    def get_relation_influence_factors(self, query: str) -> Dict[str, Dict[str, float]]:
        """
        Generate relation influence factors (β values) for a given query.
        
        Args:
            query: The input query string
            
        Returns:
            Dictionary with structure:
            {
                "entity_entity": {
                    "HIERARCHICAL": float,
                    "TEMPORAL": float,
                    "SPATIAL": float,
                    "CAUSALITY": float,
                    "ATTRIBUTION": float
                },
                "entity_passage": {
                    "PRIMARY": float,
                    "SECONDARY": float,
                    "PERIPHERAL": float
                }
            }
            All values are normalized to sum to 1.0 within each category.
        """
        try:
            # Render the prompt template
            if self.prompt_template_manager.is_template_name_valid('relation_influence'):
                messages = self.prompt_template_manager.render(
                    name='relation_influence',
                    query=query
                )
            else:
                logger.warning("relation_influence template not found, using fallback")
                messages = self._create_fallback_prompt(query)
            
            # Call LLM
            llm_response = self.llm.infer(messages=messages)
            
            # Parse response
            # The infer method returns (message, metadata, cache_hit) due to cache decorator
            if isinstance(llm_response, tuple):
                if len(llm_response) == 3:
                    # Cache-enabled: (message, metadata, cache_hit)
                    response_messages, metadata, cache_hit = llm_response
                elif len(llm_response) == 2:
                    # Non-cached: (message, metadata)
                    response_messages, metadata = llm_response
                else:
                    # Fallback: assume first element is the message
                    response_messages = llm_response[0]
                    metadata = llm_response[1] if len(llm_response) > 1 else {}
                
                # Extract content from response_messages
                if isinstance(response_messages, str):
                    response_content = response_messages
                elif isinstance(response_messages, list) and len(response_messages) > 0:
                    response_content = response_messages[0].get('content', '') if isinstance(response_messages[0], dict) else str(response_messages[0])
                else:
                    response_content = str(response_messages)
            else:
                response_content = str(llm_response)
            
            # Extract JSON from response
            beta_values = self._parse_response(response_content)
            
            # Validate and normalize
            beta_values = self._validate_and_normalize(beta_values)
            
            logger.debug(f"Generated relation influence factors for query: {query[:50]}...")
            logger.debug(f"Entity-Entity: {beta_values['entity_entity']}")
            logger.debug(f"Entity-Passage: {beta_values['entity_passage']}")
            
            return beta_values
            
        except Exception as e:
            logger.error(f"Error generating relation influence factors: {e}")
            # Return uniform distribution as fallback
            return self._get_uniform_factors()
    
    def _parse_response(self, response_content: str) -> Dict[str, Dict[str, float]]:
        """
        Parse JSON response from LLM.
        
        Handles cases where LLM includes reasoning text before/after JSON.
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            Parsed beta values dictionary
        """
        import re
        
        # Try to extract JSON from response
        response_content = response_content.strip()
        
        # Strategy 1: Remove markdown code blocks if present
        if response_content.startswith('```'):
            lines = response_content.split('\n')
            # Remove first line (```json or ```)
            if len(lines) > 1:
                response_content = '\n'.join(lines[1:])
            # Remove last line (```)
            if response_content.endswith('```'):
                response_content = response_content[:-3].strip()
        
        # Strategy 2: Find all potential JSON objects by matching braces
        # Look for patterns that might be JSON objects
        potential_jsons = []
        i = 0
        while i < len(response_content):
            if response_content[i] == '{':
                # Found start of potential JSON, try to find matching closing brace
                brace_count = 0
                start = i
                for j in range(i, len(response_content)):
                    if response_content[j] == '{':
                        brace_count += 1
                    elif response_content[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Found complete JSON object
                            potential_jsons.append((start, j + 1))
                            i = j + 1
                            break
                else:
                    i += 1
            else:
                i += 1
        
        # Try each potential JSON object
        for start, end in potential_jsons:
            json_str = response_content[start:end]
            try:
                beta_values = json.loads(json_str)
                # Validate it has the required structure
                if isinstance(beta_values, dict) and 'entity_entity' in beta_values and 'entity_passage' in beta_values:
                    return beta_values
            except json.JSONDecodeError:
                continue
        
        # Strategy 3: Find first complete JSON object by matching braces
        start_idx = response_content.find('{')
        if start_idx >= 0:
            brace_count = 0
            end_idx = -1
            for i in range(start_idx, len(response_content)):
                if response_content[i] == '{':
                    brace_count += 1
                elif response_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i
                        break
            
            if end_idx > start_idx:
                json_str = response_content[start_idx:end_idx + 1]
                try:
                    beta_values = json.loads(json_str)
                    # Validate structure
                    if isinstance(beta_values, dict) and 'entity_entity' in beta_values and 'entity_passage' in beta_values:
                        return beta_values
                except json.JSONDecodeError:
                    pass
        
        # Strategy 4: Try to find JSON after common prefixes
        # Sometimes LLM adds "Here's the JSON:" or similar
        for prefix in ['json:', '```json', '```', 'output:', 'response:']:
            idx = response_content.lower().find(prefix.lower())
            if idx >= 0:
                # Try to find JSON after this prefix
                remaining = response_content[idx + len(prefix):].strip()
                start_idx = remaining.find('{')
                if start_idx >= 0:
                    brace_count = 0
                    end_idx = -1
                    for i in range(start_idx, len(remaining)):
                        if remaining[i] == '{':
                            brace_count += 1
                        elif remaining[i] == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_idx = i
                                break
                    
                    if end_idx > start_idx:
                        json_str = remaining[start_idx:end_idx + 1]
                        try:
                            beta_values = json.loads(json_str)
                            if isinstance(beta_values, dict) and 'entity_entity' in beta_values and 'entity_passage' in beta_values:
                                return beta_values
                        except json.JSONDecodeError:
                            continue
        
        # Strategy 5: Try parsing the entire response (in case it's just JSON)
        try:
            beta_values = json.loads(response_content)
            if isinstance(beta_values, dict) and 'entity_entity' in beta_values and 'entity_passage' in beta_values:
                return beta_values
        except json.JSONDecodeError:
            pass
        
        # If all strategies fail, log and raise error
        logger.warning(f"Failed to extract valid JSON from response")
        logger.warning(f"Response content (first 1000 chars): {response_content[:1000]}")
        raise ValueError("Could not extract valid JSON from LLM response")
    
    def _validate_and_normalize(self, beta_values: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Validate and normalize beta values to ensure they sum to 1.0.
        
        Args:
            beta_values: Raw beta values from LLM
            
        Returns:
            Normalized beta values
        """
        # Entity-Entity relation types
        entity_entity_types = ['HIERARCHICAL', 'TEMPORAL', 'SPATIAL', 'CAUSALITY', 'ATTRIBUTION']
        entity_passage_types = ['PRIMARY', 'SECONDARY', 'PERIPHERAL']
        
        # Normalize entity_entity
        entity_entity = {}
        total = 0.0
        for rtype in entity_entity_types:
            value = float(beta_values['entity_entity'].get(rtype, 0.0))
            value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
            entity_entity[rtype] = value
            total += value
        
        # Normalize to sum to 1.0
        if total > 0:
            for rtype in entity_entity_types:
                entity_entity[rtype] /= total
        else:
            # Uniform distribution if all zeros
            uniform_value = 1.0 / len(entity_entity_types)
            for rtype in entity_entity_types:
                entity_entity[rtype] = uniform_value
        
        # Normalize entity_passage
        entity_passage = {}
        total = 0.0
        for rtype in entity_passage_types:
            value = float(beta_values['entity_passage'].get(rtype, 0.0))
            value = max(0.0, min(1.0, value))  # Clamp to [0, 1]
            entity_passage[rtype] = value
            total += value
        
        # Normalize to sum to 1.0
        if total > 0:
            for rtype in entity_passage_types:
                entity_passage[rtype] /= total
        else:
            # Uniform distribution if all zeros
            uniform_value = 1.0 / len(entity_passage_types)
            for rtype in entity_passage_types:
                entity_passage[rtype] = uniform_value
        
        return {
            'entity_entity': entity_entity,
            'entity_passage': entity_passage
        }
    
    def _get_uniform_factors(self) -> Dict[str, Dict[str, float]]:
        """
        Get uniform relation influence factors as fallback.
        
        CRITICAL FIX: Use beta=1.0 for all types when manager agent fails.
        This ensures no edge downweighting occurs, matching baseline behavior.
        Previously used 1.0/len(types) which heavily downweighted all edges.
        
        Returns:
            Uniform beta values (all 1.0 to preserve baseline edge weights)
        """
        entity_entity_types = ['HIERARCHICAL', 'TEMPORAL', 'SPATIAL', 'CAUSALITY', 'ATTRIBUTION']
        entity_passage_types = ['PRIMARY', 'SECONDARY', 'PERIPHERAL']
        
        # Use 1.0 for all types - this means no modification to edge weights
        # This is better than downweighting all edges when manager agent fails
        return {
            'entity_entity': {rtype: 1.0 for rtype in entity_entity_types},
            'entity_passage': {rtype: 1.0 for rtype in entity_passage_types}
        }
    
    def _create_fallback_prompt(self, query: str) -> list:
        """
        Create a fallback prompt if template is not available.
        
        Args:
            query: The input query
            
        Returns:
            List of message dictionaries
        """
        system_prompt = """You are a relation-aware retrieval manager. Given a query, determine how relevant each relation type is for answering it.

Output a JSON object with relation influence factors (β values) that sum to 1.0 for each category.

Entity-Entity Relation Types: HIERARCHICAL, TEMPORAL, SPATIAL, CAUSALITY, ATTRIBUTION
Entity-Passage Relation Types: PRIMARY, SECONDARY, PERIPHERAL

Output ONLY valid JSON, no other text."""

        user_prompt = f"""Analyze this query and determine relation influence factors:

Query: {query}

Output ONLY a JSON object with this exact structure:
{{
  "entity_entity": {{
    "HIERARCHICAL": float,
    "TEMPORAL": float,
    "SPATIAL": float,
    "CAUSALITY": float,
    "ATTRIBUTION": float
  }},
  "entity_passage": {{
    "PRIMARY": float,
    "SECONDARY": float,
    "PERIPHERAL": float
  }}
}}"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

