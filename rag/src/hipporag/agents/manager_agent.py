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
            if isinstance(llm_response, tuple):
                response_messages, metadata = llm_response
                if isinstance(response_messages, list) and len(response_messages) > 0:
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
        
        Args:
            response_content: Raw response content from LLM
            
        Returns:
            Parsed beta values dictionary
        """
        # Try to extract JSON from response
        response_content = response_content.strip()
        
        # Remove markdown code blocks if present
        if response_content.startswith('```'):
            lines = response_content.split('\n')
            # Remove first line (```json or ```)
            if len(lines) > 1:
                response_content = '\n'.join(lines[1:])
            # Remove last line (```)
            if response_content.endswith('```'):
                response_content = response_content[:-3].strip()
        
        # Try to find JSON object
        start_idx = response_content.find('{')
        end_idx = response_content.rfind('}')
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = response_content[start_idx:end_idx + 1]
        else:
            json_str = response_content
        
        # Parse JSON
        try:
            beta_values = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.warning(f"Response content: {response_content[:500]}")
            raise ValueError(f"Invalid JSON response: {e}")
        
        # Validate structure
        if not isinstance(beta_values, dict):
            raise ValueError("Response is not a dictionary")
        
        required_keys = ['entity_entity', 'entity_passage']
        for key in required_keys:
            if key not in beta_values:
                raise ValueError(f"Missing required key: {key}")
            if not isinstance(beta_values[key], dict):
                raise ValueError(f"Value for {key} is not a dictionary")
        
        return beta_values
    
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
        
        Returns:
            Uniform beta values (all equal within each category)
        """
        entity_entity_types = ['HIERARCHICAL', 'TEMPORAL', 'SPATIAL', 'CAUSALITY', 'ATTRIBUTION']
        entity_passage_types = ['PRIMARY', 'SECONDARY', 'PERIPHERAL']
        
        uniform_ee = 1.0 / len(entity_entity_types)
        uniform_ep = 1.0 / len(entity_passage_types)
        
        return {
            'entity_entity': {rtype: uniform_ee for rtype in entity_entity_types},
            'entity_passage': {rtype: uniform_ep for rtype in entity_passage_types}
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

