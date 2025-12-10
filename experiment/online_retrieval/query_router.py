#!/usr/bin/env python3
"""
Query Router for MARA-RAG: Relation-Aware Weight Assignment

The router analyzes a query and determines:
1. Weights for each relation type (hierarchical, temporal, spatial, causality, attribution)
2. Beta value for passage node importance in PPR reset probability

This enables dynamic, query-specific graph weighting for improved retrieval.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "rag" / "src"))

from hipporag.llm.base import BaseLLM
from hipporag.utils.llm_utils import TextChatMessage


class QueryRouter:
    """
    LLM-based router that assigns relation type weights and beta value for each query.

    The router uses an LLM to analyze the query's semantic characteristics and
    determine which relation types are most relevant for retrieval.
    """

    # System prompt for relation-aware retrieval manager
    ROUTER_SYSTEM_PROMPT = """You are a relation-aware retrieval manager. Given a query, determine how relevant each relation type is for answering it.

Your task is to output a JSON object with relation influence factors (weights) for entity-entity relations that sum to 1.0, plus a beta value.

**Entity-Entity Relation Types:**

- **hierarchical**: Represents is-a, part-of, subclass-of, membership, and family relationships.
  Examples: "Paris is the capital of France", "Apple is a fruit", "John is a member of the team", "The engine is part of the car"
  Use high weight when the query involves classification, categorization, or hierarchical relationships.

- **temporal**: Represents time-based relationships, dates, events, and temporal sequences.
  Examples: "founded in 1776", "occurred during World War II", "When was X created?", "What happened after Y?"
  Use high weight when the query asks about when something happened, temporal ordering, or historical events.

- **spatial**: Represents location-based and geographic relationships.
  Examples: "located in New York", "adjacent to the river", "Where is X?", "What countries border France?"
  Use high weight when the query asks about locations, geography, or spatial relationships.

- **causality**: Represents cause-effect relationships, dependencies, and causal chains.
  Examples: "What caused X?", "X results in Y", "X leads to Y", "What enabled the revolution?"
  Use high weight when the query asks about causes, effects, or causal reasoning.

- **attribution**: Represents properties, attributes, roles, and descriptive characteristics.
  Examples: "Who painted X?", "What is X's property?", "X is owned by Y", "What are the characteristics of Z?"
  Use high weight when the query asks about attributes, properties, or descriptive information.

**Beta value** (between 0.01 and 0.1): Represents the degree to which the query relies on external supporting information from passages.
- Higher beta (0.05-0.1): Query is more retrieval-based, requiring specific information from passages
  Examples: "What is the capital of France?" (needs factual retrieval)
- Lower beta (0.01-0.05): Query is more open-ended reasoning, less dependent on specific passage content
  Examples: "Why did the war start?" (requires reasoning over multiple facts)
- Default: 0.05 for balanced queries (matches original HippoRAG)

**CRITICAL:**
- All relation weights must be between 0.0 and 1.0
- Sum of hierarchical, temporal, spatial, causality, and attribution must equal 1.0
- Beta value must be between 0.01 and 0.1
- Output ONLY valid JSON, no other text"""

    # User prompt template with examples - STRICT JSON FORMAT
    ROUTER_USER_PROMPT_TEMPLATE = """Analyze this query and determine relation influence factors.

Query: {query}

You MUST output ONLY a valid JSON object. No explanations, no reasoning, no markdown, no code blocks. Just pure JSON.

Required JSON structure:
{{
  "hierarchical": <number>,
  "temporal": <number>,
  "spatial": <number>,
  "causality": <number>,
  "attribution": <number>,
  "beta": <number>
}}

The five relation weights (hierarchical, temporal, spatial, causality, attribution) must sum to exactly 1.0.
Beta must be between 0.01 and 0.1 (original HippoRAG range).

Output ONLY the JSON object, nothing else."""

    def __init__(self, llm: BaseLLM):
        """
        Initialize the query router with an LLM.

        Args:
            llm: LLM instance for generating routing decisions
        """
        self.llm = llm
        # Detect LLM type for prompt formatting
        self.llm_name = getattr(llm, 'llm_name', None) or getattr(llm, 'model_name', None) or ""
        self.is_openai = 'gpt' in self.llm_name.lower() or 'openai' in str(type(llm)).lower()
        self.is_qwen = 'qwen' in self.llm_name.lower()
        # Detect LLM type for prompt formatting
        self.llm_name = getattr(llm, 'llm_name', None) or getattr(llm, 'model_name', None) or ""
        self.is_openai = 'gpt' in self.llm_name.lower() or 'openai' in str(type(llm)).lower()
        self.is_qwen = 'qwen' in self.llm_name.lower()

    def route(self, query: str) -> Dict[str, float]:
        """
        Analyze a query and return relation type weights and beta value.

        Args:
            query: The input query string

        Returns:
            Dictionary with keys:
                - 'hierarchical': weight for hierarchical relations
                - 'temporal': weight for temporal relations
                - 'spatial': weight for spatial relations
                - 'causality': weight for causality relations
                - 'attribution': weight for attribution relations
                - 'beta': passage node weight for PPR

        Example:
            >>> router = QueryRouter(llm)
            >>> weights = router.route("What caused World War I?")
            >>> weights
            {
                'hierarchical': 0.1,
                'temporal': 0.3,
                'spatial': 0.1,
                'causality': 0.5,
                'attribution': 0.0,
                'beta': 0.05
            }
        """

        # DEBUG: Print which LLM type we're using
        print(f"\nDEBUG: Using LLM type - is_openai={self.is_openai}, is_qwen={self.is_qwen}, llm_name={self.llm_name}")
        
        # Build messages based on LLM type
        # OpenAI models work well with system/user messages
        # Qwen/vLLM models may need different formatting (some don't handle system messages well)
        if self.is_openai:
            # OpenAI format: system + user messages with few-shot examples
            messages: List[TextChatMessage] = [
                {"role": "system", "content": self.ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": self.ROUTER_USER_PROMPT_TEMPLATE.format(query="When was Radio City founded?")},
                {"role": "assistant", "content": """{
  "hierarchical": 0.1,
  "temporal": 0.7,
  "spatial": 0.05,
  "causality": 0.05,
  "attribution": 0.1,
  "beta": 0.05
}"""},
                {"role": "user", "content": self.ROUTER_USER_PROMPT_TEMPLATE.format(query="Where is Radio City located?")},
                {"role": "assistant", "content": """{
  "hierarchical": 0.1,
  "temporal": 0.05,
  "spatial": 0.7,
  "causality": 0.05,
  "attribution": 0.1,
  "beta": 0.08
}"""},
                {"role": "user", "content": self.ROUTER_USER_PROMPT_TEMPLATE.format(query="What caused Radio City to launch PlanetRadiocity.com?")},
                {"role": "assistant", "content": """{
  "hierarchical": 0.1,
  "temporal": 0.2,
  "spatial": 0.05,
  "causality": 0.5,
  "attribution": 0.15,
  "beta": 0.03
}"""},
                {"role": "user", "content": self.ROUTER_USER_PROMPT_TEMPLATE.format(query=query)}
            ]
        else:
            # Qwen/vLLM format: Combine system prompt into user message, use simpler format
            # Some vLLM servers don't handle system messages well, so we combine everything into user message
            # Use STRICT JSON format with clear examples
            combined_prompt = f"""{self.ROUTER_SYSTEM_PROMPT}

Examples (copy this exact format):

Query: When was Radio City founded?
{{
  "hierarchical": 0.1,
  "temporal": 0.7,
  "spatial": 0.05,
  "causality": 0.05,
  "attribution": 0.1,
  "beta": 0.05
}}

Query: Where is Radio City located?
{{
  "hierarchical": 0.1,
  "temporal": 0.05,
  "spatial": 0.7,
  "causality": 0.05,
  "attribution": 0.1,
  "beta": 0.08
}}

Query: What caused Radio City to launch PlanetRadiocity.com?
{{
  "hierarchical": 0.1,
  "temporal": 0.2,
  "spatial": 0.05,
  "causality": 0.5,
  "attribution": 0.15,
  "beta": 0.03
}}

Now analyze this query:

Query: {query}

CRITICAL: Output ONLY valid JSON. No text before or after. No explanations. No markdown. Just the JSON object starting with {{ and ending with }}."""
            messages: List[TextChatMessage] = [
                {"role": "user", "content": combined_prompt}
            ]
        
        # DEBUG: Print the prompt being sent (first 500 chars)
        print(f"\nDEBUG: Prompt being sent to LLM (first 500 chars):")
        print(f"{'='*80}")
        if self.is_openai:
            print(f"System: {messages[0]['content'][:200]}...")
            print(f"User (last): {messages[-1]['content'][:300]}...")
        else:
            print(f"User: {messages[0]['content'][:500]}...")
        print(f"{'='*80}\n")

        # Call LLM using HippoRAG's interface
        # Use lower temperature and sufficient tokens for complete JSON response
        try:
            # The infer method may return 2 or 3 values (with cache decorator it returns 3)
            # Use higher max_tokens for Qwen to ensure complete JSON output
            # OpenAI models are more concise, Qwen may need more tokens
            max_tokens = 1000 if self.is_qwen else 500
            infer_result = self.llm.infer(messages, max_tokens=max_tokens, temperature=0.0)
            
            # Handle both cached (3 values) and non-cached (2 values) responses
            try:
                # Try unpacking as 3 values first (cache decorator returns 3)
                response_message, metadata, cache_hit = infer_result
            except ValueError:
                # If that fails, unpack as 2 values (direct API call)
                response_message, metadata = infer_result
            
            # Extract response content
            # The response can be a string (from cache) or List[TextChatMessage] (from API)
            if isinstance(response_message, str):
                response = response_message
            elif isinstance(response_message, list) and len(response_message) > 0:
                response = response_message[0].get("content", "") if isinstance(response_message[0], dict) else str(response_message[0])
            else:
                response = str(response_message) if response_message else ""

            # DEBUG: Print full LLM response for debugging
            print(f"\n{'='*80}")
            print(f"DEBUG: Full LLM Response (length: {len(response)} chars):")
            print(f"{'='*80}")
            print(response)
            print(f"{'='*80}\n")

            # Parse JSON response (expects flat structure)
            weights = self._parse_llm_response(response)

            # Validate and normalize flat structure
            weights = self._validate_and_normalize_flat(weights)

            return weights

        except Exception as e:
            # Try one retry with a simpler prompt if first attempt fails
            if not hasattr(self, '_retry_attempted'):
                self._retry_attempted = True
                try:
                    # Simpler prompt for retry
                    simple_prompt = f"""Query: {query}

Output JSON only:
{{
  "hierarchical": 0.2,
  "temporal": 0.2,
  "spatial": 0.2,
  "causality": 0.2,
  "attribution": 0.2,
  "beta": 0.15
}}"""
                    retry_messages = [{"role": "user", "content": simple_prompt}]
                    retry_max_tokens = 1000 if self.is_qwen else 500
                    infer_result = self.llm.infer(retry_messages, max_tokens=retry_max_tokens, temperature=0.0)
                    
                    # Handle response
                    try:
                        response_message, metadata, cache_hit = infer_result
                    except ValueError:
                        response_message, metadata = infer_result
                    
                    if isinstance(response_message, str):
                        response = response_message
                    elif isinstance(response_message, list) and len(response_message) > 0:
                        response = response_message[0].get("content", "") if isinstance(response_message[0], dict) else str(response_message[0])
                    else:
                        response = str(response_message) if response_message else ""
                    
                    weights = self._parse_llm_response(response)
                    weights = self._validate_and_normalize_flat(weights)
                    self._retry_attempted = False
                    return weights
                except Exception as retry_e:
                    self._retry_attempted = False
                    pass
            
            # Fallback to default weights if routing fails
            print(f"Warning: Query routing failed ({e}), using default weights")
            return self._get_default_weights()

    def _parse_llm_response(self, response: str) -> Dict:
        """
        Parse the LLM's JSON response (expects flat structure with relation weights and beta).

        Args:
            response: Raw LLM response string

        Returns:
            Parsed dictionary with relation weights and beta
        """
        # Try to extract JSON from response
        # Handle cases where LLM includes extra text, reasoning, or markdown
        
        # Clean the response - remove any markdown code blocks
        cleaned_response = response.strip()
        
        # Remove markdown code blocks if present
        if '```' in cleaned_response:
            # Remove markdown code blocks
            cleaned_response = re.sub(r'```json\s*', '', cleaned_response)
            cleaned_response = re.sub(r'```\s*', '', cleaned_response)
            cleaned_response = cleaned_response.strip()
        
        # Remove common prefixes that models might add
        prefixes_to_remove = [
            r'^here\s+is\s+the\s+json\s*:?\s*',
            r'^the\s+json\s+is\s*:?\s*',
            r'^json\s*:?\s*',
            r'^response\s*:?\s*',
            r'^answer\s*:?\s*',
        ]
        for prefix in prefixes_to_remove:
            cleaned_response = re.sub(prefix, '', cleaned_response, flags=re.IGNORECASE)
            cleaned_response = cleaned_response.strip()
        
        # Try to find JSON object - look for balanced braces
        # Start from the first { and find the matching }
        start_idx = cleaned_response.find('{')
        if start_idx == -1:
            # Try to find JSON without braces (unlikely but handle gracefully)
            raise ValueError(f"No JSON object found in response. First 200 chars: {response[:200]}")
        
        # Find the matching closing brace
        brace_count = 0
        end_idx = start_idx
        for i in range(start_idx, len(cleaned_response)):
            if cleaned_response[i] == '{':
                brace_count += 1
            elif cleaned_response[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        
        if brace_count == 0:
            json_str = cleaned_response[start_idx:end_idx]
            try:
                weights = json.loads(json_str)
                return weights
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                # Remove trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
            try:
                weights = json.loads(json_str)
                return weights
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON: {e}\nJSON string: {json_str}\nFull response (first 500 chars): {response[:500]}")
        
        raise ValueError(f"Unbalanced braces in JSON response. First 500 chars: {response[:500]}")
    
    def _validate_and_normalize_flat(self, weights: Dict) -> Dict[str, float]:
        """
        Validate and normalize the flat weights structure from the router.

        Args:
            weights: Raw weights from LLM (flat structure)

        Returns:
            Validated and normalized weights
        """
        # Required relation types (lowercase)
        required_relation_types = ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']

        # Check all required keys are present
        for rel_type in required_relation_types:
            if rel_type not in weights:
                raise ValueError(f"Missing required relation type: {rel_type}")

        if 'beta' not in weights:
            raise ValueError("Missing 'beta' key in response")

        # Extract relation weights
        relation_weights = {k: float(weights[k]) for k in required_relation_types}
        beta = float(weights['beta'])

        # Normalize relation weights to sum to 1.0
        total = sum(relation_weights.values())
        if total > 0:
            relation_weights = {k: v / total for k, v in relation_weights.items()}
        else:
            # If all zeros, use uniform distribution
            relation_weights = {k: 0.2 for k in required_relation_types}

        # Clip beta to valid range [0.01, 0.1] (original HippoRAG range)
        beta = max(0.01, min(0.1, beta))

        # Return flat structure
        result = {**relation_weights, 'beta': beta}
        return result

    def _validate_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and normalize the weights from the router.

        Args:
            weights: Raw weights from LLM

        Returns:
            Validated and normalized weights
        """

        # Required relation types
        relation_types = ['hierarchical', 'temporal', 'spatial', 'causality', 'attribution']

        # Check all required keys are present
        for rel_type in relation_types:
            if rel_type not in weights:
                raise ValueError(f"Missing required relation type: {rel_type}")

        # Check beta is present
        if 'beta' not in weights:
            raise ValueError("Missing required 'beta' value")

        # Extract relation weights
        relation_weights = {k: float(weights[k]) for k in relation_types}
        beta = float(weights['beta'])

        # Normalize relation weights to sum to 1.0
        total = sum(relation_weights.values())
        if total > 0:
            relation_weights = {k: v / total for k, v in relation_weights.items()}
        else:
            # If all zeros, use uniform distribution
            relation_weights = {k: 0.2 for k in relation_types}

        # Clip beta to valid range [0.01, 0.1] (original HippoRAG range)
        beta = max(0.01, min(0.1, beta))

        # Reconstruct full weights dict
        result = {**relation_weights, 'beta': beta}

        # Preserve reasoning if present
        if 'reasoning' in weights:
            result['reasoning'] = weights['reasoning']

        return result

    def _get_default_weights(self) -> Dict[str, float]:
        """
        Get default weights when routing fails.
        
        Increased beta from 0.05 to 0.15 to give more weight to passage nodes,
        which should improve retrieval performance.

        Returns:
            Default uniform weights (flat structure for PPR)
        """
        return {
            'hierarchical': 0.2,
            'temporal': 0.2,
            'spatial': 0.2,
            'causality': 0.2,
            'attribution': 0.2,
            'beta': 0.05  # Original HippoRAG default
        }

    def route_batch(self, queries: list[str]) -> list[Dict[str, float]]:
        """
        Route multiple queries.

        Args:
            queries: List of query strings

        Returns:
            List of weight dictionaries, one per query
        """
        return [self.route(q) for q in queries]


def test_router():
    """Test the router with example queries (requires LLM)."""

    # This is a demonstration of how to use the router
    # In practice, you would initialize with a real LLM instance

    print("QueryRouter test examples:")
    print("\nExample 1: Causality query")
    print("Query: 'What caused World War I?'")
    print("Expected: High causality weight, moderate temporal weight")

    print("\nExample 2: Spatial query")
    print("Query: 'Where is the Eiffel Tower located?'")
    print("Expected: High spatial weight, moderate hierarchical weight")

    print("\nExample 3: Temporal query")
    print("Query: 'When was the United States founded?'")
    print("Expected: High temporal weight, moderate attribution weight")

    print("\nExample 4: Multi-hop reasoning query")
    print("Query: 'Who painted the portrait of the person who wrote Romeo and Juliet?'")
    print("Expected: High attribution weight, moderate hierarchical weight")

    print("\nNote: Actual routing requires a configured LLM instance.")


if __name__ == "__main__":
    test_router()
