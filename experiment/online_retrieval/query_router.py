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
from typing import Dict

from hipporag.llm.base import BaseLLM


class QueryRouter:
    """
    LLM-based router that assigns relation type weights and beta value for each query.

    The router uses an LLM to analyze the query's semantic characteristics and
    determine which relation types are most relevant for retrieval.
    """

    # Prompt template for the router
    ROUTER_PROMPT_TEMPLATE = """You are a query analyzer for a knowledge graph retrieval system. Given a query, you need to determine which types of relations are most important for answering it.

Analyze the following query and assign weights to each relation type. The weights should sum to 1.0 and reflect how important each relation type is for answering the query.

**Relation Types:**
- **hierarchical**: is-a, part-of, subclass-of relationships (e.g., "Paris is the capital of France")
- **temporal**: time-related relationships (e.g., "founded in 1776", "occurred during")
- **spatial**: location-based relationships (e.g., "located in", "adjacent to", "contains")
- **causality**: cause-effect relationships (e.g., "causes", "results in", "leads to")
- **attribution**: properties and characteristics (e.g., "has property", "created by", "painted by")

Also determine a **beta** value (between 0.0 and 0.3) that controls how much weight to give to passage-level dense retrieval vs. entity-level graph traversal:
- Higher beta (0.2-0.3): More emphasis on passage-level semantic similarity
- Lower beta (0.01-0.1): More emphasis on entity-level graph structure
- Use beta ≈ 0.05 as default for balanced retrieval

**Query:** {query}

**Instructions:**
1. Identify what types of information are needed to answer the query
2. Assign weights (0.0 to 1.0) to each relation type based on relevance
3. Ensure weights sum to 1.0
4. Choose an appropriate beta value
5. Return ONLY a valid JSON object with this exact format:

{{
  "hierarchical": 0.0,
  "temporal": 0.0,
  "spatial": 0.0,
  "causality": 0.0,
  "attribution": 0.0,
  "beta": 0.05,
  "reasoning": "Brief explanation of your choices"
}}

Return only the JSON object, no other text."""

    def __init__(self, llm: BaseLLM):
        """
        Initialize the query router with an LLM.

        Args:
            llm: LLM instance for generating routing decisions
        """
        self.llm = llm

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
                - 'reasoning': explanation of the routing decision (optional)

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
                'beta': 0.05,
                'reasoning': 'This is a causality question about historical events...'
            }
        """

        # Format the prompt
        prompt = self.ROUTER_PROMPT_TEMPLATE.format(query=query)

        # Call LLM
        try:
            response = self.llm.generate(prompt, max_tokens=500, temperature=0.0)

            # Parse JSON response
            weights = self._parse_llm_response(response)

            # Validate and normalize
            weights = self._validate_and_normalize(weights)

            return weights

        except Exception as e:
            # Fallback to default weights if routing fails
            print(f"Warning: Query routing failed ({e}), using default weights")
            return self._get_default_weights()

    def _parse_llm_response(self, response: str) -> Dict[str, float]:
        """
        Parse the LLM's JSON response.

        Args:
            response: Raw LLM response string

        Returns:
            Parsed dictionary
        """
        # Try to extract JSON from response
        # Handle cases where LLM includes extra text

        # Look for JSON object pattern
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)

        if json_match:
            json_str = json_match.group(0)
            weights = json.loads(json_str)
            return weights
        else:
            raise ValueError(f"Could not find valid JSON in LLM response: {response}")

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

        # Clip beta to valid range [0.0, 0.3]
        beta = max(0.0, min(0.3, beta))

        # Reconstruct full weights dict
        result = {**relation_weights, 'beta': beta}

        # Preserve reasoning if present
        if 'reasoning' in weights:
            result['reasoning'] = weights['reasoning']

        return result

    def _get_default_weights(self) -> Dict[str, float]:
        """
        Get default weights when routing fails.

        Returns:
            Default uniform weights
        """
        return {
            'hierarchical': 0.2,
            'temporal': 0.2,
            'spatial': 0.2,
            'causality': 0.2,
            'attribution': 0.2,
            'beta': 0.05,
            'reasoning': 'Default weights (routing failed)'
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
