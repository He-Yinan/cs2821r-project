"""
Prompt template for Manager Agent to generate relation influence factors.
"""

from ...utils.llm_utils import convert_format_to_template

relation_influence_system = """You are a relation-aware retrieval manager. Given a query, determine how relevant each relation type is for answering it.

Your task is to output a JSON object with relation influence factors (β values) that sum to 1.0 for each category.

Entity-Entity Relation Types:
- HIERARCHICAL: is_a, part_of, subclass_of, membership, family relations
- TEMPORAL: dates, time-based events, founded_in, started_on
- SPATIAL: location, geographic relations, located_in, adjacent_to
- CAUSALITY: causes, results_in, leads_to, enables, created_by
- ATTRIBUTION: properties, attributes, roles, painted_by, owned_by

Entity-Passage Relation Types:
- PRIMARY: Core entities that are central to the passage's main topic
- SECONDARY: Supporting entities that provide context
- PERIPHERAL: Minor entities mentioned in passing

Output format (JSON only):
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

CRITICAL: 
- All values must be between 0.0 and 1.0
- Sum of entity_entity values must equal 1.0
- Sum of entity_passage values must equal 1.0
- Output ONLY valid JSON, no other text
"""

relation_influence_frame = """Analyze this query and determine relation influence factors:

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
}}

Do NOT include reasoning. Output ONLY the JSON object."""

# Example 1: Temporal query
example_1_query = "When was Radio City founded?"
example_1_output = """{
  "entity_entity": {
    "HIERARCHICAL": 0.1,
    "TEMPORAL": 0.7,
    "SPATIAL": 0.05,
    "CAUSALITY": 0.05,
    "ATTRIBUTION": 0.1
  },
  "entity_passage": {
    "PRIMARY": 0.7,
    "SECONDARY": 0.25,
    "PERIPHERAL": 0.05
  }
}"""

# Example 2: Spatial query
example_2_query = "Where is Radio City located?"
example_2_output = """{
  "entity_entity": {
    "HIERARCHICAL": 0.1,
    "TEMPORAL": 0.05,
    "SPATIAL": 0.7,
    "CAUSALITY": 0.05,
    "ATTRIBUTION": 0.1
  },
  "entity_passage": {
    "PRIMARY": 0.75,
    "SECONDARY": 0.2,
    "PERIPHERAL": 0.05
  }
}"""

# Example 3: Causal query
example_3_query = "What caused Radio City to launch PlanetRadiocity.com?"
example_3_output = """{
  "entity_entity": {
    "HIERARCHICAL": 0.1,
    "TEMPORAL": 0.2,
    "SPATIAL": 0.05,
    "CAUSALITY": 0.5,
    "ATTRIBUTION": 0.15
  },
  "entity_passage": {
    "PRIMARY": 0.6,
    "SECONDARY": 0.3,
    "PERIPHERAL": 0.1
  }
}"""

prompt_template = [
    {"role": "system", "content": relation_influence_system},
    {"role": "user", "content": relation_influence_frame.format(query=example_1_query)},
    {"role": "assistant", "content": example_1_output},
    {"role": "user", "content": relation_influence_frame.format(query=example_2_query)},
    {"role": "assistant", "content": example_2_output},
    {"role": "user", "content": relation_influence_frame.format(query=example_3_query)},
    {"role": "assistant", "content": example_3_output},
    {"role": "user", "content": convert_format_to_template(
        original_string=relation_influence_frame,
        placeholder_mapping=None,
        static_values=None
    )}
]













