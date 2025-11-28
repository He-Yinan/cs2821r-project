from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system ="""You are a JSON-only output system. Extract triples from passages and output ONLY valid JSON.

STRICT REQUIREMENTS:
1. Output ONLY the JSON object. Nothing else.
2. DO NOT include any reasoning, explanations, thinking, or commentary.
3. DO NOT use <think>, <reasoning>, or any other tags.
4. DO NOT write any text before or after the JSON.
5. Start your response with {{ and end with }}.
6. Your entire response must be parseable as valid JSON.

REQUIRED OUTPUT FORMAT (copy this structure exactly):
{{
  "triples": [
    ["subject", "predicate", "object", "relation_type", confidence]
  ]
}}

Relation Types (MUST use one of these 5 types):
- HIERARCHICAL: is_a, part_of, subclass_of, membership, spouse, child, parent
- TEMPORAL: started_on, occurred_in, published_in, birth_date, first_appearance, last_appearance
- SPATIAL: located_in, adjacent_to, inside, contains, geographic relations
- CAUSALITY: causes, results_in, leads_to, enables, created_by, written_by
- ATTRIBUTION: owned_by, has_property, offers, plays, portrayed_by, occupation

Confidence (float 0.0-1.0):
- 0.9-1.0: explicit in passage
- 0.7-0.9: likely correct, minor ambiguity
- 0.5-0.7: uncertain evidence

Rules:
- Each triple must include at least one named entity from the provided list
- Use ONLY the 5 relation types listed above
- Output ONLY the JSON object - no other text whatsoever
"""

ner_conditioned_re_frame = """Extract triples from this passage using the named entities.

Passage:
{passage}

Named entities:
{named_entity_json}

REQUIRED OUTPUT (JSON only, no other text):
{{
  "triples": [
    ["subject", "predicate", "object", "relation_type", confidence]
  ]
}}

CRITICAL: Output ONLY the JSON object above. Do NOT include:
- Any reasoning or thinking
- Any explanations
- Any tags like <think> or <reasoning>
- Any text before or after the JSON

Start with {{ and end with }}."""


ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{
  "triples": [
    ["Radio City", "located in", "India", "SPATIAL", 0.95],
    ["Radio City", "is", "private FM radio station", "HIERARCHICAL", 0.88],
    ["Radio City", "started on", "3 July 2001", "TEMPORAL", 0.92],
    ["Radio City", "plays songs in", "Hindi", "ATTRIBUTION", 0.75],
    ["Radio City", "plays songs in", "English", "ATTRIBUTION", 0.78],
    ["Radio City", "forayed into", "New Media", "CAUSALITY", 0.70],
    ["Radio City", "launched", "PlanetRadiocity.com", "CAUSALITY", 0.85],
    ["PlanetRadiocity.com", "launched in", "May 2008", "TEMPORAL", 0.90],
    ["PlanetRadiocity.com", "is", "music portal", "HIERARCHICAL", 0.82],
    ["PlanetRadiocity.com", "offers", "news", "ATTRIBUTION", 0.80],
    ["PlanetRadiocity.com", "offers", "videos", "ATTRIBUTION", 0.80],
    ["PlanetRadiocity.com", "offers", "songs", "ATTRIBUTION", 0.80]
  ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(
        original_string=ner_conditioned_re_frame,
        placeholder_mapping=None,
        static_values=None
    )}
]
