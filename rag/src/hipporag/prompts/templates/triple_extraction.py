from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system ="""You are a knowledge extraction system. Extract factual relations from passages and output ONLY valid JSON.

CRITICAL: Output ONLY the JSON object. Do NOT include reasoning, explanations, or any other text.

Output format (JSON only):
{
  "triples": [
    ["subject", "predicate", "object", "relation_type", confidence],
    ...
  ]
}

Relation Types (choose ONE per triple):
- HIERARCHICAL: is_a, part_of, subclass_of, capital_of, membership, spouse, child, parent
- TEMPORAL: founded_in, started_on, occurred_in, birth_date, first_appearance, last_appearance
- SPATIAL: located_in, adjacent_to, inside, contains, geographic relations
- CAUSALITY: causes, results_in, leads_to, enables, created_by
- ATTRIBUTION: painted_by, owned_by, has_property, offers, plays, portrayed_by, occupation

Confidence (float 0.0-1.0):
- 0.9-1.0: explicit in passage
- 0.7-0.9: likely correct, minor ambiguity
- 0.5-0.7: uncertain evidence
- <0.5: DO NOT output

Rules:
- Each triple must include at least one named entity from the provided list
- Resolve pronouns to specific entities
- Use ONLY the 5 relation types listed above
- Output ONLY valid JSON, no other text
"""

ner_conditioned_re_frame = """Extract triples from this passage. Use the named entities provided.

Passage:
{passage}

Named entities:
{named_entity_json}

Output ONLY a JSON object with this exact structure:
{
  "triples": [
    ["subject", "predicate", "object", "relation_type", confidence]
  ]
}

Do NOT include reasoning. Output ONLY the JSON object."""


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
