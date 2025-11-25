from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system ="""Your task is to construct an RDF-style knowledge graph 
from the given passages and named entity lists.

You MUST extract factual relations and respond with a JSON object of the form:

{
  "triples": [
    ["subject", "predicate", "object", "relation_type", confidence],
    ...
  ]
}

Relation Types (choose ONE for each triple):
- HIERARCHICAL  : is_a, part_of, subclass_of, capital_of, membership relations.
- TEMPORAL      : founded_in, started_on, occurred_in, birth date, time expressions.
- SPATIAL       : located_in, adjacent_to, inside, contains, geographic relations.
- CAUSALITY     : causes, results_in, leads_to, enables.
- ATTRIBUTION   : created_by, painted_by, owned_by, has_property, offers, plays.

Confidence Score (float in [0,1]):
- 0.9–1.0 : clear and explicit in the passage.
- 0.7–0.9 : likely correct, minor ambiguity.
- 0.5–0.7 : uncertain or weak evidence.
- <0.5    : do NOT output such relations.

Requirements:
- Each triple should contain at least one, and preferably two, of the named entities in the list.
- Resolve pronouns to their specific entities.
- Use ONLY the allowed relation types above.
- The final output MUST be a single valid JSON object with the key "triples" and no extra text.
"""

ner_conditioned_re_frame = """Convert the paragraph into a JSON dict. 
The JSON dict must contain:
- a named entity list (copied from the given named_entity_json),
- a triple list named "triples", where each triple is
  ["subject", "predicate", "object", "relation_type", confidence].

```
{passage}
```

{named_entity_json}
"""


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
