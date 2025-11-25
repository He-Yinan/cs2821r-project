from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """Your task is to construct a typed RDF-style knowledge graph from the given passages and named entity lists.
You must extract factual relations as a list of JSON triples.

Requirements:
- Each triple must involve at least one, and preferably two, of the named entities in the list for each passage.
- Resolve pronouns to their specific names so that subjects and objects are explicit entity mentions.
- For every relation, you must assign:
  • "relation_type": one of ["HIERARCHICAL", "TEMPORAL", "SPATIAL", "CAUSALITY", "ATTRIBUTION"]
  • "confidence": a float between 0.0 and 1.0 indicating how certain you are about this triple.

Relation type guidelines:
- HIERARCHICAL: is_a, type_of, subclass_of, part_of, is (when defining category or role). 
  Examples: 
    "Radio City is a private FM radio station" -> HIERARCHICAL
    "PlanetRadiocity.com is a music portal" -> HIERARCHICAL
- TEMPORAL: started_on, launched_in, founded_in, occurred_in, happened_on, opened_on.
  Example:
    "Radio City started on 3 July 2001" -> TEMPORAL
- SPATIAL: located_in, based_in, in, within, part_of (for locations).
  Example:
    "Radio City is located in India" -> SPATIAL
- CAUSALITY: causes, results_in, leads_to, enables, triggered_by, launched (when indicating an action leading to a result).
- ATTRIBUTION: has_property, plays, offers, created_by, owned_by, has_attribute, describes features or properties.

Output format:
- You MUST return a single JSON object with a key "triples".
- The value of "triples" must be a list of objects.
- Each object MUST have the following keys:
  "subject", "predicate", "object", "relation_type", "confidence".
- Do NOT include any additional keys outside this structure.
"""

ner_conditioned_re_frame = """Convert the paragraph into a JSON dict with a single key "triples".
- Use the provided named entity list to guide subject and object selection.
- Each triple should be an object with the following fields:
  "subject": string,
  "predicate": string,
  "object": string,
  "relation_type": one of ["HIERARCHICAL", "TEMPORAL", "SPATIAL", "CAUSALITY", "ATTRIBUTION"],
  "confidence": float in [0.0, 1.0].

Paragraph:
{passage}
{named_entity_json}
"""

ner_conditioned_re_input = ner_conditioned_re_frame.format(
    passage=one_shot_ner_paragraph,
    named_entity_json=one_shot_ner_output
)

# One-shot example output in the new typed format
ner_conditioned_re_output = """{
  "triples": [
    {
      "subject": "Radio City",
      "predicate": "located in",
      "object": "India",
      "relation_type": "SPATIAL",
      "confidence": 0.95
    },
    {
      "subject": "Radio City",
      "predicate": "is",
      "object": "private FM radio station",
      "relation_type": "HIERARCHICAL",
      "confidence": 0.94
    },
    {
      "subject": "Radio City",
      "predicate": "started on",
      "object": "3 July 2001",
      "relation_type": "TEMPORAL",
      "confidence": 0.96
    },
    {
      "subject": "Radio City",
      "predicate": "plays songs in",
      "object": "Hindi",
      "relation_type": "ATTRIBUTION",
      "confidence": 0.93
    },
    {
      "subject": "Radio City",
      "predicate": "plays songs in",
      "object": "English",
      "relation_type": "ATTRIBUTION",
      "confidence": 0.93
    },
    {
      "subject": "Radio City",
      "predicate": "forayed into",
      "object": "New Media",
      "relation_type": "CAUSALITY",
      "confidence": 0.88
    },
    {
      "subject": "Radio City",
      "predicate": "launched",
      "object": "PlanetRadiocity.com",
      "relation_type": "CAUSALITY",
      "confidence": 0.9
    },
    {
      "subject": "PlanetRadiocity.com",
      "predicate": "launched in",
      "object": "May 2008",
      "relation_type": "TEMPORAL",
      "confidence": 0.94
    },
    {
      "subject": "PlanetRadiocity.com",
      "predicate": "is",
      "object": "music portal",
      "relation_type": "HIERARCHICAL",
      "confidence": 0.94
    },
    {
      "subject": "PlanetRadiocity.com",
      "predicate": "offers",
      "object": "news",
      "relation_type": "ATTRIBUTION",
      "confidence": 0.9
    },
    {
      "subject": "PlanetRadiocity.com",
      "predicate": "offers",
      "object": "videos",
      "relation_type": "ATTRIBUTION",
      "confidence": 0.9
    },
    {
      "subject": "PlanetRadiocity.com",
      "predicate": "offers",
      "object": "songs",
      "relation_type": "ATTRIBUTION",
      "confidence": 0.9
    }
  ]
}
"""

prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {
        "role": "user",
        "content": convert_format_to_template(
            original_string=ner_conditioned_re_frame,
            placeholder_mapping=None,
            static_values=None
        )
    }
]
