import json
import difflib
from pydantic import BaseModel, Field, TypeAdapter
from openai import OpenAI
from copy import deepcopy
from typing import Union, Optional, List, Dict, Any, Tuple, Literal
import re
import ast
from .prompts.filter_default_prompt import best_dspy_prompt

class Fact(BaseModel):
    fact: list[list[str]] = Field(description="A list of facts, each fact is a list of 3 strings: [subject, predicate, object]")


class DSPyFilter:
    def __init__(self, hipporag):
        """
        Initializes the object with the necessary configurations and templates for processing input and output messages.

        Parameters:
        hipporag : An object that provides the global configuration and the LLM model required for inference.

        Attributes:
        dspy_file_path : The file path for reranking as specified in the global configuration.
        one_input_template : A string template for formatting the input message with placeholders for specific fields.
        one_output_template : A string template for formatting the output message with specific fields.
        message_template : A template generated using the specified dspy file path.
        llm_infer_fn : A function reference for making inferences using the provided LLM model.
        model_name : The name of the language model as specified in the global configuration.
        default_gen_kwargs : A dictionary for storing the default generation keyword arguments.
        """
        dspy_file_path = hipporag.global_config.rerank_dspy_file_path
        self.one_input_template = """Question: {question}

Candidate Facts:
{fact_before_filter}

Task: Filter the facts above to select only the most relevant ones for answering the question. Output ONLY a valid JSON object in this exact format, with no additional text, reasoning, or explanation:

{{"fact": [["subject1", "predicate1", "object1"], ["subject2", "predicate2", "object2"]]}}

Output:"""
        self.one_output_template = """[[ ## fact_after_filter ## ]]
{fact_after_filter}

[[ ## completed ## ]]"""
        self.message_template = self.make_template(dspy_file_path)
        self.llm_infer_fn = hipporag.llm_model.infer
        self.model_name = hipporag.global_config.llm_name
        self.default_gen_kwargs = {}

    def make_template(self, dspy_file_path):
        if dspy_file_path is not None:
            dspy_saved = json.load(open(dspy_file_path, 'r'))
        else:
            dspy_saved = best_dspy_prompt

        system_prompt = dspy_saved['prog']['system']
        message_template = [
            {"role": "system", "content": system_prompt},
        ]
        demos = dspy_saved["prog"]["demos"]
        for demo in demos:
            message_template.append({"role": "user", "content": self.one_input_template.format(question=demo["question"], fact_before_filter=demo["fact_before_filter"])})
            message_template.append({"role": "assistant", "content": self.one_output_template.format(fact_after_filter=demo["fact_after_filter"])})
        return message_template

    def parse_filter(self, response):
        """
        Extract JSON from LLM response, handling cases where the model includes reasoning.
        The response may contain reasoning before/after the JSON, so we extract the JSON object.
        """
        parsed = []
        
        if not response or not response.strip():
            return parsed
        
        # Strategy: Find the JSON object with "fact" key anywhere in the response
        # This handles cases where the model includes reasoning before/after the JSON
        
        # Method 1: Look for field markers (old format with [[ ## ... ## ]])
        sections = [(None, [])]
        field_header_pattern = re.compile('\\[\\[ ## (\\w+) ## \\]\\]')
        for line in response.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                sections.append((match.group(1), []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]
        value_to_parse = None
        
        for k, value in sections:
            if k == "fact_after_filter":
                value_to_parse = value
                break
        
        # Method 2: If no field markers found, search entire response for JSON
        if value_to_parse is None:
            value_to_parse = response.strip()
        
        # Extract JSON object - try multiple strategies
        try:
            # Strategy 1: Find JSON object with balanced braces containing "fact" key
            # This regex finds the outermost { ... } that contains "fact"
            depth = 0
            start_idx = -1
            json_str = None
            
            for i, char in enumerate(value_to_parse):
                if char == '{':
                    if depth == 0:
                        start_idx = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_idx >= 0:
                        candidate = value_to_parse[start_idx:i+1]
                        # Check if this JSON object has "fact" key
                        if '"fact"' in candidate or "'fact'" in candidate:
                            json_str = candidate
                            break
            
            # Strategy 2: If balanced brace matching didn't work, use regex
            if json_str is None:
                # Match JSON object with "fact" key (handles nested structures)
                # This regex tries to match { ... "fact": [...] ... } with proper nesting
                json_match = re.search(r'\{[^{}]*(?:"fact"\s*:\s*\[[^\]]*\])[^{}]*\}', value_to_parse, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    # More aggressive: find any { ... } that contains "fact"
                    # Use a simpler pattern that handles nested brackets
                    json_match = re.search(r'\{[^{}]*"fact"[^{}]*\}', value_to_parse, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
            
            # Strategy 3: Try to find JSON in code blocks (```json ... ```)
            if json_str is None:
                code_block_match = re.search(r'```(?:json)?\s*(\{.*?"fact".*?\})\s*```', value_to_parse, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
            
            # Parse the extracted JSON
            if json_str:
                try:
                    parsed_value = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Try cleaning up common JSON issues
                    # Remove trailing commas, fix quotes, etc.
                    cleaned = json_str.strip()
                    # Try with ast.literal_eval as fallback (handles Python-style JSON)
                    try:
                        parsed_value = ast.literal_eval(cleaned)
                    except (ValueError, SyntaxError):
                        # Last resort: try parsing the whole value_to_parse
                        try:
                            parsed_value = json.loads(value_to_parse)
                        except:
                            raise e
                parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
            else:
                # No JSON found with "fact" key, try parsing entire response as JSON
                try:
                    parsed_value = json.loads(value_to_parse)
                    parsed = TypeAdapter(Fact).validate_python(parsed_value).fact
                except (json.JSONDecodeError, ValueError):
                    # If all parsing fails, log and return empty
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Could not extract JSON with 'fact' key from response. Response length: {len(value_to_parse)}, Preview: {value_to_parse[:300]}")
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(
                f"Error parsing filter response: {e}.\n\n\t\tResponse preview:\n```\n{value_to_parse[:500] if value_to_parse else 'EMPTY'}\n```"
            )

        return parsed

    def llm_call(self, question, fact_before_filter):
        # make prompt
        messages = deepcopy(self.message_template)
        messages.append({"role": "user", "content": self.one_input_template.format(question=question, fact_before_filter=fact_before_filter)})
        # call openai

        # Set generation parameters - let model generate full response, we'll parse JSON from it
        gen_kwargs = deepcopy(self.default_gen_kwargs)
        gen_kwargs['max_completion_tokens'] = 1024  # Increased to allow full response with reasoning
        # Remove stop parameter entirely to allow full generation
        # We'll extract JSON from the response regardless of reasoning
        if 'stop' in gen_kwargs:
            del gen_kwargs['stop']  # Remove stop sequences completely
        # Lower temperature for more deterministic output
        if 'temperature' not in gen_kwargs:
            gen_kwargs['temperature'] = 0.0

        try:
            response = self.llm_infer_fn(
                messages=messages,
                model=self.model_name,
                **gen_kwargs
            )
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"LLM call failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return ""

        # The infer method typically returns (message, metadata) where message is a string
        # But it might also return just a string or other formats
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"LLM response type: {type(response)}")
        if isinstance(response, tuple):
            logger.debug(f"LLM response tuple length: {len(response)}")
            if len(response) > 0:
                logger.debug(f"First element type: {type(response[0])}, value preview: {str(response[0])[:100] if response[0] else 'None'}")
        
        # Extract the actual message content
        if isinstance(response, tuple):
            if len(response) >= 1:
                # Extract the message (first element)
                response_message = response[0]
                # If message is a string, return it directly (most common case)
                if isinstance(response_message, str):
                    if not response_message:
                        logger.warning("LLM returned empty string response")
                    else:
                        logger.debug(f"Extracted string response, length: {len(response_message)}")
                    return response_message
                # If message is a list, extract content from first element
                elif isinstance(response_message, list) and len(response_message) > 0:
                    if isinstance(response_message[0], dict):
                        content = response_message[0].get('content', '')
                        logger.debug(f"Extracted content from dict, length: {len(content)}")
                        return content
                    else:
                        content = str(response_message[0])
                        logger.debug(f"Extracted content from list, length: {len(content)}")
                        return content
                else:
                    content = str(response_message) if response_message is not None else ""
                    logger.debug(f"Converted response_message to string, length: {len(content)}")
                    return content
            else:
                logger.warning(f"Empty tuple response: {response}")
                return ""
        elif isinstance(response, str):
            if not response:
                logger.warning("LLM returned empty string response")
            else:
                logger.debug(f"Direct string response, length: {len(response)}")
            return response
        else:
            content = str(response) if response is not None else ""
            logger.debug(f"Converted response to string, length: {len(content)}")
            return content

    def __call__(self, *args, **kwargs):
        return self.rerank(*args, **kwargs)

    def rerank(self,
               query: str,
               candidate_items: List[Tuple],
               candidate_indices: List[int],
               len_after_rerank: int =None) -> Tuple[List[int], List[Tuple], dict]:
        fact_before_filter = {"fact": [list(candidate_item) for candidate_item in candidate_items]}
        try:
            # prediction = self.program(question=query, fact_before_filter=json.dumps(fact_before_filter))
            response = self.llm_call(query, json.dumps(fact_before_filter))
            generated_facts = self.parse_filter(response)
        except Exception as e:
            print('exception', e)
            generated_facts = []
        result_indices = []
        for generated_fact in generated_facts:
            closest_matched_fact = difflib.get_close_matches(str(generated_fact), [str(i) for i in candidate_items], n=1, cutoff=0.0)
            if not closest_matched_fact:
                continue
            closest_matched_fact = closest_matched_fact[0]
            try:
                # Try to find the matching fact by comparing string representations
                # First try to parse as tuple using ast.literal_eval (safer than eval)
                try:
                    parsed_fact = ast.literal_eval(closest_matched_fact)
                    if parsed_fact in candidate_items:
                        result_indices.append(candidate_items.index(parsed_fact))
                        continue
                except (ValueError, SyntaxError):
                    pass
                
                # Fallback: find by string matching
                for idx, candidate_item in enumerate(candidate_items):
                    if str(candidate_item) == closest_matched_fact:
                        result_indices.append(idx)
                        break
            except Exception as e:
                print(f'result_indices exception for fact {generated_fact}: {e}')
                continue

        sorted_candidate_indices = [candidate_indices[i] for i in result_indices]
        sorted_candidate_items = [candidate_items[i] for i in result_indices]
        return sorted_candidate_indices[:len_after_rerank], sorted_candidate_items[:len_after_rerank], {'confidence': None}