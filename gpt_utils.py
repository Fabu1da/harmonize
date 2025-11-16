import functools
import json
import logging
from typing import Optional

import openai
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, stop_after_delay, wait_fixed, wait_random_exponential

from json_schema import ObjectSchema, Schema


logger = logging.getLogger(__name__)

client = openai.AsyncOpenAI(
    #http_client=openai.DefaultAioHttpClient(),
    http_client=openai.DefaultAsyncHttpxClient(),
)


@retry(retry=retry_if_exception_type(openai.APIError), wait=wait_fixed(1), stop=stop_after_attempt(5))
@retry(retry=retry_if_exception_type(openai.RateLimitError), wait=wait_random_exponential(min=1, max=60), stop=stop_after_delay(300))
@functools.wraps(client.responses.create)
async def create_response(*args, **kwargs):
    return await client.responses.create(*args, **kwargs)


@retry(retry=retry_if_exception_type(openai.APIError), wait=wait_fixed(1), stop=stop_after_attempt(5))
@retry(retry=retry_if_exception_type(openai.RateLimitError), wait=wait_random_exponential(min=1, max=60), stop=stop_after_delay(300))
@functools.wraps(client.responses.parse)
async def parse_response(*args, **kwargs):
    return await client.responses.parse(*args, **kwargs)


def compare_mappings(old_mapping, new_mapping):
    """
    Compares two dictionaries and identifies:
    - Unchanged mappings
    - Changed mappings
    - Newly added mappings
    - Removed mappings
    """
    unchanged = {}
    changed = {}
    added = {}
    removed = {}

    for key in old_mapping:
        if key in new_mapping:
            if old_mapping[key] == new_mapping[key]:
                unchanged[key] = old_mapping[key]  # Same mapping
            else:
                changed[key] = (old_mapping[key], new_mapping[key])  # Changed mapping
        else:
            removed[key] = old_mapping[key]  # Key was removed

    for key in new_mapping:
        if key not in old_mapping:
            added[key] = new_mapping[key]  # Key was newly added

    return {"unchanged": unchanged, "changed": changed, "added": added, "removed": removed}


async def gpt_column_mapping(source_schema: ObjectSchema, target_schema: ObjectSchema, model: str, seed: Optional[int] = None, reasoning_effort: str = "medium") -> dict[str, tuple[Optional[str], float, Optional[str]]]:
    """
    Enhanced GPT column mapping with detailed reasoning including closest match analysis.
    """
    assert source_schema.properties is not None
    assert target_schema.properties is not None
    source_columns = list(source_schema.properties.keys())
    target_columns = list(target_schema.properties.keys())
    system_message = " ".join([
        "You are an expert in schema matching.",
        "For each target column, you identify the source column that matches it (if any).",
        "Set `source_column` to `null` if the target column does not match any source column.",
    ])
    user_message = "".join([
        "### Target Schema:\n",
        f"{target_schema.model_dump_json(indent=4)}\n\n",
        "### Source Schema:\n",
        f"{source_schema.model_dump_json(indent=4)}\n\n",
    ])

    schema = {
        "type": "object",
        "properties": {
            target_column: {
                "type": "object",
                "properties": {
                    "source_column": {
                        "type": ["string", "null"],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": [
                    "source_column",
                    "confidence",
                ],
                "additionalProperties": False,
            }
            for target_column in target_columns
        },
        "required": target_columns,
        "additionalProperties": False,
    }
    response = await create_response(
        model=model,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        reasoning={
            "effort": reasoning_effort,
        },
        text={
            "format": {
                "name": "schema_matching",
                "schema": schema,
                "type": "json_schema",
                "description": "Mapping from target columns to source columns with confidence and reasoning.",
                "strict": True,
            },
            "verbosity": "low" if model.startswith("gpt-5") else "medium",
        },
        temperature=1.0 if model.startswith("gpt-5") else 0.0,  # GPT-5 requires 1.0
        # seed=None if model.startswith("gpt-5") else seed,
    )
    raw_output = json.loads(response.output_text)
    if response.reasoning is not None:
        logger.debug(response.reasoning.model_dump(mode="json"))
    return {
        target_column: (
            mapping["source_column"] if mapping["source_column"] in source_columns else None,
            mapping["confidence"],
            None,
        )
        for target_column, mapping in raw_output.items()
    }

class FieldName(BaseModel, frozen=True):
    old: str
    new: str

class FieldNames(BaseModel, frozen=True):
    names: list[FieldName]


# TODO: pass full properties and generate full properties
async def gpt_rename_fields(properties: list[str], model: str, seed: Optional[int] = None) -> dict[str, str]:
    system_message = "You think of synonyms."
    user_message = json.dumps(properties, indent=4)

    response = await parse_response(
        model=model,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        text_format=FieldNames,
        temperature=1.0 if model.startswith("gpt-5") else 0.0,
        # seed=None if model.startswith("gpt-5") else seed,
    )
    names: FieldNames = response.output_parsed
    return {
        name.old: (name.new if name.new != "" else name.old)
        for name in names.names
        if name.old in properties
    }


class Property(Schema, frozen=True):
    name: str
    required: bool

class Properties(BaseModel, frozen=True):
    properties: list[Property]


async def gpt_add_properties(schema: ObjectSchema, model: str, seed: Optional[int] = None) -> ObjectSchema:
    system_message = "You think of additional (primitive) properties."
    user_message = schema.model_dump_json(indent=4)

    response = await parse_response(
        model=model,
        input=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        text_format=Properties,
        temperature=1.0 if model.startswith("gpt-5") else 0.0,
        # seed=None if model.startswith("gpt-5") else seed,
    )
    newProperties: Properties = response.output_parsed
    return ObjectSchema(
        properties={prop.name: Schema(type=prop.type, description=prop.description, examples=prop.examples) for prop in newProperties.properties},
        required=[prop.name for prop in newProperties.properties if prop.required],
    )