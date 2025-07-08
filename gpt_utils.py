import json
from typing import Optional

import openai
from pydantic import BaseModel, Field

from json_schema import ObjectSchema, Schema



import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type,
)
from openai import RateLimitError, APIError, Timeout
from metrics import OPENAI_CALLS, OPENAI_LATENCY


# Retry on rate limits, server errors, or timeouts
@retry(
    reraise=True,
    stop=stop_after_attempt(5),                           # give up after 5 tries
    wait=wait_random_exponential(min=1, max=60),          # exponential backoff 1s→60s
    retry=retry_if_exception_type((RateLimitError, APIError, Timeout)),
)
def openai_with_retry(func: callable, *args, **kwargs):
    """Call an OpenAI client method with retries on transient errors."""
    OPENAI_CALLS.labels(api_type="chat").inc()
    with OPENAI_LATENCY.labels(api_type="chat").time():
        return func(*args, **kwargs)




def get_embeddings(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[text], model=model)
    return response.data[0].embedding

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


class ColumnMapping(BaseModel):
    target_column: str
    source_column: Optional[str]
    confidence: float

class ColumnMappings(BaseModel):
    mappings: list[ColumnMapping]

async def gpt_column_mapping(source_schema: ObjectSchema, target_schema: ObjectSchema, seed: Optional[int] = None) -> dict[str, str]:
    system_message = " ".join([
        "You are an expert in schema matching.",
        "For each target column, you identify the source column that best matches it.",
        "Set the target column to null to not match it to any source column.",
        "You also provide a confidence score for each mapping.",
        "The confidence score is a float between 0.0 and 1.0.",
        "A score of 1.0 means a perfect match, and 0.0 means no match.",
    ])
    user_message = "".join([
        "### Source Schema:\n",
        f"{source_schema.model_dump_json(indent=4)}\n\n",
        "### Target Schema:\n",
        f"{target_schema.model_dump_json(indent=4)}",
    ])

    try:
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format=ColumnMappings,
            temperature=0.0,
            seed=seed,
        )
        mappings = response.choices[0].message.parsed
        return {mapping.target_column: ((None if mapping.source_column == 'null' else mapping.source_column), mapping.confidence) for mapping in mappings.mappings}
    except Exception as e:
        print(f"Error during GPT request: {e}")
        return {target_column: None for target_column in target_schema.properties.keys()}


class FieldName(BaseModel):
    old: str
    new: str

class FieldNames(BaseModel):
    names: list[FieldName]


# TODO: pass full properties and generate full properties
async def gpt_rename_fields(properties: list[str], seed: Optional[int] = None) -> dict[str, str]:
    system_message = "You think of synonyms."
    user_message = json.dumps(properties, indent=4)

    try:
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format=FieldNames,
            temperature=0.0,
            seed=seed,
        )
        names = response.choices[0].message.parsed
        return {name.old: name.new for name in names.names}
    except Exception as e:
        print(f"Error during GPT request: {e}")
        return {p: p for p in properties}


class Property(Schema):
    name: str
    required: bool

class Properties(BaseModel):
    properties: list[Property]


# TODO: Check if high temperature is deterministic.
async def gpt_add_properties(schema: ObjectSchema, seed: Optional[int] = None) -> ObjectSchema:
    system_message = "You think of additional (primitive) properties."
    user_message = schema.model_dump_json(indent=4)

    try:
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format=Properties,
            temperature=0.0,
            seed=seed,
        )
        newProperties = response.choices[0].message.parsed
        return ObjectSchema(
            properties={prop.name: Schema(type=prop.type, description=prop.description, examples=prop.examples) for prop in newProperties.properties},
            required=[prop.name for prop in newProperties.properties if prop.required],
        )
    except Exception as e:
        print(f"Error during GPT request: {e}")
        return ObjectSchema()