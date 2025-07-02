import openai
import logging
import os
import json
from typing import Optional
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

from pydantic import BaseModel

from json_schema import ObjectSchema

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

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

async def gpt_column_mapping(source_schema: ObjectSchema, target_schema: ObjectSchema, seed: int = None) -> dict[str, str]:
    
    system_message = " ".join([
        "You are an expert in schema matching.",
        "For each target column, you identify the source column that best matches it.",
        "Use ´null´ to indicate that the target column should not be matched to any source column.",
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
            temperature=0,
            seed=seed,
        )
        mappings = response.choices[0].message.parsed
        return {mapping.target_column: (mapping.source_column, mapping.confidence) for mapping in mappings.mappings}
    except Exception as e:
        print(f"Error during GPT request: {e}")
        return {target_column: None for target_column in target_schema}


class FieldName(BaseModel):
    old: str
    new: str

class FieldNames(BaseModel):
    names: list[FieldName]


async def gpt_rename_fields(properties: list[str], seed: int = None) -> dict[str, str]:
    system_message = "You think of synonyms."
    user_message = json.dumps(sorted(properties), indent=4)

    try:
        response = openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format=FieldNames,
            temperature=0,
            seed=seed,
        )
        names = response.choices[0].message.parsed
        return {name.old: name.new for name in names.names}
    except Exception as e:
        print(f"Error during GPT request: {e}")
        return {p: p for p in properties}
