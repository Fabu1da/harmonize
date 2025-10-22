import json
from typing import Optional

import openai
from pydantic import BaseModel, Field

from json_schema import ObjectSchema, Schema

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
    reason: str = Field(description="Detailed explanation of mapping decision. For matches: explain semantic similarity and selection rationale. For non-matches: identify the closest potential source column, explain why it's closest, then specify why it's insufficient. Format: 'Closest match: [column] because [reason], but insufficient because [specific failings].'")  

class ColumnMappings(BaseModel):
    mappings: list[ColumnMapping]

async def gpt_column_mapping(source_schema: ObjectSchema, target_schema: ObjectSchema, model: str, seed: Optional[int] = None) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
    """
    Enhanced GPT column mapping with detailed reasoning including closest match analysis.
    
    For no-match decisions, GPT will:
    1. Identify the closest potential source column
    2. Explain why it seems closest 
    3. Specify why it's insufficient
    4. Consider alternative candidates
    5. Describe requirements for a valid match
    """
    system_message = " ".join([
        "You are an expert in schema matching.",
        "For each target column, you identify the source column that best matches it.",
        "Set the source_column to null to not match it to any source column.",
        "You also provide a confidence score for each mapping.",
        "The confidence score is a float between 0.0 and 1.0.",
        "A score of 1.0 means a perfect match, and 0.0 means no match.",
        "You also provide a detailed reason for each mapping decision.",
        "For MATCHES: Explain the semantic similarity, naming patterns, data types, and why you chose this specific source column over others.",
        "For NO MATCHES (null): Always identify the closest potential match from available source columns, explain why it's the closest, then explain specifically why it still doesn't qualify for a match.",
        "Consider semantic meaning, data types, naming conventions, and domain context.",
        "Be analytical and specific - avoid generic statements like 'no corresponding column'.",
        "For no matches, structure your reasoning as: 'Closest match: [column_name] because [similarity reason], but insufficient because [specific reasons why it fails].'",
        "Include your decision-making process and what would need to be present for a valid match."
    ])
    user_message = "".join([
        "### Source Schema:\n",
        f"{source_schema.model_dump_json(indent=4)}\n\n",
        "### Target Schema:\n",
        f"{target_schema.model_dump_json(indent=4)}\n\n",
        "### Example Reasoning:\n",
        "- For a match: 'birthDate matches birthDate due to identical naming and semantic meaning. Both represent date of birth with same data type. Considered other temporal fields but none as semantically precise.'\n",
        "- For no match: 'Closest match: musicianLabel because it relates to musician identity and contains musician information, but insufficient because it stores the musician name as readable text rather than a unique numeric/alphanumeric identifier that musicianID requires. Secondary candidates: musician (represents person entity, not unique ID), birthDate (temporal data, semantically unrelated to identification). A valid match would need a field containing unique identifiers like numeric IDs or alphanumeric codes.'\n\n",
        "Always identify the closest match for no-match decisions and explain the gap.",
    ])

    try:
        response = openai.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            response_format=ColumnMappings,
            temperature=0.0 if not model.startswith("gpt-5") else 1.0,  # GPT-5 requires explicit temperature, not None
            seed=seed,
        )
        mappings = response.choices[0].message.parsed
        return {mapping.target_column: ((None if mapping.source_column == "null" else mapping.source_column), mapping.confidence, mapping.reason) for mapping in mappings.mappings}
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