import logging
import json
import openai
import os

from json_schema import ObjectSchema

async def add_description(source_schema: ObjectSchema) -> ObjectSchema:
    print("🔄 Mapping columns...")

    system_message = (
        "You are an expert in schema matching and data integration. "
        "Given a source schema in JSON Schema format, your task is to add a short one-line description to each column (i.e., each property). "
        "Return the updated schema in the exact same structure, only adding the 'description' fields. "
        "Respond with ONLY the raw JSON, no markdown or code block formatting."
    )
    user_message = source_schema.model_dump_json(indent=4)

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content.strip()
        updated_schema = ObjectSchema.model_validate_json(content)
        return updated_schema

    except Exception as e:
        logging.error(f"Error during GPT request: {e}")
        return source_schema
