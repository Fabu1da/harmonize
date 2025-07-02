from typing import Optional, Union

from pydantic import BaseModel, Field

class Schema(BaseModel):
    type: str
    description: Optional[str] = None
    examples: Optional[list[Union[str, int, float, bool]]] = None

class ObjectSchema(Schema):
    type: str = Field(default="object")
    properties: Optional[dict[str, Schema]] = Field(default_factory=dict)
    required: Optional[list[str]] = Field(default_factory=list)

def merge_schemas(schema1: ObjectSchema, schema2: ObjectSchema) -> ObjectSchema:
    merged_properties = {**schema1.properties, **schema2.properties}
    merged_required = list(set(schema1.required) | set(schema2.required))
    
    return ObjectSchema(
        properties=merged_properties,
        required=merged_required
    )