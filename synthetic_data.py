import random
from typing import Optional

from gpt_utils import gpt_rename_fields
from json_schema import ObjectSchema, Schema


# async def apply_perturbations(target_schema: ObjectSchema, seed: int = None) -> tuple[ObjectSchema, dict[str, str]]:
    
#     # TODO: generate target_schema from scratch
#     properties = list(target_schema.properties.keys())
#     source_schema = target_schema.model_copy(deep=True)
#     mapping = {
#         k: k for k in properties
#     }

#     random.seed(seed)

#     if source_schema.required is None:
#         source_schema.required = []
    
#     new_names = await gpt_rename_fields(properties, seed=seed)

#     rename_properties = random.sample(list(new_names.items()), random.randint(1, len(properties)))

#     # rename properties
#     for property_name, new_property_name in rename_properties:
#         source_schema.properties[new_property_name] = source_schema.properties.pop(property_name)
#         del source_schema.properties[new_property_name].description
#         del source_schema.properties[new_property_name].examples
#         mapping[property_name] = new_property_name

#     delete_properties = random.sample(properties, random.randint(1, len(properties)))

#     # delete properties
#     for property_name in delete_properties:
#         del source_schema.properties[mapping[property_name]]
#         mapping[property_name] = None

#     # add properties
#     for _ in range(random.randint(1, len(properties))):
#         # TODO: replace with single AI call, pass all property names
#         property_name = random.choice(["foo", "bar", "lorem", "ipsum", "dolor", "sit", "amet"])
#         new_property_type = random.choice(["string", "integer", "number", "boolean", "object"])
#         # Generate a valid example based on the property type
#         if new_property_type == "string":
#             new_property_example = ["example"]
#         elif new_property_type == "integer":
#             new_property_example = [42]
#         elif new_property_type == "number":
#             new_property_example = [3.14]
#         elif new_property_type == "boolean":
#             new_property_example = [True]
#         elif new_property_type == "object":
#             new_property_example = [{"key": "value"}]
#         else:
#             new_property_example = ["example"]

#         print("new_property_type", new_property_type)
#         source_schema.properties[property_name] = Schema(
#             type=new_property_type,
#             examples=new_property_example,
#         )
#         if random.choice([True, False]):
#             source_schema.required.append(property_name)

#     return source_schema, mapping




import random
from typing import Optional, Union, Tuple, Dict

from gpt_utils import gpt_rename_fields
from json_schema import ObjectSchema, Schema


async def apply_perturbations(
    target_schema: ObjectSchema,
    seed: Optional[int] = None
) -> Tuple[ObjectSchema, Dict[str, Optional[str]]]:
    """
    Create a noisy clone of target_schema plus a ground-truth mapping.
    Renames a random subset of properties, deletes some, and adds new ones.
    Returns (source_schema, expected_mapping).
    """
    # Copy and prepare
    source_schema = target_schema.model_copy(deep=True)
    original_props = list(target_schema.properties.keys())
    mapping: Dict[str, Optional[str]] = {prop: prop for prop in original_props}

    random.seed(seed)
    if source_schema.required is None:
        source_schema.required = []

    # 1) Rename some properties via GPT suggestions
    new_names = await gpt_rename_fields(original_props, seed=seed)
    to_rename = random.sample(list(new_names.items()), random.randint(1, len(original_props)))
    for old_name, new_name in to_rename:
        # Move property
        prop_def = source_schema.properties.pop(old_name)
        source_schema.properties[new_name] = prop_def
        # Clear metadata
        source_schema.properties[new_name].description = None
        source_schema.properties[new_name].examples = None
        mapping[old_name] = new_name

    # 2) Delete a random subset of properties
    to_delete = random.sample(original_props, random.randint(1, len(original_props)))
    for prop in to_delete:
        new_key = mapping[prop]
        if new_key and new_key in source_schema.properties:
            source_schema.properties.pop(new_key)
        mapping[prop] = None

    # 3) Add some new synthetic properties with type-safe examples
    add_count = random.randint(1, len(original_props))
    candidates = ["foo","bar","lorem","ipsum","dolor","sit","amet"]
    for _ in range(add_count):
        prop = random.choice(candidates)
        jtype = random.choice(["string","integer","number","boolean","object"])
        # Provide at least one valid example per type
        if jtype == "string":
            examples: list[Union[str,int,float,bool]] = ["example"]
        elif jtype == "integer":
            examples = [0]
        elif jtype == "number":
            examples = [0.0]
        elif jtype == "boolean":
            examples = [True]
        else:
            examples = []
        # Assign new property
        source_schema.properties[prop] = Schema(type=jtype, examples=examples)
        # Optionally mark required
        if random.choice([True, False]):
            source_schema.required.append(prop)

    return source_schema, mapping



# returns (unweighted accuracy, confidence-weighted accuracy)
def score_mapping(predicted_mapping: dict[str, tuple[Optional[str], float]], expected_mapping: dict[str, Optional[str]]) -> tuple[float, float]:
    unweighted_accuracy = sum(1 for k, v in predicted_mapping.items() if v[0] == expected_mapping.get(k)) / max(len(predicted_mapping), len(expected_mapping))
    weighted_accuracy = sum((v[1] if v[0] == expected_mapping.get(k) else -v[1]) for k, v in predicted_mapping.items()) / max(len(predicted_mapping), len(expected_mapping))
    return unweighted_accuracy, weighted_accuracy
