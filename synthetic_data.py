import random
from typing import Optional

from gpt_utils import gpt_rename_fields, gpt_add_properties
from json_schema import ObjectSchema


async def apply_perturbations(target_schema: ObjectSchema, model: str, seed: Optional[int] = None) -> tuple[ObjectSchema, dict[str, Optional[str]]]:
    assert target_schema.properties is not None

    # Default configuration for more balanced perturbations
    perturbation_config = {
        "rename_ratio_min": 0.3,
        "rename_ratio_max": 0.7,
        "delete_ratio_min": 0.1,
        "delete_ratio_max": 0.4,
        "add_count_max": 3,
        "remove_metadata_prob": 0.7
    }
    
    # TODO: generate target_schema from scratch
    properties = list(target_schema.properties.keys())
    if not properties:
        return target_schema.model_copy(deep=True), {}
        
    source_schema = target_schema.model_copy(deep=True)
    assert source_schema.properties is not None
    mapping: dict[str, Optional[str]] = {
        k: k for k in properties
    }

    random.seed(seed)

    if source_schema.required is None:
        source_schema.required = []
    
    new_names = await gpt_rename_fields(properties, model=model, seed=seed)

    # More controlled renaming - rename based on config
    rename_ratio = random.uniform(
        perturbation_config.get("rename_ratio_min", 0.3), 
        perturbation_config.get("rename_ratio_max", 0.7)
    )
    rename_count = max(1, int(len(properties) * rename_ratio))
    rename_properties = random.sample(list(new_names.items()), min(rename_count, len(new_names)))

    # rename properties
    for property_name, new_property_name in rename_properties:
        if property_name in source_schema.properties:
            source_schema.properties[new_property_name] = source_schema.properties.pop(property_name)
            # Selectively remove metadata to create varying difficulty levels
            metadata_removal_prob = perturbation_config.get("remove_metadata_prob", 0.7)
            if random.random() < metadata_removal_prob:
                source_schema.properties[new_property_name].description = None
            if random.random() < metadata_removal_prob * 0.8:  # Slightly less likely to remove examples
                source_schema.properties[new_property_name].examples = None
            mapping[property_name] = new_property_name

    # More controlled deletion - delete based on config
    remaining_properties = [p for p in properties if mapping[p] == p]
    if remaining_properties:
        delete_ratio = random.uniform(
            perturbation_config.get("delete_ratio_min", 0.1), 
            perturbation_config.get("delete_ratio_max", 0.4)
        )
        delete_count = max(1, int(len(remaining_properties) * delete_ratio))
        delete_properties = random.sample(remaining_properties, min(delete_count, len(remaining_properties)))

        # delete properties
        for property_name in delete_properties:
            current_name = mapping[property_name]
            if current_name and current_name in source_schema.properties:
                del source_schema.properties[current_name]
                if current_name in source_schema.required:
                    source_schema.required.remove(current_name)
                mapping[property_name] = None

    additional_properties = await gpt_add_properties(source_schema, model=model, seed=seed)
    assert additional_properties.properties is not None
    assert additional_properties.required is not None

    for prop_name, prop_info in additional_properties.properties.items():
        if prop_name in source_schema.properties:
            continue  # Avoid overwriting existing properties
        source_schema.properties[prop_name] = prop_info
        if prop_name in additional_properties.required:
            source_schema.required.append(prop_name)

    return source_schema, mapping


def score_mapping(predicted_mapping: dict[str, tuple[Optional[str], float, Optional[str]]], expected_mapping: dict[str, Optional[str]]) -> tuple[float, float]:
    """
    Score the quality of predicted schema mappings against ground truth.
    Returns (unweighted_accuracy, confidence_weighted_accuracy).
    """
    if not predicted_mapping or not expected_mapping:
        return 0.0, 0.0
    
    # Use expected_mapping as the source of truth for total count
    total_props = len(expected_mapping)
    if total_props == 0:
        return 1.0, 1.0
    
    # Unweighted accuracy: fraction of correct predictions
    correct_predictions = 0
    total_confidence = 0.0
    weighted_score = 0.0
    
    for expected_key, expected_value in expected_mapping.items():
        if expected_key in predicted_mapping:
            prediction_tuple = predicted_mapping[expected_key]
            predicted_value, confidence, reasoning = prediction_tuple
                
            if predicted_value == expected_value:
                correct_predictions += 1
                weighted_score += confidence
            total_confidence += confidence
        else:
            # Missing prediction counts as incorrect
            pass
    
    unweighted_accuracy = correct_predictions / total_props
    
    # Normalize weighted score by total possible confidence
    if total_confidence > 0:
        weighted_accuracy = weighted_score / total_confidence
    else:
        weighted_accuracy = 0.0
    
    return unweighted_accuracy, weighted_accuracy
