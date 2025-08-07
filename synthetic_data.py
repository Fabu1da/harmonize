import random
from typing import Optional, Union, Tuple, Dict
import pandas as pd

from gpt_utils import gpt_rename_fields
from json_schema import ObjectSchema, Schema


def anonymize_dataset(df: pd.DataFrame, seed: Optional[int] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Anonymize column names in a dataset while preserving data content.
    Returns (anonymized_df, original_to_anonymous_mapping).
    """
    random.seed(seed)
    
    # Create anonymous column names
    anonymous_mapping = {}
    reverse_mapping = {}
    
    for i, original_col in enumerate(df.columns):
        anonymous_col = f"col_{i:03d}"
        anonymous_mapping[original_col] = anonymous_col
        reverse_mapping[anonymous_col] = original_col
    
    # Create anonymized dataframe
    anonymized_df = df.rename(columns=anonymous_mapping)
    
    return anonymized_df, reverse_mapping


def extract_content_features(column_data: pd.Series) -> Dict[str, any]:
    """
    Extract features from column values when names are hidden.
    These features help identify column content without relying on names.
    """
    non_null_data = column_data.dropna()
    
    if len(non_null_data) == 0:
        return {
            'data_type': 'null',
            'unique_ratio': 0.0,
            'null_ratio': 1.0,
            'sample_values': [],
            'value_patterns': [],
            'numeric_stats': None
        }
    
    # Basic statistics
    features = {
        'data_type': str(non_null_data.dtype),
        'unique_ratio': non_null_data.nunique() / len(column_data),
        'null_ratio': column_data.isnull().sum() / len(column_data),
        'sample_values': list(non_null_data.sample(min(5, len(non_null_data))).astype(str)),
        'row_count': len(column_data)
    }
    
    # Value pattern analysis
    if non_null_data.dtype == 'object':
        # String patterns
        features['avg_length'] = non_null_data.astype(str).str.len().mean()
        features['has_email_pattern'] = any('@' in str(val) and '.' in str(val) for val in non_null_data.head(100))
        features['has_url_pattern'] = any(str(val).startswith(('http://', 'https://')) for val in non_null_data.head(100))
        features['has_date_pattern'] = any('-' in str(val) or '/' in str(val) for val in non_null_data.head(100))
        features['is_categorical'] = features['unique_ratio'] < 0.1 and non_null_data.nunique() < 20
    else:
        # Numeric patterns
        features['numeric_stats'] = {
            'mean': float(non_null_data.mean()),
            'std': float(non_null_data.std()),
            'min': float(non_null_data.min()),
            'max': float(non_null_data.max()),
            'is_integer': all(float(val).is_integer() for val in non_null_data.head(100))
        }
    
    return features


def compute_content_similarity(source_col: pd.Series, target_schema_property: Schema) -> float:
    """
    Compare source column content with target schema property.
    Returns similarity score between 0 and 1.
    """
    source_features = extract_content_features(source_col)
    score = 0.0
    
    # Type compatibility
    source_type = source_features['data_type']
    target_type = target_schema_property.type
    
    type_compatibility = {
        ('object', 'string'): 0.9,
        ('int64', 'integer'): 1.0,
        ('float64', 'number'): 1.0,
        ('bool', 'boolean'): 1.0,
        ('datetime64[ns]', 'string'): 0.7,
    }
    
    score += type_compatibility.get((source_type, target_type), 0.1)
    
    # Example value similarity
    if target_schema_property.examples:
        target_examples = [str(ex) for ex in target_schema_property.examples]
        source_samples = source_features['sample_values']
        
        # Check for exact matches
        exact_matches = sum(1 for sample in source_samples if sample in target_examples)
        if source_samples:
            score += (exact_matches / len(source_samples)) * 0.5
        
        # Check for pattern similarity
        if source_features.get('has_email_pattern') and any('@' in ex for ex in target_examples):
            score += 0.3
        if source_features.get('has_url_pattern') and any(ex.startswith(('http', 'www')) for ex in target_examples):
            score += 0.3
        if source_features.get('has_date_pattern') and any('-' in ex or '/' in ex for ex in target_examples):
            score += 0.3
    
    # Numeric range similarity
    if source_features.get('numeric_stats') and target_schema_property.examples:
        try:
            target_numeric = [float(ex) for ex in target_schema_property.examples if isinstance(ex, (int, float))]
            if target_numeric:
                source_mean = source_features['numeric_stats']['mean']
                target_mean = sum(target_numeric) / len(target_numeric)
                # Similarity based on mean proximity
                mean_diff = abs(source_mean - target_mean) / max(abs(source_mean), abs(target_mean), 1)
                score += max(0, 0.3 * (1 - mean_diff))
        except:
            pass
    
    return min(1.0, score)


def content_similarity_matcher(source_data: pd.DataFrame, target_schema: ObjectSchema) -> Dict[str, Tuple[Optional[str], float]]:
    """
    Match columns based on data content similarity when names are hidden.
    Returns predictions in the format expected by score_mapping.
    """
    predictions = {}
    
    for target_col, target_props in target_schema.properties.items():
        best_match = None
        best_score = 0.0
        
        for source_col in source_data.columns:
            score = compute_content_similarity(source_data[source_col], target_props)
            if score > best_score:
                best_score = score
                best_match = source_col
        
        # Only predict if confidence is above threshold
        if best_score > 0.3:
            predictions[target_col] = (best_match, best_score)
        else:
            predictions[target_col] = (None, 0.0)
    
    return predictions


async def apply_perturbations(target_schema: ObjectSchema, 
    seed: Optional[int] = None,
    perturbation_config: Optional[Dict[str, Union[int, float]]] = None
    ) -> tuple[ObjectSchema, dict[str, str]]:
    
    # Default configuration for more balanced perturbations
    if perturbation_config is None:
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
    mapping = {
        k: k for k in properties
    }

    random.seed(seed)

    if source_schema.required is None:
        source_schema.required = []
    
    new_names = await gpt_rename_fields(properties, seed=seed)

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

    # add properties with more realistic names and examples
    realistic_properties = [
        ("created_at", "string", ["2023-01-01T00:00:00Z", "2024-03-15T14:30:00Z"]),
        ("updated_at", "string", ["2023-12-31T23:59:59Z", "2024-07-20T09:15:00Z"]),
        ("is_active", "boolean", [True, False]),
        ("version", "integer", [1, 2, 3]),
        ("score", "number", [0.85, 1.42, 99.9]),
        ("category", "string", ["electronics", "books", "clothing"]),
        ("priority", "integer", [1, 2, 3, 4, 5]),
        ("status", "string", ["pending", "approved", "rejected"]),
        ("count", "integer", [0, 10, 100, 1000]),
        ("price", "number", [9.99, 29.95, 199.00]),
        ("name", "string", ["Product A", "Service B", "Item C"]),
        ("description", "string", ["High quality item", "Premium service", "Best value"]),
        ("tags", "string", ["important", "featured", "sale"]),
        ("owner_id", "integer", [1001, 2002, 3003]),
        ("reference", "string", ["REF-001", "DOC-123", "ID-999"])
    ]
    
    add_count = random.randint(1, min(perturbation_config.get("add_count_max", 3), len(realistic_properties)))
    selected_props = random.sample(realistic_properties, min(add_count, len(realistic_properties)))
    
    for prop_name, prop_type, prop_examples in selected_props:
        # Ensure unique property name
        counter = 1
        unique_name = prop_name
        while unique_name in source_schema.properties:
            unique_name = f"{prop_name}_{counter}"
            counter += 1
        
        source_schema.properties[unique_name] = Schema(
            type=prop_type,
            examples=prop_examples,
        )
        # Make some properties required with lower probability for realism
        if random.random() < 0.3:  # 30% chance of being required
            source_schema.required.append(unique_name)

    return source_schema, mapping


def score_mapping(predicted_mapping: dict[str, tuple[Optional[str], float]], expected_mapping: dict[str, Optional[str]]) -> tuple[float, float]:
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
            predicted_value, confidence = predicted_mapping[expected_key]
            if predicted_value == expected_value:
                correct_predictions += 1
                weighted_score += confidence
            else:
                weighted_score -= confidence * 0.5  # Lighter penalty for wrong predictions
            total_confidence += confidence
        else:
            # Missing prediction counts as incorrect
            pass
    
    unweighted_accuracy = correct_predictions / total_props
    
    # Normalize weighted score by total possible confidence
    if total_confidence > 0:
        weighted_accuracy = max(0.0, weighted_score / total_confidence)
    else:
        weighted_accuracy = 0.0
    
    return unweighted_accuracy, weighted_accuracy


def test_anonymous_column_matching(source_data: pd.DataFrame, target_schema: ObjectSchema, seed: Optional[int] = None) -> Dict[str, any]:
    """
    Test column matching with anonymized column names.
    
    Args:
        source_data: DataFrame with original column names
        target_schema: Target schema to match against
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with test results including accuracy scores
    """
    # Step 1: Anonymize the source data
    anonymized_data, column_mapping = anonymize_dataset(source_data, seed=seed)
    
    # Step 2: Create ground truth mapping (original columns -> target schema properties)
    # This would normally be provided or inferred from your data
    # For demonstration, we'll assume a direct mapping exists
    ground_truth = {}
    for target_prop in target_schema.properties.keys():
        # Try to find matching column in original data
        matching_cols = [col for col in source_data.columns if col.lower() == target_prop.lower()]
        if matching_cols:
            # Map to the anonymized column name
            original_col = matching_cols[0]
            anonymous_col = column_mapping.get(original_col)
            if anonymous_col:
                ground_truth[target_prop] = anonymous_col
        else:
            ground_truth[target_prop] = None
    
    # Step 3: Run content-based matching on anonymized data
    predictions = content_similarity_matcher(anonymized_data, target_schema)
    
    # Step 4: Evaluate results
    unweighted_acc, weighted_acc = score_mapping(predictions, ground_truth)
    
    # Step 5: Create detailed results
    results = {
        'anonymized_columns': list(anonymized_data.columns),
        'original_to_anonymous_mapping': {v: k for k, v in column_mapping.items()},
        'ground_truth_mapping': ground_truth,
        'predictions': predictions,
        'unweighted_accuracy': unweighted_acc,
        'weighted_accuracy': weighted_acc,
        'total_target_properties': len(target_schema.properties),
        'successful_matches': sum(1 for pred in predictions.values() if pred[0] is not None),
    }
    
    return results
