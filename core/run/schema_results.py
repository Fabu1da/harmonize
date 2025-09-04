from typing import Dict, List


def update_target_schema_results(target_schema_results: Dict, target_table: str,
                                all_approaches: List, evaluation_mapping: Dict,
                                predicted_mapping: Dict, embed_predicted: Dict,
                                cluster_predicted: Dict, majority_predicted: Dict,
                                weighted_predicted: Dict):
    """Update cross-dataset aggregation results"""
    if target_table not in target_schema_results:
        target_schema_results[target_table] = {
            'total_combinations': 0,
            'approach_accuracies': {name: [] for name in all_approaches},
            'approach_scores': {name: [] for name in all_approaches}
        }
    
    target_schema_results[target_table]['total_combinations'] += 1
    
    approaches = [
        ("GPT", predicted_mapping),
        ("Embedding", embed_predicted),
        ("Clustering", cluster_predicted),
        ("Majority Vote", majority_predicted),
        ("Weighted Ensemble", weighted_predicted)
    ]
    
    for name, predictions in approaches:
        correct_count = sum(1 for col, expected_src in evaluation_mapping.items()
                          if col in predictions and predictions[col][0] == expected_src)
        total_count = len(evaluation_mapping)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        confidence_weighted_score = 0.0
        for col, expected_src in evaluation_mapping.items():
            if col in predictions:
                prediction_tuple = predictions[col]
                # Handle both 2-tuple and 3-tuple formats
                if len(prediction_tuple) == 2:
                    predicted_src, confidence = prediction_tuple
                elif len(prediction_tuple) == 3:
                    predicted_src, confidence, reasoning = prediction_tuple
                else:
                    continue  # Skip invalid formats
                    
                is_correct = (predicted_src == expected_src)
                confidence_weighted_score += confidence if is_correct else -confidence
        
        normalized_score = confidence_weighted_score / total_count if total_count > 0 else 0.0
        
        target_schema_results[target_table]['approach_accuracies'][name].append(accuracy)
        target_schema_results[target_table]['approach_scores'][name].append(normalized_score)
