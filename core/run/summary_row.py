from typing import Dict, List


def add_overall_summary_row(comparison_table: List, headers: List, evaluation_mapping: Dict,
                           predicted_mapping: Dict, embed_predicted: Dict, 
                           cluster_predicted: Dict, majority_predicted: Dict, 
                           weighted_predicted: Dict) -> List:
    """Add overall summary row to comparison table"""
    approaches = [
        ("GPT", predicted_mapping),
        ("Embedding", embed_predicted),
        ("Clustering", cluster_predicted),
        ("Majority Vote", majority_predicted),
        ("Weighted Ensemble", weighted_predicted)
    ]
    
    overall_row = ["Overall", " "]  # Target column = "Overall", GT column = " "
    
    for name, predictions in approaches:
        correct_count = sum(1 for col, expected_src in evaluation_mapping.items()
                          if col in predictions and predictions[col][0] == expected_src)
        total_count = len(evaluation_mapping)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        # Calculate confidence-weighted score
        confidence_weighted_score = 0.0
        for col, expected_src in evaluation_mapping.items():
            if col in predictions:
                predicted_src, confidence = predictions[col]
                is_correct = (predicted_src == expected_src)
                confidence_weighted_score += confidence if is_correct else -confidence
        
        # Normalize by total count
        normalized_score = confidence_weighted_score / total_count if total_count > 0 else 0.0
        
        # Add accuracy and score to overall row
        overall_row.extend([f"{accuracy:.3f}", f"{normalized_score:.3f}"])
    
    comparison_table.append(overall_row)
    return comparison_table