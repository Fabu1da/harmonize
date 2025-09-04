from typing import Any, Dict, List, Optional, Tuple


def get_correctness_indicator(predicted_source, ground_truth_source, confidence=0.0):
    """
    Generate a visual indicator for mapping correctness with confidence-based coloring.
    
    Args:
        predicted_source: The predicted source column name
        ground_truth_source: The actual/expected source column name  
        confidence: Confidence score (0.0 to 1.0) for the prediction
        
    Returns:
        String with ANSI color codes for terminal display
    """
    # Handle None values
    if predicted_source is None:
        predicted_source = "—"
    if predicted_source == "—" or ground_truth_source is None:
        return "—"  # No match case
    
    # Calculate background color based on confidence (low confidence = red, high = green)
    red_intensity = int(255 * (1 - confidence * 0.5))
    green_intensity = int(255 * confidence)
    
    # ANSI color codes for RGB background
    bg_color = f"\033[48;2;{red_intensity};{green_intensity};0m"
    reset_color = "\033[0m"  # Reset to default
    
    if predicted_source == ground_truth_source:
        return f"{bg_color}{predicted_source} {reset_color}"  # Correct match with background
    else:
        return f"{bg_color}{predicted_source} {reset_color}"  # Incorrect match with background


def build_comparison_table(target_schema: Any, real_gt_mapping: Optional[Dict], 
                          expected_mapping: Dict, predicted_mapping: Dict,
                          embed_predicted: Dict, cluster_predicted: Dict,
                          majority_predicted: Dict, weighted_predicted: Dict) -> Tuple[List, List]:
    """Build comprehensive comparison table for all matchers"""
    comparison_table = []
    gt_type = "Real GT" if real_gt_mapping else "Synthetic GT"
    headers = [
        "Target", gt_type,
        "GPT Match", "GPT Score", "GPT Reasoning",
        "Embed Match", "Embed Score", 
        "Cluster Match", "Cluster Score",
        "Majority Match", "Majority Score",
        "Weighted Match", "Weighted Score"
    ]

    for col in target_schema.properties.keys():
        # Use real ground truth if available, otherwise use synthetic
        ground_truth_for_col = real_gt_mapping.get(col, "—") if real_gt_mapping else expected_mapping.get(col, "—")
        
        # Handle both 2-tuple and 3-tuple formats for GPT predictions
        gpt_tuple = predicted_mapping.get(col, ("—", 0.0))
        if len(gpt_tuple) == 2:
            gpt_match, gpt_score = gpt_tuple
            gpt_reasoning = "Not available"
        elif len(gpt_tuple) == 3:
            gpt_match, gpt_score, gpt_reasoning = gpt_tuple
        else:
            gpt_match, gpt_score, gpt_reasoning = "—", 0.0, "Not available"
            
        emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
        cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
        majority_match, majority_score = majority_predicted.get(col, ("—", 0.0))
        weighted_match, weighted_score = weighted_predicted.get(col, ("—", 0.0))
        
        # Add color coding for correctness with confidence-based background
        gpt_display = get_correctness_indicator(gpt_match, ground_truth_for_col, gpt_score)
        emb_display = get_correctness_indicator(emb_match, ground_truth_for_col, emb_score)
        cluster_display = get_correctness_indicator(cluster_match, ground_truth_for_col, cluster_score)
        majority_display = get_correctness_indicator(majority_match, ground_truth_for_col, majority_score)
        weighted_display = get_correctness_indicator(weighted_match, ground_truth_for_col, weighted_score)
        
        # Truncate reasoning for table display (max 50 characters)
        truncated_reasoning = gpt_reasoning[:50] + "..." if len(gpt_reasoning) > 50 else gpt_reasoning
        
        comparison_table.append([
            col, ground_truth_for_col,
            gpt_display, f"{gpt_score:.2f}", truncated_reasoning,
            emb_display, f"{emb_score:.2f}",
            cluster_display, f"{cluster_score:.2f}",
            majority_display, f"{majority_score:.2f}",
            weighted_display, f"{weighted_score:.2f}"
        ])

    return comparison_table, headers