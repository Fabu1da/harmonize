from typing import Any, Dict, Optional, Tuple
import clustering_matcher
from embedding_utils import embedding_column_mapping
from gpt_utils import gpt_column_mapping
import os
import json
from datetime import datetime


async def run_all_matchers(source_schema: Any, target_schema: Any, seed: int, 
                          real_gt_mapping: Optional[Dict], gpt_calibrator: Any, 
                          source_name: str = None, target_name: str = None) -> Tuple[Dict, Dict, Dict]:
    """Run all matching algorithms and return their predictions"""
    # Run GPT matcher
    raw_gpt_predictions = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    # Display GPT reasoning
    print(f"\n🔍 GPT MATCHER DEBUG WITH REASONING:")
    for target_col, prediction_tuple in raw_gpt_predictions.items():
        if len(prediction_tuple) == 3:
            source_col, confidence, reasoning = prediction_tuple
            print(f"   {target_col} -> {source_col} (conf: {confidence:.3f})")
            print(f"      Reasoning: {reasoning}")
        elif len(prediction_tuple) == 2:
            source_col, confidence = prediction_tuple
            print(f"   {target_col} -> {source_col} (conf: {confidence:.3f})")
            print(f"      Reasoning: Not available")
        else:
            print(f"   {target_col} -> Invalid prediction format")
    
    # Also store raw predictions for detailed reasoning display later
    print(f"\n📝 Detailed GPT Reasoning Analysis:")
    print("=" * 60)
    for target_col, prediction_tuple in raw_gpt_predictions.items():
        if len(prediction_tuple) == 3:
            source_col, confidence, reasoning = prediction_tuple
            print(f"\n🎯 Target: {target_col}")
            print(f"   Match: {source_col}")
            print(f"   Confidence: {confidence:.3f}")
            print(f"   Reasoning: {reasoning}")
            print("-" * 40)
    
    # Collect training data if we have ground truth
    if real_gt_mapping and gpt_calibrator:
        gpt_calibrator.collect_training_data(raw_gpt_predictions, real_gt_mapping)
    
    # Apply calibration if calibrator is fitted
    if gpt_calibrator and gpt_calibrator.is_fitted:
        predicted_mapping = gpt_calibrator.calibrate_predictions(raw_gpt_predictions)
        print("🎯 Applied isotonic calibration to GPT-4 confidences")
    else:
        predicted_mapping = raw_gpt_predictions
        print("⚠️ Using raw GPT-4 confidences (calibrator not fitted)")

    # Run other matchers
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )
    
    cluster_predicted, cluster_info = clustering_matcher.clustering_matcher(source_schema, target_schema, return_cluster_info=True)
    
    # Note: cluster_info can be used here if needed for logging or stats collection
    
    # Save reasoning to JSON if we have source and target names
    if source_name and target_name:
        save_reasoning_to_json(
            raw_gpt_predictions, 
            real_gt_mapping, 
            f"{source_name}_to_{target_name}",
            source_schema,
            target_schema
        )
    
    return predicted_mapping, embed_predicted, cluster_predicted


def save_reasoning_to_json(predicted_mapping_with_reasoning, expected_mapping=None, output_name=None, source_schema=None, target_schema=None):
    """
    Save GPT reasoning data to a structured JSON file.
    """
    # Create output directory if it doesn't exist
    os.makedirs("./output/reasoning", exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = output_name if output_name else "reasoning"
    filename = f"./output/reasoning/{base_name}_{timestamp}.json"
    
    # Structure the reasoning data
    reasoning_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source_schema_columns": list(source_schema.properties.keys()) if source_schema and hasattr(source_schema, 'properties') else None,
            "target_schema_columns": list(target_schema.properties.keys()) if target_schema and hasattr(target_schema, 'properties') else None,
            "total_mappings": len(predicted_mapping_with_reasoning),
            "has_ground_truth": expected_mapping is not None
        },
        "mappings": []
    }
    
    # Process each mapping with reasoning
    for target_col, prediction_tuple in predicted_mapping_with_reasoning.items():
        if len(prediction_tuple) == 3:
            source_col, confidence, reasoning = prediction_tuple
        elif len(prediction_tuple) == 2:
            source_col, confidence = prediction_tuple
            reasoning = "No reasoning provided"
        else:
            continue
            
        mapping_entry = {
            "target_column": target_col,
            "predicted_source_column": source_col,
            "confidence": float(confidence),
            "reasoning": reasoning
        }
        
        # Add ground truth comparison if available
        if expected_mapping:
            expected_source = expected_mapping.get(target_col)
            mapping_entry["expected_source_column"] = expected_source
            mapping_entry["is_correct"] = source_col == expected_source
        
        reasoning_data["mappings"].append(mapping_entry)
    
    # Calculate summary statistics if ground truth is available
    if expected_mapping:
        correct_mappings = sum(1 for mapping in reasoning_data["mappings"] if mapping.get("is_correct", False))
        total_mappings = len(reasoning_data["mappings"])
        accuracy = correct_mappings / total_mappings if total_mappings > 0 else 0.0
        avg_confidence = sum(mapping["confidence"] for mapping in reasoning_data["mappings"]) / total_mappings if total_mappings > 0 else 0.0
        
        reasoning_data["summary"] = {
            "accuracy": accuracy,
            "correct_mappings": correct_mappings,
            "total_mappings": total_mappings,
            "average_confidence": avg_confidence
        }
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(reasoning_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 GPT Reasoning saved to: {filename}")
    return filename
