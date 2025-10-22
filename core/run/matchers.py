from typing import Any, Dict, Optional
import os
import json
from datetime import datetime

from config import APPROACHES


async def run_all_matchers(source_schema: Any, target_schema: Any, seed: int, 
                          real_gt_mapping: Optional[Dict],
                          source_name: str = None, target_name: str = None) -> Dict[str, Dict]:
    """Run all matching algorithms using approaches from config.py"""
    
    print(f"\n🔄 Running {len(APPROACHES)} approaches from config.py")
    print("=" * 60)
    
    # Store all predictions by approach name
    predictions_by_approach = {}
    
    for i, (approach, approach_name) in enumerate(APPROACHES):
        print(f"\n🔍 Running {approach_name} ({i+1}/{len(APPROACHES)})")
        try:
            predictions = await approach.predict(
                source_schema=source_schema,
                target_schema=target_schema,
                seed=seed,
            )
            predictions_by_approach[approach_name] = predictions
            
            # Show brief results
            print(f"   ✅ {approach_name}: {len(predictions)} predictions")
            
            # Save reasoning for this approach if we have names
            if source_name and target_name:
                save_reasoning_to_json(
                    predictions, 
                    real_gt_mapping, 
                    f"{source_name}_to_{target_name}_{approach_name.replace(' ', '_').replace('(', '').replace(')', '')}",
                    source_schema,
                    target_schema
                )
            
        except Exception as e:
            print(f"   ❌ {approach_name}: Error - {e}")
            predictions_by_approach[approach_name] = {}
            continue
    
    return predictions_by_approach


def save_reasoning_to_json(predicted_mapping_with_reasoning, expected_mapping=None, output_name=None, source_schema=None, target_schema=None):
    """
    Save reasoning data to a structured JSON file.
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
    
    print(f"💾 Reasoning saved to: {filename}")
    return filename
