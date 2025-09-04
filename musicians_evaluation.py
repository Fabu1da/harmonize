#!/usr/bin/env python3
"""
Evaluation script for musicians_semjoinable dataset using the expected mapping as ground truth.
"""

import json
import pandas as pd
import asyncio
from typing import Dict, Optional, Tuple
from pathlib import Path

# Import the evaluation functions
from synthetic_data import score_mapping, content_similarity_matcher
from json_schema import ObjectSchema
from gpt_utils import gpt_column_mapping
from embedding_utils import embedding_column_mapping
from clustering_matcher import clustering_matcher
from ensemble_matchers import create_ensemble_matchers
from schema_inference import infer_schema


def load_ground_truth_mapping(json_path: str) -> Dict[str, Optional[str]]:
    """
    Convert the JSON mapping format to the expected evaluation format.
    
    Args:
        json_path: Path to the JSON file with matches
        
    Returns:
        Dictionary mapping target columns to source columns
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert from JSON format to evaluation format
    expected_mapping = {}
    
    for match in data.get('matches', []):
        target_column = match['target_column']
        source_column = match['source_column']
        expected_mapping[target_column] = source_column
    
    return expected_mapping


def load_source_data(csv_path: str) -> pd.DataFrame:
    """Load the source CSV data."""
    return pd.read_csv(csv_path)


def load_target_schema(json_path: str) -> ObjectSchema:
    """Load the target JSON schema."""
    with open(json_path, 'r') as f:
        content = f.read()
    return ObjectSchema.model_validate_json(content)


async def evaluate_musicians_matching():
    """
    Run evaluation on the musicians_semjoinable dataset using the ground truth mapping.
    """
    print("🎵 Musicians Semjoinable Dataset Evaluation")
    print("=" * 60)
    
    # File paths
    base_path = Path("/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/harmonize/assets")
    source_path = base_path / "source" / "musicians_semjoinable_source.csv"
    target_path = base_path / "target" / "musicians_semjoinable_target.json"
    expected_path = base_path / "expected" / "musicians_semjoinable_mapping.json"
    
    # Load data
    print("📂 Loading data...")
    source_data = load_source_data(str(source_path))
    target_schema = load_target_schema(str(target_path))
    expected_mapping = load_ground_truth_mapping(str(expected_path))
    
    print(f"✅ Source data: {source_data.shape[0]} rows, {source_data.shape[1]} columns")
    print(f"✅ Target schema: {len(target_schema.properties)} properties")
    print(f"✅ Ground truth mappings: {len(expected_mapping)} mappings")
    
    # Display ground truth mappings
    print("\n📋 Ground Truth Mappings:")
    for target_col, source_col in expected_mapping.items():
        print(f"  {target_col} ← {source_col}")
    
    # Infer source schema from data
    source_schema = await infer_schema(source_data)
    
    # Run different matchers and evaluate
    results = []
    
    # GPT-4 based matcher
    print(f"\n🔄 Running GPT-4 Schema Matcher...")
    try:
        gpt_predicted = await gpt_column_mapping(
            source_schema=source_schema,
            target_schema=target_schema,
            source_columns=list(source_data.columns),
            target_columns=list(target_schema.properties.keys()),
            threshold=0
        )
        unweighted_acc, weighted_acc = score_mapping(gpt_predicted, expected_mapping)
        results.append({
            'matcher': 'GPT-4 Schema Matcher',
            'unweighted_accuracy': unweighted_acc,
            'weighted_accuracy': weighted_acc,
            'predictions': gpt_predicted
        })
        print(f"  � Unweighted Accuracy: {unweighted_acc:.3f}")
        print(f"  📊 Weighted Accuracy: {weighted_acc:.3f}")
    except Exception as e:
        print(f"  ❌ Error running GPT-4 matcher: {str(e)}")
        results.append({'matcher': 'GPT-4 Schema Matcher', 'error': str(e)})
    
    # Embedding-based matcher
    print(f"\n🔄 Running Embedding Similarity Matcher...")
    try:
        embed_predicted = await embedding_column_mapping(
            source_schema=source_schema,
            target_schema=target_schema,
            source_columns=list(source_data.columns),
            target_columns=list(target_schema.properties.keys()),
            threshold=0
        )
        unweighted_acc, weighted_acc = score_mapping(embed_predicted, expected_mapping)
        results.append({
            'matcher': 'Embedding Similarity',
            'unweighted_accuracy': unweighted_acc,
            'weighted_accuracy': weighted_acc,
            'predictions': embed_predicted
        })
        print(f"  📊 Unweighted Accuracy: {unweighted_acc:.3f}")
        print(f"  📊 Weighted Accuracy: {weighted_acc:.3f}")
    except Exception as e:
        print(f"  ❌ Error running embedding matcher: {str(e)}")
        results.append({'matcher': 'Embedding Similarity', 'error': str(e)})
    
    # Clustering-based matcher
    print(f"\n🔄 Running Clustering Matcher...")
    try:
        cluster_predicted, cluster_info = clustering_matcher(source_schema, target_schema, return_cluster_info=True)
        unweighted_acc, weighted_acc = score_mapping(cluster_predicted, expected_mapping)
        results.append({
            'matcher': 'Clustering Matcher',
            'unweighted_accuracy': unweighted_acc,
            'weighted_accuracy': weighted_acc,
            'predictions': cluster_predicted,
            'cluster_info': cluster_info
        })
        print(f"  📊 Unweighted Accuracy: {unweighted_acc:.3f}")
        print(f"  📊 Weighted Accuracy: {weighted_acc:.3f}")
        print(f"  📊 Clusters used: {cluster_info['n_clusters_actual']}/{cluster_info['n_clusters_requested']}")
    except Exception as e:
        print(f"  ❌ Error running clustering matcher: {str(e)}")
        results.append({'matcher': 'Clustering Matcher', 'error': str(e)})
    
    # Content similarity matcher
    print(f"\n🔄 Running Content Similarity Matcher...")
    try:
        content_predicted = content_similarity_matcher(source_data, target_schema)
        unweighted_acc, weighted_acc = score_mapping(content_predicted, expected_mapping)
        results.append({
            'matcher': 'Content Similarity',
            'unweighted_accuracy': unweighted_acc,
            'weighted_accuracy': weighted_acc,
            'predictions': content_predicted
        })
        print(f"  📊 Unweighted Accuracy: {unweighted_acc:.3f}")
        print(f"  📊 Weighted Accuracy: {weighted_acc:.3f}")
    except Exception as e:
        print(f"  ❌ Error running content similarity matcher: {str(e)}")
        results.append({'matcher': 'Content Similarity', 'error': str(e)})
    
    # Ensemble matchers (only if we have successful individual predictions)
    successful_predictions = [r for r in results if 'error' not in r and 'predictions' in r]
    
    if len(successful_predictions) >= 2:
        print(f"\n🔄 Running Ensemble Matchers...")
        try:
            # Get predictions from successful matchers
            pred_dict = {r['matcher']: r['predictions'] for r in successful_predictions}
            
            # Create ensemble matchers
            if len(successful_predictions) >= 4:
                majority_ensemble, weighted_ensemble = create_ensemble_matchers(
                    pred_dict.get('GPT-4 Schema Matcher', {}),
                    pred_dict.get('Embedding Similarity', {}),
                    pred_dict.get('Clustering Matcher', {}),
                    pred_dict.get('Content Similarity', {})
                )
            else:
                # Use whatever predictions we have
                pred_list = [p['predictions'] for p in successful_predictions]
                majority_ensemble, weighted_ensemble = create_ensemble_matchers(*pred_list)
            
            # Get ensemble predictions
            majority_predicted = majority_ensemble.predict(source_schema, target_schema)
            weighted_predicted = weighted_ensemble.predict(source_schema, target_schema)
            
            # Evaluate majority ensemble
            unweighted_acc, weighted_acc = score_mapping(majority_predicted, expected_mapping)
            results.append({
                'matcher': 'Majority Ensemble',
                'unweighted_accuracy': unweighted_acc,
                'weighted_accuracy': weighted_acc,
                'predictions': majority_predicted
            })
            print(f"  📊 Majority Ensemble - Unweighted: {unweighted_acc:.3f}, Weighted: {weighted_acc:.3f}")
            
            # Evaluate weighted ensemble
            unweighted_acc, weighted_acc = score_mapping(weighted_predicted, expected_mapping)
            results.append({
                'matcher': 'Weighted Ensemble',
                'unweighted_accuracy': unweighted_acc,
                'weighted_accuracy': weighted_acc,
                'predictions': weighted_predicted
            })
            print(f"  📊 Weighted Ensemble - Unweighted: {unweighted_acc:.3f}, Weighted: {weighted_acc:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error running ensemble matchers: {str(e)}")
            results.append({'matcher': 'Ensemble Matchers', 'error': str(e)})
    else:
        print(f"\n⚠️  Skipping ensemble matchers (need at least 2 successful individual matchers)")
    
    # Summary results
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"{'Matcher':<25} {'Unweighted Acc':<15} {'Weighted Acc':<15}")
    print("-" * 55)
    
    for result in results:
        if 'error' not in result:
            print(f"{result['matcher']:<25} {result['unweighted_accuracy']:<15.3f} {result['weighted_accuracy']:<15.3f}")
        else:
            print(f"{result['matcher']:<25} {'ERROR':<15} {'ERROR':<15}")
    
    # Show some detailed comparison for best performer
    best_result = max([r for r in results if 'error' not in r], 
                     key=lambda x: x['unweighted_accuracy'], default=None)
    
    if best_result:
        print(f"\n🏆 Best Performer: {best_result['matcher']} (Accuracy: {best_result['unweighted_accuracy']:.3f})")
        print("\n🔍 Detailed comparison for best performer:")
        print(f"{'Target Column':<20} {'Predicted':<20} {'Ground Truth':<20} {'Match'}")
        print("-" * 80)
        
        for target_col, expected_source in expected_mapping.items():
            prediction_tuple = best_result['predictions'].get(target_col, (None, 0.0))
            # Handle both 2-tuple and 3-tuple formats
            if len(prediction_tuple) == 2:
                predicted_source, confidence = prediction_tuple
            elif len(prediction_tuple) == 3:
                predicted_source, confidence, reasoning = prediction_tuple
            else:
                predicted_source, confidence = None, 0.0
                
            is_match = "✓" if predicted_source == expected_source else "✗"
            print(f"{target_col:<20} {str(predicted_source):<20} {str(expected_source):<20} {is_match}")
    
    # Export detailed results
    output_path = Path("output") / "musicians_evaluation_results.json"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'dataset': 'musicians_semjoinable',
            'ground_truth_mappings': expected_mapping,
            'evaluation_results': results
        }, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    asyncio.run(evaluate_musicians_matching())
