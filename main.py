#!/usr/bin/env python
"""
harmonize: Schema Matching and Data Harmonization System
Main entry point for the system providing schema matching capabilities using
GPT, embedding-based, and clustering-based approaches with pairwise comparison analysis.
"""

from dotenv import load_dotenv

from core.run.global_sumary import generate_global_summaries
from core.run.single_source import process_single_source_target_pair

from core.utils.comprehensive_summary import print_final_comprehensive_summary
from core.utils.data import load_gpt_calibrator
from core.utils.final_summary_tables import generate_final_summary_tables
from core.utils.save_calibrator import train_and_save_calibrator


load_dotenv(override=True)

import argparse
import asyncio

import json
from typing import  Optional
import logging
import glob
import os
from tabulate import tabulate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from gpt_utils import gpt_column_mapping
from json_schema import ObjectSchema
from schema_inference import infer_schema
from synthetic_data import apply_perturbations, score_mapping
from embedding_utils import embedding_column_mapping
from clustering_matcher import clustering_matcher 

from gpt_calibration import GPTConfidenceCalibrator
from pairwise_comparison import export_pairwise_results
from ensemble_matchers import create_ensemble_matchers
from run_pairwise_analysis import run_pairwise_analysis

#core
from core import  apply_rules, infer_rules
from core.utils.cluster_statistics import cluster_stats_collector

print(f"🔍 IMPORT DEBUG:")
try:
    from embedding_utils import embedding_column_mapping
    print(f"   ✅ embedding_column_mapping imported successfully")
except ImportError as e:
    print(f"   ❌ embedding_column_mapping import failed: {e}")

try:
    from clustering_matcher import clustering_matcher
    print(f"   ✅ clustering_matcher imported successfully")
except ImportError as e:
    print(f"   ❌ clustering_matcher import failed: {e}")





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



def setup_evaluation():
    """Initialize logging and evaluation setup"""
    logging.info("🧪 Starting Real Data Evaluation")
    logging.info("=" * 60)



async def main(args: argparse.ArgumentParser):
    """Main entry point for the harmonize system."""
    os.chdir(os.path.dirname(__file__))
    
    # Check if user wants to run pairwise analysis
    if getattr(args, 'pairwise', False):
        logging.info("Running Step 2: Pairwise Matcher Analysis...")
        await run_pairwise_analysis()
        return
    
    # Run default schema matching pipeline
    await main_test(args)




# Initialize global GPT calibrator
gpt_calibrator = GPTConfidenceCalibrator()




async def main_test(args: argparse.Namespace):
    """
    Refactored main_test function broken into smaller components.
    Enhanced main_test that uses synthetic data as ground truth for real source data evaluation.
    """
    # Initialize data structures
    results = []
    detailed_matches = []
    all_pairwise_results = []
    all_triple_results = []
    target_schema_results = {}
    all_approaches = ["GPT", "Embedding", "Clustering", "Majority Vote", "Weighted Ensemble"]

    # Setup evaluation
    setup_evaluation()
    
    # Load GPT calibrator
    gpt_calibrator = await load_gpt_calibrator()

    # Process all source CSV files
    for source_csv_path in sorted(glob.glob("./assets/source/*.csv")):
        logging.info(f"📁 Processing source: {source_csv_path}")
        
        # Process all target JSON schemas for this source
        for target_path in tqdm(sorted(glob.glob("./assets/target/*.json", recursive=True))):
            await process_single_source_target_pair(
                source_csv_path, target_path, args, gpt_calibrator,
                all_pairwise_results, all_triple_results, detailed_matches,
                results, target_schema_results, all_approaches
            )

    # Generate global summaries
    generate_global_summaries(all_triple_results, all_pairwise_results)
    
    # Train and save calibrator
    train_and_save_calibrator(gpt_calibrator)
    
    # Export pairwise comparison results
    print(f"\n🔄 STEP 2 COMPLETE: Exporting Pairwise Comparison Results")
    print("=" * 70)
    if all_pairwise_results:
        export_pairwise_results(all_pairwise_results, "complete_pairwise_analysis")
    else:
        print("⚠️ No pairwise results to export")
    
    # Generate final summary tables
    generate_final_summary_tables(results, detailed_matches, target_schema_results, all_approaches)
    
    # Print final comprehensive summary
    print_final_comprehensive_summary(all_triple_results, all_pairwise_results, results)


async def main_all_expected(args: argparse.Namespace):
    score_sum = 0
    weight_sum = 0

    for expected_path in tqdm(sorted(glob.glob("**/*.json", root_dir="./assets/expected", recursive=True))):
        print(flush=True)
        expected_name, _ = os.path.splitext(expected_path)

        with open(f"./assets/expected/{expected_name}.json") as f:
            expectation = json.load(f)

        source_table = expectation["source_table"]
        target_table = expectation["target_table"]
        expected_mappings = expectation["mappings"]
        expected_mapping = {
            mapping["target_column"]: mapping["source_column"]
            for mapping in expected_mappings
        }

        print(" ")
        print("[INFO]", expected_name, source_table, target_table)
        print("Expected Mapping:", expected_mapping)
        
        predicted_mapping, score, weight = await main_core(source_table, target_table, expected_mapping, seed=args.seed, output_name=args.output_name)
        score_sum += score[0] * weight
        weight_sum += weight

    overall_score = score_sum / weight_sum
    print("Overall Score", overall_score)


async def main_synthetic(args: argparse.ArgumentParser):
    score_sum = 0
    weight_sum = 0

    target_files = sorted(glob.glob("**/*.json", root_dir="./assets/target", recursive=True))
    print(f"🔍 Found {len(target_files)} target files: {target_files}")

    for target_path in tqdm(target_files):
        print(f"\n🎯 Processing file: {target_path}")
        print(flush=True)
        target_table, _ = os.path.splitext(target_path)

        full_path = f"./assets/target/{target_path}"
        print(f"📁 Reading: {full_path}")
        
        try:
            with open(full_path, 'r') as f:
                content = f.read()
                print(f"📄 File content length: {len(content)} characters")
                if len(content.strip()) == 0:
                    print(f"⚠️ Empty file detected: {full_path}")
                    continue
                    
                target_schema = ObjectSchema.model_validate_json(content)
                print(f"✅ Successfully parsed schema")
        except Exception as e:
            print(f"❌ Error reading {full_path}: {e}")
            continue

        source_schema, expected_mapping = await apply_perturbations(target_schema, seed=args.seed)
        # print("Source Schema:", source_schema.model_dump_json())
        # print("Expected Mapping:", expected_mapping)
        
        out_name = f"{target_table.replace('/', '_')}__synthetic.json"
        os.makedirs("./assets/expected", exist_ok=True)
        with open(f"./assets/expected/{out_name}", "w") as f:
            json.dump({
                "source_table": target_table + "_synthetic",
                "target_table": target_table,
                "synthetic": True,
                "generated_with": f"main_synthetic('{target_table}')",
                "mappings": [
                    {"source_column": src, "target_column": tgt}
                    for tgt, src in expected_mapping.items()
                    if src is not None
            ]
        }, f, indent=2)
        

        predicted_mapping, score, weight = await main_core_inner_with_ensembles(
            None, source_schema, target_schema, expected_mapping, 
            seed=args.seed, output_name=args.output_name
        )
        score_sum += score[0] * weight
        weight_sum += weight

    overall_score = score_sum / weight_sum if weight_sum > 0 else 0.0
    print(f"\n🎯 Overall Score: {overall_score}")
    return overall_score


async def main_core(source_table: str, target_table: str, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    source_data = pd.read_csv(f"./assets/source/{source_table}.csv")
    source_schema_path = f"./assets/source/{source_table}.json"
    target_schema_path = f"./assets/target/{target_table}.json"

    if os.path.exists(source_schema_path):
        with open(source_schema_path) as f:
            source_schema = ObjectSchema.model_validate_json(f.read())
    else:
        source_schema = await infer_schema(source_data)
        print("Source Schema:", source_schema.model_dump_json())

    with open(target_schema_path) as f:
        target_schema = ObjectSchema.model_validate_json(f.read())

    #expected_data = pd.read_csv(f"./assets/expected/{expected_name}.csv")

    return await main_core_inner(source_data, source_schema, target_schema, expected_mapping, seed=seed, output_name=output_name)



async def main_core_inner_with_ensembles(source_data: Optional[pd.DataFrame], source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    """
    Enhanced version of main_core_inner that includes ensemble evaluation with ground truth.
    """
    # Get individual matcher predictions with reasoning
    predicted_mapping_with_reasoning = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    # Extract just the mapping for compatibility with existing code
    predicted_mapping = {k: (v[0], v[1]) for k, v in predicted_mapping_with_reasoning.items()}
    
    print(f"\n🔍 GPT MATCHER DEBUG WITH REASONING:")
    for target_col, (source_col, confidence, reasoning) in predicted_mapping_with_reasoning.items():
        print(f"   {target_col} -> {source_col} (conf: {confidence:.3f})")
        print(f"      Reasoning: {reasoning}")
    
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )
    
   # In your main.py, replace the existing debug blocks with more detailed ones:

    # After embed_predicted = embedding_column_mapping(...)
    print(f"\n🔍 EMBEDDING MATCHER DETAILED DEBUG:")
    print(f"   Source columns: {list(source_schema.properties.keys())}")
    print(f"   Target columns: {list(target_schema.properties.keys())}")
    print(f"   Number of source columns: {len(source_schema.properties.keys())}")
    print(f"   Number of target columns: {len(target_schema.properties.keys())}")
    print(f"   Threshold: 0")
    print(f"   Function called successfully: {embed_predicted is not None}")
    print(f"   Return type: {type(embed_predicted)}")
    print(f"   Embedding predictions count: {len(embed_predicted) if embed_predicted else 'None/Empty'}")
    print(f"   Embedding results: {embed_predicted}")

 
    
    

    cluster_predicted, cluster_info = clustering_matcher(source_schema, target_schema, return_cluster_info=True)
    
    # Collect cluster statistics
    dataset_name = output_name if output_name else "unknown_dataset"
    cluster_stats_collector.add_cluster_info(cluster_info, dataset_name)
    
    # ADD THIS DEBUG BLOCK:
    # After cluster_predicted = clustering_matcher(...)
    print(f"\n🔍 CLUSTERING MATCHER DETAILED DEBUG:")
    print(f"   Source schema type: {type(source_schema)}")
    print(f"   Target schema type: {type(target_schema)}")
    print(f"   Function called successfully: {cluster_predicted is not None}")
    print(f"   Return type: {type(cluster_predicted)}")
    print(f"   Clustering predictions count: {len(cluster_predicted) if cluster_predicted else 'None/Empty'}")
    print(f"   Clusters requested: {cluster_info['n_clusters_requested']}")
    print(f"   Clusters actually used: {cluster_info['n_clusters_actual']}")
    print(f"   Total columns: {cluster_info['total_columns']}")
    print(f"   Clustering results: {cluster_predicted}")
    
    
    # Create ensemble matchers
    majority_ensemble, weighted_ensemble = create_ensemble_matchers(
        predicted_mapping,    # GPT predictions
        embed_predicted,      # Embedding predictions
        cluster_predicted,    # Clustering predictions
    )

    # Get ensemble predictions
    majority_predicted = majority_ensemble.predict(source_schema, target_schema)
    weighted_predicted = weighted_ensemble.predict(source_schema, target_schema)

    if expected_mapping is None:
        conf_sum = sum(conf for _, conf in predicted_mapping.values())
        total = len(target_schema.properties)
        fallback_score = conf_sum / total if total > 0 else 0.0
        score = (fallback_score, {"note": "proxy score based on high-confidence matches"})
        weight = total
    else:
        # Enhanced comparison table with ensembles
        comparison_table = []
        headers = [
            "Target", "Expected",
            "GPT Match", "GPT Score", "GPT Reasoning",
            "Embed Match", "Embed Score",
            "Cluster Match", "Cluster Score",
            "Majority Match", "Majority Score",
            "Weighted Match", "Weighted Score"
        ]

        for col in target_schema.properties.keys():
            expected = expected_mapping.get(col, "—")

            gpt_match, gpt_score, gpt_reasoning = predicted_mapping_with_reasoning.get(col, ("—", 0.0, "No reasoning available"))
            emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
            cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
            majority_match, majority_score = majority_predicted.get(col, ("—", 0.0))
            weighted_match, weighted_score = weighted_predicted.get(col, ("—", 0.0))
            
            # Truncate reasoning for table display
            truncated_reasoning = gpt_reasoning[:40] + "..." if len(gpt_reasoning) > 40 else gpt_reasoning
            
            comparison_table.append([
                col, expected,
                gpt_match, f"{gpt_score:.2f}", truncated_reasoning,
                emb_match, f"{emb_score:.2f}",
                cluster_match, f"{cluster_score:.2f}",
                majority_match, f"{majority_score:.2f}",
                weighted_match, f"{weighted_score:.2f}"
            ])

        print("\n📊 Complete Matcher Comparison (Including GPT Reasoning)")
        print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))
        
        # Print detailed reasoning for each match
        print("\n🧠 Detailed GPT Reasoning:")
        print("=" * 60)
        for col in target_schema.properties.keys():
            if col in predicted_mapping_with_reasoning:
                gpt_match, gpt_score, gpt_reasoning = predicted_mapping_with_reasoning[col]
                expected = expected_mapping.get(col, "—")
                correctness = "✅ CORRECT" if gpt_match == expected else "❌ INCORRECT"
                print(f"\nTarget: {col}")
                print(f"Expected: {expected} | Predicted: {gpt_match} | {correctness}")
                print(f"Confidence: {gpt_score:.3f}")
                print(f"Reasoning: {gpt_reasoning}")
                print("-" * 40)

        # Calculate accuracy for each approach
        approaches = [
            ("GPT", predicted_mapping),
            ("Embedding", embed_predicted),
            ("Clustering", cluster_predicted),
            ("Majority Vote", majority_predicted),
            ("Weighted Ensemble", weighted_predicted)
        ]
        
        accuracy_table = []
        for name, predictions in approaches:
            correct_count = sum(1 for col, expected_src in expected_mapping.items()
                              if col in predictions and predictions[col][0] == expected_src)
            total_count = len(expected_mapping)
            accuracy = correct_count / total_count if total_count > 0 else 0.0
            avg_confidence = sum(conf for _, conf in predictions.values()) / len(predictions) if predictions else 0.0
            
            accuracy_table.append([name, f"{accuracy:.3f}", f"{avg_confidence:.3f}", f"{correct_count}/{total_count}"])
        
        print("\n🎯 Accuracy Comparison")
        print(tabulate(
            accuracy_table,
            headers=["Approach", "Accuracy", "Avg Confidence", "Correct/Total"],
            tablefmt="fancy_grid"
        ))

        score = score_mapping(predicted_mapping, expected_mapping)
        weight = len(target_schema.properties.keys())
        print("Score:", score)

    if source_data is not None and output_name:
        rules = await infer_rules(predicted_mapping, target_schema)
        predicted_data = apply_rules(source_data, rules)
        predicted_data.to_csv(output_name, index=False)

    return predicted_mapping, score, weight


async def main_core_inner(source_data: Optional[pd.DataFrame], source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    predicted_mapping_with_reasoning = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    # Extract just the mapping for compatibility with existing code
    predicted_mapping = {k: (v[0], v[1]) for k, v in predicted_mapping_with_reasoning.items()}
    
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )

    cluster_predicted, cluster_info = clustering_matcher(source_schema, target_schema, return_cluster_info=True)
    
    # Collect cluster statistics  
    dataset_name = output_name if output_name else "main_core_inner"
    cluster_stats_collector.add_cluster_info(cluster_info, dataset_name)
    # gittables_predicted = gittables_matcher(list(target_schema.properties.keys()))  # <- GitTables matcher

    if expected_mapping is None:
        conf_sum = sum(conf for _, conf in predicted_mapping.values())
        total = len(target_schema.properties)
        fallback_score = conf_sum / total if total > 0 else 0.0
        score = (fallback_score, {"note": "proxy score based on high-confidence matches"})
        weight = total
    else:
        comparison_table = []
        headers = [
            "Target", "Expected",
            "GPT Match", "GPT Score", "GPT Reasoning",
            "Embed Match", "Embed Score",
            "Cluster Match", "Cluster Score",
        ]

        for col in target_schema.properties.keys():
            expected = expected_mapping.get(col, "—")

            gpt_match, gpt_score, gpt_reasoning = predicted_mapping_with_reasoning.get(col, ("—", 0.0, "No reasoning available"))
            emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
            cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
            
            # Truncate reasoning for table display
            truncated_reasoning = gpt_reasoning[:40] + "..." if len(gpt_reasoning) > 40 else gpt_reasoning
            
            comparison_table.append([
                col, expected,
                gpt_match, f"{gpt_score:.2f}", truncated_reasoning,
                emb_match, f"{emb_score:.2f}",
                cluster_match, f"{cluster_score:.2f}"
            ])

        print("\n📊 Combined Matcher Comparison (Including GPT Reasoning)")
        print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))
        
        # Print detailed reasoning for each match
        print("\n🧠 Detailed GPT Reasoning:")
        print("=" * 60)
        for col in target_schema.properties.keys():
            if col in predicted_mapping_with_reasoning:
                gpt_match, gpt_score, gpt_reasoning = predicted_mapping_with_reasoning[col]
                expected = expected_mapping.get(col, "—")
                correctness = "✅ CORRECT" if gpt_match == expected else "❌ INCORRECT"
                print(f"\nTarget: {col}")
                print(f"Expected: {expected} | Predicted: {gpt_match} | {correctness}")
                print(f"Confidence: {gpt_score:.3f}")
                print(f"Reasoning: {gpt_reasoning}")
                print("-" * 40)

        score = score_mapping(predicted_mapping, expected_mapping)
        weight = len(target_schema.properties.keys())
        print("Score:", score)

    if source_data is not None and output_name:
        rules = await infer_rules(predicted_mapping, target_schema)
        predicted_data = apply_rules(source_data, rules)
        predicted_data.to_csv(output_name, index=False)

    return predicted_mapping, score, weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize a dataset to a target schema.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--source-table", help="Name of the source schema/data", type=str)
    parser.add_argument("--target-table", help="Name of the target schema/data", type=str)
    parser.add_argument("--output-name", help="Name of the output files", type=str)
    
    # coma
    parser.add_argument("--coma-threshold", default=0.0, type=float, help="Min similarity for COMA matches")
    
    # pairwise analysis
    parser.add_argument("--pairwise", action="store_true", help="Run Step 2: Pairwise matcher comparison analysis")

    
    args = parser.parse_args()
    asyncio.run(main(args))
