#!/usr/bin/env python
import dotenv
dotenv.load_dotenv(override=True)

import argparse
import asyncio
from functools import partial
from collections.abc import Callable
import json
from typing import Any, Optional
import logging
import glob
from pathlib import Path
import os
from tabulate import tabulate

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from gpt_utils import gpt_column_mapping
from json_schema import ObjectSchema
from schema_inference import infer_schema
from synthetic_data import apply_perturbations, score_mapping
from embedding_utils import embedding_column_mapping
from clustering_matcher import clustering_matcher 

# Add this:
from ensemble_matchers import create_ensemble_matchers

def get_correctness_indicator(predicted_source, ground_truth_source, confidence=0.0):
    """Return visual indicator for correctness with color background based on confidence"""
    # Handle None values
    if predicted_source is None:
        predicted_source = "—"
    if predicted_source == "—" or ground_truth_source is None:
        return "—"  # No match case
    
    # Calculate background color based on confidence (yellow to red fade)
    # confidence: 0.0 = red, 1.0 = yellow
    red_intensity = int(255 * (1 - confidence * 0.5))  # More red for lower confidence
    green_intensity = int(255 * confidence)  # More green for higher confidence
    
    # ANSI color codes for RGB background
    bg_color = f"\033[48;2;{red_intensity};{green_intensity};0m"  # RGB background
    reset_color = "\033[0m"  # Reset to default
    
    if predicted_source == ground_truth_source:
        return f"{bg_color}{predicted_source} ✅{reset_color}"  # Correct match with background
    else:
        return f"{bg_color}{predicted_source} ❌{reset_color}"  # Incorrect match with background




async def infer_rules(column_map, target_schema: ObjectSchema) -> dict[str, Callable[[dict], Any]]:
    # Ensure column_map contains only strings
    column_map = {
        target: source
        for target, source in column_map.items()
    }

    print("🔍 Processed column_map:", column_map)  # Debugging output

    def rule(data: dict, target_column: str) -> Any:
        """
        Transformation rule: Maps a source column to a target column and applies conversion.
        """
        source_column = column_map.get(target_column)
        if not source_column:
            return None  # No source column mapped

        value = data.get(source_column, None)  # Fetch source data
        if value is None:
            return None  # No value found

        try:
            # Convert value to target schema's expected type
            jsontype = target_schema.properties[target_column].type
            datatype = {
                "string": str,
                "number": float,
                "integer": int,
                "boolean": bool,
            }.get(jsontype, str)
            value = datatype(value)
        except ValueError as e:
            logging.error(f" Value conversion error for column {target_column}: {e}")
            value = None  # Handle conversion errors gracefully

        return value

    #  Return transformation rules dictionary
    return {
        column: partial(rule, target_column=column)
        for column in target_schema.properties.keys()
    }

# TODO: improve runtime complexity from O(R*C) to O(C)
def apply_rules(dataset: pd.DataFrame, rules: dict[str, Callable[[dict], Any]]) -> pd.DataFrame:
    columns = rules.keys()
    dataset2 = pd.DataFrame(columns=columns)
    for row in dataset.itertuples():
        index = row.Index
        data = row._asdict()
        for column, rule in rules.items():
            value = rule(data)
            dataset2.loc[index, column] = value
    return dataset2

async def main(args: argparse.ArgumentParser):
    os.chdir(os.path.dirname(__file__))
    # await main_all_expected(args)
    await main_test(args)          # Real data with synthetic ground truth
    # await main_synthetic(args)   # Pure synthetic evaluation



def show_mapping_with_examples(predicted_mapping: dict, source_data: pd.DataFrame, num_examples: int = 1):
    table = []

    for target, (src, conf) in predicted_mapping.items():
        if src and src in source_data.columns:
            examples = source_data[src].dropna().astype(str).unique()[:num_examples]
            example_val = ", ".join(examples) if examples.any() else "—"
        else:
            example_val = "—"
        
        table.append([target, src if src else "—", f"{conf:.2f}", example_val])

    print(tabulate(table, headers=["Target Column", "Predicted Source", "Confidence", "Example"], tablefmt="fancy_grid"))



def export_table_as_image(data, headers, filename):
    df = pd.DataFrame(data, columns=headers)

    fig, ax = plt.subplots(figsize=(len(headers) * 2, len(data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"✅ Table saved to {filepath}")
    
    

# async def main_test(args: argparse.Namespace):
#     source_table = "Cricket"
#     results = []

#     for target_path in tqdm(sorted(glob.glob("**/*.json", root_dir="./assets/target", recursive=True))):
#         print(flush=True)
#         target_table, _ = os.path.splitext(target_path)
#         # await main_core(source_table, target_table, seed=args.seed, output_name=args.output_name)
#         predicted_mapping, score, weight = await main_core(
#             source_table,
#             target_table,
#             seed=args.seed,
#             output_name=args.output_name
#         )
        
#         show_mapping_with_examples(predicted_mapping, pd.read_csv(f"./assets/source/{source_table}.csv"))

#         results.append([
#             source_table,
#             target_table,
#             f"{score[0]:.2f}" if score else "—",
#             weight if weight is not None else "—"
#         ])

#     # Print result summary table
#     print("\n📊 Harmonization Summary Table")
#     print(tabulate(
#         results,
#         headers=["Source Table", "Target Table", "Score", "Weight"],
#         tablefmt="fancy_grid"
#     ))







async def main_test(args: argparse.Namespace):
    """
    Enhanced main_test that uses synthetic data as ground truth for real source data evaluation.
    """
    results = []
    detailed_matches = []
    
    # Add cross-dataset aggregation structure as per todo2.txt
    target_schema_results = {}  # Group results by target schema
    all_approaches = ["GPT", "Embedding", "Clustering", "Majority Vote", "Weighted Ensemble"]

    print("🧪 Starting Real Data Evaluation with Synthetic Ground Truth")
    print("=" * 60)

    # Iterate over all source CSV files
    for source_csv_path in sorted(glob.glob("./assets/source/*.csv")):
        print(f"\n📁 Processing source: {source_csv_path}")
        source_table = Path(source_csv_path).stem
        source_path = f"./assets/source/{source_table}.csv"
        source_data = pd.read_csv(source_path)
        source_schema_path = f"./assets/source/{source_table}.json"

        # Load or infer source schema
        if os.path.exists(source_schema_path):
            with open(source_schema_path) as f:
                source_schema = ObjectSchema.model_validate_json(f.read())
        else:
            source_schema = await infer_schema(source_data)

        # Iterate over all target JSON schemas
        for target_path in tqdm(sorted(glob.glob("./assets/target/*.json", recursive=True))):
            target_table, _ = os.path.splitext(os.path.basename(target_path))

            with open(target_path) as f:
                target_schema = ObjectSchema.model_validate_json(f.read())

            print(f"\n🎯 Matching {source_table} → {target_table}")

            # Generate synthetic ground truth for this target schema
            synthetic_source_schema, expected_mapping = await apply_perturbations(target_schema, seed=args.seed)
            print(f"📋 Generated {len(expected_mapping)} synthetic mappings as ground truth")

            # Run all matchers with the real source schema and target schema
            predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=args.seed)
            
            embed_predicted = embedding_column_mapping(
                source_columns=list(source_schema.properties.keys()),
                target_columns=list(target_schema.properties.keys()),
                threshold=0
            )
            
            cluster_predicted = clustering_matcher(source_schema, target_schema)
            
            # Show mapping with examples from real data
            show_mapping_with_examples(predicted_mapping, source_data)

            # Create ensemble matchers
            majority_ensemble, weighted_ensemble = create_ensemble_matchers(
                predicted_mapping,    # GPT predictions
                embed_predicted,      # Embedding predictions  
                cluster_predicted     # Clustering predictions
            )

            # Get ensemble predictions
            majority_predicted = majority_ensemble.predict(source_schema, target_schema)
            weighted_predicted = weighted_ensemble.predict(source_schema, target_schema)

            # Build comprehensive comparison table
            comparison_table = []
            headers = [
                "Target", "Synthetic GT",
                "GPT Match", "GPT Score",
                "Embed Match", "Embed Score", 
                "Cluster Match", "Cluster Score",
                "Majority Match", "Majority Score",
                "Weighted Match", "Weighted Score"
            ]

            for col in target_schema.properties.keys():
                synthetic_gt = expected_mapping.get(col, "—")
                
                gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
                emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
                cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
                majority_match, majority_score = majority_predicted.get(col, ("—", 0.0))
                weighted_match, weighted_score = weighted_predicted.get(col, ("—", 0.0))
                
                # Add color coding for correctness with confidence-based background
                gpt_display = get_correctness_indicator(gpt_match, synthetic_gt, gpt_score)
                emb_display = get_correctness_indicator(emb_match, synthetic_gt, emb_score)
                cluster_display = get_correctness_indicator(cluster_match, synthetic_gt, cluster_score)
                majority_display = get_correctness_indicator(majority_match, synthetic_gt, majority_score)
                weighted_display = get_correctness_indicator(weighted_match, synthetic_gt, weighted_score)
                
                comparison_table.append([
                    col, synthetic_gt,
                    gpt_display, f"{gpt_score:.2f}",
                    emb_display, f"{emb_score:.2f}",
                    cluster_display, f"{cluster_score:.2f}",
                    majority_display, f"{majority_score:.2f}",
                    weighted_display, f"{weighted_score:.2f}"
                ])

                # Add detailed match info for each matcher including ensembles
                for matcher_name, match, sim in [
                    ("gpt", gpt_match, gpt_score),
                    ("embed", emb_match, emb_score),
                    ("cluster", cluster_match, cluster_score),
                    ("majority", majority_match, majority_score),
                    ("weighted", weighted_match, weighted_score)
                ]:
                    if match not in ("—", None):
                        detailed_matches.append({
                            "source": f"real_{source_table}.{match}",
                            "target": f"{target_table}.{col}",
                            "similarity": round(sim, 4),
                            "src_file": f"{source_table}.csv",
                            "trg_file": f"{target_table}.json",
                            "matcher": matcher_name,
                            "synthetic_gt": synthetic_gt
                        })

            print(f"\n📊 Complete Matcher Comparison for {source_table} → {target_table}")
            
            # Calculate accuracy against synthetic ground truth for Overall row
            approaches = [
                ("GPT", predicted_mapping),
                ("Embedding", embed_predicted),
                ("Clustering", cluster_predicted),
                ("Majority Vote", majority_predicted),
                ("Weighted Ensemble", weighted_predicted)
            ]
            
            # Build Overall summary row as per todo.txt algorithm
            overall_row = ["Overall", " "]  # Target column = "Overall", GT column = " "
            
            for name, predictions in approaches:
                correct_count = sum(1 for col, expected_src in expected_mapping.items()
                                  if col in predictions and predictions[col][0] == expected_src)
                total_count = len(expected_mapping)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                # Calculate confidence-weighted score as per todo.txt algorithm
                confidence_weighted_score = 0.0
                for col, expected_src in expected_mapping.items():
                    if col in predictions:
                        predicted_src, confidence = predictions[col]
                        is_correct = (predicted_src == expected_src)
                        confidence_weighted_score += confidence if is_correct else -confidence
                
                # Normalize by total count (score = scores(approach) / len(targetSchema))
                normalized_score = confidence_weighted_score / total_count if total_count > 0 else 0.0
                
                # Add accuracy and score to overall row
                overall_row.extend([f"{accuracy:.3f}", f"{normalized_score:.3f}"])
            
            # Add Overall row to comparison table
            comparison_table.append(overall_row)
            
            print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))

            # Store results for cross-dataset aggregation (todo2.txt)
            if target_table not in target_schema_results:
                target_schema_results[target_table] = {
                    'total_combinations': 0,
                    'approach_accuracies': {name: [] for name in all_approaches},
                    'approach_scores': {name: [] for name in all_approaches}
                }
            
            target_schema_results[target_table]['total_combinations'] += 1
            
            # Store individual approach results for this target schema
            for i, (name, predictions) in enumerate(approaches):
                correct_count = sum(1 for col, expected_src in expected_mapping.items()
                                  if col in predictions and predictions[col][0] == expected_src)
                total_count = len(expected_mapping)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                confidence_weighted_score = 0.0
                for col, expected_src in expected_mapping.items():
                    if col in predictions:
                        predicted_src, confidence = predictions[col]
                        is_correct = (predicted_src == expected_src)
                        confidence_weighted_score += confidence if is_correct else -confidence
                
                normalized_score = confidence_weighted_score / total_count if total_count > 0 else 0.0
                
                target_schema_results[target_table]['approach_accuracies'][name].append(accuracy)
                target_schema_results[target_table]['approach_scores'][name].append(normalized_score)

            # Export table as image
            export_table_as_image(comparison_table, headers, f"RealData_{source_table}_to_{target_table}.png")

            # Calculate overall score
            score = score_mapping(predicted_mapping, expected_mapping)
            score_val = score[0] if score and isinstance(score, tuple) else None
            score_display = f"{score_val:.2f}" if score_val is not None else "—"
            weight_display = str(len(target_schema.properties))

            results.append([
                source_table,
                target_table,
                score_display,
                weight_display
            ])

    # Final summary table
    print("\n📊 Real Data Harmonization Summary")
    summary_headers = ["Source Table", "Target Table", "Score", "Weight"]
    print(tabulate(results, headers=summary_headers, tablefmt="fancy_grid"))
    export_table_as_image(results, summary_headers, "RealData_Harmonization_Summary.png")

    # Export detailed matches as JSON
    with open("output/real_data_detailed_matches.json", "w") as f:
        json.dump(detailed_matches, f, indent=2)
    print("✅ Real data detailed matcher results saved to output/real_data_detailed_matches.json")

    # Create cross-dataset aggregation summary table (todo2.txt Step 2)
    print("\n" + "="*80)
    print("📋 CROSS-DATASET AGGREGATION SUMMARY (todo2.txt)")
    print("="*80)
    
    if target_schema_results:
        # Build aggregation table
        aggregation_table = []
        aggregation_headers = [
            "Target Schema",
            "GPT Acc", "GPT Score",
            "Embed Acc", "Embed Score", 
            "Cluster Acc", "Cluster Score",
            "Majority Acc", "Majority Score",
            "Weighted Acc", "Weighted Score"
        ]
        
        # Calculate averages for each target schema
        overall_stats = {name: {'accuracies': [], 'scores': []} for name in all_approaches}
        
        for target_name, data in target_schema_results.items():
            row = [target_name]
            
            for approach_name in all_approaches:
                accuracies = data['approach_accuracies'][approach_name]
                scores = data['approach_scores'][approach_name]
                
                avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                avg_score = sum(scores) / len(scores) if scores else 0.0
                
                # Store for overall calculation
                overall_stats[approach_name]['accuracies'].extend(accuracies)
                overall_stats[approach_name]['scores'].extend(scores)
                
                row.extend([f"{avg_accuracy:.3f}", f"{avg_score:.3f}"])
            
            aggregation_table.append(row)
        
        # Add overall summary row across all target schemas
        overall_row = ["Overall"]
        for approach_name in all_approaches:
            all_accuracies = overall_stats[approach_name]['accuracies']
            all_scores = overall_stats[approach_name]['scores']
            
            overall_accuracy = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            overall_row.extend([f"{overall_accuracy:.3f}", f"{overall_score:.3f}"])
        
        aggregation_table.append(overall_row)
        
        print(tabulate(aggregation_table, headers=aggregation_headers, tablefmt="fancy_grid"))
        
        # Export aggregation table as image
        export_table_as_image(aggregation_table, aggregation_headers, "CrossDataset_Aggregation_Summary.png")
        print("✅ Cross-dataset aggregation table saved as image")
    else:
        print("⚠️ No target schema results to aggregate")

    



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


def compare_mappings(old_mapping, new_mapping):
    """
    Compares two dictionaries and identifies:
    - Unchanged mappings
    - Changed mappings
    - Newly added mappings
    - Removed mappings
    """
    unchanged = {}
    changed = {}
    added = {}
    removed = {}

    for key in old_mapping:
        if key in new_mapping:
            if old_mapping[key] == new_mapping[key]:
                unchanged[key] = old_mapping[key]
            else:
                changed[key] = (old_mapping[key], new_mapping[key])
        else:
            removed[key] = old_mapping[key]

    for key in new_mapping:
        if key not in old_mapping:
            added[key] = new_mapping[key]

    return {"unchanged": unchanged, "changed": changed, "added": added, "removed": removed}


def calculate_individual_scores(expected_mapping: dict, target_schema: ObjectSchema) -> list:
    """
    Calculate detailed scores for each matcher approach against ground truth.
    """
    # This would be called after running the matchers
    # For now, return empty list - you can enhance this based on your needs
    return []


async def main_core_inner_with_ensembles(source_data: Optional[pd.DataFrame], source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    """
    Enhanced version of main_core_inner that includes ensemble evaluation with ground truth.
    """
    # Get individual matcher predictions
    predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )

    cluster_predicted = clustering_matcher(source_schema, target_schema)
    
    # Create ensemble matchers
    majority_ensemble, weighted_ensemble = create_ensemble_matchers(
        predicted_mapping,    # GPT predictions
        embed_predicted,      # Embedding predictions  
        cluster_predicted     # Clustering predictions
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
            "GPT Match", "GPT Score",
            "Embed Match", "Embed Score",
            "Cluster Match", "Cluster Score",
            "Majority Match", "Majority Score",
            "Weighted Match", "Weighted Score"
        ]

        for col in target_schema.properties.keys():
            expected = expected_mapping.get(col, "—")

            gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
            emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
            cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
            majority_match, majority_score = majority_predicted.get(col, ("—", 0.0))
            weighted_match, weighted_score = weighted_predicted.get(col, ("—", 0.0))
            
            comparison_table.append([
                col, expected,
                gpt_match, f"{gpt_score:.2f}",
                emb_match, f"{emb_score:.2f}",
                cluster_match, f"{cluster_score:.2f}",
                majority_match, f"{majority_score:.2f}",
                weighted_match, f"{weighted_score:.2f}"
            ])

        print("\n📊 Complete Matcher Comparison (Including Ensembles)")
        print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))

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
    predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )

    cluster_predicted = clustering_matcher(source_schema, target_schema)
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
            "GPT Match", "GPT Score",
            "Embed Match", "Embed Score",
            "Cluster Match", "Cluster Score",
        ]

        for col in target_schema.properties.keys():
            expected = expected_mapping.get(col, "—")


            gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
            emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
            cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
            comparison_table.append([
                col, expected,
                gpt_match, f"{gpt_score:.2f}",
                emb_match, f"{emb_score:.2f}",
                cluster_match, f"{cluster_score:.2f}"
            ])

        print("\n📊 Combined Matcher Comparison (Including GitTables)")
        print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))

        score = score_mapping(predicted_mapping, expected_mapping)
        weight = len(target_schema.properties.keys())
        print("Score:", score)

    if source_data is not None and output_name:
        rules = await infer_rules(predicted_mapping, target_schema)
        predicted_data = apply_rules(source_data, rules)
        predicted_data.to_csv(output_name, index=False)

    return predicted_mapping, score, weight


# async def main_core_inner(source_data: Optional[pd.DataFrame], source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
#     predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=seed)
#     # print("Predicted Mapping:", predicted_mapping)
    
#     embed_predicted = embedding_column_mapping(
#         source_columns=list(source_schema.properties.keys()),
#         target_columns=list(target_schema.properties.keys()),
#         threshold=0
#     )

#     cluster_predicted = clustering_matcher(source_schema, target_schema)
#     gittables_predicted = gittables_matcher(list(target_schema.properties.keys()))
#     if expected_mapping is None:
#         # Proxy score: percentage of predicted mappings with confidence >= 0.5
#         threshold = 0.5
#         high_conf_count = sum(1 for _, conf in predicted_mapping.values() if conf >= threshold)
#         total = len(target_schema.properties)
#         fallback_score = round(high_conf_count / total, 2) if total > 0 else 0.0
#         score = (fallback_score, {"note": "proxy score based on high-confidence matches"})
#         weight = total
#     else:
#         for k, v in expected_mapping.items():
#             pv = predicted_mapping.get(0)
#             if pv != v:
#                 # print(f"Expected: {k} -> {v}, Actual: {k} -> {pv}")
#                 pass

#         # # Prepare data for display
#         # comparison = []
#         # for col in target_schema.properties.keys():
#         #     expected = expected_mapping.get(col, "—") if expected_mapping else "—"
#         #     predicted, conf = predicted_mapping.get(col, ("—", 0.0))
#         #     comparison.append([col, expected, predicted, f"{conf:.2f}"])
#         # # Print as table
#         # print(tabulate(comparison, headers=["Target Column", "Expected Source", "Predicted Source", "Confidence"], tablefmt="github"))
        
        
#         comparison_table = []
#         status_icons = {"unchanged": "✔", "changed": "✖", "added": "➕", "removed": "➖"}

#         # Flatten mapping to get only predicted source columns and confidence
#         flat_predicted = {k: v[0] for k, v in predicted_mapping.items()}
#         comparison = compare_mappings(expected_mapping, flat_predicted)

#         for key in set(expected_mapping.keys()).union(predicted_mapping.keys()):
#             expected = expected_mapping.get(key, "—")
#             predicted_info = predicted_mapping.get(key, ("—", 0.0))
#             predicted, conf = predicted_info
#             source_column = predicted if predicted != "—" else None

#             if key in comparison["unchanged"]:
#                 status = status_icons["unchanged"]
#             elif key in comparison["changed"]:
#                 status = status_icons["changed"]
#             elif key in comparison["added"]:
#                 status = status_icons["added"]
#             elif key in comparison["removed"]:
#                 status = status_icons["removed"]
#             else:
#                 status = "?"

#             comparison_table.append([
#                 key,                # Target
#                 expected,           # Expected Source
#                 predicted,          # Predicted Source
#                 source_column,      # Source Column
#                 f"{conf:.2f}",      # Confidence
#                 status              # Match Status
#             ])

#         # Print nicely formatted table
#         print(tabulate(
#             comparison_table,
#             headers=["Target", "Expected", "Predicted", "Source Column", "Confidence", "Column Status"],
#             tablefmt="fancy_grid"
#         ))     
        
#         comparison_table = []
#         headers = ["Target", "Expected", "GPT Match", "GPT Score", "Embed Match", "Embed Score", "Cluster Match", "Cluster Score"]

#         for col in target_schema.properties.keys():
#             expected = expected_mapping.get(col, "—") if expected_mapping else "—"

#             gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
#             emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
        
#             cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))

#             comparison_table.append([
#                 col,
#                 expected,
#                 gpt_match, f"{gpt_score:.2f}",
#                 emb_match, f"{emb_score:.2f}",
#                 cluster_match, f"{cluster_score:.2f}"
#             ])

#         print("\n📊 Combined Matcher Comparison")
#         print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))



#         score = score_mapping(predicted_mapping, expected_mapping)
#         weight = len(target_schema.properties.keys())
#         print("Score:", score)

#     if source_data is not None and output_name:
#         rules = await infer_rules(predicted_mapping, target_schema)
#         predicted_data = apply_rules(source_data, rules)
#         predicted_data.to_csv(output_name, index=False)

#     return predicted_mapping, score, weight


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize a dataset to a target schema.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--source-table", help="Name of the source schema/data", type=str)
    parser.add_argument("--target-table", help="Name of the target schema/data", type=str)
    parser.add_argument("--output-name", help="Name of the output files", type=str)
    
    # coma
    parser.add_argument("--coma-threshold", default=0.0, type=float, help="Min similarity for COMA matches")

    
    args = parser.parse_args()
    asyncio.run(main(args))
