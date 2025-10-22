#!/usr/bin/env python
"""
harmonize: Schema Matching and Data Harmonization System
Main entry point for the system providing schema matching capabilities.

IMPORTANT: main_test() uses CARTESIAN PRODUCT matching:
- Each source file is matched against EVERY target file
- With N sources and M targets, you get N×M combinations
- Example: 2 sources × 3 targets = 6 total matches
"""

from dotenv import load_dotenv

from core.run.single_source import process_single_source_target_pair
from core.utils.data import load_confidence_calibrator
from core.utils.save_calibrator import train_and_save_calibrator
from reports.report import generate_reports

from config import APPROACHES

load_dotenv(override=True)

import argparse
import asyncio

import json
from typing import Optional
import logging
import glob
import os
from perfomance import performance_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from json_schema import ObjectSchema
from schema_inference import infer_schema
from synthetic_data import apply_perturbations, score_mapping 

from confidence_calibration import ConfidenceCalibrator

#core
from core import apply_rules, infer_rules



def save_reasoning_to_json(predicted_mapping_with_reasoning, expected_mapping=None, output_name=None, source_schema=None, target_schema=None):
    """
    Save reasoning data to a structured JSON file.
    
    Args:
        predicted_mapping_with_reasoning: Dict with structure {target_col: (source_col, confidence, reasoning)}
        expected_mapping: Optional ground truth mapping for correctness analysis
        output_name: Optional base name for the output file
        source_schema: Optional source schema for metadata
        target_schema: Optional target schema for metadata
    """
    from datetime import datetime
    
    # Debug: Check if function is called and what data it receives
    print(f"\n🔍 DEBUG: save_reasoning_to_json called")
    print(f"   - Reasoning data keys: {list(predicted_mapping_with_reasoning.keys()) if predicted_mapping_with_reasoning else 'None'}")
    print(f"   - Output name: {output_name}")
    
    if not predicted_mapping_with_reasoning:
        print(f"   ⚠️  No reasoning data to save")
        return
    
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
            "source_schema_columns": list(source_schema.properties.keys()) if source_schema else None,
            "target_schema_columns": list(target_schema.properties.keys()) if target_schema else None,
            "total_mappings": len(predicted_mapping_with_reasoning),
            "has_ground_truth": expected_mapping is not None
        },
        "mappings": []
    }
    
    # Process each mapping with reasoning
    for target_col, (source_col, confidence, reasoning) in predicted_mapping_with_reasoning.items():
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
    
    print(f"\n💾 Reasoning saved to: {filename}")
    print(f"   📊 Total mappings with reasoning: {len(reasoning_data['mappings'])}")
    
    return filename


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



def load_ground_truth_for_pair(source_path: str, target_path: str, expected_dir: str = "./assets/test/expected") -> Optional[dict]:
    """
    Load ground truth for a specific source-target pair
    
    Args:
        source_path: Path to source CSV file
        target_path: Path to target JSON file
        expected_dir: Directory containing ground truth files
        
    Returns:
        Ground truth dictionary or None if not found
    """
    # Extract filenames without extensions
    source_name = os.path.splitext(os.path.basename(source_path))[0]
    target_name = os.path.splitext(os.path.basename(target_path))[0]
    
    # Try multiple naming patterns for ground truth files
    possible_gt_names = [
        f"{target_name}_mapping.json",
        f"{source_name}_mapping.json", 
        f"{source_name}_{target_name}_mapping.json",
        f"{target_name}_{source_name}_mapping.json"
    ]
    
    for gt_name in possible_gt_names:
        gt_path = os.path.join(expected_dir, gt_name)
        if os.path.exists(gt_path):
            try:
                with open(gt_path, 'r') as f:
                    ground_truth = json.load(f)
                print(f"✅ Loaded ground truth: {gt_path}")
                return ground_truth
            except Exception as e:
                print(f"⚠️ Error loading {gt_path}: {e}")
                continue
        
        # Also check rm/ subdirectory
        gt_path_rm = os.path.join(expected_dir, "rm", gt_name)
        if os.path.exists(gt_path_rm):
            try:
                with open(gt_path_rm, 'r') as f:
                    ground_truth = json.load(f)
                print(f"✅ Loaded ground truth: {gt_path_rm}")
                return ground_truth
            except Exception as e:
                print(f"⚠️ Error loading {gt_path_rm}: {e}")
                continue
    
    print(f"⚠️ No ground truth found for {source_name} → {target_name}")
    return None


def setup_evaluation():
    """Initialize logging and evaluation setup"""
    logging.info("🧪 Starting Real Data Evaluation")
    logging.info("=" * 60)


async def load_schemas(source_table: str, target_table: str, source_dir: str, target_dir: str) -> tuple[pd.DataFrame, ObjectSchema, ObjectSchema]:
    """
    Load source data and both schemas from files.
    
    Args:
        source_table: Name of source table (without extension)
        target_table: Name of target table (without extension)
        source_dir: Directory containing source files
        target_dir: Directory containing target schema files
        
    Returns:
        Tuple of (source_data, source_schema, target_schema)
    """
    source_data = pd.read_csv(f"{source_dir}/{source_table}.csv")
    source_schema_path = f"{source_dir}/{source_table}.json"
    target_schema_path = f"{target_dir}/{target_table}.json"

    if os.path.exists(source_schema_path):
        with open(source_schema_path) as f:
            source_schema = ObjectSchema.model_validate_json(f.read())
    else:
        source_schema = await infer_schema(source_data)
        print("Source Schema:", source_schema.model_dump_json())

    with open(target_schema_path) as f:
        target_schema = ObjectSchema.model_validate_json(f.read())

    return source_data, source_schema, target_schema


async def predict_mappings(source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict], seed: Optional[int], output_name: Optional[str]) -> dict[str, dict]:
    """
    Generate predictions using all configured approaches.
    
    Args:
        source_schema: Source schema object
        target_schema: Target schema object
        expected_mapping: Optional ground truth mapping
        seed: Random seed for reproducibility
        output_name: Name for output files
        
    Returns:
        Dictionary mapping approach names to their predictions
    """
    predictions_by_approach = {}
    
    for approach, approach_name in APPROACHES:
        predictions_by_approach[approach_name] = approach.predict(
            source_schema=source_schema,
            target_schema=target_schema,
            seed=seed,
        )
        save_reasoning_to_json(
            predictions_by_approach[approach_name], 
            expected_mapping, 
            output_name, 
            source_schema, 
            target_schema
        )
    
    return predictions_by_approach


async def evaluate_predictions(predicted_mapping: dict, expected_mapping: Optional[dict], target_schema: ObjectSchema) -> tuple[tuple[float, dict], int]:
    """
    Evaluate predictions against ground truth or use confidence-based fallback.
    
    Args:
        predicted_mapping: The predicted column mappings
        expected_mapping: Ground truth mapping (can be None)
        target_schema: Target schema for weight calculation
        
    Returns:
        Tuple of (score, weight)
    """
    if expected_mapping is None:
        conf_sum = sum(conf for _, conf in predicted_mapping.values())
        total = len(target_schema.properties)
        fallback_score = conf_sum / total if total > 0 else 0.0
        score = (fallback_score, {"note": "proxy score based on high-confidence matches"})
        weight = total
    else:
        score = score_mapping(predicted_mapping, expected_mapping)
        weight = len(target_schema.properties.keys())
        print("Score:", score)
    
    return score, weight


async def apply_mapping_and_save(source_data: pd.DataFrame, predicted_mapping: dict, target_schema: ObjectSchema, output_name: str):
    """
    Apply the predicted mapping to source data and save results.
    
    Args:
        source_data: Source dataframe
        predicted_mapping: The predicted column mappings
        target_schema: Target schema
        output_name: Output file path
    """
    rules = await infer_rules(predicted_mapping, target_schema)
    predicted_data = apply_rules(source_data, rules)
    predicted_data.to_csv(output_name, index=False)



async def train_confidence_calibrators(detailed_matches: list, all_ground_truth: list):
    """
    Train and save confidence calibrators for all approaches.
    
    Args:
        detailed_matches: List of detailed match results for training
    """
    for approach, approach_name in APPROACHES:
        confidence_calibrator = await load_confidence_calibrator(approach_name)
        if not confidence_calibrator:
            print(f"⚠️ No confidence calibrator found for {approach_name}, skipping training")
            continue
        
        confidence_calibrator.collect_training_data(detailed_matches, all_ground_truth)
        train_and_save_calibrator(confidence_calibrator, approach_name)


async def main(args: argparse.Namespace):
    """Main entry point for the harmonize system."""
    os.chdir(os.path.dirname(__file__))
    await main_test(args)



async def process_pair(source_path: str, target_path: str, args: argparse.Namespace) -> dict:
    """
    Process a single source-target pair and return detailed match results.
    
    Args:
        source_csv_path: Path to source CSV file
        target_path: Path to target JSON schema file
        args: Command-line arguments
    """ 
    # Initialize accumulators
    detailed_matches = []
    all_ground_truth_data = []
    dataset_names = []

    def pick(paths, name, ext):
        if name:
            p = os.path.join(args.source_dir if ext==".csv" else args.target_dir, f"{name}{ext}")
            return [p] if os.path.exists(p) else []
        return paths

    source_paths = pick(source_path, args.source_table, ".csv")
    target_paths = pick(target_path, args.target_table, ".json")

    if not source_paths:
        source_paths = source_paths
    if not target_paths:
        target_paths = target_paths
        
        # Process each source against each target (Cartesian product)
    for source_csv_path in source_paths:
        source_name = os.path.splitext(os.path.basename(source_csv_path))[0]
        logging.info(f"Processing source: {source_name}")
        
        for target_path in tqdm(target_paths, desc=f"Matching {source_name}"):
            target_name = os.path.splitext(os.path.basename(target_path))[0]
            pair_name = f"{source_name}_to_{target_name}"
            
            print(f"\n[INFO] Processing: {source_name} → {target_name}")
            
            # Process this source-target pair
            matched_predictions = await process_single_source_target_pair(
                source_csv_path, target_path, args
            )
            
            detailed_matches.append(matched_predictions)
            
            # Load corresponding ground truth for this pair (if exists)
            pair_ground_truth = load_ground_truth_for_pair(
                source_csv_path, target_path, args.expected_dir
            )
            all_ground_truth_data.append(pair_ground_truth)
            dataset_names.append(pair_name)

    return detailed_matches, all_ground_truth_data, dataset_names
        
        

async def main_test(args: argparse.Namespace):
    """
    Process all source-target pairs (Cartesian product) and generate comprehensive evaluation reports.
    For each source file, matches against each target file (N sources × M targets = N×M pairs).
    """
    setup_evaluation()
    os.makedirs("output/results", exist_ok=True)
    
  
    
    # Discover all source and target files
    source_paths = sorted(glob.glob(f"{args.source_dir}/*.csv"))
    target_paths = sorted(glob.glob(f"{args.target_dir}/*.json"))

    detailed_matches, all_ground_truth_data, dataset_names = await process_pair(source_paths, target_paths, args)

    print(f"\n✅ Processed {len(detailed_matches)} total source-target combinations")
    
    # Train calibrators
    await train_confidence_calibrators(detailed_matches, all_ground_truth_data)
    
    # Generate all reports
    generate_reports(
        detailed_matches, 
        all_ground_truth_data, 
        dataset_names, 
        args.expected_dir
    )


async def main_all_expected(args: argparse.Namespace):
    """
    Process all expected files and compute overall score.
    """
    score_sum = 0
    weight_sum = 0

    for expected_path in tqdm(sorted(glob.glob("**/*.json", root_dir="./assets/test/expected", recursive=True))):
        print(flush=True)
        expected_name, _ = os.path.splitext(expected_path)

        with open(f"./assets/test/expected/{expected_name}.json") as f:
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
        
        predicted_mapping, score, weight = await process_expectation(
            source_table, target_table, expected_mapping, 
            seed=args.seed, output_name=args.output_name, 
            source_dir=args.source_dir, target_dir=args.target_dir
        )
        score_sum += score[0] * weight
        weight_sum += weight

    overall_score = score_sum / weight_sum
    print("Overall Score", overall_score)


async def main_synthetic(args: argparse.Namespace):
    """
    Generate and process synthetic test data.
    """
    score_sum = 0
    weight_sum = 0

    target_files = sorted(glob.glob("**/*.json", root_dir=args.target_dir, recursive=True))
    print(f"🔍 Found {len(target_files)} target files: {target_files}")

    for target_path in tqdm(target_files):
        print(f"\n🎯 Processing file: {target_path}")
        print(flush=True)
        target_table, _ = os.path.splitext(target_path)

        full_path = f"{args.target_dir}/{target_path}"
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
        
        out_name = f"{target_table.replace('/', '_')}__synthetic.json"
        os.makedirs("./assets/test/expected", exist_ok=True)
        with open(f"./assets/test/expected/{out_name}", "w") as f:
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
        

        predicted_mapping, score, weight = await process_schemas(
            None, source_schema, target_schema, expected_mapping, 
            seed=args.seed, output_name=args.output_name
        )
        score_sum += score[0] * weight
        weight_sum += weight

    overall_score = score_sum / weight_sum if weight_sum > 0 else 0.0
    print(f"\n🎯 Overall Score: {overall_score}")
    return overall_score


async def process_expectation(source_table: str, target_table: str, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None, source_dir: str = "./assets/test/source", target_dir: str = "./assets/test/target"):
    """
    Process a single expectation by loading schemas and delegating to process_schemas.
    """
    source_data, source_schema, target_schema = await load_schemas(
        source_table, target_table, source_dir, target_dir
    )
    
    return await process_schemas(
        source_data, source_schema, target_schema, expected_mapping, 
        seed=seed, output_name=output_name
    )


async def process_schemas(source_data: Optional[pd.DataFrame], source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    """
    Core processing logic: predict mappings, evaluate, and optionally apply transformations.
    """
    predictions_by_approach = await predict_mappings(
        source_schema, target_schema, expected_mapping, seed, output_name
    )
    
    # Use the first approach's predictions for evaluation (or implement ensemble logic)
    predicted_mapping = predictions_by_approach[APPROACHES[0][1]]
    
    score, weight = await evaluate_predictions(
        predicted_mapping, expected_mapping, target_schema
    )
    
    if source_data is not None and output_name:
        await apply_mapping_and_save(source_data, predicted_mapping, target_schema, output_name)

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
    
    # test mode (quiet evaluation)
    parser.add_argument("--quiet", action="store_true", help="Run in quiet test mode (minimal output)")
    
    # ensemble weight training
    parser.add_argument("--training", action="store_true", help="Train optimal ensemble weights using current results")
    
    # configurable data directories
    parser.add_argument("--source-dir", default="./assets/test/source", help="Directory containing source CSV files")
    parser.add_argument("--target-dir", default="./assets/test/target", help="Directory containing target JSON schema files")
    parser.add_argument("--expected-dir", default="./assets/test/expected", help="Directory containing expected ground truth mapping files")

    args = parser.parse_args()

    asyncio.run(main(args))