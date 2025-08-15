#!/usr/bin/env python
"""
Hamonize: Schema Matching and Data Harmonization System
Main entry point for the system providing schema matching capabilities using
GPT, embedding-based, and clustering-based approaches with pairwise comparison analysis.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import argparse
import asyncio
from functools import partial
from collections.abc import Callable, MutableMapping
import json
from typing import Any, Dict, List, Optional
import logging
import glob
from pathlib import Path
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
from pairwise_comparison import print_triple_comparison_table, run_all_pairwise_comparisons, print_pairwise_comparison_table,  run_triple_comparison
from ensemble_matchers import create_ensemble_matchers
from run_pairwise_analysis import run_pairwise_analysis


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




async def infer_rules(column_map, target_schema: ObjectSchema) -> dict[str, Callable[[dict], Any]]:
    # Ensure column_map contains only strings
    column_map = {
        target: source
        for target, source in column_map.items()
    }

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
    """
    Apply transformation rules to convert source dataset to target format.
    
    Args:
        dataset: Source dataframe to transform
        rules: Dictionary mapping target columns to transformation functions
        
    Returns:
        Transformed dataframe with target schema
        
    Note:
        Current implementation has O(R*C) complexity where R=rows, C=columns.
        Could be optimized to O(C) by vectorizing operations.
    """
    columns = rules.keys()
    transformed_df = pd.DataFrame(columns=columns)
    
    for row in dataset.itertuples():
        index = row.Index
        data = row._asdict()
        for column, rule in rules.items():
            value = rule(data)
            transformed_df.loc[index, column] = value
            
    return transformed_df

async def main(args: argparse.ArgumentParser):
    """Main entry point for the Hamonize system."""
    os.chdir(os.path.dirname(__file__))
    
    # Check if user wants to run pairwise analysis
    if getattr(args, 'pairwise', False):
        logging.info("Running Step 2: Pairwise Matcher Analysis...")
        await run_pairwise_analysis()
        return
    
    # Run default schema matching pipeline
    await main_test(args)



def show_mapping_with_examples(predicted_mapping: dict, source_data: pd.DataFrame, num_examples: int = 1):
    """Display schema mapping results with example values in a formatted table."""
    table = []

    for target, (src, conf) in predicted_mapping.items():
        if src and src in source_data.columns:
            examples = source_data[src].dropna().astype(str).unique()[:num_examples]
            example_val = ", ".join(examples) if len(examples) > 0 else "—"
        else:
            example_val = "—"
        
        table.append([target, src if src else "—", f"{conf:.2f}", example_val])

    headers = ["Target Column", "Predicted Source", "Confidence", "Example"]
    print(tabulate(table, headers=headers, tablefmt="fancy_grid"))



def export_table_as_image(data, headers, filename):
    """Export table data as a high-quality image file."""
    df = pd.DataFrame(data, columns=headers)

    fig, ax = plt.subplots(figsize=(len(headers) * 2, len(data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"Table exported to {filepath}")
    
    

def export_table_as_latex(data, headers, filename, caption="", label=""):
    """Export table data as LaTeX table format with proper escaping."""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    
    def escape_latex(text):
        """Escape special LaTeX characters and remove ANSI codes."""
        import re
        text = str(text)
        # Remove ANSI color codes
        text = re.sub(r'\033\[[0-9;]*m', '', text)
        # Escape LaTeX special characters
        latex_special_chars = {
            '&': '\\&', '%': '\\%', '$': '\\$', '#': '\\#',
            '_': '\\_', '{': '\\{', '}': '\\}', '~': '\\textasciitilde{}',
            '^': '\\textasciicircum{}'
        }
        for char, escape in latex_special_chars.items():
            text = text.replace(char, escape)
        return text
    
    with open(filepath, 'w') as f:
        num_cols = len(headers)
        col_spec = 'l' * num_cols
        
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        
        # Write headers
        escaped_headers = [escape_latex(h) for h in headers]
        f.write(" & ".join(escaped_headers) + " \\\\\n")
        f.write("\\midrule\n")
        
        # Write data rows
        for row in data:
            escaped_row = [escape_latex(cell) for cell in row]
            f.write(" & ".join(escaped_row) + " \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        
        f.write("\\end{table}\n")
    
    logging.info(f"LaTeX table exported to {filepath}")
    return filepath

def export_table_as_latex_landscape(data, headers, filename, caption="", label=""):
    """Export wide table as LaTeX landscape table with smaller font"""
    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    
    with open(filepath, 'w') as f:
        # Landscape table for wide tables
        f.write("\\begin{landscape}\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\small\n")  # Smaller font for wide tables
        
        # Dynamic column specification based on content width
        num_cols = len(headers)
        if num_cols > 8:
            col_spec = 'p{1.5cm}' * num_cols  # Fixed width columns for very wide tables
        else:
            col_spec = 'l' * num_cols
        
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\toprule\n")
        
        # Write headers with line breaks for long headers
        formatted_headers = []
        for header in headers:
            # Break long headers
            if len(header) > 10:
                header = header.replace(' ', '\\\\ ')
            formatted_headers.append(header)
        
        header_row = " & ".join(formatted_headers) + " \\\\\n"
        f.write(header_row)
        f.write("\\midrule\n")
        
        # Write data rows
        for row in data:
            cleaned_row = []
            for cell in row:
                cell_str = str(cell)
                # Escape LaTeX special characters
                cell_str = cell_str.replace('&', '\\&')
                cell_str = cell_str.replace('%', '\\%')
                cell_str = cell_str.replace('$', '\\$')
                cell_str = cell_str.replace('#', '\\#')
                cell_str = cell_str.replace('_', '\\_')
                cell_str = cell_str.replace('{', '\\{')
                cell_str = cell_str.replace('}', '\\}')
                # Remove ANSI color codes and emoji
                import re
                cell_str = re.sub(r'\033\[[0-9;]*m', '', cell_str)
                cell_str = re.sub(r'[✅❌]', '', cell_str)  # Remove checkmarks
                cleaned_row.append(cell_str)
            
            data_row = " & ".join(cleaned_row) + " \\\\\n"
            f.write(data_row)
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        
        if caption:
            f.write(f"\\caption{{{caption}}}\n")
        if label:
            f.write(f"\\label{{{label}}}\n")
        
        f.write("\\end{table}\n")
        f.write("\\end{landscape}\n")
    
    print(f"✅ LaTeX landscape table saved to {filepath}")
    return filepath


# Initialize global GPT calibrator
gpt_calibrator = GPTConfidenceCalibrator()




def create_triple_comparison_summary(all_triple_results: List[Dict[str, MutableMapping[str, Any]]]) -> Dict[str, Any]:
    """Create a comprehensive summary of all triple comparison results across datasets."""
    if not all_triple_results:
        return {"error": "No triple comparison results to summarize"}
    
    # Aggregate statistics across all datasets
    total_columns = 0
    total_all_correct = 0
    total_two_correct = 0
    total_one_correct = 0
    total_none_correct = 0
    total_agreements = 0
    total_majority_correct = 0
    
    dataset_summaries = []
    
    for result in all_triple_results:
        if 'error' in result:
            continue
            
        stats = result['summary_stats']
        source_table = result.get('source_table', 'Unknown')
        target_table = result.get('target_table', 'Unknown')
        
        # Aggregate totals
        dataset_total = stats['total_columns']
        total_columns += dataset_total
        total_all_correct += len(result['patterns']['all_correct'])
        total_two_correct += len(result['patterns']['two_correct'])
        total_one_correct += len(result['patterns']['one_correct'])
        total_none_correct += len(result['patterns']['none_correct'])
        total_majority_correct += len(result['patterns']['majority_correct'])
        
        # Calculate agreement count from detailed results
        agreement_count = sum(1 for detail in result['detailed_results'] if detail['all_agree'])
        total_agreements += agreement_count
        
        # Store dataset summary
        dataset_summaries.append({
            'source_table': source_table,
            'target_table': target_table,
            'total_columns': dataset_total,
            'all_correct_rate': stats['all_correct_rate'],
            'agreement_rate': stats['agreement_rate'],
            'majority_accuracy': stats['majority_accuracy']
        })
    
    # Calculate overall statistics
    overall_stats = {
        'total_columns': total_columns,
        'total_datasets': len([r for r in all_triple_results if 'error' not in r]),
        'all_correct_rate': total_all_correct / total_columns if total_columns > 0 else 0,
        'two_correct_rate': total_two_correct / total_columns if total_columns > 0 else 0,
        'one_correct_rate': total_one_correct / total_columns if total_columns > 0 else 0,
        'none_correct_rate': total_none_correct / total_columns if total_columns > 0 else 0,
        'agreement_rate': total_agreements / total_columns if total_columns > 0 else 0,
        'majority_accuracy': total_majority_correct / total_columns if total_columns > 0 else 0
    }
    
    return {
        'overall_stats': overall_stats,
        'dataset_summaries': dataset_summaries
    }

def print_triple_comparison_summary(summary: Dict):
    """Print formatted summary of all triple comparison results."""
    
    if 'error' in summary:
        print(f"❌ {summary['error']}")
        return
    
    overall = summary['overall_stats']
    
    print(f"\n🎯 TRIPLE COMPARISON GLOBAL SUMMARY")
    print("=" * 80)
    print(f"📊 Overall Statistics Across {overall['total_datasets']} Datasets:")
    print(f"   Total Columns Analyzed: {overall['total_columns']}")
    print(f"   All 3 Matchers Correct: {overall['all_correct_rate']:.1%}")
    print(f"   2/3 Matchers Correct:   {overall['two_correct_rate']:.1%}")
    print(f"   1/3 Matchers Correct:   {overall['one_correct_rate']:.1%}")
    print(f"   0/3 Matchers Correct:   {overall['none_correct_rate']:.1%}")
    print(f"   Agreement Rate:         {overall['agreement_rate']:.1%}")
    print(f"   Majority Vote Accuracy: {overall['majority_accuracy']:.1%}")
    
    # Dataset-by-dataset breakdown
    print(f"\n📋 Dataset Breakdown:")
    dataset_table = []
    headers = ["Source", "Target", "Columns", "All Correct", "Agreement", "Majority Acc"]
    
    for ds in summary['dataset_summaries']:
        dataset_table.append([
            ds['source_table'],
            ds['target_table'], 
            ds['total_columns'],
            f"{ds['all_correct_rate']:.1%}",
            f"{ds['agreement_rate']:.1%}",
            f"{ds['majority_accuracy']:.1%}"
        ])
    
    print(tabulate(dataset_table, headers=headers, tablefmt="fancy_grid"))
    
    # Export results
    print(f"\n💾 Exporting detailed summary...")
    os.makedirs("output", exist_ok=True)
    
    # Export as JSON
    with open("output/triple_comparison_global_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Export dataset table as LaTeX
    export_table_as_latex(
        dataset_table,
        headers,
        "triple_comparison_summary.tex",
        caption="Triple comparison summary across all datasets",
        label="tab:triple_comparison_summary"
    )
    
    print("✅ Global triple comparison summary exported to:")
    print("   📁 output/triple_comparison_global_summary.json")
    print("   📄 output/triple_comparison_summary.tex")



async def main_test(args: argparse.Namespace):
    """
    Enhanced main_test that uses synthetic data as ground truth for real source data evaluation.
    """
    results = []
    detailed_matches = []
    all_pairwise_results = []  # Collect all pairwise comparison results
    all_triple_results = []  # Collect all triple comparison results

    # Add cross-dataset aggregation structure as per todo2.txt
    target_schema_results = {}  # Group results by target schema
    all_approaches = ["GPT", "Embedding", "Clustering", "Majority Vote", "Weighted Ensemble"]

    logging.info("🧪 Starting Real Data Evaluation with Synthetic Ground Truth")
    logging.info("=" * 60)
    
    # Load pre-trained GPT calibrator if it exists
    calibrator_path = "./models/gpt_isotonic_calibrator.pkl"
    if os.path.exists(calibrator_path):
        global gpt_calibrator
        gpt_calibrator = GPTConfidenceCalibrator.load(calibrator_path)
        logging.info("✅ Loaded pre-trained GPT isotonic calibrator")
    else:
        logging.warning("⚠️ No pre-trained GPT calibrator found, starting from scratch")

    # Iterate over all source CSV files
    for source_csv_path in sorted(glob.glob("./assets/source/*.csv")):
        logging.info(f"📁 Processing source: {source_csv_path}")
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

            logging.info(f"🎯 Matching {source_table} → {target_table}")

            # Generate synthetic ground truth for this target schema
            synthetic_source_schema, expected_mapping = await apply_perturbations(target_schema, seed=args.seed)
            logging.info(f"📋 Generated {len(expected_mapping)} synthetic mappings as ground truth")

            # Load expected mapping for this target schema first
            target_prefix = "_".join(target_table.split("_")[:2])
            real_gt_path = f"./assets/expected/{target_prefix}_mapping.json"
            logging.info(f"🔍 Loading expected mapping from: {real_gt_path}")

            # Initialize real_gt_mapping as None
            real_gt_mapping = None
            
            # Try to load real ground truth
            try:
                with open(real_gt_path) as f:
                    real_gt_data = json.load(f)
                
                # Extract the actual mapping from the loaded data
                if "mappings" in real_gt_data:
                    # Format: {"mappings": [{"source_column": "src", "target_column": "tgt"}]}
                    real_gt_mapping = {
                        mapping["target_column"]: mapping["source_column"]
                        for mapping in real_gt_data["mappings"]
                    }
                elif "matches" in real_gt_data:
                    # Format: {"matches": [{"source_column": "src", "target_column": "tgt"}]}
                    real_gt_mapping = {
                        mapping["target_column"]: mapping["source_column"]
                        for mapping in real_gt_data["matches"]
                    }
                elif isinstance(real_gt_data, dict) and all(isinstance(v, str) for v in real_gt_data.values()):
                    # Format: {"target_col": "source_col"}
                    real_gt_mapping = real_gt_data
                else:
                    print(f"⚠️ Unknown real ground truth format in {real_gt_path}")
                    real_gt_mapping = None
                    
                print(f"✅ Loaded real ground truth with {len(real_gt_mapping)} mappings")
            except FileNotFoundError:
                print(f"⚠️ Real ground truth file not found: {real_gt_path}")
                real_gt_mapping = None
            except Exception as e:
                print(f"⚠️ Error loading real ground truth: {e}")
                real_gt_mapping = None

            # Run all matchers with the real source schema and target schema
            raw_gpt_predictions = await gpt_column_mapping(source_schema, target_schema, seed=args.seed)
            
            # Collect training data if we have ground truth
            if real_gt_mapping:
                gpt_calibrator.collect_training_data(raw_gpt_predictions, real_gt_mapping)
            
            # Apply calibration if calibrator is fitted
            if gpt_calibrator.is_fitted:
                predicted_mapping = gpt_calibrator.calibrate_predictions(raw_gpt_predictions)
                print("🎯 Applied isotonic calibration to GPT-4 confidences")
            else:
                predicted_mapping = raw_gpt_predictions
                print("⚠️ Using raw GPT-4 confidences (calibrator not fitted)")

            embed_predicted = embedding_column_mapping(
                source_columns=list(source_schema.properties.keys()),
                target_columns=list(target_schema.properties.keys()),
                threshold=0
            )
            
            
            cluster_predicted = clustering_matcher(source_schema, target_schema)
            
            # NEW: Step 2 - Pairwise Comparisons as shown in your sketch
            print(f"\n🔄 Step 2: Pairwise Matcher Comparisons")
            
            pairwise_results, pairwise_result_entry = run_all_pairwise_comparisons(
                gpt_predictions=predicted_mapping,
                embedding_predictions=embed_predicted,
                clustering_predictions=cluster_predicted,
                ground_truth=real_gt_mapping,  # Use real ground truth if available
                source_table=source_table,
                target_table=target_table
            )
            
            # Display pairwise comparison results
            print_pairwise_comparison_table(
                pairwise_results, source_table, target_table
            )
            
            # Store pairwise result for later export
            all_pairwise_results.append(pairwise_result_entry)
            
            
            # NEW: Step 3 - Triple Comparison Analysis
            print(f"\n🔄 Step 3: Triple Matcher Comparison Against Ground Truth")
            if real_gt_mapping:
                triple_results = run_triple_comparison(
                    gpt_predictions=predicted_mapping,
                    embedding_predictions=embed_predicted,
                    clustering_predictions=cluster_predicted,
                    ground_truth=real_gt_mapping,
                    source_table=source_table,
                    target_table=target_table
                )
                
                # ADD: Store source and target info
                triple_results['source_table'] = source_table
                triple_results['target_table'] = target_table
                
                # ADD: Store for global summary
                all_triple_results.append(triple_results)
    
                
                print(f"📊 Triple Comparison Results for {source_table} → {target_table}")
                print_triple_comparison_table(triple_results, source_table, target_table)


            else:
                print("⚠️ No real ground truth available for triple comparison")
            
            
            # Show mapping with examples from real data
            show_mapping_with_examples(predicted_mapping, source_data)

            # Create ensemble matchers
            majority_ensemble, weighted_ensemble = create_ensemble_matchers(
                predicted_mapping,    # GPT predictions
                embed_predicted,      # Embedding predictions  
                cluster_predicted,    # Clustering predictions
            )

            # Get ensemble predictions
            majority_predicted = majority_ensemble.predict(source_schema, target_schema)
            weighted_predicted = weighted_ensemble.predict(source_schema, target_schema)

            # Build comprehensive comparison table
            comparison_table = []
            gt_type = "Real GT" if real_gt_mapping else "Synthetic GT"
            headers = [
                "Target", gt_type,
                "GPT Match", "GPT Score",
                "Embed Match", "Embed Score", 
                "Cluster Match", "Cluster Score",
                "Majority Match", "Majority Score",
                "Weighted Match", "Weighted Score"
            ]

            for col in target_schema.properties.keys():
                # Use real ground truth if available, otherwise use synthetic
                ground_truth_for_col = real_gt_mapping.get(col, "—") if real_gt_mapping else expected_mapping.get(col, "—")
                
                gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
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
                
                comparison_table.append([
                    col, ground_truth_for_col,
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
                            "ground_truth": ground_truth_for_col
                        })

            print(f"\n📊 Complete Matcher Comparison for {source_table} → {target_table}")
            export_table_as_latex_landscape(
                comparison_table, 
                headers, 
                f"RealData_{source_table}_to_{target_table}.tex",
                caption=f"Matcher comparison for {source_table} to {target_table}",
                label=f"tab:{source_table}_{target_table}"
            )
            
            # Calculate accuracy against real or synthetic ground truth for Overall row
            approaches = [
                ("GPT", predicted_mapping),
                ("Embedding", embed_predicted),
                ("Clustering", cluster_predicted),
                ("Majority Vote", majority_predicted),
                ("Weighted Ensemble", weighted_predicted)
            ]
            
            # Use real ground truth if available, otherwise synthetic
            evaluation_mapping = real_gt_mapping if real_gt_mapping else expected_mapping
            
            # Build Overall summary row as per todo.txt algorithm
            overall_row = ["Overall", " "]  # Target column = "Overall", GT column = " "
            
            for name, predictions in approaches:
                correct_count = sum(1 for col, expected_src in evaluation_mapping.items()
                                  if col in predictions and predictions[col][0] == expected_src)
                total_count = len(evaluation_mapping)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                # Calculate confidence-weighted score as per todo.txt algorithm
                confidence_weighted_score = 0.0
                for col, expected_src in evaluation_mapping.items():
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
                correct_count = sum(1 for col, expected_src in evaluation_mapping.items()
                                  if col in predictions and predictions[col][0] == expected_src)
                total_count = len(evaluation_mapping)
                accuracy = correct_count / total_count if total_count > 0 else 0.0
                
                confidence_weighted_score = 0.0
                for col, expected_src in evaluation_mapping.items():
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
            score = score_mapping(predicted_mapping, evaluation_mapping)
            score_val = score[0] if score and isinstance(score, tuple) else None
            score_display = f"{score_val:.2f}" if score_val is not None else "—"
            weight_display = str(len(target_schema.properties))

            results.append([
                source_table,
                target_table,
                score_display,
                weight_display
            ])

    print(f"\n🔄 STEP 2 COMPLETE: Creating Global Triple Comparison Summary")
    print("=" * 80)
    create_triple_comparison_summary(all_triple_results)

    # NEW: Create and display pairwise comparison summary
    print(f"\n🔄 STEP 2 COMPLETE: Creating Global Pairwise Comparison Summary")
    print("=" * 80)

    if all_pairwise_results:
        # Use the existing export function to create the summary
        from pairwise_comparison import export_pairwise_results
        
        summary_data, headers = export_pairwise_results(
            all_pairwise_results, 
            "global_pairwise_comparison_results"
        )
        
        print(f"✅ Pairwise comparison summary exported for {len(all_pairwise_results)} dataset pairs")
    else:
        print("⚠️ No pairwise comparison results to summarize")
        
        
   
    
   # The above code snippet is training a GPT Isotonic Calibrator using the `fit()` method. After
   # training the calibrator, it saves the trained model to a specified path, generates a calibration
   # curve plot, generates LaTeX code for the calibration plot, saves the LaTeX code to a file, and
   # prints calibration statistics including Raw ECE (Expected Calibration Error), Calibrated ECE,
   # Improvement, and the number of training samples used.
    # After processing all data, train the calibrator
    print("\n🎯 Training GPT Isotonic Calibrator...")
    if gpt_calibrator.fit():
        # Save the trained calibrator
        os.makedirs("./models", exist_ok=True)
        gpt_calibrator.save(calibrator_path)
        
        # Generate calibration plots
        os.makedirs("./output", exist_ok=True)
        # Generate calibration plot
        fig = gpt_calibrator.plot_calibration_curve("./output/gpt_calibration_curve.png")
        if fig:
            plt.close(fig)  # Close to free memory
        
        # Generate LaTeX calibration plot code
        latex_code = gpt_calibrator.generate_latex_calibration_plot()
        
        # Print calibration statistics
        stats = gpt_calibrator.get_calibration_stats()
        if stats:
            print(f"📊 GPT Calibration Results:")
            print(f"   Raw ECE: {stats['raw_ece']:.4f}")
            print(f"   Calibrated ECE: {stats['calibrated_ece']:.4f}")
            print(f"   Improvement: {stats['improvement']:.4f}")
            print(f"   Training samples: {stats['n_samples']}")
            
    # NEW: Export complete pairwise comparison results
    print(f"\n🔄 STEP 2 COMPLETE: Exporting Pairwise Comparison Results")
    print("=" * 70)
    if all_pairwise_results:
        export_pairwise_results(all_pairwise_results, "complete_pairwise_analysis")
    else:
        print("⚠️ No pairwise results to export")
    

    # Final summary table
    print("\n📊 Real Data Harmonization Summary")
    summary_headers = ["Source Table", "Target Table", "Score", "Weight"]
    print(tabulate(results, headers=summary_headers, tablefmt="fancy_grid"))
    export_table_as_image(results, summary_headers, "RealData_Harmonization_Summary.png")
    
    export_table_as_latex(
        results, 
        summary_headers, 
        "RealData_Harmonization_Summary.tex",
        caption="Real data harmonization summary across all source-target combinations",
        label="tab:harmonization_summary"
    )

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
        export_table_as_latex(
            aggregation_table, 
            aggregation_headers, 
            "CrossDataset_Aggregation_Summary.tex",
            caption="Cross-dataset aggregation summary",
            label="tab:cross_dataset_aggregation"
        )
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
    
    
    print(f"\n🔍 GPT MATCHER DEBUG############################:")
    
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

 
    
    

    cluster_predicted = clustering_matcher(source_schema, target_schema)
    # ADD THIS DEBUG BLOCK:
    # After cluster_predicted = clustering_matcher(...)
    print(f"\n🔍 CLUSTERING MATCHER DETAILED DEBUG:")
    print(f"   Source schema type: {type(source_schema)}")
    print(f"   Target schema type: {type(target_schema)}")
    print(f"   Function called successfully: {cluster_predicted is not None}")
    print(f"   Return type: {type(cluster_predicted)}")
    print(f"   Clustering predictions count: {len(cluster_predicted) if cluster_predicted else 'None/Empty'}")
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
