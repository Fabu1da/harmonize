import argparse
from typing import Any, Dict, List

from core.create_ensembles import create_ensembles
from core.export_to_latex import export_table_as_latex_landscape
from core.run.comparison_table import build_comparison_table
from core.run.detailed_matches import collect_detailed_matches
from core.run.matchers import run_all_matchers
from core.run.pairwise import run_pairwise_analysis
from core.run.schema_results import update_target_schema_results
from core.run.summary_row import add_overall_summary_row
from core.run.triple import run_triple_analysis
from core.table_as_image import export_table_as_image
from core.utils.data import load_real_ground_truth, load_source_data, load_target_schema
from core.utils.synthetic import generate_synthetic_ground_truth
from synthetic_data import score_mapping
from tabulate import tabulate

import logging


async def process_single_source_target_pair(source_csv_path: str, target_path: str, 
                                           args: argparse.Namespace, gpt_calibrator: Any,
                                           all_pairwise_results: List, all_triple_results: List,
                                           detailed_matches: List, results: List,
                                           target_schema_results: Dict, all_approaches: List):
    """Process a single source-target schema pair"""
    # Load data and schemas
    source_table, source_data, source_schema = await load_source_data(source_csv_path)
    target_table, target_schema = load_target_schema(target_path)
    
    logging.info(f"🎯 Matching {source_table} → {target_table}")
    
    # Generate synthetic ground truth
    synthetic_source_schema, expected_mapping = await generate_synthetic_ground_truth(target_schema, args.seed)
    
    # Load real ground truth
    real_gt_mapping = load_real_ground_truth(target_table)
    
    # Run all matchers
    predicted_mapping, embed_predicted, cluster_predicted = await run_all_matchers(
        source_schema, target_schema, args.seed, real_gt_mapping, gpt_calibrator
    )
    
    # Run pairwise analysis
    pairwise_results, pairwise_result_entry = run_pairwise_analysis(
        predicted_mapping, embed_predicted, cluster_predicted, source_table, target_table,
        real_gt_mapping  # Pass the real ground truth
    )
    all_pairwise_results.append(pairwise_result_entry)
    
    # Run triple analysis
    triple_results = run_triple_analysis(
        predicted_mapping, embed_predicted, cluster_predicted, 
        real_gt_mapping, source_table, target_table
    )
    if triple_results:
        all_triple_results.append(triple_results)
    
    # Create ensembles
    majority_ensemble, weighted_ensemble, majority_predicted, weighted_predicted = create_ensembles(
        predicted_mapping, embed_predicted, cluster_predicted, source_schema, target_schema
    )
    
    # Choose evaluation mapping (real or synthetic)
    evaluation_mapping = real_gt_mapping if real_gt_mapping else expected_mapping
    
    # Build comparison table
    comparison_table, headers = build_comparison_table(
        target_schema, real_gt_mapping, expected_mapping, predicted_mapping,
        embed_predicted, cluster_predicted, majority_predicted, weighted_predicted
    )
    
    # Add overall summary row
    comparison_table = add_overall_summary_row(
        comparison_table, headers, evaluation_mapping, predicted_mapping,
        embed_predicted, cluster_predicted, majority_predicted, weighted_predicted
    )
    
    # Collect detailed matches
    collect_detailed_matches(
        detailed_matches, source_table, target_table, target_schema,
        predicted_mapping, embed_predicted, cluster_predicted,
        majority_predicted, weighted_predicted, evaluation_mapping
    )
    
    # Update target schema results for cross-dataset aggregation
    update_target_schema_results(
        target_schema_results, target_table, all_approaches, evaluation_mapping,
        predicted_mapping, embed_predicted, cluster_predicted,
        majority_predicted, weighted_predicted
    )
    
    # Display and export results
    print(f"\n📊 Complete Matcher Comparison for {source_table} → {target_table}")
    export_table_as_latex_landscape(
        comparison_table, headers, f"RealData_{source_table}_to_{target_table}.tex",
        caption=f"Matcher comparison for {source_table} to {target_table}",
        label=f"tab:{source_table}_{target_table}"
    )
    
    print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))
    export_table_as_image(comparison_table, headers, f"RealData_{source_table}_to_{target_table}.png")
    
    # Calculate overall score for results summary
    score = score_mapping(predicted_mapping, evaluation_mapping)
    score_val = score[0] if score and isinstance(score, tuple) else None
    score_display = f"{score_val:.2f}" if score_val is not None else "—"
    weight_display = str(len(target_schema.properties))
    
    results.append([source_table, target_table, score_display, weight_display])
