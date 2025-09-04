import json
import logging
import os
from typing import Dict, List

from core.export_to_latex import export_table_as_latex
from core.table_as_image import export_table_as_image
from tabulate import tabulate

def generate_final_summary_tables(results: List, detailed_matches: List, 
                                target_schema_results: Dict, all_approaches: List):
    """Generate final summary tables and exports"""
    # Final summary table
    print("\n📊 Real Data Harmonization Summary")
    summary_headers = ["Source Table", "Target Table", "Score", "Weight"] #weight number of fields, F1 = 2 × (Precision × Recall) / (Precision + Recall)
    print(tabulate(results, headers=summary_headers, tablefmt="fancy_grid"))
    export_table_as_image(results, summary_headers, "RealData_Harmonization_Summary.png")

    export_table_as_latex(
        results, summary_headers, "RealData_Harmonization_Summary.tex",
        caption="Real data harmonization summary across all source-target combinations",
        label="tab:harmonization_summary"
    )

    # Export detailed matches as JSON
    with open("output/real_data_detailed_matches.json", "w") as f:
        json.dump(detailed_matches, f, indent=2)
    print("✅ Real data detailed matcher results saved to output/real_data_detailed_matches.json")

    # Create cross-dataset aggregation summary table
    generate_cross_dataset_aggregation_table(target_schema_results, all_approaches)




def generate_cross_dataset_aggregation_table(target_schema_results: Dict, all_approaches: List):
    """Generate cross-dataset aggregation summary table"""
    print("\n" + "="*80)
    print("📋 CROSS-DATASET AGGREGATION SUMMARY (todo2.txt)")
    print("="*80)
    
    if not target_schema_results:
        print("⚠️ No target schema results to aggregate")
        return
        
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
    
    # Export aggregation table
    export_table_as_image(aggregation_table, aggregation_headers, "CrossDataset_Aggregation_Summary.png")
    export_table_as_latex(
        aggregation_table, aggregation_headers, "CrossDataset_Aggregation_Summary.tex",
        caption="Cross-dataset aggregation summary", label="tab:cross_dataset_aggregation"
    )
    print("✅ Cross-dataset aggregation table saved")