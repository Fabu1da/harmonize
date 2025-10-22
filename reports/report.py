import json
import os
from analysis.error_analysis import run_error_analysis
from cross_dataset_summary import run_cross_dataset_analysis
from analysis.error_analysis import run_error_analysis
from mcnemar import mcnemar_analysis
from perfomance import performance_report
import logging

from mcnemar import mcnemar_analysis

logging.basicConfig(level=logging.INFO)

def generate_reports(detailed_matches: list, all_ground_truth_data: list, dataset_names: list, expected_dir: str):
    """
    Generate all comprehensive reports from collected data.
    
    Args:
        detailed_matches: List of detailed match results from all evaluations
        all_ground_truth_data: List of ground truth data for each pair
        dataset_names: Names of datasets processed
        expected_dir: Directory containing expected ground truth files
    """
    if not detailed_matches:
        print("⚠️ No matches to generate reports for")
        return
    
    # Save detailed matches
    with open("./output/real_data_detailed_matches.json", "w") as f:
        json.dump(detailed_matches, f, indent=2)
    
    print("\n🎯 GENERATING COMPREHENSIVE PERFORMANCE REPORT...")
    performance_report(detailed_matches, ground_truth=None)
    
    print("\n📊 RUNNING MCNEMAR ANALYSIS...")
    mcnemar_analysis(detailed_matches, expected_dir=expected_dir)
    
    print("\n🔍 RUNNING COMPREHENSIVE ERROR ANALYSIS...")
    print(f"📊 Processing {len(all_ground_truth_data)} ground truth files")
    run_error_analysis(detailed_matches, ground_truth=all_ground_truth_data)
    
    print("\n📋 RUNNING CROSS-DATASET ANALYSIS...")
    report_path = run_cross_dataset_analysis(
        detailed_matches_list=detailed_matches,
        ground_truth_list=all_ground_truth_data,
        dataset_names=dataset_names
    )
    
    print(f"\n✅ All reports generated successfully!")
    print(f"📄 Cross-dataset report: {report_path}")



