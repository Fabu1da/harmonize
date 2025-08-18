#!/usr/bin/env python3
"""
COMA Baseline Dataset Type Breakdown Analysis
Creates the specific table format requested for thesis:

Table~\ref{tab:dataset_breakdown_coma}: COMA baseline

Dataset Type	Coverage	Best F1	Precision	Recall	MRR
Joinable	...	...	...	...	...
Sem-joinable	...	...	...	...	...
Unionable	...	...	...	...	...
View-union	...	...	...	...	...
"""

import json
import os
import sys
from typing import Dict, List, Any, Set, Tuple
import numpy as np
from collections import defaultdict

# Add analysis directory to path
sys.path.append('analysis')
from normalization import build_gt_set, normalize_coma, load_ground_truth
from evaluation import compute_candidate_coverage, eval_set_based, eval_ranking_based


def load_coma_results() -> List[Dict]:
    """Load COMA evaluation results from the bridge output"""
    results_path = "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/hamonize/assets/output/matches.json"
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"COMA results not found at {results_path}. Please run COMA evaluation first.")
    
    with open(results_path, 'r') as f:
        content = f.read()
        
    # Fix European decimal format - replace comma decimal separators with dots
    # Need to be careful to only replace commas in similarity values, not JSON structure
    import re
    # Replace pattern like "similarity":0,1234 with "similarity":0.1234
    content = re.sub(r'("similarity":)(\d+),(\d+)', r'\1\2.\3', content)
    
    try:
        coma_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print("First 500 chars of content:")
        print(content[:500])
        raise
    
    print(f"✓ Loaded {len(coma_data)} COMA match results")
    return coma_data


def categorize_by_dataset_type(coma_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Categorize COMA results by dataset type based on file names
    
    Args:
        coma_data: List of COMA match results
    
    Returns:
        Dictionary with dataset type as key and list of matches as value
    """
    
    # Dataset type mapping based on file naming patterns
    # Order matters! Check more specific patterns first
    type_mapping = [
        ('semjoinable', 'sem-joinable'),  # Check semjoinable before joinable
        ('joinable', 'joinable'),
        ('viewunion', 'view-union'),
        ('unionable', 'unionable')
    ]
    
    categorized = {
        'joinable': [],
        'sem-joinable': [],
        'unionable': [],
        'view-union': []
    }
    
    for match in coma_data:
        src_file = str(match.get('src_file', '')).lower()
        trg_file = str(match.get('trg_file', '')).lower()
        
        # Determine dataset type from file names
        dataset_type = None
        for pattern, type_name in type_mapping:
            if pattern in src_file or pattern in trg_file:
                dataset_type = type_name
                break
        
        if dataset_type:
            categorized[dataset_type].append(match)
        else:
            print(f"⚠️ Could not categorize: {src_file} -> {trg_file}")
    
    # Print categorization summary
    print(f"\n📊 Dataset Type Categorization:")
    for dtype, matches in categorized.items():
        print(f"  {dtype.title()}: {len(matches)} matches")
    
    return categorized


def evaluate_coma_by_dataset_type(categorized_data: Dict[str, List[Dict]], 
                                  gt_data: Any) -> Dict[str, Dict]:
    """
    Evaluate COMA performance for each dataset type
    
    Args:
        categorized_data: COMA results categorized by dataset type
        gt_data: Ground truth data
    
    Returns:
        Dictionary with evaluation metrics for each dataset type
    """
    
    # Build ground truth set
    G = build_gt_set(gt_data)
    print(f"✓ Loaded ground truth with {len(G)} pairs")
    
    results = {}
    
    for dataset_type, matches in categorized_data.items():
        if not matches:
            print(f"⚠️ No matches found for {dataset_type}")
            continue
            
        print(f"\n🔍 Evaluating {dataset_type} ({len(matches)} matches)...")
        
        try:
            # Convert COMA matches to normalized format
            # First, ensure similarity values are strings for the normalize_coma function
            matches_for_normalization = []
            for match in matches:
                match_copy = match.copy()
                # Convert float similarity back to string for normalize_coma function
                match_copy['similarity'] = str(match['similarity'])
                matches_for_normalization.append(match_copy)
            
            coma_run = normalize_coma(matches_for_normalization, run_id=f"COMA-{dataset_type}")
            print(f"  ✓ Normalized COMA run: {len(coma_run.pairs)} pairs")
            
            # Filter ground truth for this dataset type
            gt_filtered = filter_ground_truth_by_type(G, dataset_type)
            print(f"  Ground truth pairs for {dataset_type}: {len(gt_filtered)}")
            
            if not gt_filtered:
                print(f"  ⚠️ No ground truth for {dataset_type}, skipping...")
                continue
            
            # Calculate metrics
            coverage = compute_candidate_coverage(coma_run, gt_filtered)
            print(f"  ✓ Coverage calculated: {coverage:.3f}")
            
            # Set-based evaluation (Precision, Recall, F1)
            thresholds = np.linspace(0.0, 1.0, 101)
            pr_curve, f1_curve, best_metrics = eval_set_based(coma_run, gt_filtered, thresholds)
            print(f"  ✓ Set-based evaluation completed")
            
            # Ranking-based evaluation (MRR)
            ks = [1, 3, 5]  # Standard k values for ranking evaluation
            ranking_metrics = eval_ranking_based(coma_run, gt_filtered, ks)
            print(f"  ✓ Ranking-based evaluation completed")
            
            results[dataset_type] = {
                'coverage': coverage,
                'best_f1': best_metrics['best_F1'],
                'precision': best_metrics['precision_at_best'],
                'recall': best_metrics['recall_at_best'],
                'mrr': ranking_metrics['MRR'],
                'hits_at_1': ranking_metrics['Hits@1'],
                'num_matches': len(matches),
                'num_gt_pairs': len(gt_filtered)
            }
            
            print(f"  ✓ Coverage: {coverage:.3f}")
            print(f"  ✓ Best F1: {best_metrics['best_F1']:.3f}")
            print(f"  ✓ Precision: {best_metrics['precision_at_best']:.3f}")
            print(f"  ✓ Recall: {best_metrics['recall_at_best']:.3f}")
            print(f"  ✓ MRR: {ranking_metrics['MRR']:.3f}")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {dataset_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def filter_ground_truth_by_type(gt_set: Set, dataset_type: str) -> Set:
    """
    Filter ground truth pairs to only include those matching the dataset type
    
    Args:
        gt_set: Set of ground truth pairs
        dataset_type: Dataset type to filter for
    
    Returns:
        Filtered ground truth set
    """
    
    # Mapping from dataset type to file pattern
    type_patterns = {
        'joinable': 'joinable',
        'sem-joinable': 'semjoinable',
        'unionable': 'unionable', 
        'view-union': 'viewunion'
    }
    
    pattern = type_patterns.get(dataset_type, '')
    if not pattern:
        return set()
    
    filtered_gt = set()
    for pair in gt_set:
        # Check if the source or target table contains the pattern
        source_table = str(getattr(pair, 'source_table', '')).lower()
        target_table = str(getattr(pair, 'target_table', '')).lower()
        
        if (pattern in source_table or pattern in target_table):
            filtered_gt.add(pair)
    
    return filtered_gt


def generate_latex_table(results: Dict[str, Dict]) -> str:
    """
    Generate LaTeX table for the COMA baseline results
    
    Args:
        results: Evaluation results by dataset type
    
    Returns:
        LaTeX table content
    """
    
    # Order of dataset types for the table
    ordered_types = ['joinable', 'sem-joinable', 'unionable', 'view-union']
    
    # Generate table rows
    latex_rows = []
    for dtype in ordered_types:
        if dtype in results:
            r = results[dtype]
            row = f"{dtype.replace('-', '-').title()} & {r['coverage']:.3f} & {r['best_f1']:.3f} & {r['precision']:.3f} & {r['recall']:.3f} & {r['mrr']:.3f} \\\\"
            latex_rows.append(row)
        else:
            # Add empty row if no data
            row = f"{dtype.replace('-', '-').title()} & --- & --- & --- & --- & --- \\\\"
            latex_rows.append(row)
    
    latex_content = f"""% COMA Baseline Dataset Type Breakdown
\\begin{{table}}[h]
\\centering
\\caption{{COMA baseline performance breakdown by dataset type}}
\\label{{tab:dataset_breakdown_coma}}
\\begin{{tabular}}{{|l|c|c|c|c|c|}}
\\hline
Dataset Type & Coverage & Best F1 & Precision & Recall & MRR \\\\
\\hline
{chr(10).join(latex_rows)}
\\hline
\\end{{tabular}}
\\end{{table}}"""
    
    return latex_content


def print_summary_table(results: Dict[str, Dict]) -> None:
    """Print a nicely formatted summary table"""
    
    print(f"\n📋 COMA BASELINE - DATASET TYPE BREAKDOWN")
    print("=" * 80)
    
    print(f"{'Dataset Type':<15} {'Coverage':<10} {'Best F1':<10} {'Precision':<10} {'Recall':<10} {'MRR':<10}")
    print("-" * 80)
    
    ordered_types = ['joinable', 'sem-joinable', 'unionable', 'view-union']
    
    for dtype in ordered_types:
        if dtype in results:
            r = results[dtype]
            print(f"{dtype.title():<15} {r['coverage']:<10.3f} {r['best_f1']:<10.3f} {r['precision']:<10.3f} {r['recall']:<10.3f} {r['mrr']:<10.3f}")
        else:
            print(f"{dtype.title():<15} {'---':<10} {'---':<10} {'---':<10} {'---':<10} {'---':<10}")


def main():
    """Run complete COMA dataset breakdown analysis"""
    
    print("🔬 COMA BASELINE DATASET TYPE BREAKDOWN ANALYSIS")
    print("=" * 80)
    
    try:
        # Step 1: Load COMA results
        print("\n📂 Step 1: Loading COMA results...")
        coma_data = load_coma_results()
        
        # Step 2: Load ground truth
        print("\n📂 Step 2: Loading ground truth...")
        gt_data = load_ground_truth()
        
        # Step 3: Categorize by dataset type
        print("\n📊 Step 3: Categorizing by dataset type...")
        categorized_data = categorize_by_dataset_type(coma_data)
        
        # Step 4: Evaluate each dataset type
        print("\n🎯 Step 4: Evaluating performance by dataset type...")
        results = evaluate_coma_by_dataset_type(categorized_data, gt_data)
        
        # Step 5: Generate outputs
        print("\n📄 Step 5: Generating outputs...")
        
        # Print summary table
        print_summary_table(results)
        
        # Generate LaTeX table
        latex_content = generate_latex_table(results)
        
        # Save LaTeX table
        os.makedirs("output", exist_ok=True)
        latex_path = "output/dataset_breakdown_coma.tex"
        with open(latex_path, "w") as f:
            f.write(latex_content)
        
        print(f"\n✅ LaTeX table saved to: {latex_path}")
        print(f"   Use in thesis: \\input{{dataset_breakdown_coma}}")
        print(f"   Reference as: Table~\\ref{{tab:dataset_breakdown_coma}}")
        
        # Save detailed results as JSON
        results_path = "output/coma_dataset_breakdown_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Detailed results saved to: {results_path}")
        
        print(f"\n🎉 COMA baseline analysis complete!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
