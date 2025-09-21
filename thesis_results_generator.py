#!/usr/bin/env python3
"""
Thesis Results Generator
Generates academic-quality results analysis for master's thesis from FP/FN analysis.
"""

import os
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple
from fp_fn_analysis import (
    load_pairwise_data, 
    load_detailed_matches, 
    analyze_fp_fn_from_detailed_matches,
    generate_specific_fp_fn_examples,
    analyze_fp_fn_patterns
)

def generate_thesis_performance_summary(output_dir: str) -> Dict:
    """Generate comprehensive performance summary for thesis results."""
    
    # Load pairwise comparison data
    pairwise_data = load_pairwise_data(output_dir)
    if not pairwise_data:
        return {}
    
    # Aggregate results by method comparison
    method_performance = {}
    dataset_performance = {}
    
    for dataset_pair in pairwise_data:
        source = dataset_pair['source_table']
        target = dataset_pair['target_table']
        comparisons = dataset_pair['comparisons']
        
        dataset_key = f"{source} → {target}"
        dataset_performance[dataset_key] = {}
        
        for comparison_type, metrics_str in comparisons.items():
            # Parse metrics
            import re
            precision_match = re.search(r'precision=([0-9.]+)', metrics_str)
            recall_match = re.search(r'recall=([0-9.]+)', metrics_str)
            f1_match = re.search(r'f1_score=([0-9.]+)', metrics_str)
            
            metrics = {
                'precision': float(precision_match.group(1)) if precision_match else 0.0,
                'recall': float(recall_match.group(1)) if recall_match else 0.0,
                'f1': float(f1_match.group(1)) if f1_match else 0.0
            }
            
            # Store by method
            if comparison_type not in method_performance:
                method_performance[comparison_type] = []
            method_performance[comparison_type].append(metrics)
            
            # Store by dataset
            dataset_performance[dataset_key][comparison_type] = metrics
    
    # Calculate aggregate statistics
    method_stats = {}
    for method, results in method_performance.items():
        if results:
            method_stats[method] = {
                'mean_precision': np.mean([r['precision'] for r in results]),
                'std_precision': np.std([r['precision'] for r in results]),
                'mean_recall': np.mean([r['recall'] for r in results]),
                'std_recall': np.std([r['recall'] for r in results]),
                'mean_f1': np.mean([r['f1'] for r in results]),
                'std_f1': np.std([r['f1'] for r in results]),
                'count': len(results)
            }
    
    return {
        'method_performance': method_stats,
        'dataset_performance': dataset_performance,
        'raw_data': method_performance
    }

def create_thesis_visualizations(output_dir: str, performance_summary: Dict):
    """Create publication-quality visualizations for thesis."""
    
    # Set academic plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    # Figure 1: Method Comparison - Box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = list(performance_summary['raw_data'].keys())
    method_names = [m.replace('_vs_', ' vs ') for m in methods]
    
    # Precision comparison
    precision_data = [performance_summary['raw_data'][m] for m in methods]
    precision_values = [[r['precision'] for r in data] for data in precision_data]
    
    bp1 = axes[0].boxplot(precision_values, labels=method_names, patch_artist=True)
    axes[0].set_title('Precision Comparison Across Methods')
    axes[0].set_ylabel('Precision')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Recall comparison
    recall_values = [[r['recall'] for r in data] for data in precision_data]
    bp2 = axes[1].boxplot(recall_values, labels=method_names, patch_artist=True)
    axes[1].set_title('Recall Comparison Across Methods')
    axes[1].set_ylabel('Recall')
    axes[1].tick_params(axis='x', rotation=45)
    
    # F1 comparison
    f1_values = [[r['f1'] for r in data] for data in precision_data]
    bp3 = axes[2].boxplot(f1_values, labels=method_names, patch_artist=True)
    axes[2].set_title('F1-Score Comparison Across Methods')
    axes[2].set_ylabel('F1-Score')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = sns.color_palette("husl", len(methods))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'thesis_method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Error Analysis - FP/FN Distribution
    fp_fn_examples = generate_specific_fp_fn_examples(output_dir)
    if fp_fn_examples:
        patterns = analyze_fp_fn_patterns(fp_fn_examples)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # FP similarity distribution
        fp_ranges = patterns['false_positive_patterns']['similarity_ranges']
        ranges = list(fp_ranges.keys())
        fp_counts = list(fp_ranges.values())
        
        axes[0,0].bar(ranges, fp_counts, color='lightcoral', alpha=0.7)
        axes[0,0].set_title('False Positive Distribution by Similarity Range')
        axes[0,0].set_xlabel('Similarity Range')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # FN similarity distribution
        fn_ranges = patterns['false_negative_patterns']['similarity_ranges']
        fn_ranges_keys = list(fn_ranges.keys())
        fn_counts = list(fn_ranges.values())
        
        axes[0,1].bar(fn_ranges_keys, fn_counts, color='lightblue', alpha=0.7)
        axes[0,1].set_title('False Negative Distribution by Similarity Range')
        axes[0,1].set_xlabel('Similarity Range')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Matcher performance comparison
        matcher_perf = patterns['matcher_performance']
        matchers = list(matcher_perf.keys())
        fp_counts_by_matcher = [matcher_perf[m]['fp_count'] for m in matchers]
        fn_counts_by_matcher = [matcher_perf[m]['fn_count'] for m in matchers]
        
        x = np.arange(len(matchers))
        width = 0.35
        
        axes[1,0].bar(x - width/2, fp_counts_by_matcher, width, label='False Positives', color='lightcoral', alpha=0.7)
        axes[1,0].bar(x + width/2, fn_counts_by_matcher, width, label='False Negatives', color='lightblue', alpha=0.7)
        axes[1,0].set_title('Error Distribution by Matcher Type')
        axes[1,0].set_xlabel('Matcher')
        axes[1,0].set_ylabel('Error Count')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(matchers)
        axes[1,0].legend()
        
        # Error rate by matcher
        total_errors = [(fp_counts_by_matcher[i] + fn_counts_by_matcher[i]) for i in range(len(matchers))]
        error_rates = [fp_counts_by_matcher[i] / max(1, total_errors[i]) for i in range(len(matchers))]
        
        axes[1,1].bar(matchers, error_rates, color='orange', alpha=0.7)
        axes[1,1].set_title('False Positive Rate by Matcher')
        axes[1,1].set_xlabel('Matcher')
        axes[1,1].set_ylabel('FP Rate (FP / Total Errors)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'thesis_error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()

def generate_thesis_results_table(output_dir: str, performance_summary: Dict) -> str:
    """Generate LaTeX table for thesis results section."""
    
    method_stats = performance_summary['method_performance']
    
    latex_table = """
\\begin{table}[htbp]
\\centering
\\caption{Performance Comparison of Schema Matching Methods}
\\label{tab:method_performance}
\\begin{tabular}{|l|c|c|c|c|}
\\hline
\\textbf{Method} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} & \\textbf{N} \\\\
\\hline
"""
    
    for method, stats in method_stats.items():
        method_name = method.replace('_vs_', ' vs ').replace('_', ' ').title()
        precision = f"{stats['mean_precision']:.3f} ± {stats['std_precision']:.3f}"
        recall = f"{stats['mean_recall']:.3f} ± {stats['std_recall']:.3f}"
        f1 = f"{stats['mean_f1']:.3f} ± {stats['std_f1']:.3f}"
        count = stats['count']
        
        latex_table += f"{method_name} & {precision} & {recall} & {f1} & {count} \\\\\n"
    
    latex_table += """\\hline
\\end{tabular}
\\end{table}
"""
    
    return latex_table

def generate_thesis_insights_text(output_dir: str) -> str:
    """Generate academic text summarizing key insights for thesis."""
    
    fp_fn_examples = generate_specific_fp_fn_examples(output_dir)
    if not fp_fn_examples:
        return "No detailed analysis data available."
    
    patterns = analyze_fp_fn_patterns(fp_fn_examples)
    
    total_fp = fp_fn_examples['summary']['total_fp_examples']
    total_fn = fp_fn_examples['summary']['total_fn_examples']
    
    # Key insights
    high_sim_fps = len(patterns['false_positive_patterns']['high_similarity_fp'])
    low_sim_fns = len(patterns['false_negative_patterns']['low_similarity_fn'])
    threshold_misses = len(patterns['false_negative_patterns']['threshold_misses'])
    
    # Matcher analysis
    matcher_perf = patterns['matcher_performance']
    best_precision_matcher = min(matcher_perf.keys(), key=lambda x: matcher_perf[x]['fp_count'])
    best_recall_matcher = min(matcher_perf.keys(), key=lambda x: matcher_perf[x]['fn_count'])
    
    insights_text = f"""
\\section{{Error Analysis and Performance Insights}}

\\subsection{{False Positive and False Negative Analysis}}

Our comprehensive error analysis revealed {total_fp} false positive cases and {total_fn} false negative cases across all method comparisons. This analysis provides crucial insights into the strengths and limitations of different schema matching approaches.

\\textbf{{False Positive Patterns:}} 
The analysis identified {high_sim_fps} high-confidence false positives (similarity ≥ 0.7), indicating cases where the models were highly confident but incorrect. These cases suggest potential issues with semantic understanding, where syntactically similar terms may have different semantic meanings in their respective contexts.

\\textbf{{False Negative Patterns:}}
We observed {low_sim_fns} cases of very low similarity false negatives (similarity < 0.3), representing instances where obvious matches were completely missed by the models. Additionally, {threshold_misses} near-threshold false negatives (0.4-0.5 similarity) suggest that threshold optimization could significantly improve recall performance.

\\subsection{{Method-Specific Performance Characteristics}}

\\textbf{{Precision Leaders:}} The {best_precision_matcher} approach demonstrated the lowest false positive rate, making it suitable for applications where match precision is critical.

\\textbf{{Recall Leaders:}} The {best_recall_matcher} approach showed the lowest false negative rate, indicating better coverage of true matches.

\\subsection{{Implications for Schema Matching Practice}}

These findings suggest several important considerations for practical schema matching applications:

\\begin{{enumerate}}
    \\item \\textbf{{Threshold Sensitivity:}} The high number of near-threshold false negatives indicates that performance could be significantly improved through careful threshold tuning.
    \\item \\textbf{{Method Selection:}} Different methods excel in different scenarios, suggesting that ensemble or hybrid approaches may yield optimal results.
    \\item \\textbf{{Semantic Limitations:}} High-confidence false positives highlight the need for improved semantic understanding in automated matching systems.
\\end{{enumerate}}
"""
    
    return insights_text

def main():
    """Generate comprehensive thesis results analysis."""
    print("=== Generating Thesis Results Analysis ===\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    # Generate performance summary
    print("Generating performance summary...")
    performance_summary = generate_thesis_performance_summary(output_dir)
    
    if not performance_summary:
        print("No performance data available. Please run pairwise analysis first.")
        return
    
    # Create visualizations
    print("Creating thesis visualizations...")
    create_thesis_visualizations(output_dir, performance_summary)
    
    # Generate LaTeX table
    print("Generating LaTeX results table...")
    latex_table = generate_thesis_results_table(output_dir, performance_summary)
    
    # Save LaTeX table
    table_file = os.path.join(output_dir, "thesis_results_table.tex")
    with open(table_file, 'w') as f:
        f.write(latex_table)
    
    # Generate insights text
    print("Generating academic insights text...")
    insights_text = generate_thesis_insights_text(output_dir)
    
    # Save insights text
    insights_file = os.path.join(output_dir, "thesis_insights_text.tex")
    with open(insights_file, 'w') as f:
        f.write(insights_text)
    
    # Generate complete results summary
    results_summary = {
        'performance_summary': performance_summary,
        'generated_files': {
            'visualizations': [
                'thesis_method_comparison.png',
                'thesis_error_analysis.png'
            ],
            'latex_table': 'thesis_results_table.tex',
            'insights_text': 'thesis_insights_text.tex'
        }
    }
    
    # Save complete summary
    summary_file = os.path.join(output_dir, "thesis_results_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n=== Thesis Results Generation Complete ===")
    print(f"Performance visualizations saved to: {output_dir}")
    print(f"LaTeX table saved to: {table_file}")
    print(f"Academic insights saved to: {insights_file}")
    print(f"Complete summary saved to: {summary_file}")
    
    # Print preview of key statistics
    print("\n=== Key Statistics for Thesis ===")
    for method, stats in performance_summary['method_performance'].items():
        method_name = method.replace('_vs_', ' vs ')
        print(f"{method_name}:")
        print(f"  Precision: {stats['mean_precision']:.3f} ± {stats['std_precision']:.3f}")
        print(f"  Recall: {stats['mean_recall']:.3f} ± {stats['std_recall']:.3f}")
        print(f"  F1-Score: {stats['mean_f1']:.3f} ± {stats['std_f1']:.3f}")
        print(f"  Experiments: {stats['count']}")
        print()

if __name__ == "__main__":
    main()
