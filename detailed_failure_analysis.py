#!/usr/bin/env python3
"""
Detailed Failure Pattern Analysis Script
Analyzes specific patterns and creates visualizations for FP/FN analysis.
"""

import os
import pandas as pd
import numpy as np
from collections import defaultdict
import re

def load_and_analyze_detailed_results():
    """Load the detailed results and perform deeper analysis."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    csv_file = os.path.join(output_dir, "fp_fn_detailed_results.csv")
    
    if not os.path.exists(csv_file):
        print(f"Please run fp_fn_analysis.py first to generate {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    return df

def analyze_dataset_patterns(df):
    """Analyze patterns based on dataset characteristics."""
    
    # Extract dataset characteristics from source/target names
    def extract_dataset_info(name):
        # Pattern: assays_both_50_X_acY_ev
        parts = name.replace('\_', '_').split('_')
        if len(parts) >= 6:
            try:
                size_param = int(parts[3])  # The X in 50_X
                accuracy_param = int(parts[4][2:])  # The Y in acY
                return size_param, accuracy_param
            except:
                pass
        return None, None
    
    df['source_size'], df['source_acc'] = zip(*df['source'].apply(extract_dataset_info))
    df['target_size'], df['target_acc'] = zip(*df['target'].apply(extract_dataset_info))
    
    # Analyze performance by dataset characteristics
    analysis = {}
    
    # Performance by source dataset size
    size_analysis = df.groupby('source_size').agg({
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'FP': ['mean', 'std'],
        'FN': ['mean', 'std']
    }).round(3)
    analysis['by_source_size'] = size_analysis
    
    # Performance by accuracy parameter
    acc_analysis = df.groupby('source_acc').agg({
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'FP': ['mean', 'std'],
        'FN': ['mean', 'std']
    }).round(3)
    analysis['by_accuracy'] = acc_analysis
    
    # Cross-dataset type analysis (same vs different datasets)
    df['same_dataset'] = df['source'] == df['target']
    same_diff_analysis = df.groupby('same_dataset').agg({
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std', 'count'],
        'FP': ['mean', 'std'],
        'FN': ['mean', 'std']
    }).round(3)
    analysis['same_vs_different'] = same_diff_analysis
    
    return analysis

def identify_worst_cases(df, top_n=10):
    """Identify the worst performing cases for detailed analysis."""
    
    worst_cases = {}
    
    # Worst precision cases (highest FP)
    worst_precision = df.nsmallest(top_n, 'precision')[
        ['source', 'target', 'comparison', 'precision', 'recall', 'f1', 'FP', 'FN']
    ]
    worst_cases['worst_precision'] = worst_precision
    
    # Worst recall cases (highest FN)
    worst_recall = df.nsmallest(top_n, 'recall')[
        ['source', 'target', 'comparison', 'precision', 'recall', 'f1', 'FP', 'FN']
    ]
    worst_cases['worst_recall'] = worst_recall
    
    # Worst overall F1 cases
    worst_f1 = df.nsmallest(top_n, 'f1')[
        ['source', 'target', 'comparison', 'precision', 'recall', 'f1', 'FP', 'FN']
    ]
    worst_cases['worst_f1'] = worst_f1
    
    return worst_cases

def generate_detailed_latex_analysis(df, dataset_analysis, worst_cases, output_dir):
    """Generate comprehensive LaTeX analysis."""
    
    latex_content = ""
    
    # Dataset characteristics analysis
    latex_content += """
% Dataset Size Impact Analysis
\\begin{table}[htbp]
\\centering
\\caption{Performance Analysis by Source Dataset Size Parameter}
\\label{tab:performance_by_size}
\\begin{tabular}{lcccccc}
\\toprule
Size Param & Count & Precision & Recall & FP Mean & FN Mean & F1 Mean \\\\
\\midrule
"""
    
    for size, stats in dataset_analysis['by_source_size'].iterrows():
        if pd.notna(size):
            precision_mean = stats[('precision', 'mean')]
            recall_mean = stats[('recall', 'mean')]
            fp_mean = stats[('FP', 'mean')]
            fn_mean = stats[('FN', 'mean')]
            count = int(stats[('precision', 'count')])
            f1_mean = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0
            
            latex_content += f"{int(size)} & {count} & {precision_mean:.3f} & {recall_mean:.3f} & {fp_mean:.2f} & {fn_mean:.2f} & {f1_mean:.3f} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Same vs Different Dataset Analysis
    latex_content += """
% Same vs Different Dataset Analysis
\\begin{table}[htbp]
\\centering
\\caption{Performance: Same Dataset vs Cross-Dataset Matching}
\\label{tab:same_vs_different}
\\begin{tabular}{lcccccc}
\\toprule
Match Type & Count & Precision & Recall & FP Mean & FN Mean & F1 Mean \\\\
\\midrule
"""
    
    for same_dataset, stats in dataset_analysis['same_vs_different'].iterrows():
        match_type = "Same Dataset" if same_dataset else "Cross-Dataset"
        precision_mean = stats[('precision', 'mean')]
        recall_mean = stats[('recall', 'mean')]
        fp_mean = stats[('FP', 'mean')]
        fn_mean = stats[('FN', 'mean')]
        count = int(stats[('precision', 'count')])
        f1_mean = 2 * (precision_mean * recall_mean) / (precision_mean + recall_mean) if (precision_mean + recall_mean) > 0 else 0
        
        latex_content += f"{match_type} & {count} & {precision_mean:.3f} & {recall_mean:.3f} & {fp_mean:.2f} & {fn_mean:.2f} & {f1_mean:.3f} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Worst cases analysis
    latex_content += """
% Worst Performing Cases
\\begin{table}[htbp]
\\centering
\\caption{Worst Precision Cases (Top 5)}
\\label{tab:worst_precision}
\\begin{tabular}{llllcc}
\\toprule
Source & Target & Comparison & Precision & FP & FN \\\\
\\midrule
"""
    
    for idx, row in worst_cases['worst_precision'].head(5).iterrows():
        source = row['source'].replace('_', '\\_')
        target = row['target'].replace('_', '\\_')
        comparison = row['comparison'].replace('_', '\\_')
        latex_content += f"{source[:20]}... & {target[:20]}... & {comparison} & {row['precision']:.3f} & {row['FP']:.1f} & {row['FN']:.1f} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Performance distribution analysis
    precision_bins = pd.cut(df['precision'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], include_lowest=True)
    precision_dist = precision_bins.value_counts().sort_index()
    
    latex_content += """
% Precision Distribution Analysis
\\begin{table}[htbp]
\\centering
\\caption{Precision Score Distribution}
\\label{tab:precision_distribution}
\\begin{tabular}{lc}
\\toprule
Precision Range & Count \\\\
\\midrule
"""
    
    for bin_range, count in precision_dist.items():
        latex_content += f"{bin_range} & {count} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Error correlation analysis
    correlation_matrix = df[['precision', 'recall', 'f1', 'FP', 'FN']].corr()
    
    latex_content += """
% Correlation Analysis
\\begin{table}[htbp]
\\centering
\\caption{Correlation Matrix: Performance Metrics vs Errors}
\\label{tab:correlation_matrix}
\\begin{tabular}{lccccc}
\\toprule
& Precision & Recall & F1 & FP & FN \\\\
\\midrule
"""
    
    for metric in ['precision', 'recall', 'f1', 'FP', 'FN']:
        row_data = [f"{correlation_matrix.loc[metric, col]:.3f}" for col in ['precision', 'recall', 'f1', 'FP', 'FN']]
        latex_content += f"{metric.title()} & {' & '.join(row_data)} \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Save to file
    latex_file = os.path.join(output_dir, "detailed_fp_fn_analysis.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    print(f"\nDetailed FP/FN Analysis saved to: {latex_file}")
    return latex_content

def create_summary_insights(df, dataset_analysis):
    """Generate key insights for the thesis."""
    
    insights = []
    
    # Overall performance insights
    total_high_fp = len(df[df['precision'] < 0.5])
    total_high_fn = len(df[df['recall'] < 0.5])
    total_perfect = len(df[(df['precision'] == 1.0) & (df['recall'] == 1.0)])
    
    insights.append(f"False Positive Issues: {total_high_fp}/600 cases ({total_high_fp/6:.1f}%) have precision < 0.5")
    insights.append(f"False Negative Issues: {total_high_fn}/600 cases ({total_high_fn/6:.1f}%) have recall < 0.5")
    insights.append(f"Perfect Performance: {total_perfect}/600 cases ({total_perfect/6:.1f}%) achieve perfect precision and recall")
    
    # Matcher comparison insights
    by_comparison = df.groupby('comparison').agg({
        'precision': 'mean',
        'recall': 'mean',
        'FP': 'mean',
        'FN': 'mean'
    }).round(3)
    
    best_matcher = by_comparison['precision'].idxmax()
    worst_matcher = by_comparison['precision'].idxmin()
    
    insights.append(f"Best Performing Matcher (Precision): {best_matcher} ({by_comparison.loc[best_matcher, 'precision']:.3f})")
    insights.append(f"Worst Performing Matcher (Precision): {worst_matcher} ({by_comparison.loc[worst_matcher, 'precision']:.3f})")
    
    # Dataset size impact
    if 'by_source_size' in dataset_analysis:
        size_stats = dataset_analysis['by_source_size']
        if len(size_stats) > 1:
            size_precision = size_stats[('precision', 'mean')]
            best_size = size_precision.idxmax()
            worst_size = size_precision.idxmin()
            insights.append(f"Best Dataset Size Parameter: {best_size} (Precision: {size_precision[best_size]:.3f})")
            insights.append(f"Worst Dataset Size Parameter: {worst_size} (Precision: {size_precision[worst_size]:.3f})")
    
    # Same vs different dataset performance
    if 'same_vs_different' in dataset_analysis:
        same_diff = dataset_analysis['same_vs_different']
        same_precision = same_diff.loc[True, ('precision', 'mean')] if True in same_diff.index else 0
        diff_precision = same_diff.loc[False, ('precision', 'mean')] if False in same_diff.index else 0
        insights.append(f"Same Dataset Matching: {same_precision:.3f} precision")
        insights.append(f"Cross-Dataset Matching: {diff_precision:.3f} precision")
        if same_precision > diff_precision:
            insights.append("Same dataset matching performs better than cross-dataset matching")
        else:
            insights.append("Cross-dataset matching performs surprisingly well")
    
    return insights

def main():
    """Main function for detailed analysis."""
    print("=== Detailed False Positive/False Negative Analysis ===\n")
    
    # Load data
    df = load_and_analyze_detailed_results()
    if df is None:
        return
    
    print(f"Loaded {len(df)} comparison results for detailed analysis")
    
    # Analyze dataset patterns
    dataset_analysis = analyze_dataset_patterns(df)
    
    # Identify worst cases
    worst_cases = identify_worst_cases(df)
    
    # Generate insights
    insights = create_summary_insights(df, dataset_analysis)
    
    # Generate LaTeX analysis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    latex_content = generate_detailed_latex_analysis(df, dataset_analysis, worst_cases, output_dir)
    
    # Print key insights
    print("\n=== Key Insights for Thesis ===")
    for i, insight in enumerate(insights, 1):
        print(f"{i}. {insight}")
    
    print(f"\n=== Generated LaTeX Analysis ===")
    print("="*60)
    print(latex_content[:1000] + "..." if len(latex_content) > 1000 else latex_content)
    print("="*60)

if __name__ == "__main__":
    main()
