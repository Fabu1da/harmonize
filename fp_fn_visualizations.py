#!/usr/bin/env python3
"""
FP/FN Visualization Script
Creates LaTeX charts for false positive and false negative analysis.
"""

import os
import pandas as pd
import numpy as np

def create_fp_fn_charts(output_dir):
    """Create LaTeX charts for FP/FN analysis."""
    
    # Load the detailed results
    csv_file = os.path.join(output_dir, "fp_fn_detailed_results.csv")
    if not os.path.exists(csv_file):
        print(f"Please run fp_fn_analysis.py first to generate {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Calculate summary statistics by comparison type
    by_comparison = df.groupby('comparison').agg({
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std'],
        'FP': ['mean', 'std'],
        'FN': ['mean', 'std']
    }).round(3)
    
    # Chart 1: Precision vs Recall by Matcher
    precision_recall_chart = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=12cm,
    height=8cm,
    ylabel={{Score}},
    xlabel={{Matcher Comparison}},
    symbolic x coords={{GPT vs Embedding, GPT vs Clustering, Embedding vs Clustering}},
    xtick=data,
    x tick label style={{rotate=45, anchor=east}},
    ymin=0,
    ymax=1,
    bar width=15pt,
    legend pos=north west,
    grid=major,
    grid style={{dashed,gray!30}},
]]
\\addplot[fill=blue!60] coordinates {{
    (GPT vs Embedding, {by_comparison.loc['GPT vs Embedding', ('precision', 'mean')]:.3f})
    (GPT vs Clustering, {by_comparison.loc['GPT vs Clustering', ('precision', 'mean')]:.3f})
    (Embedding vs Clustering, {by_comparison.loc['Embedding vs Clustering', ('precision', 'mean')]:.3f})
}};
\\addplot[fill=red!60] coordinates {{
    (GPT vs Embedding, {by_comparison.loc['GPT vs Embedding', ('recall', 'mean')]:.3f})
    (GPT vs Clustering, {by_comparison.loc['GPT vs Clustering', ('recall', 'mean')]:.3f})
    (Embedding vs Clustering, {by_comparison.loc['Embedding vs Clustering', ('recall', 'mean')]:.3f})
}};
\\legend{{Precision, Recall}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Precision vs Recall by Matcher Comparison}}
\\label{{fig:precision_recall_by_matcher}}
\\end{{figure}}"""
    
    # Chart 2: False Positives vs False Negatives
    fp_fn_chart = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=12cm,
    height=8cm,
    ylabel={{Average Count}},
    xlabel={{Matcher Comparison}},
    symbolic x coords={{GPT vs Embedding, GPT vs Clustering, Embedding vs Clustering}},
    xtick=data,
    x tick label style={{rotate=45, anchor=east}},
    ymin=0,
    ymax=20,
    bar width=15pt,
    legend pos=north west,
    grid=major,
    grid style={{dashed,gray!30}},
]]
\\addplot[fill=orange!60] coordinates {{
    (GPT vs Embedding, {by_comparison.loc['GPT vs Embedding', ('FP', 'mean')]:.2f})
    (GPT vs Clustering, {by_comparison.loc['GPT vs Clustering', ('FP', 'mean')]:.2f})
    (Embedding vs Clustering, {by_comparison.loc['Embedding vs Clustering', ('FP', 'mean')]:.2f})
}};
\\addplot[fill=purple!60] coordinates {{
    (GPT vs Embedding, {by_comparison.loc['GPT vs Embedding', ('FN', 'mean')]:.2f})
    (GPT vs Clustering, {by_comparison.loc['GPT vs Clustering', ('FN', 'mean')]:.2f})
    (Embedding vs Clustering, {by_comparison.loc['Embedding vs Clustering', ('FN', 'mean')]:.2f})
}};
\\legend{{False Positives, False Negatives}}
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{False Positives vs False Negatives by Matcher Comparison}}
\\label{{fig:fp_fn_by_matcher}}
\\end{{figure}}"""
    
    # Chart 3: Error Distribution
    # Calculate error categories
    high_fp_count = len(df[df['precision'] < 0.5])
    high_fn_count = len(df[df['recall'] < 0.5])
    perfect_count = len(df[(df['precision'] == 1.0) & (df['recall'] == 1.0)])
    partial_count = len(df) - high_fp_count - high_fn_count - perfect_count
    
    error_dist_chart = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=10cm,
    height=7cm,
    ylabel={{Count}},
    xlabel={{Error Category}},
    symbolic x coords={{High FP, High FN, Perfect, Other}},
    xtick=data,
    x tick label style={{rotate=45, anchor=east}},
    ymin=0,
    ymax=400,
    bar width=20pt,
    nodes near coords,
    nodes near coords align={{vertical}},
    grid=major,
    grid style={{dashed,gray!30}},
]]
\\addplot[fill=green!60] coordinates {{
    (High FP, {high_fp_count})
    (High FN, {high_fn_count})
    (Perfect, {perfect_count})
    (Other, {partial_count})
}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Distribution of Error Types (N=600)}}
\\label{{fig:error_distribution}}
\\end{{figure}}"""
    
    # Combine all charts
    full_latex = precision_recall_chart + "\n\n" + fp_fn_chart + "\n\n" + error_dist_chart
    
    # Save to file
    latex_file = os.path.join(output_dir, "fp_fn_charts.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)
    
    print(f"FP/FN Charts saved to: {latex_file}")
    return full_latex

def create_performance_matrix_chart(output_dir):
    """Create a performance matrix visualization."""
    
    csv_file = os.path.join(output_dir, "fp_fn_detailed_results.csv")
    df = pd.read_csv(csv_file)
    
    # Create precision/recall matrix for each comparison type
    comparisons = df['comparison'].unique()
    
    matrix_data = []
    for comp in comparisons:
        comp_data = df[df['comparison'] == comp]
        precision_mean = comp_data['precision'].mean()
        recall_mean = comp_data['recall'].mean()
        f1_mean = comp_data['f1'].mean()
        matrix_data.append([comp, precision_mean, recall_mean, f1_mean])
    
    # Create LaTeX table for the matrix
    matrix_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Performance Matrix: Summary of All Metrics}}
\\label{{tab:performance_matrix}}
\\begin{{tabular}}{{lccc}}
\\toprule
Matcher Comparison & Precision & Recall & F1-Score \\\\
\\midrule
"""
    
    for comp, precision, recall, f1 in matrix_data:
        comp_clean = comp.replace('_', '\\_')
        matrix_table += f"{comp_clean} & {precision:.3f} & {recall:.3f} & {f1:.3f} \\\\\n"
    
    matrix_table += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Save to file
    latex_file = os.path.join(output_dir, "performance_matrix.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(matrix_table)
    
    print(f"Performance Matrix saved to: {latex_file}")
    return matrix_table

def main():
    """Main function for visualization generation."""
    print("=== FP/FN Visualization Generation ===\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    # Create charts
    charts_latex = create_fp_fn_charts(output_dir)
    matrix_latex = create_performance_matrix_chart(output_dir)
    
    print("\n=== Generated LaTeX Charts ===")
    print("="*60)
    print(charts_latex[:800] + "..." if len(charts_latex) > 800 else charts_latex)
    print("="*60)
    
    print("\n=== Performance Matrix ===")
    print(matrix_latex)

if __name__ == "__main__":
    main()
