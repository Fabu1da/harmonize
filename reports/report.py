import shutil
from typing import Any

import numpy as np

from tabulate import tabulate
import matplotlib.patches
from matplotlib import pyplot as plt

from approaches.cluster import ClusteringApproach
from approaches.embedding import EmbeddingApproach
from approaches.ensemble import BaseEnsembleApproach
from approaches.llm import LLMApproach
from config import APPROACHES, APPROACH_NAMES


categories = {
    'Baseline': [],
    'LLM': [],
    'Embedding': [],
    'Ensemble': [],
}

for approach, approach_name in APPROACHES:
    if isinstance(approach, LLMApproach):
        categories['LLM'].append(approach_name)
    elif isinstance(approach, EmbeddingApproach) or isinstance(approach, ClusteringApproach):
        categories['Embedding'].append(approach_name)
    elif isinstance(approach, BaseEnsembleApproach):
        categories['Ensemble'].append(approach_name)
    else:
        categories['Baseline'].append(approach_name)


def make_overall_bar_chart(base_methods: dict[str, Any], calibrated_methods: dict[str, Any]):
    import os
    os.makedirs('./assets/reports/variables', exist_ok=True)
    
    # Prepare data for bar chart
    methods = base_methods.keys()
    uncal_cwa = [base_methods[m].get("confidence_weighted_accuracy", 0.0) for m in methods]
    uncal_ua = [base_methods[m].get("unweighted_accuracy", 0.0) for m in methods]
    cal_cwa = [calibrated_methods[m].get("confidence_weighted_accuracy", 0.0) for m in methods]

    # Set up bar chart
    bar_width = 0.2
    index = np.arange(len(methods))
    
    plt.figure(figsize=(12, 8))  # Increased height for legend
    
    # Plot bars
    plt.bar(index - bar_width*0.5, uncal_ua, bar_width, label='Unweighted', color='#ff0000')
    plt.bar(index - bar_width*1.5, uncal_cwa, bar_width, label='Uncalibrated Confidence Weighted', color='#00ff00')
    plt.bar(index + bar_width*0.5, cal_cwa, bar_width, label='Calibrated Confidence Weighted', color='#0000ff')
    
    # Customize plot
    plt.xlabel('Overall Performance of Approaches')
    plt.ylabel('Accuracy')
    plt.title('Overall Accuracy Comparison of Schema Matching Methods')
    plt.xticks(index, [f'Appr. {i+1}' for i in range(len(methods))])  # Use indices as labels
    plt.legend(loc='lower right')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add method mapping below the chart
    method_mapping = "\n".join([f"{i+1}. {m}" for i, m in enumerate(methods)])
    plt.figtext(0.1, -0.1, f"Approaches:\n{method_mapping}", ha='left', fontsize=10, wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)  # Make room for the text
    
    # Save the plot
    plt.savefig('./assets/reports/overall_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Bar chart saved to overall_performance.png")

    shutil.rmtree('./assets/reports/variables/cal', ignore_errors=True)
    os.makedirs('./assets/reports/variables/cal', exist_ok=True)
    
    # Create separate LaTeX files for each method's calibrated and uncalibrated values
    for i, method in enumerate(methods):
        method_clean = method.replace(" ", "_").replace("-", "_").lower()
        
        # Uncalibrated confidence weighted accuracy
        with open(f"./assets/reports/variables/cal/{method_clean}_uncalibrated_cwa.tex", "w") as f:
            f.write(f"\\text{{{uncal_cwa[i]:.4f}}}")
        
        # Uncalibrated unweighted accuracy
        with open(f"./assets/reports/variables/cal/{method_clean}_uncalibrated_ua.tex", "w") as f:
            f.write(f"\\text{{{uncal_ua[i]:.4f}}}")
        
        # Calibrated confidence weighted accuracy
        with open(f"./assets/reports/variables/cal/{method_clean}_calibrated_cwa.tex", "w") as f:
            f.write(f"\\text{{{cal_cwa[i]:.4f}}}")
        
        print(f"Created LaTeX files for {method}: uncalibrated_cwa, uncalibrated_ua, calibrated_cwa")
    


def overall_report(data: dict[str, Any]):
    # Separate calibrated and non-calibrated methods
    base_methods = {}
    calibrated_methods = {}
    
    for key, value in data.items():
        print(f"Processing key: {key}")
        if key.startswith("calibrated/"):
            method_name = key.replace("calibrated/", "")
            calibrated_methods[method_name] = value.get("overall", {})
        else:
            base_methods[key] = value.get("overall", {})

    make_overall_bar_chart(base_methods, calibrated_methods)

    # Build the combined report
    report_data = []
    for method, base_overall in base_methods.items():
        calibrated_overall = calibrated_methods[method]
        row = [
            method,
            base_overall.get("confidence_weighted_accuracy", 0.0),
            base_overall.get("unweighted_accuracy", 0.0),
            calibrated_overall.get("confidence_weighted_accuracy", 0.0),
            calibrated_overall.get("unweighted_accuracy", 0.0)
        ]
        report_data.append(row)
    
    # Save LaTeX table to file
        
    latex = """\\begin{table}[h]
    \\centering\n"
    \\begin{tabular}{|l|c|c|c|c|}
    \\hline
    \\Method & Unweighted & Uncalibrated Confidence Weighted & Calibrated Confidence Weighted
    \\hline
    """

    for row in report_data:
        method = row[0].replace("&", "\\&").replace("_", "\\_")
        latex += f"{method} & {row[1]:.4f} & {row[2]:.4f} & {row[3]:.4f} & {row[4]:.4f}\\\\\n"

    latex += """
    \\hline
    \\end{tabular}
    \\caption{Overall Accuracy Comparison of Schema Matching Methods}
    \\label{tab:accuracies}
    \\end{table}
    """
        
    with open("./assets/reports/overall_perfomance.tex", "w") as f:
        f.write(latex)
    
    print("LaTeX table saved to accuracy_table.tex")
    
    
def per_dataset_accuracy_report(data: dict[str, Any]):
    datasets = ['ChEMBL', 'Magellan', 'OpenData', 'TPC-DI', 'Wikidata', 'synthetic']

    breakdown = {approach_name: {} for approach_name in APPROACH_NAMES}
    
    for method_key, method_data in data.items():
        if method_key.startswith('calibrated/'):
            continue  # Skip calibrated methods
        method_name = method_key
        if method_name not in APPROACH_NAMES:
            continue
        
        per_table = method_data.get('per_table', {})
        dataset_sums = {ds: {'weighted_sum': 0.0, 'total_weight': 0.0} for ds in datasets}
        
        for table_key, table_data in per_table.items():
            parts = table_key.split('/')
            if len(parts) > 0 and parts[0] in datasets:
                ds = parts[0]
                acc = table_data['unweighted_accuracy']
                weight = table_data['table_weight']
                dataset_sums[ds]['weighted_sum'] += acc * weight
                dataset_sums[ds]['total_weight'] += weight
            if len(parts) > 1 and parts[1] in datasets:
                ds = parts[1]
                acc = table_data['unweighted_accuracy']
                weight = table_data['table_weight']
                dataset_sums[ds]['weighted_sum'] += acc * weight
                dataset_sums[ds]['total_weight'] += weight
        
        for ds in datasets:
            total_weight = dataset_sums[ds]['total_weight']
            if total_weight > 0:
                avg = dataset_sums[ds]['weighted_sum'] / total_weight
                breakdown[method_name][ds] = avg * 100  # Convert to percentage
            else:
                breakdown[method_name][ds] = 0.0

    # Build table data
    headers = ['Dataset'] + APPROACH_NAMES
    table_data = []
    for ds in datasets:
        row = [ds]
        for method in APPROACH_NAMES:
            row.append(breakdown[method][ds])
        table_data.append(row)
    
    # Print as grid with bold formatting for highest values
    print("\nPer-Dataset Accuracy Breakdown:")
    formatted_table_data = []
    for row in table_data:
        formatted_row = [row[0]]  # Dataset name
        values = row[1:]
        max_val = max(values) if values else 0
        for val in values:
            if val == max_val and max_val > 0:
                formatted_row.append(f"**{val:.1f}**")  # Bold markdown for console
            else:
                formatted_row.append(f"{val:.1f}")
        formatted_table_data.append(formatted_row)
    print(tabulate(formatted_table_data, headers=headers, tablefmt="grid"))
    
    # Save to LaTeX with bold formatting for highest values
    with open("./assets/reports/per_dataset_breakdown.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|" + "c|" * len(APPROACH_NAMES) + "}\n")
        f.write("\\hline\n")
        f.write("Dataset & " + " & ".join([m.replace("_", "\\_") for m in APPROACH_NAMES]) + " \\\\\n")
        f.write("\\hline\n")
        for row in table_data:
            ds = row[0]
            values = row[1:]
            max_val = max(values) if values else 0
            formatted_values = []
            for val in values:
                if val == max_val and max_val > 0:
                    formatted_values.append(f"\\textbf{{{val:.1f}}}")
                else:
                    formatted_values.append(f"{val:.1f}")
            values_str = " & ".join(formatted_values)
            f.write(f"{ds} & {values_str} \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Per-Dataset Accuracy Breakdown Showing Method Robustness}\n")
        f.write("\\label{tab:per_dataset}\n")
        f.write("\\end{table}\n")
    
    print("LaTeX table saved to per_dataset_breakdown.tex")
    
    # Generate grouped bar chart
    make_per_dataset_bar_chart(breakdown, datasets, APPROACH_NAMES)
    make_per_dataset_bar_chart2(breakdown, datasets, APPROACH_NAMES)


def make_per_dataset_bar_chart(breakdown, datasets, methods):
    # Grouped bar chart: datasets as groups, methods as bars
    n_datasets = len(datasets)
    n_methods = len(methods)
    bar_width = 0.8 / n_methods  # Thin bars to fit
    index = np.arange(n_datasets)
    
    plt.figure(figsize=(15, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']  # Colorblind-friendly
    
    for i, method in enumerate(methods):
        accs = [breakdown[method][ds] / 100 for ds in datasets]  # Convert back to decimal
        plt.bar(index + i * bar_width, accs, bar_width, label=f'{i+1}. {method}', color=colors[i % len(colors)])
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Per-Dataset Accuracy Breakdown')
    plt.xticks(index + bar_width * (n_methods - 1) / 2, datasets)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('./assets/reports/per_dataset_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Per-dataset bar chart saved to per_dataset_breakdown.png")
    
def make_per_dataset_bar_chart2(breakdown, datasets, methods):
    # Grouped bar chart: datasets as groups, methods as bars
    n_datasets = len(datasets)
    n_methods = len(methods)
    bar_width = 0.8 / n_datasets  # Thin bars to fit
    index = np.arange(n_methods)
    
    plt.figure(figsize=(15, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']  # Colorblind-friendly
    
    for i, dataset in enumerate(datasets):
        accs = [breakdown[m][dataset] / 100 for m in methods]  # Convert back to decimal
        plt.bar(index + i * bar_width, accs, bar_width, label=f'{i+1}. {dataset}', color=colors[i % len(colors)])
    
    plt.xlabel('Approach')
    plt.ylabel('Accuracy')
    plt.title('Per-Dataset Accuracy Breakdown Group By Approach')
    plt.xticks(index + bar_width * (n_datasets - 1) / 2, [f"Appr. {i+1}" for i in range(n_methods)])
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Add method mapping below the chart
    method_mapping = "\n".join([f"{i+1}. {m}" for i, m in enumerate(methods)])
    plt.figtext(0.1, 0, f"Approaches:\n{method_mapping}", ha='left', fontsize=12, wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.35)  # Make room for the text
    
    plt.savefig('./assets/reports/per_dataset_breakdown2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Per-dataset bar chart saved to per_dataset_breakdown2.png")


def category_comparison_table(data):
    # Get overall accuracies
    accuracies = {}
    for method_key, method_data in data.items():
        if not method_key.startswith('calibrated/'):
            accuracies[method_key] = method_data.get('overall', {}).get('unweighted_accuracy', 0.0)

    max_baseline_acc = max((accuracies[method] for method in categories["Baseline"]), default=0.0)

    # Compute category stats
    category_data = []
    for cat, methods in categories.items():
        cat_accs = [accuracies[m] for m in methods if m in accuracies]
        if cat_accs:
            max_acc = max(cat_accs)
            best_method = max(methods, key=lambda m: accuracies[m])
            improvement = max_acc - max_baseline_acc
            category_data.append([cat, max_acc * 100, best_method, improvement * 100])

    # Build table
    headers = ['Category', 'Max Accuracy (%)', 'Best Method', 'Improvement over Baseline (%)']
    table_data = category_data
    
    # Print as grid
    print("\nMethod Category Comparison:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=["", ".1f", "", ".1f"]))
    
    # Save to LaTeX
    with open("./assets/reports/method_category_comparison.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|c|l|c|}\n")
        f.write("\\hline\n")
        f.write("Category & Max Accuracy (\%) & Best Method & Improvement over Baseline (\%) \\\\\n")
        f.write("\\hline\n")
        for row in table_data:
            cat = row[0]
            max_acc = row[1]
            best = row[2].replace("_", "\\_")
            imp = row[3]
            f.write(f"{cat} & {max_acc:.1f} & {best} & {imp:.1f} \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Method Category Comparison Showing Trends Across Approach Types}\n")
        f.write("\\label{tab:category_comparison}\n")
        f.write("\\end{table}\n")
    
    print("LaTeX table saved to method_category_comparison.tex")
    
    # Generate stacked bar chart
    make_category_stacked_bar_chart(category_data)


def make_category_stacked_bar_chart(category_data):
    import os
    os.makedirs('./assets/reports/variables', exist_ok=True)
    
    data = [(cat, max_acc - imp, imp) for cat, max_acc, _, imp in category_data]
    labels = [row[0] for row in data]
    bottoms = [row[1] for row in data]
    tops = [row[2] for row in data]
    
    plt.figure(figsize=(8, 6))

    plt.bar(labels, bottoms, color='#1f77b4', label='Max Accuracy (%)')
    plt.bar(labels, tops, bottom=bottoms, color='#ff7f0e', label='Improvement over Baseline (%)')
    
    plt.xlabel('Category')
    plt.ylabel('Percentage')
    plt.title('Method Category Comparison')
    plt.legend(loc="lower right")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('./assets/reports/method_category_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Category comparison stacked bar chart saved to method_category_comparison.png")
    
    # Create separate LaTeX files for each category's main and improvement values
    for i, (cat, max_acc, _, imp) in enumerate(category_data):
        # Main value file (max accuracy)
        with open(f"./assets/reports/variables/{cat.lower()}_main.tex", "w") as f:
            f.write(f"\\text{{{max_acc:.1f}}}")  # Just the value as text
        
        # Improvement value file
        with open(f"./assets/reports/variables/{cat.lower()}_improvement.tex", "w") as f:
            f.write(f"\\text{{{imp:.1f}}}")  # Just the value as text
        
        print(f"Created {cat.lower()}_main.tex and {cat.lower()}_improvement.tex")
    

def calibrated_impact_summary_table(data):
    # Get accuracies
    impact_data = []
    for method in APPROACH_NAMES:
        base_key = method
        calibrated_key = f"calibrated/{method}"

        unweighted_acc = data[base_key]["overall"]['unweighted_accuracy'] * 100
        base_acc = data[base_key]["overall"]['confidence_weighted_accuracy'] * 100
        calibrated_acc = data[calibrated_key]["overall"]['confidence_weighted_accuracy'] * 100

        abs_change = calibrated_acc - base_acc
        rel_change = (abs_change / base_acc * 100) if base_acc != 0 else 0

        impact_data.append([method, unweighted_acc, base_acc, calibrated_acc, abs_change, rel_change])

    # Build table
    headers = ['Method', 'Unweighted (%)', 'Uncalibrated (%)', 'Calibrated (%)', 'Absolute Change (%)', 'Relative Change (%)']
    table_data = impact_data
    
    # Print as grid
    print("\nCalibration Impact Summary:")
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=["", ".1f", ".1f", "+.1f", "+.1f"]))
    
    # Save to LaTeX
    with open("./assets/reports/calibration_impact.tex", "w") as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\\begin{tabular}{|p{4.2cm}|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Method & Unweighted (\%) & Uncalibrated (\%) & Calibrated (\%) & Absolute Change (\%) & Relative Change (\%) \\\\\n")
        f.write("\\hline\n")
        for row in table_data:
            method = row[0].replace("_", "\\_")
            unweighted = row[1]
            uncal = row[2]
            cal = row[3]
            abs_ch = row[4]
            rel_ch = row[5]
            f.write(f"{method} & {unweighted:.1f} & {uncal:.1f} & {cal:.1f} & {abs_ch:+.1f} & {rel_ch:+.1f} \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Impact of Confidence Calibration on Accuracy}\n")
        f.write("\\label{tab:calibration_impact}\n")
        f.write("\\end{table}\n")
    
    print("LaTeX table saved to calibration_impact.tex")

def top_methods_ranking(data):
    # Get overall accuracies and sort top 5
    accuracies = {}
    for method_key, method_data in data.items():
        if not method_key.startswith('calibrated/'):
            accuracies[method_key] = method_data.get('overall', {}).get('unweighted_accuracy', 0.0)
    
    # Sort by accuracy descending
    methods = accuracies.items()

    # Generate vertical bar chart
    make_top_ranking_bar_chart(methods)


def make_top_ranking_bar_chart(methods_items):
    ranks = [rank for rank, (method, acc) in enumerate(methods_items, 1)]
    methods = [method for method, acc in methods_items]
    accuracies = [acc * 100 for method, acc in methods_items]  # To percentage



    # Define colors for each category
    category_colors = {
        'LLM': '#1f77b4',      # Blue
        'Embedding': '#2ca02c', # Green  
        'Baseline': '#d62728',  # Red
        'Ensemble': '#ff7f0e'   # Orange
    }
    
    # Determine category for each method
    method_categories = {}
    for method in methods:
        if method in categories['LLM']:
            method_categories[method] = 'LLM'
        elif method in categories['Embedding']:
            method_categories[method] = 'Embedding'
        elif method in categories['Ensemble']:
            method_categories[method] = 'Ensemble'
        elif method in categories['Baseline']:
            method_categories[method] = 'Baseline'
        else:
            raise ValueError(f"Method {method} not found in any category.")
        

    latex_counts_table = f"""\\begin{{table}}[h]
    \\centering
    \\caption{{top approaches ranking}}
    \\label{{tab:top_approaches_ranking}}
    \\begin{{tabular}}{{|c|c|c|}}
    \\hline
    \\textbf{{Methods Correct}} & \\textbf{{Number of Pairs}} & \\textbf{{Percentage}} \\\\
    \\hline
    """

    for method, acc in methods_items:
        latex_counts_table += f"""{method} & {acc*100:.2f}\\% \\\\
        \\hline
        """
       
    latex_counts_table += f"""
    \\end{{tabular}}
    \\end{{table}}"""

    with open('./assets/reports/top_approaches_ranking.tex', 'w') as f:
        f.write(latex_counts_table)

    
    # Get colors for each bar
    bar_colors = [category_colors[method_categories[method]] for method in methods]

    plt.figure(figsize=(12, 10))
    bars = plt.bar(ranks, accuracies, color=bar_colors)

    plt.xlabel('Approach')
    plt.ylabel('Accuracy (%)')
    plt.title('Approach Accuracy (Grouped by Category)')
    plt.xticks(ranks, [f'Appr. {i+1}' for i in range(len(methods))])

    plt.grid(True, axis='y', linestyle='--', alpha=0.7, which='both')

    # Add legend for categories with descriptions
    legend_labels = ['LLM Approaches', 'Embedding Approaches', 'Baseline', 'Ensemble Approaches']
    legend_elements = [matplotlib.patches.Rectangle((0,0),1,1, facecolor=color, edgecolor='none') 
                      for color in category_colors.values()]
    plt.legend(legend_elements, legend_labels, title='Category', 
              loc='lower right')

    # Add approach mapping below the chart
    approach_mapping = "\n".join([f"{i+1}. {m} ({method_categories[m]})" for i, m in enumerate(methods)])
    plt.figtext(0.1, 0, f"Approaches:\n{approach_mapping}", ha='left', fontsize=12, wrap=True)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Reduce space for the text

    plt.savefig('./assets/reports/top_approaches_ranking.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Top approaches ranking bar chart saved to top_approaches_ranking.png")


