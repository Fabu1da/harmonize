#!/usr/bin/env python3
"""
Generate Performance Table with Accuracies, 95% CI, Ranks, and Significance
for Calibrated Methods in LaTeX format.
"""

import json
import numpy as np
from scipy import stats
from itertools import combinations
import os
import sys
from config import APPROACHES, APPROACH_NAMES



def load_scores(scores_path):
    with open(scores_path) as f:
        return json.load(f)

def extract_accuracies(scores_data, methods):
    """Extract unweighted accuracies per column for each method."""
    acc_data = {}
    for method in methods:
        if method in scores_data:
            per_column = scores_data[method]['per_column']
            accs = [info['unweighted_accuracy'] for info in per_column.values()]
            acc_data[method] = np.array(accs)
    return acc_data

def bootstrap_ci(acc_data, method, n_boot=1000, ci=95):
    """Compute bootstrap confidence interval for a method's mean accuracy."""
    accs = acc_data[method]
    if len(accs) == 0:
        return None
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(accs, size=len(accs), replace=True)
        means.append(np.mean(sample))
    lower = np.percentile(means, (100 - ci) / 2)
    upper = np.percentile(means, 100 - (100 - ci) / 2)
    return {'mean': np.mean(accs), 'ci_lower': lower, 'ci_upper': upper}

def paired_t_tests(acc_data):
    """Perform paired t-tests between all pairs of methods."""
    results = {}
    for m1, m2 in combinations(acc_data.keys(), 2):
        acc1 = acc_data[m1]
        acc2 = acc_data[m2]
        if len(acc1) == len(acc2) and len(acc1) > 1:
            t_stat, p_val = stats.ttest_rel(acc1, acc2)
            results[(m1, m2)] = p_val
            results[(m2, m1)] = p_val  # symmetric
    return results

def count_significant_better(method, acc_data, t_results, mean_accs):
    """Count how many other methods this method is significantly better than."""
    count = 0
    for other in acc_data.keys():
        if other == method:
            continue
        if mean_accs[method] > mean_accs[other]:
            p_val = t_results.get((method, other), 1.0)
            if p_val < 0.05:
                count += 1
    return count

def generate_latex_table(unweighted_acc_data, confidence_interval_results, ranks: dict, sig_counts):
    """Generate LaTeX table."""
    latex = """\\begin{table}[H]
\\centering
\\scriptsize
\\caption{Method Performance and Statistical Validation}
\\label{tab:combined_results}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{|p{4.2cm}|c|c|c|c|}
\\hline
\\textbf{Method} & \\textbf{Accuracy} & \\textbf{95\\% CI} & \\textbf{Rank} & \\textbf{Sig. Better Than} \\\\
\\hline
"""
    for method in APPROACH_NAMES:
        rank = ranks[method]
        conf_int_per_method = confidence_interval_results[method]
        if conf_int_per_method:
            acc = f"{conf_int_per_method['mean']:.4f}"
            ci_str = f"[{conf_int_per_method['ci_lower']:.3f}, {conf_int_per_method['ci_upper']:.3f}]"
        else:
            acc = "N/A"
            ci_str = "N/A"
        sig = sig_counts[method]
        # Clean method name
        clean_method = method.replace("calibrated/", "")
        latex += f"{clean_method} & {acc} & {ci_str} & {rank} & {sig} \\\\\n\\hline\n"

    latex += "\\end{tabular}\n}\n\\end{table}"
    return latex

def performance_table():
    scores_path = "assets/predicted/scores.json"
    scores_data = load_scores(scores_path)

    methods = APPROACH_NAMES

    unweighted_acc_data = extract_accuracies(scores_data, methods)

    # Compute CIs
    confidence_interval_results = {}
    mean_accs = {}
    for method in methods:
        confidence_interval = bootstrap_ci(unweighted_acc_data, method)
        confidence_interval_results[method] = confidence_interval
        if confidence_interval:
            mean_accs[method] = confidence_interval['mean']

    # Perform t-tests
    t_results = paired_t_tests(unweighted_acc_data)

    # Compute ranks
    sorted_methods = sorted(mean_accs.items(), key=lambda x: x[1], reverse=True)
    ranks = {method: rank + 1 for rank, (method, _) in enumerate(sorted_methods)}

    # Compute sig better than
    sig_counts = {}
    for method in methods:
        sig_counts[method] = count_significant_better(method, unweighted_acc_data, t_results, mean_accs)

    # Generate LaTeX
    latex = generate_latex_table(unweighted_acc_data, confidence_interval_results, ranks, sig_counts)

    # Write to file
    output_path = "assets/reports/performance_table.tex"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(latex)

    print(f"Table generated at {output_path}")