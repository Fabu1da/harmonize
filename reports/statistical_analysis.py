#!/usr/bin/env python3
"""
Statistical Significance and Robustness Tests for Schema Matching Thesis
Performs t-tests, ANOVA, bootstrapping on method accuracies.
"""

import json
import numpy as np
from scipy import stats
from itertools import combinations

def load_scores(scores_path):
    with open(scores_path) as f:
        return json.load(f)

def extract_accuracies(scores_data, methods):
    """Extract unweighted accuracies per table for each method."""
    acc_data = {}
    for method in methods:
        if method in scores_data:
            per_table = scores_data[method]['per_table']
            accs = [info['unweighted_accuracy'] for info in per_table.values() if 'unweighted_accuracy' in info]
            acc_data[method] = np.array(accs)
    return acc_data

def paired_t_tests(acc_data):
    """Perform paired t-tests between all pairs of methods."""
    results = {}
    for m1, m2 in combinations(acc_data.keys(), 2):
        acc1 = acc_data[m1]
        acc2 = acc_data[m2]
        if len(acc1) == len(acc2) and len(acc1) > 1:
            t_stat, p_val = stats.ttest_rel(acc1, acc2)
            effect_size = (np.mean(acc1) - np.mean(acc2)) / np.std(acc1 - acc2) if np.std(acc1 - acc2) > 0 else 0
            results[f"{m1} vs {m2}"] = {'t_stat': t_stat, 'p_val': p_val, 'effect_size': effect_size}
    return results

def anova_test(acc_data):
    """Perform one-way ANOVA across methods."""
    data = [acc_data[m] for m in acc_data.keys() if len(acc_data[m]) > 1]
    if len(data) > 1:
        f_stat, p_val = stats.f_oneway(*data)
        return {'f_stat': f_stat, 'p_val': p_val}
    return None

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

def generate_latex_table(t_test_results, anova_result, ci_results):
    """Generate LaTeX table for statistical tests."""
    latex = """
\\begin{table}[h]
\\caption{Statistical Significance and Robustness Tests}
\\label{tab:stat_tests}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Comparison} & \\textbf{p-value} & \\textbf{Effect Size} & \\textbf{Significance} \\\\
\\hline
"""
    for comp, res in t_test_results.items():
        sig = "Significant" if res['p_val'] < 0.05 else "Not Significant"
        latex += f"{comp} & {res['p_val']:.4f} & {res['effect_size']:.4f} & {sig} \\\\\n\\hline\n"

    if anova_result:
        sig = "Significant" if anova_result['p_val'] < 0.05 else "Not Significant"
        latex += f"ANOVA (all methods) & {anova_result['p_val']:.4f} & - & {sig} \\\\\n\\hline\n"

    latex += "\\end{tabular}\n\\end{table}\n\n"

    # Bootstrap CIs
    latex += "\\begin{table}[h]\n\\caption{Bootstrap Confidence Intervals for Method Accuracies}\n\\label{tab:bootstrap_ci}\n\\begin{tabular}{|l|c|c|c|}\n\\hline\n\\textbf{Method} & \\textbf{Mean Accuracy} & \\textbf{95\\% CI Lower} & \\textbf{95\\% CI Upper} \\\\\n\\hline\n"
    for method, ci in ci_results.items():
        if ci:
            latex += f"{method} & {ci['mean']:.4f} & {ci['ci_lower']:.4f} & {ci['ci_upper']:.4f} \\\\\n\\hline\n"
    latex += "\\end{tabular}\n\\end{table}"
    
    
    with open("assets/reports/statistical_analysis.tex", "w") as f:
        f.write(latex)

