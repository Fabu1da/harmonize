#!/usr/bin/env python3
"""
Error Analysis Script for Schema Matching Thesis
Generates failure analysis, confusion matrices, and explanation patterns.
"""

import json
import os
from collections import defaultdict


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def compute_confusion(approach_name: str, scores_data, predicted_dir, expected_dir):
    """Compute correct, incorrect, missed mappings for a method."""
    total_correct = 0
    total_incorrect = 0
    total_missed = 0
    total_expected = 0

    per_table = scores_data[approach_name]['per_table']
    for table in per_table:
        expected_path = os.path.join(expected_dir, f"{table}.json")
        if not os.path.exists(expected_path):
            continue
        expected = load_json(expected_path)
        num_expected = len(expected['mappings'])
        total_expected += num_expected

        pred_path = os.path.join(predicted_dir, approach_name, f"{table}.json")
        if not os.path.exists(pred_path):
            missed = num_expected
            total_missed += missed
            continue

        pred = load_json(pred_path)
        correct = 0
        incorrect = 0

        # Build expected map
        expected_map = {m['target_column']: m['source_column'] for m in expected['mappings']}

        for target, (src, conf, expl) in pred.items():
            if expected_map.get(target) == src:
                correct += 1
            else:
                incorrect += 1

        missed = num_expected - correct
        total_correct += correct
        total_incorrect += incorrect
        total_missed += missed

    return {
        'correct': total_correct,
        'incorrect': total_incorrect,
        'missed': total_missed,
        'total_expected': total_expected
    }

def find_failure(approach_name: str, scores_data, predicted_dir, expected_dir, num_analysis=5):
    """Find tables with low accuracy and extract failure analysis."""
    per_table = scores_data[approach_name]['per_table']
    low_acc_tables = [(table, scores['unweighted_accuracy']) for table, scores in per_table.items() if scores['unweighted_accuracy'] < 0.5]
    low_acc_tables.sort(key=lambda x: x[1])  # lowest first

    failure = []
    for table, acc in low_acc_tables[:num_analysis]:
        expected_path = os.path.join(expected_dir, f"{table}.json")
        pred_path = os.path.join(predicted_dir, approach_name, f"{table}.json")
        if not os.path.exists(expected_path) or not os.path.exists(pred_path):
            continue

        expected = load_json(expected_path)
        pred = load_json(pred_path)

        expected_map = {m['target_column']: m['source_column'] for m in expected['mappings']}

        for target, (src, conf, expl) in pred.items():
            expected_src = expected_map.get(target)
            if expected_src and src != expected_src:
                failure.append({
                    'expected': f"{expected_src.replace('_', '\\_')} $\\rightarrow$ {target.replace('_', '\\_')}",
                    'predicted': f"{src.replace('_', '\\_')} $\\rightarrow$ {target.replace('_', '\\_')}" if src else None,
                    'explanation': expl
                })
                break  # one per table

    return failure


def generate_latex_confusion(confusion_data):
    """Generate LaTeX for confusion matrix."""
    latex = """\\begin{table}[h]
\\caption{Confusion Matrix Summary for Schema Matching Methods}
\\label{tab:confusion_matrix}
\\begin{tabular}{|l|c|c|c|}
\\hline
\\textbf{Method} & \\textbf{Correct Matches} & \\textbf{Incorrect Matches} & \\textbf{Missed Matches} \\\\
\\hline
"""
    for method, data in confusion_data.items():
        latex += f"{method} & {data['correct']} & {data['incorrect']} & {data['missed']} \\\\\n\\hline\n"
    latex += "\\end{tabular}\n\\end{table}"


    with open("./assets/reports/failure_confusion.tex", "w") as f:
        f.write(latex)




def generate_latex_failure(failure):
    """Generate LaTeX for failure analysis."""
    latex = """\\begin{table}[h]
\\small
\\centering
\\caption{Failure from Schema Matching Evaluation}
\\label{tab:failure_analysis}
\\resizebox{\\textwidth}{!}{
\\begin{tabular}{|p{4cm}|p{4cm}|p{8cm}|}
\\ Expected Match & Predicted Match & Explanation
"""
    for ex in failure:
        latex += f"{ex['expected']} & {"-" if ex['predicted'] is None else ex['predicted']} & {ex['explanation']} \\\\\n\\hline\n"
    latex += """\\end{tabular}
}
\\end{table} """
    
    
    
    # Generate LaTeX
    
    with open("./assets/reports/failure_analysis.tex", "w") as f:
        f.write(latex)
   



