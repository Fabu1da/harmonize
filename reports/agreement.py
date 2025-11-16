
import itertools
import json
import numpy as np
from scipy.stats import pearsonr

from config import APPROACHES, APPROACH_NAMES


def create_agreement_report(data):
    
    methods = APPROACH_NAMES
    method_accs = {}

    for method in methods:
        if method in data:
            accs = [info['unweighted_accuracy'] for info in data[method]['per_column'].values()]
            method_accs[method] = np.array(accs)
        else:
            print(f"Error: Method {method} not found in scores.json")
            exit(1)

    print(method_accs)

    # Ensure all have the same number of pairs
    if not all(len(accs) == len(method_accs[methods[0]]) for accs in method_accs.values()):
        print("Error: Methods have different numbers of pairs.")
        exit(1)

    # Count agreement levels: how many pairs had k methods with accuracy > 0.5
    agreement_counts = [0 for _ in range(len(APPROACH_NAMES) + 1)]
    total_pairs = len(method_accs[methods[0]])

    for i in range(total_pairs):
        correct_count = sum(1 for method in methods if method_accs[method][i] > 0.5)
        agreement_counts[correct_count] += 1

    print(f"Agreement counts (accuracy > 0.5):")

    for correct_count, sample_count in enumerate(agreement_counts):
        print(f"{correct_count} methods correct: {sample_count} samples")

    # Generate LaTeX table for agreement counts
    latex_counts_table = f"""\\begin{{table}}[h]
    \\centering
    \\caption{{Agreement Counts by Number of Correct Methods}}
    \\label{{tab:agreement_counts}}
    \\begin{{tabular}}{{|c|c|c|}}
    \\hline
    \\textbf{{Methods Correct}} & \\textbf{{Number of Pairs}} & \\textbf{{Percentage}} \\\\
    \\hline
    """

    for correct_count, sample_count in enumerate(agreement_counts):
        latex_counts_table += f"""{correct_count} & {sample_count} & {sample_count/total_pairs*100:.1f}\\% \\\\
        \\hline
        """
       
    latex_counts_table += f"""
    \\end{{tabular}}
    \\end{{table}}"""

    with open('reports/agreement_counts_table.tex', 'w') as f:
        f.write(latex_counts_table)

    print("LaTeX table for agreement counts saved to reports/agreement_counts_table.tex")

    # Compute pairwise correlations
    correlations = {}
    p_values = {}
   
    pairs = list(itertools.combinations(APPROACH_NAMES, 2))

    print(pairs)


    for m1, m2 in pairs:
        corr, p = pearsonr(method_accs[m1], method_accs[m2])
        correlations[f'{m1} vs {m2}'] = corr
        p_values[f'{m1} vs {m2}'] = p

    # Output
    total_pairs = len(method_accs[methods[0]])
    print(f"Total pairs analyzed: {total_pairs}")
    for pair in correlations:
        print(f"Correlation ({pair}): {correlations[pair]:.3f} (p-value: {p_values[pair]:.2e})")

    # Output
    total_pairs = len(method_accs[methods[0]])
    print(f"Total pairs analyzed: {total_pairs}")
    for pair, corr in correlations.items():
        print(f"Correlation ({pair}): {corr:.3f}")

    # Generate LaTeX table dynamically
    n_methods = len(APPROACH_NAMES)
    method_short = [m.split()[0] for m in APPROACH_NAMES]
    full_names = list(APPROACH_NAMES)

    # Create correlation matrix sized to number of methods
    corr_matrix = np.eye(n_methods)
    for i, m1 in enumerate(APPROACH_NAMES):
        for j, m2 in enumerate(APPROACH_NAMES):
            if i == j:
                continue
            # prefer stored key order, otherwise try flipped
            corr_matrix[i, j] = correlations.get(f'{m1} vs {m2}', correlations.get(f'{m2} vs {m1}', 0.0))

    # Build LaTeX table with dynamic columns
    header_cells = ' & '.join([f"\\textbf{{{s}}}" for s in method_short])
    col_format = '|l|' + 'c|' * n_methods
    rows = []
    for i, name in enumerate(full_names):
        row_vals = ' & '.join(f"{corr_matrix[i, j]:.3f}" for j in range(n_methods))
        rows.append(f"{name} & {row_vals} \\\\ \hline")

    latex_table = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\caption{Correlation Matrix of Method Accuracies}\n"
        "\\label{tab:method_correlations}\n"
        f"\\begin{{tabular}}{{{col_format}}}\n"
        "\\hline\n"
        f"\\textbf{{Method}} & {header_cells} \\ \n"
        "\\hline\n"
        + '\n'.join(rows)
        + "\n\\end{tabular}\n\\end{table}"
    )

    with open('reports/agreement_table.tex', 'w') as f:
        f.write(latex_table)

    print("Results saved to reports/agreement_results.txt")
    print("LaTeX table saved to reports/agreement_table.tex")