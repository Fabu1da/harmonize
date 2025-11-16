#!/usr/bin/env python3
import dotenv

dotenv.load_dotenv(override=True)

import os
import json
import glob
from itertools import combinations
from collections import defaultdict

from tqdm import tqdm
from scipy.stats import binomtest
from config import APPROACH_NAMES, APPROACHES

APPROACH_NAMES = APPROACH_NAMES


def mcnemar_counts(corr_A, corr_B):
    """
    Given correctness vectors for A and B (dict[target] -> 0/1),
    return n01 (A wrong, B right), n10 (A right, B wrong).
    """
    n01 = 0
    n10 = 0
    for t in corr_A.keys() & corr_B.keys():
        a = corr_A[t]
        b = corr_B[t]
        if a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
    return n01, n10


def mcnemar_from_counts(n01, n10):
    """
    Compute McNemar statistics from aggregated counts.
    Returns dict with chi2, p_value, winner ("A"/"B"/"tie").
    """
    n = n01 + n10
    if n == 0:
        return {
            "chi2": 0.0,
            "p_value": 1.0,
            "winner": "tie",
        }

    chi2 = ((abs(n01 - n10) - 1) ** 2) / n
    p = binomtest(
        min(n01, n10),
        n,
        p=0.5,
        alternative="two-sided"
    ).pvalue

    if n10 > n01:
        winner = "A"   # A has more unique corrects
    elif n01 > n10:
        winner = "B"   # B has more unique corrects
    else:
        winner = "tie"

    return {
        "chi2": chi2,
        "p_value": p,
        "winner": winner,
    }


def invert_predictions_to_target_to_source(predictions):
    """
    predictions: {source_col: [pred_target_col or None, ...]}
    Returns: {target_col: source_col}
    """
    inv = {}
    for src_col, values in predictions.items():
        if not values:
            continue
        tgt_col = values[0]
        if tgt_col is not None:
            inv[tgt_col] = src_col
    return inv


def correctness_vector(predictions, expected_mapping):
    """
    expected_mapping: {target_col: true_source_col}
    predictions: raw predictions dict for one approach/table.
    Returns: {target_col: 0/1}
    """
    if predictions is None:
        return {t: 0 for t in expected_mapping.keys()}

    pred_inv = invert_predictions_to_target_to_source(predictions)
    correct = {}
    for t, gt_src in expected_mapping.items():
        pred_src = pred_inv.get(t)
        correct[t] = int(pred_src == gt_src)
    return correct


def latex_escape(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("\\", "\\textbackslash{}")
         .replace("_", "\\_")
         .replace("%", "\\%")
         .replace("&", "\\&")
    )

def generate_latex_dataset(aggregated_results):
    """
    aggregated_results: list of dicts with keys:
      table, method1, method2, n01, n10, n, chi2, p_value, winner, significant
    """
    aggregated_results = sorted(
        aggregated_results,
        key=lambda r: (r["method1"], r["method2"], r["dataset"])
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{McNemar's test per table between Edit Distance and Random (column-wise correctness).}",
        "\\label{tab:mcnemar-per-table}",
        "\\small",
        "\\begin{tabular}{l l c c c c c c}",
        "\\toprule",
        "Table & Method Pair & $n_{10}$ & $n_{01}$ & $n$ & $\\chi^2$ & p-value & Winner \\\\",
        "\\midrule",
    ]

    for r in aggregated_results:
        table_name = latex_escape(r["dataset"])
        pair = f"{r['method1']}~vs~{r['method2']}"
        winner = r["winner"]
        lines.append(
            f"{table_name} & {pair} & "
            f"{r['n10']} & {r['n01']} & {r['n']} & "
            f"{r['chi2']:.3f} & {r['p_value']:.4f} & {winner} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)


def generate_latex_table(aggregated_results):

    print(aggregated_results)

    """
    aggregated_results: list of dicts with keys:
      table, method1, method2, n01, n10, n, chi2, p_value, winner, significant
    """
    aggregated_results = sorted(
        aggregated_results,
        key=lambda r: (r["method1"], r["method2"], r["table"])
    )

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{McNemar's test per table between Edit Distance and Random (column-wise correctness).}",
        "\\label{tab:mcnemar-per-table}",
        "\\small",
        "\\begin{tabular}{l l c c c c c c}",
        "\\toprule",
        "Table & Method Pair & $n_{10}$ & $n_{01}$ & $n$ & $\\chi^2$ & p-value & Winner \\\\",
        "\\midrule",
    ]

    for r in aggregated_results:
        table_name = latex_escape(r["table"])
        pair = f"{r['method1']}~vs~{r['method2']}"
        winner = r["winner"]
        lines.append(
            f"{table_name} & {pair} & "
            f"{r['n10']} & {r['n01']} & {r['n']} & "
            f"{r['chi2']:.3f} & {r['p_value']:.4f} & {winner} \\\\"
        )

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])
    return "\n".join(lines)




def main():
    expected_paths = sorted(
        glob.glob("**/*.json", root_dir="./assets/expected", recursive=True)
    )

    # First, collect raw per-file counts
    per_file = []  # each: dict(table, method1, method2, n01, n10)

    for expected_rel in tqdm(expected_paths, desc="Tables"):
        expected_full = os.path.join("./assets/expected", expected_rel)

        with open(expected_full) as f:
            expectation = json.load(f)

        expected_mappings = expectation.get("mappings", [])
        expected_mapping = {
            m["target_column"]: m["source_column"]
            for m in expected_mappings
            if m.get("target_column") is not None
            and m.get("source_column") is not None
        }

        if not expected_mapping:
            continue

        table_corr = {}
        for approach in APPROACH_NAMES:
            predicted_full = os.path.join("./assets/predicted", approach, expected_rel)
            if not os.path.exists(predicted_full):
                continue
            with open(predicted_full) as f:
                predictions = json.load(f)
            table_corr[approach] = correctness_vector(predictions, expected_mapping)

        for A, B in combinations(APPROACH_NAMES, 2):
            if A not in table_corr or B not in table_corr:
                continue
            n01, n10 = mcnemar_counts(table_corr[A], table_corr[B])
            if n01 + n10 == 0:
                continue

            per_file.append({
                "table": expectation.get("target_table", expected_rel),
                "method1": A,
                "method2": B,
                "n01": n01,
                "n10": n10,
            })

    # Now aggregate by (table, method1, method2)
    agg = defaultdict(lambda: {"n01": 0, "n10": 0})

    for r in per_file:
        key = (r["table"], r["method1"], r["method2"])
        print(key)
        agg[key]["n01"] += r["n01"]
        agg[key]["n10"] += r["n10"]

    # print("###### Par table", agg)

    aggregated_results = []
    for (table, m1, m2), cnts in agg.items():
        n01 = cnts["n01"]
        n10 = cnts["n10"]
        n = n01 + n10
        if n == 0:
            continue

        stats = mcnemar_from_counts(n01, n10)

        # Map winner label from "A"/"B"/"tie" to method names
        if stats["winner"] == "A":
            winner_label = m1
        elif stats["winner"] == "B":
            winner_label = m2
        else:
            winner_label = "Tie"

        aggregated_results.append({
            "table": table,
            "method1": m1,
            "method2": m2,
            "n01": n01,
            "n10": n10,
            "n": n,
            "chi2": stats["chi2"],
            "p_value": stats["p_value"],
            "significant": stats["p_value"],
            "winner": winner_label,
        })


    # ====================== Per-dataset aggregation ======================

    agg_dataset = defaultdict(lambda: {"n01": 0, "n10": 0})

    for r in per_file:
        table_key = r["table"]  # e.g. "valentine/ChEMBL/Joinable/assays_both_50_1_ac1_ev"
        parts = table_key.split("/")

        # dataset is the segment right after "valentine"
        if len(parts) > 1 and parts[0] == "valentine":
            dataset = parts[1]
        else:
            # fallback if format is slightly different
            dataset = parts[0]

        key = (dataset, r["method1"], r["method2"])
        agg_dataset[key]["n01"] += r["n01"]
        agg_dataset[key]["n10"] += r["n10"]

    aggregated_results_dataset = []
    for (dataset, m1, m2), cnts in agg_dataset.items():
        n01 = cnts["n01"]
        n10 = cnts["n10"]
        n = n01 + n10
        if n == 0:
            continue

        stats = mcnemar_from_counts(n01, n10)

        if stats["winner"] == "A":
            winner_label = m1
        elif stats["winner"] == "B":
            winner_label = m2
        else:
            winner_label = "Tie"

        aggregated_results_dataset.append({
            "scope": "dataset",
            "dataset": dataset,
            "method1": m1,
            "method2": m2,
            "n01": n01,
            "n10": n10,
            "n": n,
            "chi2": stats["chi2"],
            "p_value": stats["p_value"],
            "significant": stats["p_value"],
            "winner": winner_label,
        })

    for r in sorted(aggregated_results_dataset, key=lambda x: (x["method1"], x["method2"], x["dataset"])):
        sig_marker = " ***" if r["significant"] else ""
        print(
            f"[{r['dataset']}] {r['method1']} vs {r['method2']}: "
            f"n10={r['n10']}, n01={r['n01']}, n={r['n']}, "
            f"chi2={r['chi2']:.3f}, p={r['p_value']:.4f}, "
            f"winner={r['winner']}{sig_marker}"
        )

    latex = generate_latex_dataset(aggregated_results_dataset)
    os.makedirs("reports", exist_ok=True)
    with open("reports/mcnemar_confidence_dataset.tex", "w") as f:
        f.write(latex)


    


    # Console summary
    for r in sorted(aggregated_results, key=lambda x: (x["method1"], x["method2"], x["table"])):
        sig_marker = " ***" if r["significant"] else ""
        # print(
        #     f"[{r['table']}] {r['method1']} vs {r['method2']}: "
        #     f"n10={r['n10']}, n01={r['n01']}, n={r['n']}, "
        #     f"chi2={r['chi2']:.3f}, p={r['p_value']:.4f}, "
        #     f"winner={r['winner']}{sig_marker}"
        # )

    # LaTeX table from aggregated results
    latex = generate_latex_table(aggregated_results)
    print("=" * 70)
    print("LATEX TABLE")
    print("=" * 70)
    # print(latex)

    os.makedirs("reports", exist_ok=True)
    with open("reports/mcnemar_confidence_table.tex", "w") as f:
        f.write(latex)

    sig_count = sum(1 for r in aggregated_results if r["significant"])
    print(f"\nSummary: {sig_count}/{len(aggregated_results)} aggregated comparisons significant (p < 0.05)")


if __name__ == "__main__":
    main()
