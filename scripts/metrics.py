import dotenv

dotenv.load_dotenv(override=True)

import os
import glob
import json
import logging
import pandas as pd
from tqdm import tqdm
from typing import TypedDict

from config import APPROACH_NAMES, APPROACHES

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Metrics(TypedDict):
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float

APPROACH_NAMES = APPROACH_NAMES


from itertools import combinations

# def pairwise_metrics(predictions_by_model):
#     """
#     predictions_by_model: dict of {model_name: set_of_predictions}
#     Returns: list of dicts with pairwise precision, recall, f1
#     """
#     results = []
#     for A, B in combinations(predictions_by_model.keys(), 2):
#         preds_A = predictions_by_model[A]
#         preds_B = predictions_by_model[B]

#         tp = len(preds_A & preds_B)
#         fp = len(preds_A - preds_B)
#         fn = len(preds_B - preds_A)

#         precision = tp / (tp + fp) if (tp + fp) else 0
#         recall = tp / (tp + fn) if (tp + fn) else 0
#         f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

#         results.append({
#             "model1": A,
#             "model2": B,
#             "tp": tp,
#             "fp": fp,
#             "fn": fn,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#         })
#     return results



def get_metric(predictions: dict, expected_mapping: dict) -> Metrics:
    """Compute TP/FP/FN + precision/recall/F1 for one table."""
    tp = fp = fn = 0

    for target_col, expected_source in expected_mapping.items():
        if target_col not in predictions:
            fn += 1
            continue

        try:
            predicted_source, raw_confidence, reasoning = predictions[target_col]
        except (TypeError, ValueError):
            fn += 1
            continue

        if predicted_source == expected_source:
            tp += 1
        else:
            fp += 1
            fn += 1

    for target_col in predictions.keys():
        if target_col not in expected_mapping:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return Metrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1)


def get_metrics():
    

    expected_paths = sorted(
        glob.glob("**/*.json", root_dir="./assets/expected", recursive=True)
    )

    # Global totals per approach
    global_counts: dict[str, dict[str, int]] = {
        a: {"tp": 0, "fp": 0, "fn": 0} for a in APPROACH_NAMES
    }

    total = 0

    for expected_rel in tqdm(expected_paths, desc="Evaluating Tables"):
        expected_full = os.path.join("./assets/expected", expected_rel)
        with open(expected_full) as f:
            expectation = json.load(f)

        expected_mappings = expectation.get("mappings", [])
        expected_mapping = {
            m["target_column"]: m["source_column"]
            for m in expected_mappings
            if m.get("target_column") and m.get("source_column")
        }

        if not expected_mapping:
            continue

        for approach in APPROACH_NAMES:
            predicted_full = os.path.join("./assets/predicted", approach, expected_rel)
            if not os.path.exists(predicted_full):
                continue

            with open(predicted_full) as f:
                predictions = json.load(f)

            metrics = get_metric(predictions["per_columns"], expected_mapping)

            global_counts[approach]["tp"] += metrics["tp"]
            global_counts[approach]["fp"] += metrics["fp"]
            global_counts[approach]["fn"] += metrics["fn"]


            # pair_comparison = pairwise_metrics(predictions)
    print("*****", total)

    # Compute final global metrics for each approach
    global_rows = []
    for approach, counts in global_counts.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        global_rows.append(
            {
                "Approach": approach,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3),
            }
        )


    df_global = pd.DataFrame(global_rows)

    # --- Generate LaTeX output ---
    os.makedirs("./results", exist_ok=True)
    latex_global = df_global.to_latex(
        index=False,
        caption="Global evaluation summary across all tables.",
        label="tab:global_summary",
        column_format="l r r r r r r",
        float_format="%.3f",
        escape=False,
    )

    with open("./results/global_summary.tex", "w") as f:
        f.write(latex_global)

    logger.info("✅ Exported global LaTeX table to ./results/global_summary.tex")


