#!/usr/bin/env python3
import json
import glob
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

# ─── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Load ground-truth ─────────────────────────────────────────────────────────
def load_expected(pattern: str) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    recs = []
    for fp in files:
        data = json.load(open(fp, encoding="utf-8"))
        # derive table key from filename (or from data["source_table"] if you prefer)
        tbl = Path(fp).stem.lower()
        # strip leading "www." if you want
        tbl = tbl.removeprefix("www.")
        maps = data.get("mappings") or data.get("mapping")
        if maps is None:
            raise KeyError(f"{fp} has neither 'mappings' nor 'mapping'")
        for m in maps:
            recs.append({
                "source_table":  tbl,
                "source_column": m["source_column"].lower(),
                "target_column": m["target_column"].lower(),
                "origin_file":   Path(fp).name,
            })
    df = pd.DataFrame(recs)
    logger.info("Loaded %d GT mappings from %d files",
                len(df), df.origin_file.nunique())
    # duplicate check
    dup = df.duplicated(subset=["source_table","source_column","target_column"], keep=False)
    if dup.any():
        logger.warning("Duplicate GT mappings:\n%s", df[dup])
    return df

# ─── Load predictions ─────────────────────────────────────────────────────────
def load_predictions(path: str) -> pd.DataFrame:
    raw = json.load(open(path, encoding="utf-8"))
    recs = []
    for rec in raw:
        tbl = rec["source_table"].lower()
        tbl = tbl.removeprefix("www.")  # same normalization
        maps = rec.get("mappings") or rec.get("mapping")
        if maps is None:
            raise KeyError(f"Record for {tbl} missing 'mappings'/'mapping'")
        for m in maps:
            recs.append({
                "source_table":  tbl,
                "source_column": m["source_column"].lower(),
                "target_column": m["target_column"].lower(),
                "score":         float(m["similarity"]),
            })
    df = pd.DataFrame(recs)
    logger.info("Loaded %d predicted mappings", len(df))
    return df

# ─── Labeling & debug ──────────────────────────────────────────────────────────
def attach_labels(df_pred: pd.DataFrame, df_gt: pd.DataFrame) -> pd.DataFrame:
    # debug: show which tables don’t align
    pred_tables = set(df_pred.source_table)
    gt_tables   = set(df_gt.source_table)
    missing_gt  = pred_tables - gt_tables
    missing_pr  = gt_tables   - pred_tables
    if missing_gt:
        logger.warning("Prediction tables not in GT: %s", missing_gt)
    if missing_pr:
        logger.warning("GT tables not in predictions: %s", missing_pr)

    gt_set = set(zip(
        df_gt.source_table,
        df_gt.source_column,
        df_gt.target_column,
    ))
    df = df_pred.copy()
    df["true_label"] = df.apply(
        lambda r: int((r.source_table, r.source_column, r.target_column) in gt_set),
        axis=1
    )
    pos = df.true_label.sum()
    logger.info("Marked %d positives and %d negatives",
                pos, len(df)-pos)
    return df

# ─── Metrics ────────────────────────────────────────────────────────────────────
def sweep_metrics(df: pd.DataFrame, n_steps: int = 100):
    y_true  = df.true_label.values
    y_score = df.score.values

    precision, recall, thresh = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    df_pr = pd.DataFrame({
        "threshold": thresh,
        "precision": precision[:-1],
        "recall":    recall[:-1],
    })
    df_pr["f1"] = 2 * (df_pr.precision * df_pr.recall) / (
        np.clip(df_pr.precision + df_pr.recall, 1e-8, None)
    )

    # sample evenly if there are more points than requested
    if len(df_pr) > n_steps:
        idx = np.linspace(0, len(df_pr)-1, n_steps).round().astype(int)
        df_pr = df_pr.iloc[idx].reset_index(drop=True)
    return df_pr, ap

# ─── Plot ───────────────────────────────────────────────────────────────────────
def plot_pr_curve(df_pr: pd.DataFrame, ap: float):
    plt.figure(figsize=(6,6))
    plt.plot(df_pr.recall, df_pr.precision, lw=2)
    plt.title(f"Precision–Recall Curve (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ─── Main ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("-e","--expected", default="expected/*.json",
                   help="glob pattern for GT JSONs")
    p.add_argument("-p","--pred",      default="detailed_matches.json",
                   help="pipeline output JSON")
    p.add_argument("-s","--steps",     type=int, default=50,
                   help="how many thresholds to sample")
    p.add_argument("--plot",           action="store_true",
                   help="show PR curve")
    p.add_argument("-o","--out",
                   help="where to write threshold-vs-metrics CSV")
    args = p.parse_args()

    df_gt   = load_expected(args.expected)
    df_pred = load_predictions(args.pred)
    df_all  = attach_labels(df_pred, df_gt)

    df_pr, ap = sweep_metrics(df_all, n_steps=args.steps)
    logger.info("Average Precision (area under PR curve): %.3f", ap)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_pr.to_csv(out_path, index=False)
        logger.info("Wrote metrics to %s", out_path)

    print("\nSample metrics:\n", df_pr.head(10).to_string(index=False))
    if args.plot:
        plot_pr_curve(df_pr, ap)

if __name__ == "__main__":
    main()
