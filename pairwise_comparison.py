from typing import Any, Dict, Optional, Tuple, List
import pandas as pd
from dataclasses import dataclass
from tabulate import tabulate
import json
import os

@dataclass
class PairwiseMetrics:
    """Metrics for comparing two matchers"""
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    false_negatives: int
    agreement_rate: float
    
    
def calculate_triple_comparison_metrics(
    gpt_predictions: Dict[str, Tuple[str, float]],
    embedding_predictions: Dict[str, Tuple[str, float]], 
    clustering_predictions: Dict[str, Tuple[str, float]],
    ground_truth: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """
    Perform triple comparison analysis where all three matchers are compared
    against ground truth to identify agreement/disagreement patterns.
    
    Returns:
        Dictionary containing triple comparison metrics and patterns
    """
    if not ground_truth:
        return {"error": "Ground truth required for triple comparison"}
    
    # Get all target columns
    all_columns = set()
    all_columns.update(gpt_predictions.keys())
    all_columns.update(embedding_predictions.keys()) 
    all_columns.update(clustering_predictions.keys())
    all_columns.update(ground_truth.keys())
    
    # Analysis patterns
    patterns = {
        'all_correct': [],           # All 3 matchers correct
        'two_correct': [],           # Exactly 2 matchers correct
        'one_correct': [],           # Exactly 1 matcher correct  
        'none_correct': [],          # All 3 matchers wrong
        'all_agree_correct': [],     # All agree and correct
        'all_agree_wrong': [],       # All agree but wrong
        'majority_correct': [],      # Majority vote correct
        'majority_wrong': [],        # Majority vote wrong
    }
    
    detailed_results = []
    
    for col in all_columns:
        if col not in ground_truth:
            continue
            
        gt_source = ground_truth[col]
        
        # Get predictions (default to no match if missing)
        gpt_tuple = gpt_predictions.get(col, ("—", 0.0))
        if len(gpt_tuple) == 2:
            gpt_pred, gpt_conf = gpt_tuple
            gpt_reasoning = "Not available"
        elif len(gpt_tuple) == 3:
            gpt_pred, gpt_conf, gpt_reasoning = gpt_tuple
        else:
            gpt_pred, gpt_conf, gpt_reasoning = "—", 0.0, "Not available"
            
        emb_pred, emb_conf = embedding_predictions.get(col, ("—", 0.0))
        clust_pred, clust_conf = clustering_predictions.get(col, ("—", 0.0))
        
        # Check correctness
        gpt_correct = (gpt_pred == gt_source)
        emb_correct = (emb_pred == gt_source) 
        clust_correct = (clust_pred == gt_source)
        
        correct_count = sum([gpt_correct, emb_correct, clust_correct])
        
        # Check agreement (all three predict same source)
        all_agree = (gpt_pred == emb_pred == clust_pred) and gpt_pred != "—"
        
        # Majority vote
        predictions = [gpt_pred, emb_pred, clust_pred]
        confidences = [gpt_conf, emb_conf, clust_conf]
        
        # Simple majority (most frequent prediction)
        from collections import Counter
        pred_counts = Counter(p for p in predictions if p != "—")
        majority_pred = pred_counts.most_common(1)[0][0] if pred_counts else "—"
        majority_correct = (majority_pred == gt_source)
        
        # Store detailed result
        result = {
            'target_column': col,
            'ground_truth': gt_source,
            'gpt_prediction': gpt_pred,
            'gpt_confidence': gpt_conf,
            'gpt_reasoning': gpt_reasoning,
            'gpt_correct': gpt_correct,
            'embedding_prediction': emb_pred,
            'embedding_confidence': emb_conf,
            'embedding_correct': emb_correct,
            'clustering_prediction': clust_pred,
            'clustering_confidence': clust_conf,
            'clustering_correct': clust_correct,
            'correct_count': correct_count,
            'all_agree': all_agree,
            'majority_prediction': majority_pred,
            'majority_correct': majority_correct
        }
        detailed_results.append(result)
        
        # Categorize patterns
        if correct_count == 3:
            patterns['all_correct'].append(col)
        elif correct_count == 2:
            patterns['two_correct'].append(col)
        elif correct_count == 1:
            patterns['one_correct'].append(col)
        else:
            patterns['none_correct'].append(col)
            
        if all_agree:
            if gpt_correct:  # If they all agree, they're all either correct or wrong
                patterns['all_agree_correct'].append(col)
            else:
                patterns['all_agree_wrong'].append(col)
                
        if majority_correct:
            patterns['majority_correct'].append(col)
        else:
            patterns['majority_wrong'].append(col)
    
    # Calculate summary statistics
    total_columns = len(detailed_results)
    
    # Calculate P/R/F1 for each matcher using the same logic as pairwise comparison
    def calculate_matcher_metrics(predictions_dict, ground_truth_dict, matcher_name):
        """Calculate P/R/F1 for a single matcher against ground truth"""
        # Filter valid predictions (same as pairwise)
        valid_predictions = {k: v for k, v in predictions_dict.items() if v[0] is not None and v[0] != "—"}
        common_columns = set(valid_predictions.keys()) & set(ground_truth_dict.keys())
        
        true_positives = 0
        false_positives = 0 
        false_negatives = 0
        
        for col in common_columns:
            true_match = ground_truth_dict.get(col)
            predicted_match = valid_predictions.get(col, (None, 0.0))[0]
            
            if predicted_match == true_match and true_match is not None:
                true_positives += 1
            elif predicted_match is not None and predicted_match != true_match:
                false_positives += 1
            elif true_match is not None and (predicted_match is None or predicted_match != true_match):
                false_negatives += 1
        
        # Same calculation as pairwise
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1_score, true_positives, false_positives, false_negatives
    
    # Calculate for each individual matcher
    gpt_p, gpt_r, gpt_f1, gpt_tp, gpt_fp, gpt_fn = calculate_matcher_metrics(gpt_predictions, ground_truth, "GPT")
    emb_p, emb_r, emb_f1, emb_tp, emb_fp, emb_fn = calculate_matcher_metrics(embedding_predictions, ground_truth, "Embedding")
    clust_p, clust_r, clust_f1, clust_tp, clust_fp, clust_fn = calculate_matcher_metrics(clustering_predictions, ground_truth, "Clustering")
    
    # Calculate for majority vote (create majority predictions dict)
    majority_predictions = {}
    for result in detailed_results:
        col = result['target_column']
        majority_pred = result['majority_prediction']
        majority_predictions[col] = (majority_pred, 1.0)  # Use confidence 1.0 for majority
    
    maj_p, maj_r, maj_f1, maj_tp, maj_fp, maj_fn = calculate_matcher_metrics(majority_predictions, ground_truth, "Majority")
    
    summary_stats = {
        'total_columns': total_columns,
        'all_correct_rate': len(patterns['all_correct']) / total_columns if total_columns > 0 else 0,
        'two_correct_rate': len(patterns['two_correct']) / total_columns if total_columns > 0 else 0,
        'one_correct_rate': len(patterns['one_correct']) / total_columns if total_columns > 0 else 0,
        'none_correct_rate': len(patterns['none_correct']) / total_columns if total_columns > 0 else 0,
        'agreement_rate': len([r for r in detailed_results if r['all_agree']]) / total_columns if total_columns > 0 else 0,
        'agreement_accuracy': len(patterns['all_agree_correct']) / len([r for r in detailed_results if r['all_agree']]) if len([r for r in detailed_results if r['all_agree']]) > 0 else 0,
        'majority_accuracy': len(patterns['majority_correct']) / total_columns if total_columns > 0 else 0,
        # Add P/R/F1 metrics
        'gpt_precision': gpt_p,
        'gpt_recall': gpt_r,
        'gpt_f1_score': gpt_f1,
        'embedding_precision': emb_p,
        'embedding_recall': emb_r,
        'embedding_f1_score': emb_f1,
        'clustering_precision': clust_p,
        'clustering_recall': clust_r,
        'clustering_f1_score': clust_f1,
        'majority_precision': maj_p,
        'majority_recall': maj_r,
        'majority_f1_score': maj_f1,
        # Add combined/average metrics
        'combined_precision': (gpt_p + emb_p + clust_p) / 3,
        'combined_recall': (gpt_r + emb_r + clust_r) / 3,
        'combined_f1_score': (gpt_f1 + emb_f1 + clust_f1) / 3,
    }
    
    return {
        'patterns': patterns,
        'detailed_results': detailed_results,
        'summary_stats': summary_stats
    }

def print_triple_comparison_table(triple_results: Dict[str, Any], source_table: str, target_table: str):
    """Print formatted table showing triple comparison results."""
    if 'error' in triple_results:
        print(f"❌ {triple_results['error']}")
        return
    
    print(f"\n🔄 TRIPLE COMPARISON: {source_table} → {target_table}")
    print("=" * 80)
    
    # Summary statistics
    stats = triple_results['summary_stats']
    print(f"📊 Summary Statistics:")
    print(f"   All 3 Correct: {stats['all_correct_rate']:.1%}")
    print(f"   2/3 Correct:   {stats['two_correct_rate']:.1%}")
    print(f"   1/3 Correct:   {stats['one_correct_rate']:.1%}")
    print(f"   0/3 Correct:   {stats['none_correct_rate']:.1%}")
    print(f"   Agreement Rate: {stats['agreement_rate']:.1%}")
    print(f"   Agreement Accuracy: {stats['agreement_accuracy']:.1%}")
    print(f"   Majority Vote Accuracy: {stats['majority_accuracy']:.1%}")
    
    # Display P/R/F1 metrics (combined average)
    print(f"\n📈 Combined Matcher Performance:")
    print(f"   Average:    P={stats.get('combined_precision', 0):.3f}, R={stats.get('combined_recall', 0):.3f}, F1={stats.get('combined_f1_score', 0):.3f}")
    print(f"\n🗳️ Ensemble Performance:")
    print(f"   Majority Vote: P={stats.get('majority_precision', 0):.3f}, R={stats.get('majority_recall', 0):.3f}, F1={stats.get('majority_f1_score', 0):.3f}")
    
    
    # Detailed table
    table_data = []
    headers = [
        "Target", "Ground Truth",
        "GPT Pred", "GPT ✓", "GPT Conf", "GPT Reasoning",
        "Emb Pred", "Emb ✓", "Emb Conf", 
        "Clust Pred", "Clust ✓", "Clust Conf",
        "Correct Count", "All Agree", "Majority ✓"
    ]
    
    for result in triple_results['detailed_results']:
        # Truncate reasoning for table readability
        reasoning_truncated = result.get('gpt_reasoning', 'N/A')
        if len(reasoning_truncated) > 50:
            reasoning_truncated = reasoning_truncated[:50] + "..."
            
        row = [
            result['target_column'],
            result['ground_truth'],
            result['gpt_prediction'],
            "✅" if result['gpt_correct'] else "❌",
            f"{result['gpt_confidence']:.2f}",
            reasoning_truncated,
            result['embedding_prediction'], 
            "✅" if result['embedding_correct'] else "❌",
            f"{result['embedding_confidence']:.2f}",
            result['clustering_prediction'],
            "✅" if result['clustering_correct'] else "❌", 
            f"{result['clustering_confidence']:.2f}",
            f"{result['correct_count']}/3",
            "✅" if result['all_agree'] else "❌",
            "✅" if result['majority_correct'] else "❌"
        ]
        table_data.append(row)
    
    from tabulate import tabulate
    print(f"\n📋 Detailed Triple Comparison:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # P/R/F1 Summary Table (single combined row)
    print(f"\n📊 Combined Precision, Recall, F1-Score:")
    metrics_table_data = [
        ["Combined Average", f"{stats.get('combined_precision', 0):.3f}", f"{stats.get('combined_recall', 0):.3f}", f"{stats.get('combined_f1_score', 0):.3f}"]
    ]
    metrics_headers = ["Matcher", "Precision", "Recall", "F1-Score"]
    print(tabulate(metrics_table_data, headers=metrics_headers, tablefmt="grid"))

def run_triple_comparison(
    gpt_predictions: Dict[str, Tuple[str, float]],
    embedding_predictions: Dict[str, Tuple[str, float]],
    clustering_predictions: Dict[str, Tuple[str, float]], 
    ground_truth: Optional[Dict[str, str]] = None,
    source_table: str = "",
    target_table: str = ""
) -> Dict[str, Any]:
    """
    Run complete triple comparison analysis and display results.
    
    Returns:
        Triple comparison results dictionary
    """
    # Calculate triple comparison metrics
    triple_results = calculate_triple_comparison_metrics(
        gpt_predictions, embedding_predictions, clustering_predictions, ground_truth
    )
    
    # Display results
    print_triple_comparison_table(triple_results, source_table, target_table)
    
    return triple_results






def calculate_pairwise_metrics(
    matcher1_predictions: Dict[str, Tuple[str, float]], 
    matcher2_predictions: Dict[str, Tuple[str, float]],
    matcher1_name: str,
    matcher2_name: str,
    ground_truth: Dict[str, str] = None
) -> PairwiseMetrics:
    """
    Calculate P, R, F1 for two matchers.
    
    If ground_truth is provided, uses it as reference.
    Otherwise, treats matcher1 as the reference (ground truth).
    """
    
    # Get all target columns that both matchers have predictions for
    # Filter out None predictions
    matcher1_valid = {k: v for k, v in matcher1_predictions.items() if v[0] is not None}
    matcher2_valid = {k: v for k, v in matcher2_predictions.items() if v[0] is not None}
    
    common_columns = set(matcher1_valid.keys()) & set(matcher2_valid.keys())
    
    # ADD THIS DEBUG BLOCK
    print(f"\n🔍 DEBUG: {matcher1_name} vs {matcher2_name}")
    print(f"   {matcher1_name} total: {len(matcher1_predictions)}, valid: {len(matcher1_valid)}")
    print(f"   {matcher2_name} total: {len(matcher2_predictions)}, valid: {len(matcher2_valid)}")
    print(f"   Common columns: {len(common_columns)}")
    if len(common_columns) <= 5:  # Show details if small number
        print(f"   Common: {sorted(common_columns)}")
        print(f"   {matcher1_name} keys: {sorted(matcher1_valid.keys())}")
        print(f"   {matcher2_name} keys: {sorted(matcher2_valid.keys())}")
    
    if len(common_columns) == 0:
        print(f"   ⚠️ No common predictions between {matcher1_name} and {matcher2_name}")
        return PairwiseMetrics(0.0, 0.0, 0.0, 0, 0, 0, 0.0)
    
    if ground_truth:
        # Use actual ground truth as reference
        reference = ground_truth
        common_columns = common_columns & set(ground_truth.keys())
    else:
        # Use matcher1 as reference (matcher1 vs matcher2 agreement)
        reference = {col: pred[0] for col, pred in matcher1_valid.items()}
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_agreements = 0
    
    for col in common_columns:
        if ground_truth:
            # Ground truth evaluation
            true_match = reference.get(col)
            matcher2_match = matcher2_valid.get(col, (None, 0.0))[0]
            
            if matcher2_match == true_match and true_match is not None:
                true_positives += 1
            elif matcher2_match is not None and matcher2_match != true_match:
                false_positives += 1
            elif true_match is not None and (matcher2_match is None or matcher2_match != true_match):
                false_negatives += 1
        else:
            # Matcher agreement evaluation
            matcher1_match = matcher1_valid.get(col, (None, 0.0))[0]
            matcher2_match = matcher2_valid.get(col, (None, 0.0))[0]
            
            if matcher1_match == matcher2_match and matcher1_match is not None:
                total_agreements += 1
    
    if ground_truth:
        # Standard P, R, F1 calculation
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        agreement_rate = true_positives / len(common_columns) if common_columns else 0.0
    else:
        # Agreement-based metrics
        agreement_rate = total_agreements / len(common_columns) if common_columns else 0.0
        # For agreement, precision = recall = f1 = agreement rate
        precision = recall = f1_score = agreement_rate
        true_positives = total_agreements
        false_positives = len(common_columns) - total_agreements
        false_negatives = 0
    
    return PairwiseMetrics(
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        agreement_rate=agreement_rate
    )
    
def run_all_pairwise_comparisons(
    gpt_predictions: Dict[str, Tuple[str, float]],
    embedding_predictions: Dict[str, Tuple[str, float]], 
    clustering_predictions: Dict[str, Tuple[str, float]],
    ground_truth: Dict[str, str] = None,
    source_table: str = "",
    target_table: str = ""
) -> Tuple[Dict[str, PairwiseMetrics], Dict]:
    """
    Run all three pairwise comparisons as shown in your sketch Step 2
    Returns comparisons and result entry for storing
    """
    
    comparisons = {}
    
    # GPT vs Embedding
    comparisons["GPT_vs_Embedding"] = calculate_pairwise_metrics(
        gpt_predictions, embedding_predictions, "GPT", "Embedding", ground_truth
    )
    
    # GPT vs Clustering
    comparisons["GPT_vs_Clustering"] = calculate_pairwise_metrics(
        gpt_predictions, clustering_predictions, "GPT", "Clustering", ground_truth
    )
    
    # Embedding vs Clustering
    comparisons["Embedding_vs_Clustering"] = calculate_pairwise_metrics(
        embedding_predictions, clustering_predictions, "Embedding", "Clustering", ground_truth
    )
    
    # Store results for aggregation
    result_entry = {
        "source_table": source_table,
        "target_table": target_table,
        "comparisons": comparisons,
        "has_ground_truth": ground_truth is not None
    }
    
    return comparisons, result_entry
    
def print_pairwise_comparison_table(
    comparisons: Dict[str, PairwiseMetrics],
    source_table: str = "",
    target_table: str = ""
):
        """Print a formatted table of pairwise comparison results"""
        
        table_data = []
        for comparison_name, metrics in comparisons.items():
            table_data.append([
                comparison_name.replace("_", " "),
                f"{metrics.precision:.3f}",
                f"{metrics.recall:.3f}",
                f"{metrics.f1_score:.3f}",
                f"{metrics.agreement_rate:.3f}",
                f"{metrics.true_positives}/{metrics.true_positives + metrics.false_positives + metrics.false_negatives}"
            ])
        
        headers = ["Comparison", "Precision", "Recall", "F1 Score", "Agreement", "TP/Total"]
        
        print(f"\n🔄 Pairwise Matcher Comparison: {source_table} → {target_table}")
        print("=" * 80)
        print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
        
        return table_data, headers
    
def export_pairwise_results(comparison_results: List[Dict], filename: str = "pairwise_comparison_results"):
        """Export all pairwise comparison results"""
        
        # Create summary table across all datasets
        summary_data = []
        
        for result in comparison_results:
            source = result["source_table"]
            target = result["target_table"]
            
            for comp_name, metrics in result["comparisons"].items():
                summary_data.append([
                    source,
                    target,
                    comp_name.replace("_", " "),
                    f"{metrics.precision:.3f}",
                    f"{metrics.recall:.3f}",
                    f"{metrics.f1_score:.3f}",
                    f"{metrics.agreement_rate:.3f}",
                    "Yes" if result["has_ground_truth"] else "No"
                ])
        
        headers = ["Source", "Target", "Comparison", "Precision", "Recall", "F1", "Agreement", "GT Available"]
        
        # Export as JSON
        os.makedirs("output", exist_ok=True)
        with open(f"output/{filename}.json", "w") as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        # Export summary table
        print(f"\n📊 COMPLETE PAIRWISE COMPARISON SUMMARY")
        print("=" * 100)
        print(tabulate(summary_data, headers=headers, tablefmt="fancy_grid"))
        
        # Export as LaTeX
        _export_latex_table(summary_data, headers, f"{filename}_summary.tex")
        
        # Calculate overall averages
        _calculate_overall_averages(comparison_results)
        
        return summary_data, headers
    
def _export_latex_table(data, headers, filename):
        """Export pairwise results as LaTeX table"""
        os.makedirs("output", exist_ok=True)
        filepath = os.path.join("output", filename)
        
        with open(filepath, 'w') as f:
            num_cols = len(headers)
            col_spec = 'l' * num_cols
            
            f.write("\\begin{landscape}\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\small\n")
            f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
            f.write("\\toprule\n")
            
            # Headers
            header_row = " & ".join(headers) + " \\\\\n"
            f.write(header_row)
            f.write("\\midrule\n")
            
            # Data rows
            for row in data:
                cleaned_row = []
                for cell in row:
                    cell_str = str(cell).replace('&', '\\&').replace('_', '\\_').replace('%', '\\%')
                    cleaned_row.append(cell_str)
                data_row = " & ".join(cleaned_row) + " \\\\\n"
                f.write(data_row)
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Pairwise matcher comparison results across all datasets}\n")
            f.write("\\label{tab:pairwise_comparison}\n")
            f.write("\\end{table}\n")
            f.write("\\end{landscape}\n")
        
        print(f"LaTeX pairwise comparison table saved to {filepath}")
    
def _calculate_overall_averages(comparison_results: List[Dict]):
        """Calculate and display overall average metrics across all comparisons"""
        
        comparison_types = ["GPT_vs_Embedding", "GPT_vs_Clustering", "Embedding_vs_Clustering"]
        
        averages_table = []
        
        for comp_type in comparison_types:
            precisions = []
            recalls = []
            f1_scores = []
            agreements = []
            
            for result in comparison_results:
                if comp_type in result["comparisons"]:
                    metrics = result["comparisons"][comp_type]
                    precisions.append(metrics.precision)
                    recalls.append(metrics.recall)
                    f1_scores.append(metrics.f1_score)
                    agreements.append(metrics.agreement_rate)
            
            if precisions:  # Only calculate if we have data
                avg_precision = sum(precisions) / len(precisions)
                avg_recall = sum(recalls) / len(recalls)
                avg_f1 = sum(f1_scores) / len(f1_scores)
                avg_agreement = sum(agreements) / len(agreements)
                
                averages_table.append([
                    comp_type.replace("_", " "),
                    f"{avg_precision:.3f}",
                    f"{avg_recall:.3f}",
                    f"{avg_f1:.3f}",
                    f"{avg_agreement:.3f}",
                    len(precisions)
                ])
        
      
        
        return averages_table

def run_complete_pairwise_analysis(
    datasets: List[Dict],
    export_filename: str = "pairwise_comparison_results"
) -> List[Dict]:
    """
    Run complete pairwise analysis on multiple datasets
    
    Args:
        datasets: List of dictionaries containing:
            - gpt_predictions: Dict[str, Tuple[str, float]]
            - embedding_predictions: Dict[str, Tuple[str, float]]
            - clustering_predictions: Dict[str, Tuple[str, float]]
            - ground_truth: Dict[str, str] (optional)
            - source_table: str
            - target_table: str
        export_filename: Name for output files
    
    Returns:
        List of all comparison results
    """
    all_results = []
    
    for dataset in datasets:
        comparisons, result_entry = run_all_pairwise_comparisons(
            gpt_predictions=dataset.get('gpt_predictions', {}),
            embedding_predictions=dataset.get('embedding_predictions', {}),
            clustering_predictions=dataset.get('clustering_predictions', {}),
            ground_truth=dataset.get('ground_truth'),
            source_table=dataset.get('source_table', ''),
            target_table=dataset.get('target_table', '')
        )
        
        all_results.append(result_entry)
        
        # Print individual comparison for this dataset
        print_pairwise_comparison_table(
            comparisons,
            dataset.get('source_table', ''),
            dataset.get('target_table', '')
        )
    
    # Export all results
    if all_results:
        export_pairwise_results(all_results, export_filename)
    
    return all_results

