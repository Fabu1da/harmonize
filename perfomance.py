from typing import List, Dict, Union
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime

from tabulate import tabulate
from settings.gt_format import load_ground_truth_from_matches




def performance_report(detailed_matches: List[Dict], ground_truth: Union[Dict[str, str], Dict, str] = None):
    """
    Generate comprehensive performance report from detailed matches.
    
    Args:
        detailed_matches: List of dicts with approach names as keys and predictions as values
        ground_truth: Ground truth in your format (dict with 'matches'), simple mapping, or file path
    """
    
    
    # Handle your ground truth format
    if ground_truth is not None:
        if isinstance(ground_truth, dict) and 'matches' in ground_truth:
            ground_truth = load_ground_truth_from_matches(ground_truth)
        elif isinstance(ground_truth, str):
            ground_truth = load_ground_truth_from_matches(ground_truth)
    else:
        print("🔍 DEBUG: No ground truth provided!")
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE PERFORMANCE REPORT")
    print("="*80)
    
    # 1. Summary Performance Table
    print("\n" + "─"*80)
    print("TABLE 1: APPROACH PERFORMANCE SUMMARY")
    print("─"*80)
    generate_summary_table(detailed_matches, ground_truth)
    
    # 2. Detailed Column Mapping Table
    print("\n" + "─"*80)
    print("TABLE 2: DETAILED COLUMN MAPPINGS")
    print("─"*80)
    generate_detailed_mapping_table(detailed_matches, ground_truth)
    
    # 3. Confidence Score Analysis
    print("\n" + "─"*80)
    print("TABLE 3: CONFIDENCE SCORE ANALYSIS")
    print("─"*80)
    generate_confidence_analysis_table(detailed_matches)
    
    # 4. Type Awareness Analysis
    print("\n" + "─"*80)
    print("TABLE 4: TYPE AWARENESS ANALYSIS")
    print("─"*80)
    generate_type_awareness_table(detailed_matches)
    
    # 5. Agreement Analysis (for ensembles)
    print("\n" + "─"*80)
    print("TABLE 5: COMPONENT AGREEMENT MATRIX")
    print("─"*80)
    generate_agreement_matrix(detailed_matches)

    # 6. Comprehensive Error Analysis
    if ground_truth:
        print("\n" + "─"*80)
        print("TABLE 6: COMPREHENSIVE ERROR ANALYSIS")
        print("─"*80)
        try:
            from analysis.error_analysis import run_error_analysis
            print("🔍 Running comprehensive error analysis...")
            error_analysis_results = run_error_analysis(detailed_matches, ground_truth)
            print("✅ Error analysis complete - detailed report and visualizations generated")
        except ImportError:
            print("⚠️ Error analysis module not available")
        except Exception as e:
            print(f"⚠️ Error analysis failed: {e}")
    else:
        print("\n⚠️ Skipping error analysis - no ground truth provided")
    
    print("\n" + "="*80)
    print("✅ Report Complete")
    print("="*80 + "\n")


def generate_summary_table(detailed_matches: List[Dict], ground_truth: Dict[str, str] = None):
    """Generate summary performance table."""
    
    summary_data = []
    
    for match_dict in detailed_matches:
        for approach_name, predictions in match_dict.items():
            
            # Calculate metrics
            total_columns = len(predictions)
            confidences = []
            correct_count = 0
            perfect_conf_count = 0
            predictions_made = 0
            
            for target_col, prediction in predictions.items():
                source_col, confidence, explanation = prediction
                
                if source_col is not None:
                    predictions_made += 1
                    confidences.append(confidence)
                    
                    # Check correctness if ground truth available
                    if ground_truth and target_col in ground_truth:
                        if source_col == ground_truth[target_col]:
                            correct_count += 1
                    
                    # Count perfect confidence scores (with floating point tolerance)
                    if abs(confidence - 1.0) < 1e-10:
                        perfect_conf_count += 1
            
            avg_confidence = np.mean(confidences) if confidences else 0.0
            # Compute additional metrics as needed
            accuracy = (correct_count / total_columns * 100) if ground_truth else None
            
            
            summary_data.append({
                'Approach': approach_name,
                'Accuracy': f"{correct_count}/{total_columns} ({accuracy:.0f}%)" if accuracy is not None else "N/A",
                'Predictions Made': f"{predictions_made}/{total_columns}",
                'Avg Confidence': f"{avg_confidence:.3f}",
                'Perfect (1.0)': f"{perfect_conf_count}/{total_columns}",
                'Min Conf': f"{min(confidences):.3f}" if confidences else "N/A",
                'Max Conf': f"{max(confidences):.3f}" if confidences else "N/A"
            })
    
    print(tabulate(summary_data, headers="keys", tablefmt="grid"))
    # Save as LaTeX
    try:
        os.makedirs("output/results", exist_ok=True)
        df = pd.DataFrame(summary_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"output/results/{ts}_table1_summary.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved LaTeX summary table: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to save LaTeX summary table: {e}")


def generate_detailed_mapping_table(detailed_matches: List[Dict], ground_truth: Dict[str, str] = None):
    """Generate detailed column-by-column mapping table."""
    
    # Collect all approaches and columns
    match_dict = detailed_matches[0]  # Assuming single schema comparison
    approaches = list(match_dict.keys())
    columns = list(next(iter(match_dict.values())).keys())
    
    detailed_data = []
    
    for col in columns:
        row = {'Target Column': col}
        
        if ground_truth and col in ground_truth:
            row['Ground Truth'] = ground_truth[col]
        
        for approach in approaches:
            predictions = match_dict[approach]
            
            # Handle missing predictions gracefully
            if col in predictions:
                source_col, confidence, explanation = predictions[col]
                
                # Format: source (confidence)
                if source_col:
                    formatted = f"{source_col} ({confidence:.2f})"
                    # Add checkmark if correct
                    if ground_truth and col in ground_truth:
                        if source_col == ground_truth[col]:
                            formatted = "✓ " + formatted
                        else:
                            formatted = "✗ " + formatted
                else:
                    formatted = f"None ({confidence:.2f})"
            else:
                # No prediction available for this column
                formatted = "— (No Prediction)"
            
            row[approach] = formatted
        
        detailed_data.append(row)
    
    print(tabulate(detailed_data, headers="keys", tablefmt="grid"))
    # Save as LaTeX
    try:
        os.makedirs("output/results", exist_ok=True)
        df = pd.DataFrame(detailed_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"output/results/{ts}_table2_detailed_mappings.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved LaTeX detailed mapping table: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to save LaTeX detailed mapping table: {e}")


def generate_confidence_analysis_table(detailed_matches: List[Dict]):
    """Generate confidence score distribution analysis."""
    
    match_dict = detailed_matches[0]
    columns = list(next(iter(match_dict.values())).keys())
    
    confidence_data = []
    
    for col in columns:
        row = {'Column': col}
        
        for approach_name, predictions in match_dict.items():
            # Handle missing predictions gracefully
            if col in predictions:
                source_col, confidence, explanation = predictions[col]
                row[approach_name] = f"{confidence:.3f}"
            else:
                row[approach_name] = "N/A"
        
        confidence_data.append(row)
    
    # Add statistics row
    stats_row = {'Column': '──────────'}
    for approach_name in match_dict.keys():
        # Only include predictions that exist for confidence calculations
        confidences = []
        for col in columns:
            if col in match_dict[approach_name]:
                pred = match_dict[approach_name][col]
                if len(pred) >= 2:  # Ensure tuple has at least 2 elements
                    confidences.append(pred[1])
        
        if confidences:
            avg = np.mean(confidences)
            std = np.std(confidences)
            stats_row[approach_name] = f"μ={avg:.3f}, σ={std:.3f}"
        else:
            stats_row[approach_name] = "N/A"
    confidence_data.append(stats_row)
    
    print(tabulate(confidence_data, headers="keys", tablefmt="grid"))
    # Save as LaTeX
    try:
        os.makedirs("output/results", exist_ok=True)
        df = pd.DataFrame(confidence_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"output/results/{ts}_table3_confidence_analysis.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved LaTeX confidence analysis table: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to save LaTeX confidence analysis table: {e}")


def generate_type_awareness_table(detailed_matches: List[Dict]):
    """Analyze which approaches are type-aware."""
    
    match_dict = detailed_matches[0]
    
    # Heuristic: Check if confidence varies for same-name columns
    # Perfect matches (name + type) should have higher confidence than name-only matches
    
    type_awareness_data = []
    
    for approach_name, predictions in match_dict.items():
        
        perfect_matches = []  # Same name, assume same type
        type_mismatches = []  # Same name, different type (if we can detect)
        
        for target_col, prediction in predictions.items():
            source_col, confidence, explanation = prediction
            
            if source_col and source_col.lower() == target_col.lower():
                # Check explanation for type mentions
                if 'type' in explanation.lower() or 'integer' in explanation.lower() or 'string' in explanation.lower():
                    if 'mismatch' in explanation.lower() or 'differ' in explanation.lower():
                        type_mismatches.append(confidence)
                    else:
                        perfect_matches.append(confidence)
                else:
                    perfect_matches.append(confidence)
        
        avg_perfect = np.mean(perfect_matches) if perfect_matches else None
        avg_mismatch = np.mean(type_mismatches) if type_mismatches else None
        penalty = avg_perfect - avg_mismatch if (avg_perfect and avg_mismatch) else 0.0
        
        type_awareness_data.append({
            'Approach': approach_name,
            'Perfect Match Conf': f"{avg_perfect:.3f}" if avg_perfect else "N/A",
            'Type Mismatch Conf': f"{avg_mismatch:.3f}" if avg_mismatch else "N/A",
            'Confidence Penalty': f"{penalty:.3f}" if penalty else "N/A"
        })
    
    print(tabulate(type_awareness_data, headers="keys", tablefmt="grid"))
    # Save as LaTeX
    try:
        os.makedirs("output/results", exist_ok=True)
        df = pd.DataFrame(type_awareness_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"output/results/{ts}_table4_type_awareness.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved LaTeX type awareness table: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to save LaTeX type awareness table: {e}")


def generate_agreement_matrix(detailed_matches: List[Dict]):
    """Generate component agreement matrix."""
    
    match_dict = detailed_matches[0]
    approaches = list(match_dict.keys())
    columns = list(next(iter(match_dict.values())).keys())
    
    agreement_data = []
    
    for col in columns:
        row = {'Column': col}
        
        # Collect all predictions for this column
        predictions = {}
        for approach in approaches:
            if col in match_dict[approach]:
                source_col, confidence, explanation = match_dict[approach][col]
                predictions[approach] = source_col
            else:
                predictions[approach] = None
        
        # Count unique predictions
        unique_predictions = set(p for p in predictions.values() if p is not None)
        
        # Find the most common prediction
        from collections import Counter
        prediction_counts = Counter(predictions.values())
        most_common_prediction = prediction_counts.most_common(1)[0][0]
        agreement_count = prediction_counts[most_common_prediction]
        
        row['Predictions'] = ", ".join([f"{k}: {v}" for k, v in predictions.items()])
        row['Agreement'] = f"{agreement_count}/{len(approaches)} ({agreement_count/len(approaches)*100:.0f}%)"
        row['Unanimous'] = "✓" if agreement_count == len(approaches) else "✗"
        
        agreement_data.append(row)
    
    print(tabulate(agreement_data, headers="keys", tablefmt="grid", maxcolwidths=[None, 60, None, None]))
    # Save as LaTeX
    try:
        os.makedirs("output/results", exist_ok=True)
        df = pd.DataFrame(agreement_data)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"output/results/{ts}_table5_agreement_matrix.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved LaTeX agreement matrix table: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to save LaTeX agreement matrix table: {e}")

def generate_latex_tables(detailed_matches: List[Dict], ground_truth: Dict[str, str] = None):
    """Generate LaTeX code for tables (bonus function)."""
    
    print("\n" + "="*80)
    print("📄 LATEX TABLE CODE")
    print("="*80 + "\n")
    
    # Summary Table in LaTeX
    print("% TABLE 1: Summary Performance")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Approach Performance Summary}")
    print("\\label{tab:performance_summary}")
    print("\\begin{tabular}{lcccccc}")
    print("\\toprule")
    print("\\textbf{Approach} & \\textbf{Accuracy} & \\textbf{Avg Conf} & \\textbf{Perfect (1.0)} & \\textbf{Min} & \\textbf{Max} \\\\")
    print("\\midrule")
    
    for match_dict in detailed_matches:
        for approach_name, predictions in match_dict.items():
            confidences = [pred[1] for pred in predictions.values() if pred[0] is not None]
            perfect_count = sum(1 for c in confidences if c == 1.0)
            
            avg_conf = np.mean(confidences) if confidences else 0.0
            min_conf = min(confidences) if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0
            
            print(f"{approach_name} & N/A & {avg_conf:.3f} & {perfect_count}/{len(predictions)} & {min_conf:.3f} & {max_conf:.3f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")
    print()

    # Also save this LaTeX snippet as a file
    try:
        os.makedirs("output/results", exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"output/results/{ts}_table1_summary_snippet.tex"
        with open(out_path, 'w') as f:
            # recreate simple latex table from DataFrame of summary
            rows = []
            for match_dict in detailed_matches:
                for approach_name, predictions in match_dict.items():
                    confidences = [pred[1] for pred in predictions.values() if pred[0] is not None]
                    avg_conf = np.mean(confidences) if confidences else 0.0
                    perfect_count = sum(1 for c in confidences if c == 1.0)
                    rows.append({
                        'Approach': approach_name,
                        'Avg Conf': f"{avg_conf:.3f}",
                        'Perfect': f"{perfect_count}/{len(predictions)}"
                    })
            df = pd.DataFrame(rows)
            f.write(df.to_latex(index=False, longtable=True))
        print(f"📄 Saved LaTeX snippet: {out_path}")
    except Exception as e:
        print(f"⚠️ Failed to save LaTeX snippet: {e}")


