import numpy as np
import pandas as pd
from scipy.stats import chi2
from normalization import load_data, load_ground_truth, build_gt_set, normalize_coma, normalize_harmonize, aggregate_max_by_pair
from evaluation import compute_candidate_coverage, eval_set_based, eval_ranking_based, eval_hungarian_one_to_one




def mcnemar_test_top1(run1, run2, gt_set):
    """McNemar test comparing Top-1 correctness per source between two runs"""
    # Get all sources that have ground truth
    gt_sources = set()
    for pair in gt_set:
        gt_sources.add(f"{pair.source_table}.{pair.source_column}")
    
    # For each source, check if run1 and run2 get top-1 correct
    run1_correct = {}
    run2_correct = {}
    
    for source in gt_sources:
        # Get ground truth targets for this source
        gt_targets = set()
        for pair in gt_set:
            if f"{pair.source_table}.{pair.source_column}" == source:
                gt_targets.add(f"{pair.target_table}.{pair.target_column}")
        
        # Get top-1 prediction for run1
        run1_top1 = get_top1_prediction(run1, source)
        run1_correct[source] = 1 if run1_top1 in gt_targets else 0
        
        # Get top-1 prediction for run2
        run2_top1 = get_top1_prediction(run2, source)
        run2_correct[source] = 1 if run2_top1 in gt_targets else 0
    
    # Create McNemar contingency table
    # a: both correct, b: run1 correct, run2 wrong
    # c: run1 wrong, run2 correct, d: both wrong
    a = b = c = d = 0
    
    for source in gt_sources:
        r1_correct = run1_correct.get(source, 0)
        r2_correct = run2_correct.get(source, 0)
        
        if r1_correct == 1 and r2_correct == 1:
            a += 1
        elif r1_correct == 1 and r2_correct == 0:
            b += 1
        elif r1_correct == 0 and r2_correct == 1:
            c += 1
        else:  # both wrong
            d += 1
    
    # McNemar test statistic
    if b + c == 0:
        # No disagreements, can't compute test
        p_value = 1.0
        statistic = 0.0
    else:
        statistic = (abs(b - c) - 1)**2 / (b + c)  # With continuity correction
        p_value = 1 - chi2.cdf(statistic, df=1)
    
    result = {
        'contingency_table': {'a': a, 'b': b, 'c': c, 'd': d},
        'statistic': statistic,
        'p_value': p_value,
        'run1_correct': sum(run1_correct.values()),
        'run2_correct': sum(run2_correct.values()),
        'total_sources': len(gt_sources),
        'significant': p_value < 0.05
    }
    
    return result


def bootstrap_F1_CI(run, gt_set, thresholds, n_boot=1000, sample_unit="source"):
    """Bootstrap confidence intervals for F1 at best threshold"""
    # Get the best threshold from the original evaluation
    _, _, best_metrics = eval_set_based(run, gt_set, thresholds)
    best_threshold = best_metrics['best_threshold']
    
    # Group data by source for source-based resampling
    if sample_unit == "source":
        # Group pairs and ground truth by source
        source_data = {}
        all_sources = set()
        
        # Collect all sources from run pairs
        for pair in run.pairs:
            source_key = f"{pair.source_table}.{pair.source_column}"
            all_sources.add(source_key)
            if source_key not in source_data:
                source_data[source_key] = {'run_pairs': [], 'gt_pairs': []}
            source_data[source_key]['run_pairs'].append(pair)
        
        # Collect ground truth pairs by source
        for pair in gt_set:
            source_key = f"{pair.source_table}.{pair.source_column}"
            all_sources.add(source_key)
            if source_key not in source_data:
                source_data[source_key] = {'run_pairs': [], 'gt_pairs': []}
            source_data[source_key]['gt_pairs'].append(pair)
        
        sources = list(all_sources)
        
        # Bootstrap by resampling sources
        f1_scores = []
        
        for boot_iter in range(n_boot):
            # Resample sources with replacement
            boot_sources = np.random.choice(sources, size=len(sources), replace=True)
            
            # Create bootstrap datasets
            boot_run_pairs = []
            boot_gt_pairs = []
            
            for source in boot_sources:
                if source in source_data:
                    boot_run_pairs.extend(source_data[source]['run_pairs'])
                    boot_gt_pairs.extend(source_data[source]['gt_pairs'])
            
            # Convert to sets
            boot_run_set = set(boot_run_pairs)
            boot_gt_set = set(boot_gt_pairs)
            
            # Calculate F1 at best threshold for this bootstrap sample
            predicted_pairs = set()
            for pair in boot_run_pairs:
                if run.scores.get(pair, 0) >= best_threshold:
                    predicted_pairs.add(pair)
            
            # Calculate TP, FP, FN
            tp = len(predicted_pairs.intersection(boot_gt_set))
            fp = len(predicted_pairs - boot_gt_set)
            fn = len(boot_gt_set - predicted_pairs)
            
            # Calculate F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
    
    else:  # sample_unit == "pair"
        # Bootstrap by resampling individual pairs
        run_pairs = list(run.pairs)
        gt_pairs = list(gt_set)
        
        f1_scores = []
        
        for boot_iter in range(n_boot):
            # Resample run pairs and gt pairs
            boot_run_pairs = list(np.random.choice(run_pairs, size=len(run_pairs), replace=True))
            boot_gt_pairs = list(np.random.choice(gt_pairs, size=len(gt_pairs), replace=True))
            
            boot_gt_set = set(boot_gt_pairs)
            
            # Calculate F1 at best threshold
            predicted_pairs = set()
            for pair in boot_run_pairs:
                if run.scores.get(pair, 0) >= best_threshold:
                    predicted_pairs.add(pair)
            
            # Calculate TP, FP, FN
            tp = len(predicted_pairs.intersection(boot_gt_set))
            fp = len(predicted_pairs - boot_gt_set)
            fn = len(boot_gt_set - predicted_pairs)
            
            # Calculate F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
    
    # Calculate confidence intervals
    f1_scores = np.array(f1_scores)
    ci_lower = np.percentile(f1_scores, 2.5)
    ci_upper = np.percentile(f1_scores, 97.5)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    result = {
        'original_f1': best_metrics['best_F1'],
        'bootstrap_mean': mean_f1,
        'bootstrap_std': std_f1,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'n_boot': n_boot,
        'sample_unit': sample_unit,
        'all_scores': f1_scores
    }
    
    return result


def get_top1_prediction(run, source):
    """Get top-1 prediction for a source from a run"""
    # Get all pairs for this source
    source_pairs = []
    for pair in run.pairs:
        if f"{pair.source_table}.{pair.source_column}" == source:
            source_pairs.append((pair, run.scores[pair]))
    
    if not source_pairs:
        return None
    
    # Sort by score and get top-1
    source_pairs.sort(key=lambda x: x[1], reverse=True)
    top_pair = source_pairs[0][0]
    return f"{top_pair.target_table}.{top_pair.target_column}"



def debug_data_formats(runs, gt_set):
    """Debug function to examine data formats and find mismatches"""
    print("\n=== DEBUGGING DATA FORMATS ===")
    
    print(f"\nGround Truth Sample (first 5 pairs):")
    for i, pair in enumerate(list(gt_set)[:5]):
        print(f"  GT {i+1}: {pair}")
    
    print(f"\nPrediction Sample for each run (first 3 pairs):")
    for run in runs:
        print(f"\n{run.run_id}:")
        for i, pair in enumerate(run.pairs[:3]):
            score = run.scores.get(pair, 'No score')
            print(f"  {i+1}: {pair} (score: {score})")
    
    # Check for exact matches
    print(f"\n=== CHECKING FOR EXACT MATCHES ===")
    for run in runs:
        run_pairs_set = set(run.pairs)
        intersection = gt_set.intersection(run_pairs_set)
        print(f"{run.run_id}: {len(intersection)} exact matches found")
        if len(intersection) > 0:
            print(f"  Sample matches:")
            for i, pair in enumerate(list(intersection)[:3]):
                print(f"    {i+1}: {pair}")

def benchmark():
    """Create comprehensive visualizations for thesis"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Load data again for visualization
    coma_data, harmonize_data = load_data()
    gt_data = load_ground_truth()
    
    # ---------- 1) Load & normalize ----------
    print("\n=== STEP 1: NORMALIZATION ===")
    G = build_gt_set(gt_data)
    coma_run = normalize_coma(coma_data, run_id="COMA")
    harm_runs = normalize_harmonize(harmonize_data)
    harm_max_run = aggregate_max_by_pair(harm_runs, run_id="HARMONIZE-max")

    runs = [coma_run] + harm_runs + [harm_max_run]
    
    # Debug the data formats
    debug_data_formats(runs, G)
    
    print(f"\nTotal ground truth pairs: {len(G)}")
    for run in runs:
        print(f"{run.run_id}: {len(run.pairs)} pairs")
    
    # Check if we have any matches before proceeding
    has_matches = False
    for run in runs:
        run_pairs_set = set(run.pairs)
        if len(G.intersection(run_pairs_set)) > 0:
            has_matches = True
            break
    
    if not has_matches:
        print("\n❌ NO MATCHES FOUND! Stopping evaluation.")
        print("Check the data format alignment between predictions and ground truth.")
        return
    else:
        print("\n✅ Matches found, proceeding with evaluation...")
    
    # ---------- 2A) Candidate coverage ----------
    print("\n=== STEP 2A: CANDIDATE COVERAGE ===")
    for R in runs:
        R.coverage = compute_candidate_coverage(R, G)
    
    # ---------- 2B) Precision-Recall sweep + F1 vs threshold ----------
    print("\n=== STEP 2B: PR CURVES AND F1 ANALYSIS ===")
    thresholds = np.linspace(0.0, 1.0, 101)
    for R in runs:
        print(f"Computing PR curve for {R.run_id}...")
        R.pr_curve, R.f1_curve, R.best = eval_set_based(R, G, thresholds)
        print(f"  Best F1: {R.best['best_F1']:.3f} at threshold {R.best['best_threshold']:.3f}")
        print(f"  P/R at best: {R.best['precision_at_best']:.3f}/{R.best['recall_at_best']:.3f}")
    
    # ---------- 2C) Ranking metrics per source ----------
    print("\n=== STEP 2C: RANKING METRICS ===")
    ks = [1, 3, 5]
    for R in runs:
        print(f"Computing ranking metrics for {R.run_id}...")
        R.ranking = eval_ranking_based(R, G, ks)
        print(f"  MRR: {R.ranking['MRR']:.3f}")
        print(f"  Hits@1/3/5: {R.ranking['Hits@1']:.3f}/{R.ranking['Hits@3']:.3f}/{R.ranking['Hits@5']:.3f}")
        print(f"  Recall@1/3/5: {R.ranking['Recall@1']:.3f}/{R.ranking['Recall@3']:.3f}/{R.ranking['Recall@5']:.3f}")
    
    # ---------- 2D) Hungarian assignment ----------
    print("\n=== STEP 2D: HUNGARIAN 1-TO-1 ASSIGNMENT ===")
    for R in runs:
        print(f"Computing Hungarian assignment for {R.run_id}...")
        R.hungarian = eval_hungarian_one_to_one(R, G)
        print(f"  Hungarian F1: {R.hungarian['f1']:.3f}")
        print(f"  Hungarian P/R: {R.hungarian['precision']:.3f}/{R.hungarian['recall']:.3f}")
        print(f"  Assignments: {R.hungarian['num_assignments']} pairs")
    
    # ---------- 3A) McNemar test for Top-1 correctness ----------
    print("\n=== STEP 3A: MCNEMAR TEST (TOP-1 CORRECTNESS) ===")
    mcnemar_result = mcnemar_test_top1(coma_run, harm_max_run, G)
    print(f"COMA vs HARMONIZE-max McNemar test:")
    print(f"  COMA correct: {mcnemar_result['run1_correct']}/{mcnemar_result['total_sources']}")
    print(f"  HARMONIZE-max correct: {mcnemar_result['run2_correct']}/{mcnemar_result['total_sources']}")
    print(f"  Contingency: a={mcnemar_result['contingency_table']['a']}, b={mcnemar_result['contingency_table']['b']}, c={mcnemar_result['contingency_table']['c']}, d={mcnemar_result['contingency_table']['d']}")
    print(f"  Test statistic: {mcnemar_result['statistic']:.3f}")
    print(f"  P-value: {mcnemar_result['p_value']:.6f}")
    print(f"  Significant (p < 0.05): {mcnemar_result['significant']}")
    
    # ---------- 3B) Bootstrap confidence intervals for F1 ----------
    print("\n=== STEP 3B: BOOTSTRAP F1 CONFIDENCE INTERVALS ===")
    for R in runs:
        print(f"Computing bootstrap CI for {R.run_id}...")
        R.f1_ci = bootstrap_F1_CI(R, G, thresholds, n_boot=500, sample_unit="source")  # Reduced for speed
        print(f"  Original F1: {R.f1_ci['original_f1']:.3f}")
        print(f"  Bootstrap mean: {R.f1_ci['bootstrap_mean']:.3f} ± {R.f1_ci['bootstrap_std']:.3f}")
        print(f"  95% CI: [{R.f1_ci['ci_lower']:.3f}, {R.f1_ci['ci_upper']:.3f}]")
        print(f"  CI width: {R.f1_ci['ci_width']:.3f}")

    # ========== ORGANIZED RESULTS PRESENTATION ==========
    print("\n" + "="*80)
    print("                    COMPREHENSIVE BENCHMARK RESULTS")
    print("="*80)
    
    present_organized_results(runs, mcnemar_result, G)


def present_organized_results(runs, mcnemar_result, gt_set):
    """Present all benchmark results in organized, presentable tables"""
    
    # 1. Coverage and Basic Stats
    print("\n1. DATASET OVERVIEW")
    print("-" * 50)
    coverage_data = []
    for run in runs:
        coverage_data.append({
            'Method': run.run_id,
            'Total Pairs': len(run.pairs),
            'Coverage (%)': f"{run.coverage * 100:.1f}%"
        })
    
    coverage_df = pd.DataFrame(coverage_data)
    print(coverage_df.to_string(index=False))
    print(f"\nGround Truth Pairs: {len(gt_set)}")
    
    # 2. Set-based Evaluation Results
    print("\n\n2. SET-BASED EVALUATION (Best F1 Threshold)")
    print("-" * 70)
    setbased_data = []
    for run in runs:
        setbased_data.append({
            'Method': run.run_id,
            'Best Threshold': f"{run.best['best_threshold']:.3f}",
            'Precision': f"{run.best['precision_at_best']:.3f}",
            'Recall': f"{run.best['recall_at_best']:.3f}",
            'F1-Score': f"{run.best['best_F1']:.3f}",
            'F1 CI (95%)': f"[{run.f1_ci['ci_lower']:.3f}, {run.f1_ci['ci_upper']:.3f}]"
        })
    
    setbased_df = pd.DataFrame(setbased_data)
    print(setbased_df.to_string(index=False))
    
    # 3. Ranking-based Evaluation Results
    print("\n\n3. RANKING-BASED EVALUATION")
    print("-" * 60)
    ranking_data = []
    for run in runs:
        ranking_data.append({
            'Method': run.run_id,
            'MRR': f"{run.ranking['MRR']:.3f}",
            'Hits@1': f"{run.ranking['Hits@1']:.3f}",
            'Hits@3': f"{run.ranking['Hits@3']:.3f}",
            'Hits@5': f"{run.ranking['Hits@5']:.3f}",
            'Recall@1': f"{run.ranking['Recall@1']:.3f}",
            'Recall@3': f"{run.ranking['Recall@3']:.3f}",
            'Recall@5': f"{run.ranking['Recall@5']:.3f}"
        })
    
    ranking_df = pd.DataFrame(ranking_data)
    print(ranking_df.to_string(index=False))
    
    # 4. Hungarian Assignment Results
    print("\n\n4. HUNGARIAN ONE-TO-ONE ASSIGNMENT")
    print("-" * 55)
    hungarian_data = []
    for run in runs:
        hungarian_data.append({
            'Method': run.run_id,
            'Precision': f"{run.hungarian['precision']:.3f}",
            'Recall': f"{run.hungarian['recall']:.3f}",
            'F1-Score': f"{run.hungarian['f1']:.3f}",
            'Assignments': run.hungarian['num_assignments']
        })
    
    hungarian_df = pd.DataFrame(hungarian_data)
    print(hungarian_df.to_string(index=False))
    
    # 5. Statistical Significance Test
    print("\n\n5. STATISTICAL SIGNIFICANCE (McNEMAR TEST)")
    print("-" * 60)
    print("Comparison: COMA vs HARMONIZE-max (Top-1 Correctness)")
    print(f"COMA Correct Sources: {mcnemar_result['run1_correct']}/{mcnemar_result['total_sources']} ({mcnemar_result['run1_correct']/mcnemar_result['total_sources']*100:.1f}%)")
    print(f"HARMONIZE-max Correct Sources: {mcnemar_result['run2_correct']}/{mcnemar_result['total_sources']} ({mcnemar_result['run2_correct']/mcnemar_result['total_sources']*100:.1f}%)")
    print(f"Test Statistic: {mcnemar_result['statistic']:.3f}")
    print(f"P-value: {mcnemar_result['p_value']:.6f}")
    print(f"Significant (α=0.05): {'Yes' if mcnemar_result['significant'] else 'No'}")
    
    # Contingency table
    ct = mcnemar_result['contingency_table']
    print(f"\nContingency Table:")
    print(f"                  HARMONIZE-max")
    print(f"                Correct | Wrong")
    print(f"COMA Correct      {ct['a']:3d}   |  {ct['b']:3d}")
    print(f"COMA Wrong        {ct['c']:3d}   |  {ct['d']:3d}")
    
    # 6. Summary Table - Key Metrics Only
    print("\n\n6. SUMMARY - KEY PERFORMANCE METRICS")
    print("-" * 60)
    summary_data = []
    for run in runs:
        summary_data.append({
            'Method': run.run_id,
            'F1-Score': f"{run.best['best_F1']:.3f}",
            'MRR': f"{run.ranking['MRR']:.3f}",
            'Hits@1': f"{run.ranking['Hits@1']:.3f}",
            'Hungarian F1': f"{run.hungarian['f1']:.3f}",
            'Coverage': f"{run.coverage:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # 7. Export results to CSV
    print("\n\n7. EXPORTING RESULTS")
    print("-" * 30)
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('../output', exist_ok=True)
    
    # Export all tables
    coverage_df.to_csv('../output/benchmark_coverage.csv', index=False)
    setbased_df.to_csv('../output/benchmark_setbased.csv', index=False)
    ranking_df.to_csv('../output/benchmark_ranking.csv', index=False)
    hungarian_df.to_csv('../output/benchmark_hungarian.csv', index=False)
    summary_df.to_csv('../output/benchmark_summary.csv', index=False)
    
    # Export McNemar test results
    mcnemar_df = pd.DataFrame([{
        'Comparison': 'COMA vs HARMONIZE-max',
        'COMA_Correct': mcnemar_result['run1_correct'],
        'HARMONIZE_Correct': mcnemar_result['run2_correct'],
        'Total_Sources': mcnemar_result['total_sources'],
        'Test_Statistic': mcnemar_result['statistic'],
        'P_Value': mcnemar_result['p_value'],
        'Significant': mcnemar_result['significant']
    }])
    mcnemar_df.to_csv('../output/benchmark_mcnemar.csv', index=False)
    
    print("✓ Coverage results exported to: ../output/benchmark_coverage.csv")
    print("✓ Set-based results exported to: ../output/benchmark_setbased.csv")
    print("✓ Ranking results exported to: ../output/benchmark_ranking.csv")
    print("✓ Hungarian results exported to: ../output/benchmark_hungarian.csv")
    print("✓ Summary results exported to: ../output/benchmark_summary.csv")
    print("✓ McNemar test results exported to: ../output/benchmark_mcnemar.csv")
    
    print("\n" + "="*80)
    print("                    BENCHMARK ANALYSIS COMPLETE")
    print("="*80)

benchmark()