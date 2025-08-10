
import numpy as np
from scipy.optimize import linear_sum_assignment
def compute_candidate_coverage(run, gt_set):
    """Compute how many GT pairs exist in candidate list regardless of threshold"""
    # Convert run pairs to set for faster lookup
    candidate_pairs = set(run.pairs)
    
    # Count how many GT pairs are found in candidates
    covered_pairs = gt_set.intersection(candidate_pairs)
    
    coverage = len(covered_pairs) / len(gt_set) if len(gt_set) > 0 else 0.0
    
    print(f"{run.run_id} coverage: {len(covered_pairs)}/{len(gt_set)} = {coverage:.3f}")
    return coverage

def eval_set_based(run, gt_set, thresholds):
    """Compute PR curve, F1 curve, and best metrics"""
    precisions = []
    recalls = []
    f1_scores = []
    
    best_f1 = 0.0
    best_threshold = 0.0
    best_metrics = {}
    
    for threshold in thresholds:
        # Get pairs above threshold
        predicted_pairs = set()
        for pair in run.pairs:
            if run.scores[pair] >= threshold:
                predicted_pairs.add(pair)
        
        # Calculate TP, FP, FN
        tp = len(predicted_pairs.intersection(gt_set))
        fp = len(predicted_pairs - gt_set)
        fn = len(gt_set - predicted_pairs)
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        
        # Track best F1
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'best_F1': f1,
                'best_threshold': threshold,
                'precision_at_best': precision,
                'recall_at_best': recall,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
    
    pr_curve = {'thresholds': thresholds, 'precisions': precisions, 'recalls': recalls}
    f1_curve = {'thresholds': thresholds, 'f1_scores': f1_scores}
    
    return pr_curve, f1_curve, best_metrics


def eval_ranking_based(run, gt_set, ks):
    """Compute ranking metrics per source (Hits@k, MRR, Recall@k)"""
    # Group pairs by source table.column
    source_groups = {}
    for pair in run.pairs:
        source_key = f"{pair.source_table}.{pair.source_column}"
        if source_key not in source_groups:
            source_groups[source_key] = []
        source_groups[source_key].append((pair, run.scores[pair]))
    
    # Sort each source group by score (descending)
    for source_key in source_groups:
        source_groups[source_key].sort(key=lambda x: x[1], reverse=True)
    
    # Compute metrics for each source
    all_hits = {k: [] for k in ks}
    all_mrr = []
    all_recall_at_k = {k: [] for k in ks}
    
    for source_key, ranked_pairs in source_groups.items():
        # Find ground truth targets for this source
        gt_targets_for_source = set()
        for gt_pair in gt_set:
            if f"{gt_pair.source_table}.{gt_pair.source_column}" == source_key:
                gt_targets_for_source.add(f"{gt_pair.target_table}.{gt_pair.target_column}")
        
        if len(gt_targets_for_source) == 0:
            continue  # Skip sources with no ground truth
        
        # Create ranked list of predicted targets
        predicted_targets = []
        for pair, score in ranked_pairs:
            target_key = f"{pair.target_table}.{pair.target_column}"
            predicted_targets.append(target_key)
        
        # Compute Hits@k and Recall@k
        for k in ks:
            top_k_targets = set(predicted_targets[:k])
            hits = len(gt_targets_for_source.intersection(top_k_targets))
            
            # Hits@k: 1 if at least one correct target in top-k, 0 otherwise
            all_hits[k].append(1 if hits > 0 else 0)
            
            # Recall@k: fraction of GT targets found in top-k
            recall_k = hits / len(gt_targets_for_source)
            all_recall_at_k[k].append(recall_k)
        
        # Compute MRR (Mean Reciprocal Rank)
        reciprocal_rank = 0.0
        for rank, target in enumerate(predicted_targets, 1):
            if target in gt_targets_for_source:
                reciprocal_rank = 1.0 / rank
                break
        all_mrr.append(reciprocal_rank)
    
    # Average metrics across all sources
    ranking_metrics = {}
    for k in ks:
        ranking_metrics[f'Hits@{k}'] = np.mean(all_hits[k]) if all_hits[k] else 0.0
        ranking_metrics[f'Recall@{k}'] = np.mean(all_recall_at_k[k]) if all_recall_at_k[k] else 0.0
    
    ranking_metrics['MRR'] = np.mean(all_mrr) if all_mrr else 0.0
    ranking_metrics['num_sources'] = len(source_groups)
    
    return ranking_metrics

def eval_hungarian_one_to_one(run, gt_set):
    """Compute 1-to-1 global assignment using Hungarian algorithm"""
    # Get all unique source and target columns
    sources = set()
    targets = set()
    
    for pair in run.pairs:
        sources.add(f"{pair.source_table}.{pair.source_column}")
        targets.add(f"{pair.target_table}.{pair.target_column}")
    
    # Convert to sorted lists for consistent indexing
    sources = sorted(list(sources))
    targets = sorted(list(targets))
    
    # Create cost matrix (we use negative scores since Hungarian minimizes cost)
    cost_matrix = np.full((len(sources), len(targets)), 1.0)  # Default high cost
    
    # Fill in the actual costs (negative scores)
    for pair in run.pairs:
        source_idx = sources.index(f"{pair.source_table}.{pair.source_column}")
        target_idx = targets.index(f"{pair.target_table}.{pair.target_column}")
        cost_matrix[source_idx, target_idx] = -run.scores[pair]  # Negative for minimization
    
    # Apply Hungarian algorithm
    source_indices, target_indices = linear_sum_assignment(cost_matrix)
    
    # Create the 1-to-1 assignment
    assignment_pairs = set()
    for s_idx, t_idx in zip(source_indices, target_indices):
        source_key = sources[s_idx]
        target_key = targets[t_idx]
        
        # Find the corresponding Pair object
        for pair in run.pairs:
            if (f"{pair.source_table}.{pair.source_column}" == source_key and 
                f"{pair.target_table}.{pair.target_column}" == target_key):
                assignment_pairs.add(pair)
                break
    
    # Calculate metrics for the 1-to-1 assignment
    tp = len(assignment_pairs.intersection(gt_set))
    fp = len(assignment_pairs - gt_set)
    fn = len(gt_set - assignment_pairs)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    hungarian_metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'num_assignments': len(assignment_pairs),
        'assignment_pairs': assignment_pairs
    }
    
    return hungarian_metrics