from typing import Any, Dict, Tuple

from pairwise_comparison import print_pairwise_comparison_table, run_all_pairwise_comparisons


def run_pairwise_analysis(gpt_predictions: Dict, embedding_predictions: Dict, 
                         clustering_predictions: Dict, source_table: str, target_table: str,
                         ground_truth: Dict = None) -> Tuple[Any, Dict]:
    """Run pairwise comparison analysis between matchers"""
    print(f"\n🔄 Step 2: Pairwise Matcher Comparisons")
    
    pairwise_results, pairwise_result_entry = run_all_pairwise_comparisons(
        gpt_predictions=gpt_predictions,
        embedding_predictions=embedding_predictions,
        clustering_predictions=clustering_predictions,
        ground_truth=ground_truth,  # Pass the actual ground truth
        source_table=source_table,
        target_table=target_table
    )
    
    # Display pairwise comparison results
    print_pairwise_comparison_table(pairwise_results, source_table, target_table)
    
    return pairwise_results, pairwise_result_entry