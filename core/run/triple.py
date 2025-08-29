from typing import Dict, Optional

from pairwise_comparison import print_triple_comparison_table, run_triple_comparison


def run_triple_analysis(gpt_predictions: Dict, embedding_predictions: Dict, 
                       clustering_predictions: Dict, ground_truth: Optional[Dict],
                       source_table: str, target_table: str) -> Optional[Dict]:
    """Run triple comparison analysis against ground truth"""
    print(f"\n🔄 Step 3: Triple Matcher Comparison Against Ground Truth")
    
    if not ground_truth:
        print("⚠️ No real ground truth available for triple comparison")
        return None
        
    triple_results = run_triple_comparison(
        gpt_predictions=gpt_predictions,
        embedding_predictions=embedding_predictions,
        clustering_predictions=clustering_predictions,
        ground_truth=ground_truth,
        source_table=source_table,
        target_table=target_table
    )
    
    # Store source and target info
    triple_results['source_table'] = source_table
    triple_results['target_table'] = target_table
    
    print(f"📊 Triple Comparison Results for {source_table} → {target_table}")
    print_triple_comparison_table(triple_results, source_table, target_table)
    
    return triple_results