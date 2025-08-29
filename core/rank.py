
from typing import Dict, List


def create_individual_matcher_ranking(all_triple_results: List[Dict]) -> Dict[str, float]:
    """Create ranking of individual matchers based on overall performance across all datasets."""
    if not all_triple_results:
        return {}
    
    gpt_correct = 0
    embedding_correct = 0
    clustering_correct = 0
    total_predictions = 0
    
    # Aggregate performance across all datasets
    for triple_result in all_triple_results:
        if 'error' not in triple_result and 'detailed_results' in triple_result:
            for detail in triple_result['detailed_results']:
                total_predictions += 1
                if detail.get('gpt_correct', False):
                    gpt_correct += 1
                if detail.get('embedding_correct', False):
                    embedding_correct += 1
                if detail.get('clustering_correct', False):
                    clustering_correct += 1
    
    if total_predictions > 0:
        rankings = {
            "GPT": gpt_correct / total_predictions,
            "Embedding": embedding_correct / total_predictions, 
            "Clustering": clustering_correct / total_predictions,
        }
        
        return rankings
    return {}





def print_individual_matcher_ranking(rankings: Dict[str, float], total_predictions: int):
    """Print formatted individual matcher performance ranking."""
    if not rankings:
        print("❌ No individual matcher rankings available")
        return
    
    # Sort by performance
    sorted_rankings = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 INDIVIDUAL MATCHER PERFORMANCE RANKING")
    print("=" * 60)
    print(f"📊 Based on {total_predictions} total column predictions across all datasets:")
    print()
    
    for i, (matcher, accuracy) in enumerate(sorted_rankings, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        print(f"   {medal} {i}. {matcher:12}: {accuracy:.1%} accuracy")
    
    # Calculate performance gaps
    if len(sorted_rankings) >= 2:
        best_score = sorted_rankings[0][1]
        worst_score = sorted_rankings[-1][1]
        gap = best_score - worst_score
        print(f"\n📈 Performance Analysis:")
        print(f"   Best vs Worst Gap: {gap:.1%}")
        
        if gap < 0.1:  # Less than 10% difference
            print("   💡 All matchers perform similarly - ensemble methods recommended")
        elif gap > 0.3:  # More than 30% difference
            print(f"   💡 Clear winner: {sorted_rankings[0][0]} significantly outperforms others")
        else:
            print("   💡 Moderate performance differences - ensemble can help")