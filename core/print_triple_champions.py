from typing import Any, Dict, List, Optional


def print_triple_champions(triple_analysis: Dict):
    """Print triple comparison champions and ensemble analysis."""
    if 'error' in triple_analysis:
        print(f"❌ {triple_analysis['error']}")
        return
        
    pattern_stats = triple_analysis.get('pattern_stats', {})
    ensemble_stats = triple_analysis.get('ensemble_stats', {})
    contributions = triple_analysis.get('individual_contributions', {})
    
    total_cols = pattern_stats.get('total_columns', 0)
    if total_cols == 0:
        print("❌ No triple comparison data available")
        return
    
    print(f"\n🎯 TRIPLE COMPARISON CHAMPIONS & ENSEMBLE ANALYSIS")
    print("=" * 70)
    
    # Pattern performance
    print(f"📊 Collaboration Success Patterns (across {total_cols} predictions):")
    print(f"   🎯 Perfect Harmony (3/3 correct): {pattern_stats.get('all_correct_rate', 0):.1%}")
    print(f"   🤝 Strong Majority (2/3 correct): {pattern_stats.get('two_correct_rate', 0):.1%}") 
    print(f"   ⚡ Solo Success (1/3 correct):    {pattern_stats.get('one_correct_rate', 0):.1%}")
    print(f"   💥 Total Failure (0/3 correct):  {pattern_stats.get('none_correct_rate', 0):.1%}")
    print(f"   🔄 Full Agreement Rate:          {pattern_stats.get('all_agree_rate', 0):.1%}")
    
    # Ensemble performance  
    majority_acc = ensemble_stats.get('majority_vote_accuracy', 0)
    consensus_acc = ensemble_stats.get('consensus_accuracy', 0)
    
    print(f"\n🏆 ENSEMBLE CHAMPIONS:")
    print(f"   👑 Majority Vote Accuracy:       {majority_acc:.1%}")
    print(f"   🎪 Consensus (All Agree) Accuracy: {consensus_acc:.1%}")
    
    # Determine champion approach
    individual_max = max([
        pattern_stats.get('all_correct_rate', 0),
        pattern_stats.get('two_correct_rate', 0)
    ])
    
    if majority_acc > individual_max:
        print(f"   🏅 CHAMPION APPROACH: Majority Vote Ensemble")
        print(f"     Outperforms individual patterns by {(majority_acc - individual_max):.1%}")
    else:
        print(f"   🏅 CHAMPION APPROACH: Individual Matcher Performance")
        print(f"     Best individual pattern: {individual_max:.1%}")
    
    # Individual contributions to success
    total_successful = triple_analysis.get('total_successful_cases', 0)
    if total_successful > 0:
        print(f"\n🌟 Success Contribution Analysis (in {total_successful} successful cases):")
        gpt_contrib = contributions.get('gpt_in_success_rate', 0)
        emb_contrib = contributions.get('embedding_in_success_rate', 0) 
        clust_contrib = contributions.get('clustering_in_success_rate', 0)
        
        # Sort contributors
        contrib_list = [
            ('GPT', gpt_contrib),
            ('Embedding', emb_contrib), 
            ('Clustering', clust_contrib)
        ]
        contrib_list.sort(key=lambda x: x[1], reverse=True)
        
        for i, (matcher, rate) in enumerate(contrib_list, 1):
            medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"   {medal} {matcher}: Contributes to {rate:.1%} of successful cases")
    
    # Recommendations
    print(f"\n💡 TRIPLE COMPARISON INSIGHTS:")
    if pattern_stats.get('all_agree_rate', 0) > 0.7:
        print("   🤝 High agreement rate - matchers are consistent")
    elif pattern_stats.get('all_agree_rate', 0) < 0.3:
        print("   🔀 Low agreement rate - matchers are diverse (good for ensembles)")
    
    if majority_acc > 0.8:
        print("   ✨ Majority vote is highly effective - use ensemble approaches")
    elif majority_acc < 0.5:
        print("   ⚠️ Consider individual matcher selection or weighted ensembles")
