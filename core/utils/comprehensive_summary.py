from typing import List

from core.analyze_triple_champions import analyze_triple_champions
from core.pairwise import analyze_pairwise_champions
from core.rank import create_individual_matcher_ranking
from core.utils.cluster_statistics import cluster_stats_collector


def print_final_comprehensive_summary(all_triple_results: List, all_pairwise_results: List, results: List):
    """Print the final comprehensive summary"""
    print(f"\n" + "🎉" + "="*78 + "🎉")
    print("🏁 harmonize EVALUATION COMPLETE - FINAL SUMMARY")
    print("🎉" + "="*78 + "🎉")
    
    # Summary statistics
    total_datasets = len([r for r in all_triple_results if 'error' not in r])
    total_pairwise = len(all_pairwise_results)
    
    print(f"📊 Evaluation Scope:")
    print(f"   • {total_datasets} dataset pairs evaluated")
    print(f"   • {total_pairwise} pairwise comparisons performed")
    print(f"   • {len(results)} schema matching tasks completed")
    
    # Add cluster statistics
    cluster_stats_collector.print_cluster_summary()
    
    if all_triple_results:
        # Show performance summaries
        individual_rankings = create_individual_matcher_ranking(all_triple_results)
        if individual_rankings:
            sorted_individual = sorted(individual_rankings.items(), key=lambda x: x[1], reverse=True)
            print(f"\n🏆 Individual Matcher Performance:")
            for i, (matcher, accuracy) in enumerate(sorted_individual, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"   {medal} {matcher}: {accuracy:.1%}")
        
        # Show pairwise champion
        if all_pairwise_results:
            pairwise_perf = analyze_pairwise_champions(all_pairwise_results)
            if pairwise_perf and 'error' not in pairwise_perf:
                sorted_pairs = sorted(pairwise_perf.items(), key=lambda x: x[1]['avg_f1'], reverse=True)
                print(f"\n🤝 Pairwise Agreement Performance:")
                for i, (pair_name, metrics) in enumerate(sorted_pairs, 1):
                    medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                    print(f"   {medal} {pair_name.replace('_', ' ')}: F1 {metrics['avg_f1']:.3f}")
        
        # Show ensemble performance
        triple_analysis = analyze_triple_champions(all_triple_results)
        if triple_analysis and 'error' not in triple_analysis:
            ensemble_stats = triple_analysis.get('ensemble_stats', {})
            majority_acc = ensemble_stats.get('majority_vote_accuracy', 0)
            consensus_acc = ensemble_stats.get('consensus_accuracy', 0)
            
            print(f"\n🎯 Ensemble Performance:")
            print(f"   🏆 Majority Vote: {majority_acc:.1%}")
            print(f"   🤝 Consensus (All Agree): {consensus_acc:.1%}")
        
        # Overall champion summary
        if individual_rankings:
            champion_individual = max(individual_rankings.items(), key=lambda x: x[1])
            print(f"\n👑 OVERALL CHAMPIONS:")
            print(f"   🏆 Best Matcher: {champion_individual[0]} ({champion_individual[1]:.1%})")
        
        if all_pairwise_results and pairwise_perf:
            f1_champion = max(pairwise_perf.items(), key=lambda x: x[1]['avg_f1'])
            print(f"   🤝 Best Pair: {f1_champion[0].replace('_', ' ')} (F1: {f1_champion[1]['avg_f1']:.3f})")
        
        if triple_analysis and 'error' not in triple_analysis:
            print(f"   🎯 Best Ensemble: Majority Vote ({majority_acc:.1%})")
        
        # Best ensemble recommendation
        print(f"\n✨ Recommended Approach: Ensemble methods")
        print(f"   Use Majority Vote for robust predictions")
        print(f"   Use Weighted Ensemble for confidence-aware results")
    
    print(f"\n📁 Generated Outputs:")
    print(f"   📊 Pairwise comparison results: output/global_pairwise_comparison_results.*")
    print(f"   🔀 Triple comparison summary: output/triple_comparison_global_summary.json")
    print(f"   🏆 Individual matcher rankings: displayed above")
    print(f"   📈 Cross-dataset aggregation: output/CrossDataset_Aggregation_Summary.*")
    print(f"   📋 Detailed results: output/real_data_detailed_matches.json")
    print(f"   🎯 LaTeX tables: output/*.tex files")
    
    print(f"\n✨ Analysis Complete! All results exported for thesis integration.")
    print("🎉" + "="*78 + "🎉\n")