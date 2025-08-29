
from typing import Dict, List

from tabulate import tabulate


def analyze_pairwise_champions(all_pairwise_results: List[Dict]) -> Dict:
    """Analyze pairwise comparison results and identify champion pairs."""
    if not all_pairwise_results:
        return {"error": "No pairwise results to analyze"}
    
    # Track performance by comparison type
    comparison_performance = {}
    comparison_types = ["GPT_vs_Embedding", "GPT_vs_Clustering", "Embedding_vs_Clustering"]
    
    for comp_type in comparison_types:
        f1_scores = []
        agreements = []
        precisions = []
        recalls = []
        
        for result in all_pairwise_results:
            if comp_type in result["comparisons"]:
                metrics = result["comparisons"][comp_type]
                f1_scores.append(metrics.f1_score)
                agreements.append(metrics.agreement_rate)
                precisions.append(metrics.precision)
                recalls.append(metrics.recall)
        
        if f1_scores:
            comparison_performance[comp_type] = {
                'avg_f1': sum(f1_scores) / len(f1_scores),
                'avg_agreement': sum(agreements) / len(agreements),
                'avg_precision': sum(precisions) / len(precisions),
                'avg_recall': sum(recalls) / len(recalls),
                'count': len(f1_scores)
            }
    
    return comparison_performance



def print_pairwise_champions(pairwise_performance: Dict):
    """Print pairwise comparison champions and analysis."""
    if 'error' in pairwise_performance:
        print(f"❌ {pairwise_performance['error']}")
        return
    
    if not pairwise_performance:
        print("❌ No pairwise performance data available")
        return
    
    print(f"\n🤝 PAIRWISE COMPARISON CHAMPIONS")
    print("=" * 60)
    
    # Find champions by different metrics
    f1_champion = max(pairwise_performance.items(), key=lambda x: x[1]['avg_f1'])
    agreement_champion = max(pairwise_performance.items(), key=lambda x: x[1]['avg_agreement'])
    precision_champion = max(pairwise_performance.items(), key=lambda x: x[1]['avg_precision'])
    recall_champion = max(pairwise_performance.items(), key=lambda x: x[1]['avg_recall'])
    
    print(f"🏆 Best F1 Score:     {f1_champion[0].replace('_', ' ')} ({f1_champion[1]['avg_f1']:.3f})")
    print(f"🤝 Best Agreement:    {agreement_champion[0].replace('_', ' ')} ({agreement_champion[1]['avg_agreement']:.3f})")
    print(f"🎯 Best Precision:    {precision_champion[0].replace('_', ' ')} ({precision_champion[1]['avg_precision']:.3f})")
    print(f"📊 Best Recall:       {recall_champion[0].replace('_', ' ')} ({recall_champion[1]['avg_recall']:.3f})")
    
    # Overall champion (based on F1 score)
    overall_champion = f1_champion[0].replace('_', ' ')
    print(f"\n👑 OVERALL PAIRWISE CHAMPION: {overall_champion}")
    print(f"   F1: {f1_champion[1]['avg_f1']:.3f} | Agreement: {f1_champion[1]['avg_agreement']:.3f}")
    
    # Performance analysis
    f1_scores = [perf['avg_f1'] for perf in pairwise_performance.values()]
    best_f1 = max(f1_scores)
    worst_f1 = min(f1_scores)
    f1_gap = best_f1 - worst_f1
    
    print(f"\n📈 Pairwise Performance Analysis:")
    print(f"   F1 Score Range: {worst_f1:.3f} - {best_f1:.3f} (gap: {f1_gap:.3f})")
    
    if f1_gap < 0.1:
        print("   💡 All matcher pairs perform similarly")
    elif f1_gap > 0.3:
        print(f"   💡 Clear pairwise winner: {overall_champion}")
    else:
        print("   💡 Moderate differences between matcher pairs")
    
    # Detailed breakdown table
    print(f"\n📋 Detailed Pairwise Performance:")
    table_data = []
    headers = ["Pair", "Avg F1", "Avg Precision", "Avg Recall", "Avg Agreement", "Datasets"]
    
    for comp_type, metrics in pairwise_performance.items():
        table_data.append([
            comp_type.replace("_", " "),
            f"{metrics['avg_f1']:.3f}",
            f"{metrics['avg_precision']:.3f}",
            f"{metrics['avg_recall']:.3f}",
            f"{metrics['avg_agreement']:.3f}",
            metrics['count']
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))