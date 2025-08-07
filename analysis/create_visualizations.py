import json
import matplotlib.pyplot as plt
import numpy as np

def create_comparison_charts():
    """Create charts for thesis comparison"""
    
    # Data from our analysis
    algorithms = ['GPT', 'EMBED', 'CLUSTER', 'MAJORITY', 'WEIGHTED']
    accuracies = [0.731, 0.723, 0.723, 0.723, 0.723]
    
    # Chart 1: Algorithm Accuracy Comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, accuracies, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7'])
    plt.title('Hamonize Algorithm Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12)
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylim(0.7, 0.75)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('algorithm_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Chart 2: Scale Comparison
    plt.figure(figsize=(8, 6))
    methods = ['COMA', 'Hamonize']
    match_counts = [18, 886]
    colors = ['#FF6B6B', '#4ECDC4']
    
    bars = plt.bar(methods, match_counts, color=colors)
    plt.title('Schema Matching Scale Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Number of Matches', fontsize=12)
    plt.xlabel('Method', fontsize=12)
    
    # Add value labels
    for bar, count in zip(bars, match_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('scale_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Chart 3: Precision at Different Thresholds
    plt.figure(figsize=(12, 8))
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Precision data from our analysis
    gpt_precision = [0.731, 0.731, 0.731, 0.738, 0.705]
    embed_precision = [0.755, 0.760, 0.778, 0.769, 0.824]
    cluster_precision = [0.790, 0.872, 0.824, 0.824, 0.824]
    weighted_precision = [0.730, 0.754, 0.778, 0.824, 0.769]
    
    plt.plot(thresholds, gpt_precision, 'o-', label='GPT', linewidth=2, markersize=8)
    plt.plot(thresholds, embed_precision, 's-', label='EMBED', linewidth=2, markersize=8)
    plt.plot(thresholds, cluster_precision, '^-', label='CLUSTER', linewidth=2, markersize=8)
    plt.plot(thresholds, weighted_precision, 'd-', label='WEIGHTED', linewidth=2, markersize=8)
    
    plt.title('Precision vs Similarity Threshold', fontsize=16, fontweight='bold')
    plt.xlabel('Similarity Threshold', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.45, 0.95)
    plt.ylim(0.65, 0.9)
    
    plt.tight_layout()
    plt.savefig('precision_threshold_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization charts created:")
    print("  - algorithm_accuracy_comparison.png")
    print("  - scale_comparison.png") 
    print("  - precision_threshold_comparison.png")

if __name__ == "__main__":
    create_comparison_charts()
