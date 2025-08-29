# MATPLOTLIB VISUALIZATION CODE FOR THESIS
# Run this when matplotlib is available

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend to avoid display issues
matplotlib.use('Agg')

def create_thesis_visualizations():
    """Create all visualizations for thesis"""
    
    print("🎨 Creating thesis visualizations...")
    
    # Data from analysis
    algorithms = ['GPT', 'EMBED', 'CLUSTER', 'MAJORITY', 'WEIGHTED']
    accuracies = [0.731, 0.723, 0.723, 0.723, 0.723]
    match_counts = [163, 886]  # COMA, harmonize
    
    # Set style
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    
    # Figure 1: Algorithm Accuracy
    plt.figure(figsize=(12, 8))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#7209B7']
    bars = plt.bar(algorithms, accuracies, color=colors, alpha=0.8, edgecolor='black')
    plt.title('harmonize Algorithm Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylim(0.7, 0.75)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/IMG/algorithm_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print("✅ Created: algorithm_accuracy.png")
    
    # Figure 2: Scale Comparison
    plt.figure(figsize=(10, 6))
    methods = ['COMA', 'harmonize']
    colors_scale = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(methods, match_counts, color=colors_scale, alpha=0.8, edgecolor='black', linewidth=2)
    plt.title('Schema Matching Scale Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Number of Matches', fontsize=14)
    plt.xlabel('Method', fontsize=14)
    
    for bar, count in zip(bars, match_counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{count}', ha='center', va='bottom', fontweight='bold', fontsize=16)
    
    ratio = match_counts[1] / match_counts[0]
    plt.text(0.5, max(match_counts) * 0.8, f'{ratio:.1f}x more matches', 
             ha='center', va='center', fontsize=14,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('scale_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print("✅ Created: scale_comparison.png")
    
    # Figure 3: Precision vs Threshold
    plt.figure(figsize=(12, 8))
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    gpt_precision = [0.731, 0.731, 0.731, 0.738, 0.705]
    embed_precision = [0.755, 0.760, 0.778, 0.769, 0.824]
    cluster_precision = [0.790, 0.872, 0.824, 0.824, 0.824]
    weighted_precision = [0.730, 0.754, 0.778, 0.824, 0.769]
    
    plt.plot(thresholds, gpt_precision, 'o-', label='GPT', linewidth=3, markersize=8, color='#2E86AB')
    plt.plot(thresholds, embed_precision, 's-', label='EMBED', linewidth=3, markersize=8, color='#A23B72')
    plt.plot(thresholds, cluster_precision, '^-', label='CLUSTER', linewidth=3, markersize=8, color='#F18F01')
    plt.plot(thresholds, weighted_precision, 'd-', label='WEIGHTED', linewidth=3, markersize=8, color='#7209B7')
    
    plt.title('Precision vs Similarity Threshold Analysis', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Similarity Threshold', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.45, 0.95)
    plt.ylim(0.65, 0.9)
    
    plt.tight_layout()
    plt.savefig('precision_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print("✅ Created: precision_analysis.png")
    
    # Figure 4: Comprehensive Dashboard
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Algorithm accuracy
    bars1 = ax1.bar(algorithms, accuracies, color=colors, alpha=0.8)
    ax1.set_title('Algorithm Accuracy', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0.7, 0.75)
    for bar, acc in zip(bars1, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Match counts
    bars2 = ax2.bar(['COMA', 'harmonize'], match_counts, color=colors_scale, alpha=0.8)
    ax2.set_title('Total Matches', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Number of Matches')
    for bar, count in zip(bars2, match_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # Ground truth coverage
    gt_counts = [40, 642]  # Total ground truth mappings, harmonize validated
    bars3 = ax3.bar(['Total Ground Truth', 'harmonize Validated'], gt_counts, 
                    color=['lightgray', '#4ECDC4'], alpha=0.8)
    ax3.set_title('Ground Truth Coverage', fontweight='bold', fontsize=14)
    ax3.set_ylabel('Number of Mappings')
    for bar, count in zip(bars3, gt_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                f'{count}', ha='center', va='bottom', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # Performance metrics summary
    metrics = ['Accuracy', 'Scale', 'Validation', 'Granularity']
    coma_scores = [0, 0.18, 0, 0.3]  # Normalized scores
    harmonize_scores = [0.73, 1.0, 1.0, 1.0]  # Normalized scores
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars4a = ax4.bar(x - width/2, coma_scores, width, label='COMA', color='#FF6B6B', alpha=0.8)
    bars4b = ax4.bar(x + width/2, harmonize_scores, width, label='harmonize', color='#4ECDC4', alpha=0.8)
    
    ax4.set_title('Overall Performance Comparison', fontweight='bold', fontsize=14)
    ax4.set_ylabel('Normalized Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    print("✅ Created: comprehensive_dashboard.png")
    
    print("\n🎉 ALL THESIS VISUALIZATIONS CREATED!")
    print("Files generated:")
    print("• algorithm_accuracy.png")
    print("• scale_comparison.png") 
    print("• precision_analysis.png")
    print("• comprehensive_dashboard.png")

if __name__ == "__main__":
    create_thesis_visualizations()
