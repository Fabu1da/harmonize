"""Cluster statistics collection and analysis for HARMONIZE evaluation"""
from typing import List, Dict, Any
import statistics

class ClusterStatsCollector:
    """Collects and analyzes clustering statistics across evaluations"""
    
    def __init__(self):
        self.cluster_data = []
    
    def add_cluster_info(self, cluster_info: Dict[str, Any], dataset_name: str = ""):
        """Add cluster information from a single evaluation"""
        entry = {
            'dataset': dataset_name,
            'n_clusters_requested': cluster_info.get('n_clusters_requested', 0),
            'n_clusters_actual': cluster_info.get('n_clusters_actual', 0),
            'total_columns': cluster_info.get('total_columns', 0),
            'source_columns': cluster_info.get('source_columns', 0),
            'target_columns': cluster_info.get('target_columns', 0),
            'cluster_assignments': cluster_info.get('cluster_assignments', {})
        }
        self.cluster_data.append(entry)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics across all evaluations"""
        if not self.cluster_data:
            return {}
        
        n_clusters_requested = [d['n_clusters_requested'] for d in self.cluster_data]
        n_clusters_actual = [d['n_clusters_actual'] for d in self.cluster_data]
        total_columns = [d['total_columns'] for d in self.cluster_data]
        
        summary = {
            'total_evaluations': len(self.cluster_data),
            'clusters_requested': {
                'min': min(n_clusters_requested) if n_clusters_requested else 0,
                'max': max(n_clusters_requested) if n_clusters_requested else 0,
                'mean': statistics.mean(n_clusters_requested) if n_clusters_requested else 0,
                'median': statistics.median(n_clusters_requested) if n_clusters_requested else 0
            },
            'clusters_actual': {
                'min': min(n_clusters_actual) if n_clusters_actual else 0,
                'max': max(n_clusters_actual) if n_clusters_actual else 0,
                'mean': statistics.mean(n_clusters_actual) if n_clusters_actual else 0,
                'median': statistics.median(n_clusters_actual) if n_clusters_actual else 0
            },
            'total_columns_processed': sum(total_columns),
            'avg_columns_per_evaluation': statistics.mean(total_columns) if total_columns else 0
        }
        
        return summary
    
    def print_cluster_summary(self):
        """Print formatted cluster statistics summary"""
        stats = self.get_summary_stats()
        
        if not stats:
            print("   📊 No cluster statistics available")
            return
        
        print(f"   📊 Clustering Statistics:")
        print(f"      • {stats['total_evaluations']} evaluations with clustering")
        print(f"      • {stats['total_columns_processed']} total columns processed")
        print(f"      • Avg columns per evaluation: {stats['avg_columns_per_evaluation']:.1f}")
        print(f"      • Clusters requested: {stats['clusters_requested']['min']}-{stats['clusters_requested']['max']} (avg: {stats['clusters_requested']['mean']:.1f})")
        print(f"      • Clusters actually used: {stats['clusters_actual']['min']}-{stats['clusters_actual']['max']} (avg: {stats['clusters_actual']['mean']:.1f})")

# Global instance for collecting cluster stats
cluster_stats_collector = ClusterStatsCollector()
