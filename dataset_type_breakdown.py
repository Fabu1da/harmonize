"""
Dataset Type Breakdown Analysis for harmonize
Creates tables showing performance breakdown by dataset type (Joinable, Sem-joinable, Unionable, View-union)
for thesis analysis and LaTeX table generation.
"""

import json
import os
from typing import Dict, List, Any
from tabulate import tabulate


def create_dataset_type_breakdown(all_results: List[Dict], all_triple_results: List[Dict]) -> Dict:
    """
    Create breakdown table by dataset type for thesis.
    
    Args:
        all_results: List of result dictionaries from main evaluation
        all_triple_results: List of triple comparison results
    
    Returns:
        Dictionary with breakdown by dataset type
    """
    
    # Define dataset type mapping based on file naming patterns - match actual filenames
    # Order matters: more specific matches first
    dataset_type_mapping = {
        'sem-joinable': ['semjoinable'],  # Must come before 'joinable'
        'view-union': ['viewunion'],      # Must come before 'unionable'  
        'joinable': ['joinable'],
        'unionable': ['unionable']
    }
    
    # Initialize breakdown by type
    breakdown = {}
    
    # Process triple results to extract performance metrics
    for triple_result in all_triple_results:
        if 'error' in triple_result:
            continue
            
        source_table = triple_result.get('source_table', '').lower()
        target_table = triple_result.get('target_table', '').lower()
        
        # Determine dataset type - classify based on the target table type
        # Since the target table defines the schema relationship scenario
        dataset_type = 'unknown'
        for type_name, keywords in dataset_type_mapping.items():
            if any(keyword.lower() in target_table for keyword in keywords):
                dataset_type = type_name
                break
        
        # # Debug print to see classification
        # print(f"DEBUG: Source: '{source_table}' | Target: '{target_table}' → Type: '{dataset_type}'")
        
        if dataset_type not in breakdown:
            breakdown[dataset_type] = {
                'datasets': [],
                'gpt_accuracies': [],
                'gpt_coverages': [],
                'embed_accuracies': [],
                'embed_coverages': [],
                'cluster_accuracies': [],
                'cluster_coverages': [],
                'majority_accuracies': [],
                'majority_coverages': []
            }
        
        # Extract metrics from detailed results
        detailed_results = triple_result.get('detailed_results', [])
        if not detailed_results:
            continue
            
        total_columns = len(detailed_results)
        
        # Calculate individual accuracies for this dataset
        gpt_correct = sum(1 for detail in detailed_results if detail.get('gpt_correct', False))
        embed_correct = sum(1 for detail in detailed_results if detail.get('embedding_correct', False))
        cluster_correct = sum(1 for detail in detailed_results if detail.get('clustering_correct', False))
        majority_correct = sum(1 for detail in detailed_results if detail.get('majority_correct', False))
        
        # Calculate coverages (how many columns each matcher predicted)
        gpt_coverage = sum(1 for detail in detailed_results if detail.get('gpt_prediction') not in [None, '', '—'])
        embed_coverage = sum(1 for detail in detailed_results if detail.get('embedding_prediction') not in [None, '', '—'])
        cluster_coverage = sum(1 for detail in detailed_results if detail.get('clustering_prediction') not in [None, '', '—'])
        majority_coverage = sum(1 for detail in detailed_results if detail.get('majority_prediction') not in [None, '', '—'])
        
        # Store results
        breakdown[dataset_type]['datasets'].append(f"{source_table} → {target_table}")
        breakdown[dataset_type]['gpt_accuracies'].append(gpt_correct / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['gpt_coverages'].append(gpt_coverage / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['embed_accuracies'].append(embed_correct / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['embed_coverages'].append(embed_coverage / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['cluster_accuracies'].append(cluster_correct / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['cluster_coverages'].append(cluster_coverage / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['majority_accuracies'].append(majority_correct / total_columns if total_columns > 0 else 0)
        breakdown[dataset_type]['majority_coverages'].append(majority_coverage / total_columns if total_columns > 0 else 0)
    
    return breakdown


def print_dataset_type_breakdown_table(breakdown: Dict) -> List[List[str]]:
    """
    Print the table in the format needed for thesis.
    
    Args:
        breakdown: Dataset type breakdown dictionary
    
    Returns:
        Table data for further processing
    """
    
    print(f"\n📊 DATASET TYPE BREAKDOWN - Harmonize Variants")
    print("=" * 80)
    
    table_data = []
    
    # Sort dataset types for consistent ordering
    ordered_types = ['joinable', 'sem-joinable', 'unionable', 'view-union', 'unknown']
    
    for dataset_type in ordered_types:
        if dataset_type not in breakdown or not breakdown[dataset_type]['datasets']:
            continue
            
        data = breakdown[dataset_type]
        
        # Calculate averages
        avg_gpt_acc = sum(data['gpt_accuracies']) / len(data['gpt_accuracies']) if data['gpt_accuracies'] else 0
        avg_gpt_cov = sum(data['gpt_coverages']) / len(data['gpt_coverages']) if data['gpt_coverages'] else 0
        
        avg_embed_acc = sum(data['embed_accuracies']) / len(data['embed_accuracies']) if data['embed_accuracies'] else 0
        avg_embed_cov = sum(data['embed_coverages']) / len(data['embed_coverages']) if data['embed_coverages'] else 0
        
        avg_cluster_acc = sum(data['cluster_accuracies']) / len(data['cluster_accuracies']) if data['cluster_accuracies'] else 0
        avg_cluster_cov = sum(data['cluster_coverages']) / len(data['cluster_coverages']) if data['cluster_coverages'] else 0
        
        avg_majority_acc = sum(data['majority_accuracies']) / len(data['majority_accuracies']) if data['majority_accuracies'] else 0
        avg_majority_cov = sum(data['majority_coverages']) / len(data['majority_coverages']) if data['majority_coverages'] else 0
        
        table_data.append([
            dataset_type.title(),
            f"{avg_gpt_acc:.3f} / {avg_gpt_cov:.2f}",
            f"{avg_embed_acc:.3f} / {avg_embed_cov:.2f}",
            f"{avg_cluster_acc:.3f} / {avg_cluster_cov:.2f}",
            f"{avg_majority_acc:.3f} / {avg_majority_cov:.2f}"
        ])
    
    headers = ["Dataset Type", "GPT F1 / Cov.", "Embed F1 / Cov.", "Cluster F1 / Cov.", "Max-Votes F1 / Cov."]
    print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
    
    return table_data


def export_latex_breakdown_table(table_data: List[List[str]], filename: str = "dataset_breakdown_harmonize") -> str:
    """
    Export the breakdown table as LaTeX for thesis inclusion.
    
    Args:
        table_data: The table data from print_dataset_type_breakdown_table
        filename: Output filename (without extension)
    
    Returns:
        LaTeX table content
    """
    
    # Create LaTeX table rows
    latex_rows = []
    for row in table_data:
        latex_row = " & ".join(row) + " \\\\"
        latex_rows.append(latex_row)
    
    latex_content = f"""\\begin{{table}}[h]
\\centering
\\caption{{Harmonize variants performance breakdown by dataset type}}
\\label{{tab:dataset_breakdown_harmonize}}
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
Dataset Type & GPT F1 / Cov. & Embed F1 / Cov. & Cluster F1 / Cov. & Max-Votes F1 / Cov. \\\\
\\hline
{chr(10).join(latex_rows)}
\\hline
\\end{{tabular}}
\\end{{table}}"""
    
    # Save LaTeX table
    os.makedirs("output", exist_ok=True)
    filepath = f"output/{filename}.tex"
    with open(filepath, "w") as f:
        f.write(latex_content)
    
    print(f"\n📄 LaTeX table saved to: {filepath}")
    print(f"   Use in thesis: \\input{{{filename}}}")
    print(f"   Reference as: \\ref{{tab:dataset_breakdown_harmonize}}")
    
    return latex_content


def export_detailed_breakdown_analysis(breakdown: Dict, filename: str = "dataset_type_breakdown_detailed") -> Dict:
    """
    Export detailed breakdown data for thesis analysis.
    
    Args:
        breakdown: Dataset type breakdown dictionary
        filename: Output filename (without extension)
    
    Returns:
        Summary statistics dictionary
    """
    
    # Create summary statistics
    summary = {}
    
    for dataset_type, data in breakdown.items():
        if not data['datasets']:
            continue
            
        summary[dataset_type] = {
            'num_datasets': len(data['datasets']),
            'dataset_names': data['datasets'],
            'gpt': {
                'avg_accuracy': sum(data['gpt_accuracies']) / len(data['gpt_accuracies']) if data['gpt_accuracies'] else 0,
                'avg_coverage': sum(data['gpt_coverages']) / len(data['gpt_coverages']) if data['gpt_coverages'] else 0,
                'accuracy_range': [min(data['gpt_accuracies']), max(data['gpt_accuracies'])] if data['gpt_accuracies'] else [0, 0],
                'accuracy_std': _calculate_std(data['gpt_accuracies']) if len(data['gpt_accuracies']) > 1 else 0
            },
            'embedding': {
                'avg_accuracy': sum(data['embed_accuracies']) / len(data['embed_accuracies']) if data['embed_accuracies'] else 0,
                'avg_coverage': sum(data['embed_coverages']) / len(data['embed_coverages']) if data['embed_coverages'] else 0,
                'accuracy_range': [min(data['embed_accuracies']), max(data['embed_accuracies'])] if data['embed_accuracies'] else [0, 0],
                'accuracy_std': _calculate_std(data['embed_accuracies']) if len(data['embed_accuracies']) > 1 else 0
            },
            'clustering': {
                'avg_accuracy': sum(data['cluster_accuracies']) / len(data['cluster_accuracies']) if data['cluster_accuracies'] else 0,
                'avg_coverage': sum(data['cluster_coverages']) / len(data['cluster_coverages']) if data['cluster_coverages'] else 0,
                'accuracy_range': [min(data['cluster_accuracies']), max(data['cluster_accuracies'])] if data['cluster_accuracies'] else [0, 0],
                'accuracy_std': _calculate_std(data['cluster_accuracies']) if len(data['cluster_accuracies']) > 1 else 0
            },
            'majority_vote': {
                'avg_accuracy': sum(data['majority_accuracies']) / len(data['majority_accuracies']) if data['majority_accuracies'] else 0,
                'avg_coverage': sum(data['majority_coverages']) / len(data['majority_coverages']) if data['majority_coverages'] else 0,
                'accuracy_range': [min(data['majority_accuracies']), max(data['majority_accuracies'])] if data['majority_accuracies'] else [0, 0],
                'accuracy_std': _calculate_std(data['majority_accuracies']) if len(data['majority_accuracies']) > 1 else 0
            }
        }
    
    # Export as JSON for detailed analysis
    filepath = f"output/{filename}.json"
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Detailed breakdown exported to: {filepath}")
    
    return summary


def analyze_dataset_type_insights(summary: Dict) -> None:
    """
    Print insights about performance across dataset types.
    
    Args:
        summary: Summary statistics from export_detailed_breakdown_analysis
    """
    
    print(f"\n💡 DATASET TYPE PERFORMANCE INSIGHTS:")
    print("=" * 60)
    
    for dataset_type, data in summary.items():
        if dataset_type == 'unknown':
            continue
            
        # Find best method for this dataset type
        methods = [
            ('GPT', data['gpt']['avg_accuracy']),
            ('Embedding', data['embedding']['avg_accuracy']), 
            ('Clustering', data['clustering']['avg_accuracy']),
            ('Majority Vote', data['majority_vote']['avg_accuracy'])
        ]
        
        best_method = max(methods, key=lambda x: x[1])
        worst_method = min(methods, key=lambda x: x[1])
        
        print(f"\n📈 {dataset_type.upper()}:")
        print(f"   🏆 Best: {best_method[0]} ({best_method[1]:.1%})")
        print(f"   📉 Worst: {worst_method[0]} ({worst_method[1]:.1%})")
        print(f"   📊 Performance Gap: {(best_method[1] - worst_method[1]):.1%}")
        
        # Check if ensemble helps
        individual_best = max([
            data['gpt']['avg_accuracy'],
            data['embedding']['avg_accuracy'], 
            data['clustering']['avg_accuracy']
        ])
        ensemble_performance = data['majority_vote']['avg_accuracy']
        
        if ensemble_performance > individual_best:
            improvement = ensemble_performance - individual_best
            print(f"   ✨ Ensemble Improvement: +{improvement:.1%}")
        elif ensemble_performance < individual_best:
            degradation = individual_best - ensemble_performance
            print(f"   ⚠️ Ensemble Degradation: -{degradation:.1%}")
        else:
            print(f"   ⚖️ Ensemble Same as Best Individual")


def _calculate_std(values: List[float]) -> float:
    """Calculate standard deviation of a list of values."""
    if not values:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def run_complete_dataset_type_analysis(all_results: List[Dict], all_triple_results: List[Dict]) -> Dict:
    """
    Run complete dataset type breakdown analysis.
    
    Args:
        all_results: List of result dictionaries from main evaluation
        all_triple_results: List of triple comparison results
    
    Returns:
        Complete analysis results
    """
    
    print(f"\n🔬 STEP 4: Dataset Type Performance Analysis")
    print("=" * 80)
    
    # Debug: Show first few dataset names to understand naming patterns
    for i, triple_result in enumerate(all_triple_results[:5]):
        if 'error' not in triple_result:
            source = triple_result.get('source_table', '')
            target = triple_result.get('target_table', '')
            print(f"  {i+1}. Source: '{source}' | Target: '{target}'")
    
    # Create breakdown
    breakdown = create_dataset_type_breakdown(all_results, all_triple_results)
    
    if not breakdown:
        print("⚠️ No dataset breakdown data available")
        return {}
    
    # Show what dataset types were found
    print(f"\n📊 Found dataset types: {list(breakdown.keys())}")
    for dtype, data in breakdown.items():
        print(f"  - {dtype}: {len(data['datasets'])} datasets")
    
    # Print table
    table_data = print_dataset_type_breakdown_table(breakdown)
    
    # Export LaTeX table
    latex_content = export_latex_breakdown_table(table_data)
    
    # Export detailed analysis
    summary = export_detailed_breakdown_analysis(breakdown)
    
    # Print insights
    analyze_dataset_type_insights(summary)
    
    return {
        'breakdown': breakdown,
        'table_data': table_data,
        'latex_content': latex_content,
        'summary': summary
    }