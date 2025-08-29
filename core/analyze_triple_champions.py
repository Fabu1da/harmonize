import json
import os

from typing import Any, Dict, List, Optional
from collections.abc import Callable, MutableMapping

from core.export_to_latex import export_table_as_latex
from json_schema import ObjectSchema
from tabulate import tabulate



def analyze_triple_champions(all_triple_results: List[Dict]) -> Dict:
    """Analyze triple comparison results and identify champion performance patterns."""
    if not all_triple_results:
        return {"error": "No triple results to analyze"}
    
    # Track different success patterns
    pattern_stats = {
        'all_correct': 0,
        'two_correct': 0, 
        'one_correct': 0,
        'none_correct': 0,
        'all_agree': 0,
        'majority_correct': 0,
        'total_columns': 0
    }
    
    # Track ensemble performance
    ensemble_stats = {
        'majority_vote_correct': 0,
        'all_agree_and_correct': 0,
        'best_individual_vs_majority': {'individual': 0, 'majority': 0}
    }
    
    # Track individual contributions to success
    individual_contributions = {
        'gpt_in_success': 0,
        'embedding_in_success': 0, 
        'clustering_in_success': 0
    }
    
    total_successful_cases = 0  # Cases where at least 2/3 are correct
    
    for result in all_triple_results:
        if 'error' in result:
            continue
            
        # Aggregate pattern statistics
        patterns = result.get('patterns', {})
        pattern_stats['all_correct'] += len(patterns.get('all_correct', []))
        pattern_stats['two_correct'] += len(patterns.get('two_correct', []))
        pattern_stats['one_correct'] += len(patterns.get('one_correct', []))
        pattern_stats['none_correct'] += len(patterns.get('none_correct', []))
        pattern_stats['majority_correct'] += len(patterns.get('majority_correct', []))
        
        # Analyze detailed results
        for detail in result.get('detailed_results', []):
            pattern_stats['total_columns'] += 1
            
            if detail.get('all_agree', False):
                pattern_stats['all_agree'] += 1
                
            # Count successful cases (2/3 or 3/3 correct)
            correct_count = sum([
                detail.get('gpt_correct', False),
                detail.get('embedding_correct', False), 
                detail.get('clustering_correct', False)
            ])
            
            if correct_count >= 2:
                total_successful_cases += 1
                
                # Track individual contributions to success
                if detail.get('gpt_correct', False):
                    individual_contributions['gpt_in_success'] += 1
                if detail.get('embedding_correct', False):
                    individual_contributions['embedding_in_success'] += 1
                if detail.get('clustering_correct', False):
                    individual_contributions['clustering_in_success'] += 1
            
            # Majority vote performance  
            if detail.get('majority_correct', False):
                ensemble_stats['majority_vote_correct'] += 1
                
            # All agree and correct cases
            if detail.get('all_agree', False) and correct_count == 3:
                ensemble_stats['all_agree_and_correct'] += 1
    
    # Calculate percentages
    total_cols = pattern_stats['total_columns']
    if total_cols > 0:
        # Create a copy of keys to avoid "dictionary changed size during iteration" error
        pattern_keys = list(pattern_stats.keys())
        for key in pattern_keys:
            if key != 'total_columns':
                pattern_stats[f'{key}_rate'] = pattern_stats[key] / total_cols
        
        ensemble_stats['majority_vote_accuracy'] = ensemble_stats['majority_vote_correct'] / total_cols
        ensemble_stats['consensus_accuracy'] = ensemble_stats['all_agree_and_correct'] / total_cols
        
        # Individual contribution rates in successful cases
        if total_successful_cases > 0:
            # Create a copy of keys to avoid iteration error
            contrib_keys = list(individual_contributions.keys())
            for key in contrib_keys:
                individual_contributions[f'{key}_rate'] = individual_contributions[key] / total_successful_cases
    
    return {
        'pattern_stats': pattern_stats,
        'ensemble_stats': ensemble_stats,
        'individual_contributions': individual_contributions,
        'total_successful_cases': total_successful_cases
    }




def create_triple_comparison_summary(all_triple_results: List[Dict[str, MutableMapping[str, Any]]]) -> Dict[str, Any]:
    """Create a comprehensive summary of all triple comparison results across datasets."""
    if not all_triple_results:
        return {"error": "No triple comparison results to summarize"}
    
    # Aggregate statistics across all datasets
    total_columns = 0
    total_all_correct = 0
    total_two_correct = 0
    total_one_correct = 0
    total_none_correct = 0
    total_agreements = 0
    total_majority_correct = 0
    
    dataset_summaries = []
    
    for result in all_triple_results:
        if 'error' in result:
            continue
            
        stats = result['summary_stats']
        source_table = result.get('source_table', 'Unknown')
        target_table = result.get('target_table', 'Unknown')
        
        # Aggregate totals
        dataset_total = stats['total_columns']
        total_columns += dataset_total
        total_all_correct += len(result['patterns']['all_correct'])
        total_two_correct += len(result['patterns']['two_correct'])
        total_one_correct += len(result['patterns']['one_correct'])
        total_none_correct += len(result['patterns']['none_correct'])
        total_majority_correct += len(result['patterns']['majority_correct'])
        
        # Calculate agreement count from detailed results
        agreement_count = sum(1 for detail in result['detailed_results'] if detail['all_agree'])
        total_agreements += agreement_count
        
        # Store dataset summary
        dataset_summaries.append({
            'source_table': source_table,
            'target_table': target_table,
            'total_columns': dataset_total,
            'all_correct_rate': stats['all_correct_rate'],
            'agreement_rate': stats['agreement_rate'],
            'majority_accuracy': stats['majority_accuracy']
        })
    
    # Calculate overall statistics
    overall_stats = {
        'total_columns': total_columns,
        'total_datasets': len([r for r in all_triple_results if 'error' not in r]),
        'all_correct_rate': total_all_correct / total_columns if total_columns > 0 else 0,
        'two_correct_rate': total_two_correct / total_columns if total_columns > 0 else 0,
        'one_correct_rate': total_one_correct / total_columns if total_columns > 0 else 0,
        'none_correct_rate': total_none_correct / total_columns if total_columns > 0 else 0,
        'agreement_rate': total_agreements / total_columns if total_columns > 0 else 0,
        'majority_accuracy': total_majority_correct / total_columns if total_columns > 0 else 0
    }
    
    return {
        'overall_stats': overall_stats,
        'dataset_summaries': dataset_summaries
    }
    
    
    
    
def print_triple_comparison_summary(summary: Dict):
    """Print formatted summary of all triple comparison results."""
    
    if 'error' in summary:
        print(f"❌ {summary['error']}")
        return
    
    overall = summary['overall_stats']
    
    print(f"\n🎯 TRIPLE COMPARISON GLOBAL SUMMARY")
    print("=" * 80)
    print(f"📊 Overall Statistics Across {overall['total_datasets']} Datasets:")
    print(f"   Total Columns Analyzed: {overall['total_columns']}")
    print(f"   All 3 Matchers Correct: {overall['all_correct_rate']:.1%}")
    print(f"   2/3 Matchers Correct:   {overall['two_correct_rate']:.1%}")
    print(f"   1/3 Matchers Correct:   {overall['one_correct_rate']:.1%}")
    print(f"   0/3 Matchers Correct:   {overall['none_correct_rate']:.1%}")
    print(f"   Agreement Rate:         {overall['agreement_rate']:.1%}")
    print(f"   Majority Vote Accuracy: {overall['majority_accuracy']:.1%}")
    
    # Dataset-by-dataset breakdown
    print(f"\n📋 Dataset Breakdown:")
    dataset_table = []
    headers = ["Source", "Target", "GT Columns", "All Correct", "Agreement", "Majority Acc"]
    
    for ds in summary['dataset_summaries']:
        dataset_table.append([
            ds['source_table'],
            ds['target_table'], 
            ds['total_columns'],
            f"{ds['all_correct_rate']:.1%}",
            f"{ds['agreement_rate']:.1%}",
            f"{ds['majority_accuracy']:.1%}"
        ])
    
    print(tabulate(dataset_table, headers=headers, tablefmt="fancy_grid"))
    
    # Export results
    print(f"\n💾 Exporting detailed summary...")
    os.makedirs("output", exist_ok=True)
    
    # Export as JSON
    with open("output/triple_comparison_global_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Export dataset table as LaTeX
    export_table_as_latex(
        dataset_table,
        headers,
        "triple_comparison_summary.tex",
        caption="Triple comparison summary across all datasets",
        label="tab:triple_comparison_summary"
    )
    
    print("✅ Global triple comparison summary exported to:")
    print("   📁 output/triple_comparison_global_summary.json")
    print("   📄 output/triple_comparison_summary.tex")