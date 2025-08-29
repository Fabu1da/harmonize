from typing import List

from core import analyze_triple_champions, print_triple_champions
from core.pairwise import analyze_pairwise_champions, print_pairwise_champions
from core.rank import create_individual_matcher_ranking, print_individual_matcher_ranking
from dataset_type_breakdown import create_dataset_type_breakdown, export_latex_breakdown_table, print_dataset_type_breakdown_table


def generate_global_summaries(all_triple_results: List, all_pairwise_results: List):
    """Generate global summary reports"""
    # Triple comparison summary
    print(f"\n🔄 STEP 3 COMPLETE: Creating Global Triple Comparison Summary")
    print("=" * 80)
    
    if all_triple_results:
        global_triple_summary = analyze_triple_champions.create_triple_comparison_summary(all_triple_results)
        analyze_triple_champions.print_triple_comparison_summary(global_triple_summary)
        
        # Individual Matcher Performance Ranking
        individual_rankings = create_individual_matcher_ranking(all_triple_results)
        total_predictions = sum(len(result.get('detailed_results', [])) 
                              for result in all_triple_results 
                              if 'error' not in result)
        print_individual_matcher_ranking(individual_rankings, total_predictions)
        
        # Triple Champions & Ensemble Analysis
        triple_analysis = analyze_triple_champions.analyze_triple_champions(all_triple_results)
        print_triple_champions.print_triple_champions(triple_analysis)
        
        # Dataset Type Breakdown Analysis
        print(f"\n📊 DATASET TYPE BREAKDOWN ANALYSIS")
        print("=" * 80)
        dataset_breakdown = create_dataset_type_breakdown([], all_triple_results)
        table_data = print_dataset_type_breakdown_table(dataset_breakdown)
        
        # Export LaTeX table for thesis
        export_latex_breakdown_table(table_data, "dataset_breakdown_harmonize")
        print(f"✅ LaTeX table exported to output/dataset_breakdown_harmonize.tex")
    else:
        print("⚠️ No triple comparison results to summarize")

    # Pairwise comparison summary
    print(f"\n🔄 STEP 2 COMPLETE: Creating Global Pairwise Comparison Summary")
    print("=" * 80)

    if all_pairwise_results:
        from pairwise_comparison import export_pairwise_results
        
        summary_data, headers = export_pairwise_results(
            all_pairwise_results, "global_pairwise_comparison_results"
        )
        
        # Analyze and display pairwise champions
        pairwise_performance = analyze_pairwise_champions(all_pairwise_results)
        print_pairwise_champions(pairwise_performance)
        
        print(f"✅ Pairwise comparison summary exported for {len(all_pairwise_results)} dataset pairs")
    else:
        print("⚠️ No pairwise comparison results to summarize")