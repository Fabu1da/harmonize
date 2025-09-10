#!/usr/bin/env python3
"""
Voting Consensus Analysis Script
Analyzes voting patterns: 3/3, 2/3, 1/3, 0/3 agreement across the three matchers.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

def parse_tex_file_for_voting(file_path):
    """Parse a LaTeX table file and extract voting data for each source-target pair."""
    voting_data = []
    current_pair_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip comments and LaTeX commands
            if line.strip().startswith('%') or line.strip().startswith('\\'):
                continue
            
            # Skip lines without proper table format
            if '&' not in line:
                continue
                
            # Match table rows with data
            parts = line.split('&')
            if len(parts) >= 7:
                try:
                    source = parts[0].strip()
                    target = parts[1].strip()
                    comparison = parts[2].strip()
                    precision_str = parts[3].strip()
                    recall_str = parts[4].strip()
                    f1_str = parts[5].strip()
                    agreement_str = parts[6].strip()
                    
                    # Clean up LaTeX formatting
                    agreement_str = re.sub(r'\\.*$', '', agreement_str).strip()
                    
                    # Skip header rows
                    if (agreement_str and agreement_str != 'Agreement' and 
                        'vs' in comparison.lower()):
                        
                        agreement = float(agreement_str)
                        pair_key = f"{source}_TO_{target}"
                        
                        # Store the agreement for this comparison type
                        if pair_key not in current_pair_data:
                            current_pair_data[pair_key] = {
                                'source': source,
                                'target': target,
                                'agreements': {}
                            }
                        
                        current_pair_data[pair_key]['agreements'][comparison] = agreement
                        
                except (ValueError, IndexError):
                    continue
    
    # Convert to voting analysis format
    for pair_key, data in current_pair_data.items():
        agreements = data['agreements']
        
        # We expect three comparisons: GPT vs Embedding, GPT vs Clustering, Embedding vs Clustering
        if len(agreements) == 3:
            # Get the three agreement values
            gpt_embedding = agreements.get('GPT vs Embedding', 0)
            gpt_clustering = agreements.get('GPT vs Clustering', 0)
            embedding_clustering = agreements.get('Embedding vs Clustering', 0)
            
            # Analyze voting patterns
            votes = [gpt_embedding, gpt_clustering, embedding_clustering]
            
            # Count perfect agreements (1.0)
            perfect_votes = sum(1 for v in votes if v == 1.0)
            
            # Count zero agreements (0.0)
            zero_votes = sum(1 for v in votes if v == 0.0)
            
            # Count partial agreements (between 0 and 1)
            partial_votes = sum(1 for v in votes if 0 < v < 1.0)
            
            # Determine consensus level
            consensus_level = determine_consensus_level(votes)
            
            voting_data.append({
                'source': data['source'],
                'target': data['target'],
                'pair_key': pair_key,
                'gpt_embedding': gpt_embedding,
                'gpt_clustering': gpt_clustering,
                'embedding_clustering': embedding_clustering,
                'perfect_votes': perfect_votes,
                'zero_votes': zero_votes,
                'partial_votes': partial_votes,
                'consensus_level': consensus_level,
                'mean_agreement': np.mean(votes),
                'std_agreement': np.std(votes),
                'min_agreement': np.min(votes),
                'max_agreement': np.max(votes)
            })
    
    return voting_data

def determine_consensus_level(votes):
    """Determine the consensus level based on voting patterns."""
    perfect_count = sum(1 for v in votes if v == 1.0)
    zero_count = sum(1 for v in votes if v == 0.0)
    partial_count = sum(1 for v in votes if 0 < v < 1.0)
    
    # Perfect consensus patterns
    if perfect_count == 3:
        return "3/3 Perfect Agreement"
    elif zero_count == 3:
        return "3/3 No Agreement"
    elif partial_count == 3:
        return "3/3 Partial Agreement"
    
    # 2/3 consensus patterns
    elif perfect_count == 2:
        return "2/3 Perfect Agreement"
    elif zero_count == 2:
        return "2/3 No Agreement"
    elif partial_count == 2:
        return "2/3 Partial Agreement"
    
    # 1/3 patterns (mostly disagreement)
    elif perfect_count == 1 and zero_count == 1:
        return "1/3 Each (Perfect/Zero/Partial)"
    elif perfect_count == 1 and zero_count == 0:
        return "1/3 Perfect, 2/3 Partial"
    elif zero_count == 1 and perfect_count == 0:
        return "1/3 Zero, 2/3 Partial"
    
    # 0/3 perfect agreement (all different levels)
    else:
        return "0/3 Consensus (Mixed)"

def analyze_voting_patterns(voting_data):
    """Analyze the voting patterns and generate statistics."""
    
    df = pd.DataFrame(voting_data)
    
    # Count consensus levels
    consensus_counts = df['consensus_level'].value_counts()
    
    # Group into main categories
    three_out_of_three = 0
    two_out_of_three = 0
    one_out_of_three = 0
    zero_out_of_three = 0
    
    for consensus, count in consensus_counts.items():
        if "3/3" in consensus:
            three_out_of_three += count
        elif "2/3" in consensus:
            two_out_of_three += count
        elif "1/3" in consensus:
            one_out_of_three += count
        elif "0/3" in consensus:
            zero_out_of_three += count
    
    total = len(df)
    
    # Detailed analysis
    analysis = {
        'total_pairs': total,
        'consensus_summary': {
            '3/3_agreement': three_out_of_three,
            '2/3_agreement': two_out_of_three,
            '1/3_agreement': one_out_of_three,
            '0/3_agreement': zero_out_of_three
        },
        'consensus_percentages': {
            '3/3_agreement': (three_out_of_three / total) * 100,
            '2/3_agreement': (two_out_of_three / total) * 100,
            '1/3_agreement': (one_out_of_three / total) * 100,
            '0/3_agreement': (zero_out_of_three / total) * 100
        },
        'detailed_breakdown': consensus_counts.to_dict(),
        'statistics': {
            'mean_agreement_overall': df['mean_agreement'].mean(),
            'std_agreement_overall': df['mean_agreement'].std(),
            'perfect_votes_total': df['perfect_votes'].sum(),
            'zero_votes_total': df['zero_votes'].sum(),
            'partial_votes_total': df['partial_votes'].sum()
        }
    }
    
    return analysis, df

def generate_voting_latex_analysis(analysis, df, output_dir):
    """Generate LaTeX tables and charts for voting analysis."""
    
    # Main voting consensus table
    consensus_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Voting Consensus Analysis: Agreement Patterns Across Three Matchers}}
\\label{{tab:voting_consensus}}
\\begin{{tabular}}{{lcc}}
\\toprule
Consensus Level & Count & Percentage \\\\
\\midrule
3/3 Agreement & {analysis['consensus_summary']['3/3_agreement']} & {analysis['consensus_percentages']['3/3_agreement']:.1f}\\% \\\\
2/3 Agreement & {analysis['consensus_summary']['2/3_agreement']} & {analysis['consensus_percentages']['2/3_agreement']:.1f}\\% \\\\
1/3 Agreement & {analysis['consensus_summary']['1/3_agreement']} & {analysis['consensus_percentages']['1/3_agreement']:.1f}\\% \\\\
0/3 Agreement & {analysis['consensus_summary']['0/3_agreement']} & {analysis['consensus_percentages']['0/3_agreement']:.1f}\\% \\\\
\\midrule
Total & {analysis['total_pairs']} & 100.0\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

"""
    
    # Detailed breakdown table
    detailed_table = """\\begin{table}[htbp]
\\centering
\\caption{Detailed Voting Pattern Breakdown}
\\label{tab:detailed_voting}
\\begin{tabular}{lc}
\\toprule
Specific Pattern & Count \\\\
\\midrule
"""
    
    # Sort by count for better presentation
    sorted_breakdown = sorted(analysis['detailed_breakdown'].items(), key=lambda x: x[1], reverse=True)
    
    for pattern, count in sorted_breakdown:
        pattern_clean = pattern.replace('_', '\\_')
        detailed_table += f"{pattern_clean} & {count} \\\\\n"
    
    detailed_table += """\\bottomrule
\\end{tabular}
\\end{table}

"""
    
    # Voting statistics table
    stats_table = f"""\\begin{{table}}[htbp]
\\centering
\\caption{{Voting Statistics Summary}}
\\label{{tab:voting_statistics}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
Total Dataset Pairs & {analysis['total_pairs']} \\\\
Mean Agreement Score & {analysis['statistics']['mean_agreement_overall']:.3f} \\\\
Agreement Std Deviation & {analysis['statistics']['std_agreement_overall']:.3f} \\\\
Total Perfect Votes (1.0) & {analysis['statistics']['perfect_votes_total']} \\\\
Total Zero Votes (0.0) & {analysis['statistics']['zero_votes_total']} \\\\
Total Partial Votes (0.0-1.0) & {analysis['statistics']['partial_votes_total']} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

"""
    
    # Generate voting distribution chart
    chart = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=10cm,
    height=7cm,
    ylabel={{Count}},
    xlabel={{Consensus Level}},
    symbolic x coords={{3/3 Agreement, 2/3 Agreement, 1/3 Agreement, 0/3 Agreement}},
    xtick=data,
    x tick label style={{rotate=45, anchor=east}},
    ymin=0,
    ymax={max(analysis['consensus_summary'].values()) + 20},
    bar width=20pt,
    nodes near coords,
    nodes near coords align={{vertical}},
    every node near coord/.append style={{font=\\small}},
    grid=major,
    grid style={{dashed,gray!30}},
]]
\\addplot[fill=blue!60] coordinates {{
    (3/3 Agreement, {analysis['consensus_summary']['3/3_agreement']})
    (2/3 Agreement, {analysis['consensus_summary']['2/3_agreement']})
    (1/3 Agreement, {analysis['consensus_summary']['1/3_agreement']})
    (0/3 Agreement, {analysis['consensus_summary']['0/3_agreement']})
}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Voting Consensus Distribution (N={analysis['total_pairs']})}}
\\label{{fig:voting_consensus_distribution}}
\\end{{figure}}

"""
    
    # Combine all LaTeX content
    full_latex = consensus_table + detailed_table + stats_table + chart
    
    # Save to file
    latex_file = os.path.join(output_dir, "voting_consensus_analysis.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(full_latex)
    
    print(f"\nVoting Consensus Analysis saved to: {latex_file}")
    return full_latex

def main():
    """Main function to run the voting consensus analysis."""
    # Set the output directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    print("=== Voting Consensus Analysis ===")
    print(f"Analyzing files in: {output_dir}\n")
    
    # Find the main summary files
    target_files = [
        "complete_pairwise_analysis_summary.tex",
        "global_pairwise_comparison_results_summary.tex"
    ]
    
    all_voting_data = []
    
    for filename in target_files:
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            print(f"Processing: {filename}")
            voting_data = parse_tex_file_for_voting(file_path)
            all_voting_data.extend(voting_data)
            print(f"  - Found {len(voting_data)} dataset pairs")
        else:
            print(f"  - File not found: {filename}")
    
    if not all_voting_data:
        print("No voting data found!")
        return
    
    print(f"\nTotal dataset pairs analyzed: {len(all_voting_data)}")
    
    # Analyze voting patterns
    analysis, df = analyze_voting_patterns(all_voting_data)
    
    # Print summary
    print("\n=== Voting Consensus Summary ===")
    print(f"3/3 Agreement: {analysis['consensus_summary']['3/3_agreement']} pairs ({analysis['consensus_percentages']['3/3_agreement']:.1f}%)")
    print(f"2/3 Agreement: {analysis['consensus_summary']['2/3_agreement']} pairs ({analysis['consensus_percentages']['2/3_agreement']:.1f}%)")
    print(f"1/3 Agreement: {analysis['consensus_summary']['1/3_agreement']} pairs ({analysis['consensus_percentages']['1/3_agreement']:.1f}%)")
    print(f"0/3 Agreement: {analysis['consensus_summary']['0/3_agreement']} pairs ({analysis['consensus_percentages']['0/3_agreement']:.1f}%)")
    
    print(f"\n=== Detailed Breakdown ===")
    for pattern, count in sorted(analysis['detailed_breakdown'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / analysis['total_pairs']) * 100
        print(f"{pattern}: {count} pairs ({percentage:.1f}%)")
    
    # Generate LaTeX analysis
    latex_output = generate_voting_latex_analysis(analysis, df, output_dir)
    
    # Save detailed results to CSV
    csv_file = os.path.join(output_dir, "voting_consensus_detailed.csv")
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed voting data saved to: {csv_file}")
    
    print("\n" + "="*60)
    print("Generated LaTeX Analysis:")
    print("="*60)
    print(latex_output[:1000] + "..." if len(latex_output) > 1000 else latex_output)

if __name__ == "__main__":
    main()
