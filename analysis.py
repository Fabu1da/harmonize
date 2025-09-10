#!/usr/bin/env python3
"""
Analysis script to generate LaTeX table showing agreement distribution 
from pairwise analysis output files.
"""

import os
import re
import glob
from collections import Counter

def parse_tex_file(file_path):
    
   
    
    """Parse a LaTeX table file and extract agreement values."""
    agreement_values = []
    comparison_types = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Skip comments and LaTeX commands
            if line.strip().startswith('%') or line.strip().startswith('\\'):
                continue
            
            # Skip lines without proper table format
            if '&' not in line:
                continue
                
            # Match table rows with agreement data
            parts = line.split('&')
            if len(parts) >= 7:
                try:
                    # Extract agreement value (7th column, index 6)
                    agreement_str = parts[6].strip()
                    
                    # Remove any trailing LaTeX commands like \\
                    agreement_str = re.sub(r'\\.*$', '', agreement_str).strip()
                    
                    # Extract comparison type (3rd column, index 2)
                    comparison_str = parts[2].strip()
                    
                    # Skip header rows and invalid data
                    if (agreement_str and 
                        agreement_str != 'Agreement' and 
                        comparison_str != 'Comparison' and
                        'vs' in comparison_str.lower()):  # Only process matcher comparisons
                        
                        agreement = float(agreement_str)
                        # Only accept values between 0 and 1 (valid agreement scores)
                        if 0.0 <= agreement <= 1.0:
                            agreement_values.append(agreement)
                            comparison_types.append(comparison_str)
                        
                except (ValueError, IndexError):
                    continue
    
    return agreement_values, comparison_types

def analyze_output_directory(output_dir):
    """Analyze all .tex files in the output directory."""
    tex_files = glob.glob(os.path.join(output_dir, "*.tex"))
    
    all_agreements = []
    all_comparisons = []
    file_data = {}
    
    print(f"Found {len(tex_files)} .tex files to analyze")
    
    for tex_file in tex_files:
        filename = os.path.basename(tex_file)
        print(f"Processing: {filename}")
        print(f"Parsing file: {tex_file}")
        agreements, comparisons = parse_tex_file(tex_file)
        
        print(agreements)
        print(comparisons)
                
        if agreements:
            all_agreements.extend(agreements)
            all_comparisons.extend(comparisons)
            file_data[filename] = {
                'agreements': agreements,
                'comparisons': comparisons
            }
            print(f"  - Found {len(agreements)} agreement values")
        else:
            print(f"  - No data found")
    
    return all_agreements, all_comparisons, file_data

def generate_latex_chart(agreements, comparisons, output_dir):
    """Generate LaTeX chart showing agreement distribution using pgfplots."""
    
    # Calculate the three main categories
    perfect_count = agreements.count(1.0)
    zero_count = agreements.count(0.0)
    partial_count = len(agreements) - perfect_count - zero_count
    total = len(agreements)
    
    # Calculate percentages
    perfect_pct = (perfect_count / total) * 100 if total > 0 else 0
    zero_pct = (zero_count / total) * 100 if total > 0 else 0
    partial_pct = (partial_count / total) * 100 if total > 0 else 0
    
    # Generate LaTeX chart code using pgfplots
    latex_code = f"""\\begin{{figure}}[htbp]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar,
    width=10cm,
    height=7cm,
    ylabel={{Percentage (\\%)}},
    xlabel={{Agreement Category}},
    symbolic x coords={{Perfect Agreements, No Agreements, Partial Agreements}},
    xtick=data,
    x tick label style={{rotate=45, anchor=east}},
    ymin=0,
    ymax=50,
    bar width=20pt,
    nodes near coords,
    nodes near coords align={{vertical}},
    every node near coord/.append style={{font=\\small}},
    legend style={{at={{(0.5,-0.15)}},anchor=north,legend columns=-1}},
    grid=major,
    grid style={{dashed,gray!30}},
]
\\addplot[fill=blue!60] coordinates {{
    (Perfect Agreements, {perfect_pct:.1f})
    (No Agreements, {zero_pct:.1f})
    (Partial Agreements, {partial_pct:.1f})
}};
\\end{{axis}}
\\end{{tikzpicture}}
\\caption{{Agreement Distribution Analysis (N={total})}}
\\label{{fig:agreement_distribution}}
\\end{{figure}}

% Alternative table version for reference
\\begin{{table}}[htbp]
\\centering
\\caption{{Agreement Distribution Analysis - Detailed}}
\\label{{tab:agreement_distribution_detailed}}
\\begin{{tabular}}{{lcc}}
\\toprule
Agreement Category & Count & Percentage \\\\
\\midrule
Perfect Agreements (1.0) & {perfect_count} & {perfect_pct:.1f}\\% \\\\
No Agreements (0.0) & {zero_count} & {zero_pct:.1f}\\% \\\\
Partial Agreements (0.0 < x < 1.0) & {partial_count} & {partial_pct:.1f}\\% \\\\
\\midrule
Total & {total} & 100.0\\% \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""
    
    # Save LaTeX code to file
    latex_file = os.path.join(output_dir, "agreement_distribution_chart.tex")
    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    
    print(f"\nLaTeX chart saved to: {latex_file}")
    print("\nGenerated LaTeX code:")
    print("=" * 60)
    print(latex_code)
    print("=" * 60)
    
    return {
        'mean': sum(agreements) / len(agreements) if agreements else 0,
        'perfect_count': perfect_count,
        'zero_count': zero_count,
        'partial_count': partial_count,
        'total': total
    }

def main():
    """Main function to run the analysis."""
    # Set the output directory path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    if not os.path.exists(output_dir):
        print(f"Output directory not found: {output_dir}")
        return
    
    print("=== Agreement Distribution Analysis ===")
    print(f"Analyzing files in: {output_dir}\n")
    
    # Analyze all .tex files
    agreements, comparisons, file_data = analyze_output_directory(output_dir)
    
    if not agreements:
        print("No agreement data found in any .tex files!")
        return
    
    print(f"\nTotal agreement values found: {len(agreements)}")
    print(f"Unique comparison types: {set(comparisons)}")
    
    # Generate LaTeX chart
    stats = generate_latex_chart(agreements, comparisons, output_dir)
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Perfect agreements (1.0): {stats['perfect_count']}/{stats['total']} ({stats['perfect_count']/stats['total']*100:.1f}%)")
    print(f"No agreements (0.0): {stats['zero_count']}/{stats['total']} ({stats['zero_count']/stats['total']*100:.1f}%)")
    print(f"Partial agreements: {stats['partial_count']}/{stats['total']} ({stats['partial_count']/stats['total']*100:.1f}%)")
    print(f"Mean agreement: {stats['mean']:.3f}")

if __name__ == "__main__":
    main()
