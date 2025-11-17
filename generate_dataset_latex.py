#!/usr/bin/env python3
"""
Script to generate LaTeX table showing dataset names.
Creates a simple table for datasets extracted from scores.json.
"""

import json
import os
from typing import List

def load_scores_data(scores_path: str) -> dict:
    """Load the scores data from JSON file."""
    with open(scores_path, 'r') as f:
        return json.load(f)

def extract_datasets_from_scores(scores_data: dict) -> List[str]:
    """Extract unique dataset names from scores data, excluding subcategories."""
    datasets = set()
    
    # The scores_data has approach names as keys, and each approach has 'per_table' with dataset keys
    for approach_name, approach_data in scores_data.items():
        if approach_name == 'overall':
            continue  # Skip the overall summary
            
        if 'per_table' in approach_data:
            per_table_data = approach_data['per_table']
            for dataset_key in per_table_data.keys():
                # Keys are like 'synthetic/valentine/ChEMBL/Joinable/assays_both_50_1_ac1_ev'
                # or sometimes just 'valentine/ChEMBL/...'
                parts = dataset_key.split('/')
                
                # Find the index of 'valentine' and get the next part
                try:
                    valentine_index = parts.index('valentine')
                    if valentine_index + 1 < len(parts):
                        dataset_name = parts[valentine_index + 1]
                        # Exclude subcategories like 'Semantically-Joinable', 'Unionable', etc.
                        datasets.add(dataset_name)
                except ValueError:
                    continue  # 'valentine' not found in parts
    
    return sorted(list(datasets))

def generate_simple_dataset_list(datasets: List[str]) -> str:
    """Generate a simple LaTeX table showing just the dataset names in one line."""

    latex_code = """\\begin{table}[h]
\\centering
\\begin{tabular}{|l|}
\\hline
Datasets \\\\
\\hline
""" + ", ".join(datasets[:-1]) + ", and " + ", ".join(datasets[-1:]) + """ \\\\
\\hline
\\end{tabular}
\\caption{Available Datasets}
\\label{tab:datasets}
\\end{table}
"""
    return latex_code

def main():
    """Main function to generate dataset LaTeX table."""

    # Configuration
    scores_path = "./assets/predicted/scores.json"
    output_dir = "./assets/reports/variables"

    # Load scores data and extract datasets
    print("Loading scores data...")
    scores_data = load_scores_data(scores_path)
    
    print("Extracting dataset names...")
    datasets = extract_datasets_from_scores(scores_data)
    print(f"Found datasets: {datasets}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save simple dataset list
    print("Generating LaTeX table...")

    dataset_list = generate_simple_dataset_list(datasets)
    with open(f"{output_dir}/datasets.tex", 'w') as f:
        f.write(dataset_list)
    print(f"Saved dataset list to {output_dir}/datasets.tex")

    print("LaTeX table generated successfully!")

if __name__ == "__main__":
    main()