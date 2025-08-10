import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

from dataclasses import dataclass

# Load the data
def load_data():
    # Determine the directory where this script resides
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build absolute paths to the JSON files
    coma_path = os.path.normpath(os.path.join(base_dir, '..', 'assets', 'output', 'matches.json'))
    ham_path  = os.path.normpath(os.path.join(base_dir, '..', 'output', 'real_data_detailed_matches.json'))
    gt_path  = os.path.normpath(os.path.join(base_dir, '..', 'expected'))

    # Load COMA results (simple matches) and fix European decimal format
    with open(coma_path, 'r') as f:
        content = f.read()
        # Replace patterns like "similarity":0,4083 with "similarity":0.4083
        content = re.sub(r'"similarity":(\d+),(\d+)', r'"similarity":\1.\2', content)
        coma_data = json.loads(content)

    # Load Hamonize results (detailed matches)
    with open(ham_path, 'r') as f:
        hamonize_data = json.load(f)


    return coma_data, hamonize_data


# Load ground truth data
def load_ground_truth():
    
    # Determine the directory where this script resides
    base_dir = os.path.dirname(os.path.abspath(__file__))

    expected_folder = os.path.normpath(os.path.join(base_dir, '..', 'assets', 'expected'))
    ground_truth_data = {}
    
    # List all ground truth files
    if os.path.exists(expected_folder):
        print("Found ground truth files:")
        for file in os.listdir(expected_folder):
            if file.endswith('.json'):
                print(f"  - {file}")
                file_path = os.path.join(expected_folder, file)
                with open(file_path, 'r') as f:
                    ground_truth_data[file] = json.load(f)
    else:
        print(f"Expected folder not found at: {expected_folder}")
    
    return ground_truth_data

@dataclass(frozen=True)
class Pair:
    """Represents a source-target column pair"""
    source_table: str
    source_column: str
    target_table: str
    target_column: str
    
    def __str__(self):
        return f"{self.source_table}.{self.source_column} -> {self.target_table}.{self.target_column}"


def build_gt_set(gt_data):
    """Build ground truth set of Pair objects from all ground truth files"""
    gt_pairs = set()
    
    for filename, data in gt_data.items():
        print(f"Processing ground truth file: {filename}")
        matches = data.get('matches', [])
        
        for match in matches:
            pair = Pair(
                source_table=match['source_table'],
                source_column=match['source_column'],
                target_table=match['target_table'],
                target_column=match['target_column']
            )
            gt_pairs.add(pair)
    
    print(f"Built ground truth set with {len(gt_pairs)} unique pairs")
    return gt_pairs


@dataclass
class Run:
    """Represents a normalized run with pairs and scores"""
    run_id: str
    pairs: list  # List of Pair objects
    scores: dict  # Dict mapping Pair -> score

def normalize_coma(coma_data, run_id="COMA"):
    """Normalize COMA data format"""
    pairs = []
    scores = {}
    
    for item in coma_data:
        # Check if this is a column-level match (has column name) or table-level match
        source_parts = item['source'].split('.')
        target_parts = item['target'].split('.')
        
        # Skip table-level matches (no column specified)
        if len(source_parts) == 1 or len(target_parts) == 1:
            continue
            
        # Extract column names
        source_column = source_parts[-1]
        target_column = target_parts[-1]
        
        # Extract table names from the path
        source_path = source_parts[-2]
        source_path_parts = source_path.split('_')
        
        # Find the table name by looking for the pattern ending with "source"
        source_table = None
        for i in range(len(source_path_parts)):
            if source_path_parts[i] == 'source' and i + 1 < len(source_path_parts):
                # Everything after 'source' is the table name
                remaining_parts = source_path_parts[i+1:]
                source_table = '_'.join(remaining_parts)
                break
        
        # Extract target table name
        target_path = target_parts[-2]
        target_path_parts = target_path.split('_')
        
        target_table = None
        for i in range(len(target_path_parts)):
            if target_path_parts[i] == 'target' and i + 1 < len(target_path_parts):
                # Everything after 'target' is the table name
                remaining_parts = target_path_parts[i+1:]
                target_table = '_'.join(remaining_parts)
                break
        
        # Clean up target table name - remove "cvs_" prefix if present
        if target_table and target_table.startswith('cvs_'):
            target_table = target_table[4:]  # Remove "cvs_" prefix
        
        # Skip if we couldn't extract table names
        if source_table is None or target_table is None:
            continue
            
        
        # Create pair
        pair = Pair(
            source_table=source_table,
            source_column=source_column,
            target_table=target_table,
            target_column=target_column
        )
        
        # Convert similarity score (handle European decimal format)
        similarity_str = item['similarity']
        if ',' in similarity_str:
            similarity = float(similarity_str.replace(',', '.'))
        else:
            similarity = float(similarity_str)
        
        pairs.append(pair)
        scores[pair] = similarity
    return Run(run_id=run_id, pairs=pairs, scores=scores)


def normalize_harmonize(harmonize_data):
    """Normalize Hamonize data format, returns list of runs"""
    embed_pairs = []
    embed_scores = {}
    cluster_pairs = []
    cluster_scores = {}
    
    for item in harmonize_data:
        # Extract table and column names
        # Source: "real_musicians_joinable_source.musician" 
        source_parts = item['source'].split('.')
        source_column = source_parts[-1]
        # Remove "real_" prefix if present and extract table name
        source_table = source_parts[-2]
        if source_table.startswith('real_'):
            source_table = source_table[5:]  # Remove "real_" prefix
        
        # Target: "musicians_joinable_target.musicianID"
        target_parts = item['target'].split('.')
        target_column = target_parts[-1]
        target_table = target_parts[-2]
        
        # Create pair
        pair = Pair(
            source_table=source_table,
            source_column=source_column,
            target_table=target_table,
            target_column=target_column
        )
        
        # Get similarity score
        similarity = item['similarity']
        
        # Separate by matcher type
        matcher = item['matcher']
        if matcher == 'embed':
            embed_pairs.append(pair)
            embed_scores[pair] = similarity
        elif matcher == 'cluster':
            cluster_pairs.append(pair)
            cluster_scores[pair] = similarity
    
    # Create separate runs for each matcher
    embed_run = Run(run_id="HARMONIZE-embed", pairs=embed_pairs, scores=embed_scores)
    cluster_run = Run(run_id="HARMONIZE-cluster", pairs=cluster_pairs, scores=cluster_scores)

    
    return [embed_run, cluster_run]


def aggregate_max_by_pair(harm_runs, run_id="HARMONIZE-max"):
    """Aggregate max scores by pair from multiple Harmonize runs"""
    all_pairs = set()
    max_scores = {}
    
    # Collect all unique pairs and their max scores
    for run in harm_runs:
        for pair in run.pairs:
            all_pairs.add(pair)
            current_score = run.scores[pair]
            
            # Keep the maximum score for this pair
            if pair not in max_scores or current_score > max_scores[pair]:
                max_scores[pair] = current_score
    
    # Convert to list
    pairs_list = list(all_pairs)
    
    print(f"Aggregated max scores: {len(pairs_list)} unique pairs")
    return Run(run_id=run_id, pairs=pairs_list, scores=max_scores)
