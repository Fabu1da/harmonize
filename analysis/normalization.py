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
    coma_path = os.path.normpath(os.path.join(base_dir, '..', 'output', 'matches.json'))
    ham_path  = os.path.normpath(os.path.join(base_dir, '..', 'output', 'real_data_detailed_matches.json'))
    gt_path  = os.path.normpath(os.path.join(base_dir, '..', 'expected'))

    # Load COMA results (simple matches) and fix European decimal format
    with open(coma_path, 'r') as f:
        content = f.read()
        # Replace patterns like "similarity":0,4083 with "similarity":0.4083
        content = re.sub(r'"similarity":(\d+),(\d+)', r'"similarity":\1.\2', content)
        coma_data = json.loads(content)

    # Load harmonize results (detailed matches)
    with open(ham_path, 'r') as f:
        harmonize_data = json.load(f)


    return coma_data, harmonize_data


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
        # Extract information from the new COMA format
        source_table = item['source_table']
        source_column = item['source_column']
        target_table = item['target_table']
        target_column = item['target_column']
        
        # print Debug
        print(f"COMA Item - Source: {source_table}.{source_column}, Target: {target_table}.{target_column}, Similarity: {item['similarity']}")

        # COMA already provides the correct table names with _source and _target suffixes
        # that match the ground truth format, so use them directly
        source_table_name = source_table
        target_table_name = target_table
        
        # Create pair
        pair = Pair(
            source_table=source_table_name,
            source_column=source_column,
            target_table=target_table_name,
            target_column=target_column
        )
        
        # Convert similarity score (handle European decimal format)
        similarity = item['similarity']
        if isinstance(similarity, str) and ',' in similarity:
            similarity = float(similarity.replace(',', '.'))
        else:
            similarity = float(similarity)
        
        pairs.append(pair)
        scores[pair] = similarity
    return Run(run_id=run_id, pairs=pairs, scores=scores)


def normalize_harmonize(harmonize_data):
    """Normalize harmonize data format, returns list of runs"""
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
