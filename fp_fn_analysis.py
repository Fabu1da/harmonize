#!/usr/bin/env python3
"""
FP/FN Analysis Script
Generates detailed false positive and false negative analysis from pairwise comparison results.
"""

import os
import pandas as pd
import json
import re
from typing import Dict, List, Tuple

def parse_pairwise_metrics(metrics_str: str) -> Dict:
    """Parse PairwiseMetrics string representation into dictionary."""
    # Extract values using regex
    precision_match = re.search(r'precision=([0-9.]+)', metrics_str)
    recall_match = re.search(r'recall=([0-9.]+)', metrics_str) 
    f1_match = re.search(r'f1_score=([0-9.]+)', metrics_str)
    tp_match = re.search(r'true_positives=([0-9]+)', metrics_str)
    fp_match = re.search(r'false_positives=([0-9]+)', metrics_str)
    fn_match = re.search(r'false_negatives=([0-9]+)', metrics_str)
    agreement_match = re.search(r'agreement_rate=([0-9.]+)', metrics_str)
    
    return {
        'precision': float(precision_match.group(1)) if precision_match else 0.0,
        'recall': float(recall_match.group(1)) if recall_match else 0.0,
        'f1': float(f1_match.group(1)) if f1_match else 0.0,
        'TP': int(tp_match.group(1)) if tp_match else 0,
        'FP': int(fp_match.group(1)) if fp_match else 0,
        'FN': int(fn_match.group(1)) if fn_match else 0,
        'agreement_rate': float(agreement_match.group(1)) if agreement_match else 0.0
    }

def load_pairwise_data(output_dir: str) -> List[Dict]:
    """Load pairwise comparison data from JSON file."""
    json_file = os.path.join(output_dir, "complete_pairwise_analysis.json")
    
    if not os.path.exists(json_file):
        print(f"Error: {json_file} not found. Please run pairwise analysis first.")
        return []
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data

def load_detailed_matches(output_dir: str) -> List[Dict]:
    """Load detailed matching results from JSON files."""
    detailed_matches = []
    
    # Look for detailed match files
    match_files = [
        "real_data_detailed_matches.json",
        "real_data_detailed_matches 2.json", 
        "real_data_detailed_matches 3.json"
    ]
    
    for match_file in match_files:
        file_path = os.path.join(output_dir, match_file)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    detailed_matches.extend(data)
                print(f"Loaded {len(data)} matches from {match_file}")
            except Exception as e:
                print(f"Error loading {match_file}: {e}")
    
    return detailed_matches

def analyze_fp_fn_from_detailed_matches(detailed_matches: List[Dict], threshold: float = 0.5) -> Dict:
    """Analyze FP/FN from detailed matching results."""
    
    # Group matches by source/target file pairs
    file_pairs = {}
    
    for match in detailed_matches:
        src_file = match.get('src_file', 'unknown')
        trg_file = match.get('trg_file', 'unknown')
        matcher = match.get('matcher', 'unknown')
        
        key = f"{src_file}__{trg_file}__{matcher}"
        
        if key not in file_pairs:
            file_pairs[key] = []
        
        file_pairs[key].append(match)
    
    # Analyze each file pair
    analysis_results = {}
    
    for key, matches in file_pairs.items():
        src_file, trg_file, matcher = key.split('__')
        
        # Classify matches as TP, FP, FN based on threshold and ground truth
        true_positives = []
        false_positives = []
        false_negatives = []
        
        for match in matches:
            similarity = match.get('similarity', 0.0)
            ground_truth = match.get('ground_truth', '—')
            source_col = match.get('source', '')
            target_col = match.get('target', '')
            
            # Predicted as match if similarity >= threshold
            predicted_match = similarity >= threshold
            
            # True match if ground truth is not "—" or similar indicators
            actual_match = ground_truth not in ['—', '–', 'none', 'null', '', None]
            
            match_info = {
                'source': source_col,
                'target': target_col,
                'similarity': similarity,
                'ground_truth': ground_truth,
                'predicted_match': predicted_match,
                'actual_match': actual_match
            }
            
            if predicted_match and actual_match:
                true_positives.append(match_info)
            elif predicted_match and not actual_match:
                false_positives.append(match_info)
            elif not predicted_match and actual_match:
                false_negatives.append(match_info)
        
        # Calculate metrics
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        analysis_results[key] = {
            'src_file': src_file,
            'trg_file': trg_file,
            'matcher': matcher,
            'metrics': {
                'TP': tp_count,
                'FP': fp_count,
                'FN': fn_count,
                'precision': precision,
                'recall': recall,
                'f1': f1
            },
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    return analysis_results

def generate_fp_fn_detailed_results(output_dir: str) -> pd.DataFrame:
    """Generate detailed FP/FN results from pairwise comparison data."""
    
    pairwise_data = load_pairwise_data(output_dir)
    if not pairwise_data:
        return pd.DataFrame()
    
    detailed_results = []
    
    for dataset_pair in pairwise_data:
        source = dataset_pair['source_table']
        target = dataset_pair['target_table'] 
        comparisons = dataset_pair['comparisons']
        has_ground_truth = dataset_pair.get('has_ground_truth', False)
        
        # Process each comparison type
        for comparison_type, metrics_str in comparisons.items():
            metrics = parse_pairwise_metrics(metrics_str)
            
            # Clean up comparison type name
            comparison_name = comparison_type.replace('_vs_', ' vs ')
            
            result = {
                'source': source,
                'target': target,
                'comparison': comparison_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'TP': metrics['TP'],
                'FP': metrics['FP'],
                'FN': metrics['FN'],
                'agreement_rate': metrics['agreement_rate'],
                'has_ground_truth': has_ground_truth
            }
            
            detailed_results.append(result)
    
    return pd.DataFrame(detailed_results)

def generate_specific_fp_fn_examples(output_dir: str, threshold: float = 0.5) -> Dict:
    """Generate specific examples of false positives and false negatives."""
    
    detailed_matches = load_detailed_matches(output_dir)
    if not detailed_matches:
        print("No detailed match data found.")
        return {}
    
    print(f"Analyzing {len(detailed_matches)} detailed matches...")
    
    # Analyze FP/FN from detailed matches
    analysis_results = analyze_fp_fn_from_detailed_matches(detailed_matches, threshold)
    
    # Generate summary of FP/FN examples
    fp_fn_examples = {
        'summary': {
            'total_file_pairs': len(analysis_results),
            'total_fp_examples': 0,
            'total_fn_examples': 0
        },
        'false_positive_examples': [],
        'false_negative_examples': []
    }
    
    for key, result in analysis_results.items():
        fp_count = len(result['false_positives'])
        fn_count = len(result['false_negatives'])
        
        fp_fn_examples['summary']['total_fp_examples'] += fp_count
        fp_fn_examples['summary']['total_fn_examples'] += fn_count
        
        # Add examples (limit to first 5 per file pair to avoid overwhelming output)
        for fp in result['false_positives'][:5]:
            fp_example = {
                'type': 'False Positive',
                'src_file': result['src_file'],
                'trg_file': result['trg_file'],
                'matcher': result['matcher'],
                'source_column': fp['source'],
                'target_column': fp['target'],
                'similarity': fp['similarity'],
                'ground_truth': fp['ground_truth'],
                'explanation': f"Model predicted match (similarity={fp['similarity']:.3f}) but ground truth is '{fp['ground_truth']}'"
            }
            fp_fn_examples['false_positive_examples'].append(fp_example)
        
        for fn in result['false_negatives'][:5]:
            fn_example = {
                'type': 'False Negative',
                'src_file': result['src_file'],
                'trg_file': result['trg_file'], 
                'matcher': result['matcher'],
                'source_column': fn['source'],
                'target_column': fn['target'],
                'similarity': fn['similarity'],
                'ground_truth': fn['ground_truth'],
                'explanation': f"Model missed match (similarity={fn['similarity']:.3f}) but ground truth is '{fn['ground_truth']}'"
            }
            fp_fn_examples['false_negative_examples'].append(fn_example)
    
    return fp_fn_examples

def analyze_fp_fn_patterns(fp_fn_examples: Dict) -> Dict:
    """Analyze patterns in false positives and false negatives."""
    
    patterns = {
        'false_positive_patterns': {
            'high_similarity_fp': [],
            'semantic_confusion': [],
            'similarity_ranges': {'0.5-0.6': 0, '0.6-0.7': 0, '0.7-0.8': 0, '0.8-0.9': 0, '0.9-1.0': 0}
        },
        'false_negative_patterns': {
            'low_similarity_fn': [],
            'threshold_misses': [],
            'similarity_ranges': {'0.0-0.1': 0, '0.1-0.2': 0, '0.2-0.3': 0, '0.3-0.4': 0, '0.4-0.5': 0}
        },
        'matcher_performance': {}
    }
    
    # Analyze False Positives
    for fp in fp_fn_examples.get('false_positive_examples', []):
        similarity = fp['similarity']
        matcher = fp['matcher']
        source_col = fp['source_column'].split('.')[-1].lower()
        target_col = fp['target_column'].split('.')[-1].lower()
        
        # Track by matcher
        if matcher not in patterns['matcher_performance']:
            patterns['matcher_performance'][matcher] = {'fp_count': 0, 'fn_count': 0}
        patterns['matcher_performance'][matcher]['fp_count'] += 1
        
        # Similarity range analysis
        if 0.5 <= similarity < 0.6:
            patterns['false_positive_patterns']['similarity_ranges']['0.5-0.6'] += 1
        elif 0.6 <= similarity < 0.7:
            patterns['false_positive_patterns']['similarity_ranges']['0.6-0.7'] += 1
        elif 0.7 <= similarity < 0.8:
            patterns['false_positive_patterns']['similarity_ranges']['0.7-0.8'] += 1
        elif 0.8 <= similarity < 0.9:
            patterns['false_positive_patterns']['similarity_ranges']['0.8-0.9'] += 1
        elif 0.9 <= similarity <= 1.0:
            patterns['false_positive_patterns']['similarity_ranges']['0.9-1.0'] += 1
        
        # High similarity FPs (concerning cases)
        if similarity >= 0.7:
            patterns['false_positive_patterns']['high_similarity_fp'].append({
                'source': source_col,
                'target': target_col,
                'similarity': similarity,
                'matcher': matcher
            })
        
        # Semantic confusion (similar words but different meaning)
        if any(word in source_col for word in ['name', 'id', 'label']) and \
           any(word in target_col for word in ['name', 'id', 'label']):
            patterns['false_positive_patterns']['semantic_confusion'].append({
                'source': source_col,
                'target': target_col,
                'similarity': similarity,
                'type': 'name/id confusion'
            })
    
    # Analyze False Negatives
    for fn in fp_fn_examples.get('false_negative_examples', []):
        similarity = fn['similarity']
        matcher = fn['matcher']
        source_col = fn['source_column'].split('.')[-1].lower()
        target_col = fn['target_column'].split('.')[-1].lower()
        
        # Track by matcher
        if matcher not in patterns['matcher_performance']:
            patterns['matcher_performance'][matcher] = {'fp_count': 0, 'fn_count': 0}
        patterns['matcher_performance'][matcher]['fn_count'] += 1
        
        # Similarity range analysis
        if 0.0 <= similarity < 0.1:
            patterns['false_negative_patterns']['similarity_ranges']['0.0-0.1'] += 1
        elif 0.1 <= similarity < 0.2:
            patterns['false_negative_patterns']['similarity_ranges']['0.1-0.2'] += 1
        elif 0.2 <= similarity < 0.3:
            patterns['false_negative_patterns']['similarity_ranges']['0.2-0.3'] += 1
        elif 0.3 <= similarity < 0.4:
            patterns['false_negative_patterns']['similarity_ranges']['0.3-0.4'] += 1
        elif 0.4 <= similarity < 0.5:
            patterns['false_negative_patterns']['similarity_ranges']['0.4-0.5'] += 1
        
        # Low similarity FNs (very concerning cases - should have been obvious matches)
        if similarity < 0.3:
            patterns['false_negative_patterns']['low_similarity_fn'].append({
                'source': source_col,
                'target': target_col,
                'similarity': similarity,
                'matcher': matcher,
                'ground_truth': fn['ground_truth']
            })
        
        # Near-threshold misses (just below 0.5)
        if 0.4 <= similarity < 0.5:
            patterns['false_negative_patterns']['threshold_misses'].append({
                'source': source_col,
                'target': target_col,
                'similarity': similarity,
                'matcher': matcher,
                'ground_truth': fn['ground_truth']
            })
    
    return patterns

def main():
    """Main function for FP/FN analysis generation."""
    print("=== FP/FN Analysis Generation ===\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'output')
    
    # Generate detailed results from pairwise data
    df = generate_fp_fn_detailed_results(output_dir)
    
    if df.empty:
        print("No pairwise data found. Please run pairwise analysis first.")
        return
    
    print(f"Loaded {len(df)} real data points from pairwise analysis")
    
    # Note: Using only real data for thesis analysis - no synthetic augmentation
    print(f"Using {len(df)} real data points for analysis (no synthetic data added)")
    
    # Save detailed results to CSV
    csv_file = os.path.join(output_dir, "fp_fn_detailed_results.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"FP/FN detailed results saved to: {csv_file}")
    print(f"Total data points: {len(df)}")
    
    # Generate specific FP/FN examples from detailed matches
    print("\n=== Generating Specific FP/FN Examples ===")
    fp_fn_examples = generate_specific_fp_fn_examples(output_dir, threshold=0.5)
    
    if fp_fn_examples:
        # Save detailed FP/FN examples
        examples_file = os.path.join(output_dir, "fp_fn_specific_examples.json")
        with open(examples_file, 'w') as f:
            json.dump(fp_fn_examples, f, indent=2)
        
        print(f"Specific FP/FN examples saved to: {examples_file}")
        print(f"Total FP examples: {fp_fn_examples['summary']['total_fp_examples']}")
        print(f"Total FN examples: {fp_fn_examples['summary']['total_fn_examples']}")
        
        # Show some examples
        if fp_fn_examples['false_positive_examples']:
            print(f"\nSample False Positives (showing first 3):")
            for i, fp in enumerate(fp_fn_examples['false_positive_examples'][:3]):
                print(f"  {i+1}. {fp['source_column']} -> {fp['target_column']}")
                print(f"     Similarity: {fp['similarity']:.3f}, Ground Truth: '{fp['ground_truth']}'")
                print(f"     {fp['explanation']}")
        
        if fp_fn_examples['false_negative_examples']:
            print(f"\nSample False Negatives (showing first 3):")
            for i, fn in enumerate(fp_fn_examples['false_negative_examples'][:3]):
                print(f"  {i+1}. {fn['source_column']} -> {fn['target_column']}")
                print(f"     Similarity: {fn['similarity']:.3f}, Ground Truth: '{fn['ground_truth']}'")
                print(f"     {fn['explanation']}")
        
        # Analyze patterns in FP/FN
        print("\n=== Analyzing FP/FN Patterns ===")
        patterns = analyze_fp_fn_patterns(fp_fn_examples)
        
        # Save pattern analysis
        patterns_file = os.path.join(output_dir, "fp_fn_patterns_analysis.json")
        with open(patterns_file, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        print(f"Pattern analysis saved to: {patterns_file}")
        
        # Display key insights
        print("\nKey Pattern Insights:")
        print("False Positive Distribution by Similarity:")
        for range_key, count in patterns['false_positive_patterns']['similarity_ranges'].items():
            if count > 0:
                print(f"  {range_key}: {count} FPs")
        
        print("\nFalse Negative Distribution by Similarity:")
        for range_key, count in patterns['false_negative_patterns']['similarity_ranges'].items():
            if count > 0:
                print(f"  {range_key}: {count} FNs")
        
        print("\nMatcher Performance:")
        for matcher, perf in patterns['matcher_performance'].items():
            print(f"  {matcher}: {perf['fp_count']} FPs, {perf['fn_count']} FNs")
        
        # Highlight concerning cases
        high_sim_fps = len(patterns['false_positive_patterns']['high_similarity_fp'])
        low_sim_fns = len(patterns['false_negative_patterns']['low_similarity_fn'])
        threshold_misses = len(patterns['false_negative_patterns']['threshold_misses'])
        
        print(f"\nConcerning Cases:")
        print(f"  High similarity FPs (≥0.7): {high_sim_fps}")
        print(f"  Very low similarity FNs (<0.3): {low_sim_fns}")
        print(f"  Near-threshold FNs (0.4-0.5): {threshold_misses}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print("\nBy Comparison Type:")
    summary = df.groupby('comparison').agg({
        'precision': ['mean', 'std', 'count'],
        'recall': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'FP': ['mean', 'std'],
        'FN': ['mean', 'std']
    }).round(3)
    
    print(summary)
    
    print("\n=== Performance Categories ===")
    high_fp = len(df[df['precision'] < 0.5])
    high_fn = len(df[df['recall'] < 0.5]) 
    perfect = len(df[(df['precision'] == 1.0) & (df['recall'] == 1.0)])
    
    print(f"High False Positives (precision < 0.5): {high_fp}")
    print(f"High False Negatives (recall < 0.5): {high_fn}")
    print(f"Perfect matches: {perfect}")
    print(f"Other: {len(df) - high_fp - high_fn - perfect}")

if __name__ == "__main__":
    main()
