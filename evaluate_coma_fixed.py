import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set

class COMAEvaluator:
    def __init__(self, matches_file: str, ground_truth_dir: str):
        self.matches_file = Path(matches_file)
        self.ground_truth_dir = Path(ground_truth_dir)
        
    def load_matches(self) -> List[Dict]:
        """Load COMA matches from JSON"""
        with open(self.matches_file) as f:
            return json.load(f)
    
    def load_ground_truth(self) -> Set[Tuple[str, str]]:
        """Load ground truth correspondences from all files in expected directory"""
        ground_truth = set()
        
        # Process all ground truth files in the directory
        for gt_file in self.ground_truth_dir.glob("*.json"):
            try:
                with open(gt_file) as f:
                    gt_data = json.load(f)
                
                # Handle the format with 'matches' key
                if isinstance(gt_data, dict) and 'matches' in gt_data:
                    for item in gt_data['matches']:
                        # Extract source and target in table.column format
                        source_table = item.get('source_table', '')
                        source_column = item.get('source_column', '')
                        target_table = item.get('target_table', '')
                        target_column = item.get('target_column', '')
                        
                        # Create source.column format to match COMA output
                        source_elem = f"{source_table}.{source_column}"
                        target_elem = f"{target_table}.{target_column}"
                        
                        ground_truth.add((source_elem, target_elem))
                        
            except Exception as e:
                print(f"Error loading {gt_file}: {e}")
                
        return ground_truth
    
    def normalize_element_name(self, full_path: str) -> str:
        """Extract table.column from full COMA path"""
        # COMA output: "Users_..._musicians_viewunion_source.musician"
        # Target: "musicians_viewunion_source.musician"
        if '.' in full_path:
            parts = full_path.split('.')
            if len(parts) >= 2:
                column_part = parts[-1]  # Last part is column
                path_part = parts[-2]   # Part before dot
                
                # Extract table name from path by finding the relevant segments
                path_segments = path_part.split('_')
                
                # Look for patterns like musicians_viewunion_source, musicians_joinable_target, etc.
                table_patterns = ['musicians_viewunion', 'musicians_joinable', 'musicians_unionable', 'musicians_semjoinable']
                
                for pattern in table_patterns:
                    pattern_parts = pattern.split('_')
                    # Find if this pattern exists in the path
                    for i in range(len(path_segments) - len(pattern_parts) + 1):
                        if path_segments[i:i+len(pattern_parts)] == pattern_parts:
                            # Found the pattern, now get the full table name including source/target
                            start_idx = i
                            # Look for source/target after the pattern
                            if i + len(pattern_parts) < len(path_segments):
                                suffix = path_segments[i + len(pattern_parts)]
                                if suffix in ['source', 'target']:
                                    table_name = '_'.join(path_segments[start_idx:start_idx + len(pattern_parts) + 1])
                                    return f"{table_name}.{column_part}"
                
                # Fallback: look for musicians at the start and source/target at the end
                musicians_idx = -1
                for i, segment in enumerate(path_segments):
                    if segment == 'musicians':
                        musicians_idx = i
                        break
                
                if musicians_idx >= 0:
                    # Find source or target from musicians onwards
                    for i in range(musicians_idx, len(path_segments)):
                        if path_segments[i] in ['source', 'target']:
                            table_name = '_'.join(path_segments[musicians_idx:i+1])
                            return f"{table_name}.{column_part}"
                
        return full_path
    
    def calculate_metrics(self, similarity_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        matches = self.load_matches()
        ground_truth = self.load_ground_truth()
        
        # Normalize COMA matches to table.column format
        normalized_matches = []
        for match in matches:
            normalized_source = self.normalize_element_name(match['source'])
            normalized_target = self.normalize_element_name(match['target'])
            
            normalized_matches.append({
                'source': normalized_source,
                'target': normalized_target,
                'similarity': match['similarity']
            })
        
        # Sort matches by similarity (descending)
        normalized_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Filter by threshold for precision/recall/F1
        filtered_matches = [m for m in normalized_matches if m['similarity'] >= similarity_threshold]
        predicted_pairs = set((m['source'], m['target']) for m in filtered_matches)
        
        # Basic metrics
        tp = len(predicted_pairs & ground_truth)
        fp = len(predicted_pairs - ground_truth)
        fn = len(ground_truth - predicted_pairs)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        coverage = recall  # Coverage is same as recall
        
        # MRR and Hits@k
        mrr_scores = []
        hits_at_1 = 0
        hits_at_3 = 0
        hits_at_5 = 0
        
        for gt_pair in ground_truth:
            # Find rank of this ground truth pair in matches
            rank = None
            for i, match in enumerate(normalized_matches):
                if (match['source'], match['target']) == gt_pair:
                    rank = i + 1  # 1-indexed
                    break
            
            if rank is not None:
                mrr_scores.append(1.0 / rank)
                if rank == 1:
                    hits_at_1 += 1
                if rank <= 3:
                    hits_at_3 += 1
                if rank <= 5:
                    hits_at_5 += 1
            else:
                mrr_scores.append(0.0)
        
        mrr = np.mean(mrr_scores) if mrr_scores else 0
        hits_1 = hits_at_1 / len(ground_truth) if ground_truth else 0
        hits_3 = hits_at_3 / len(ground_truth) if ground_truth else 0
        hits_5 = hits_at_5 / len(ground_truth) if ground_truth else 0
        
        # Find best F1 across different thresholds
        best_f1 = 0
        for threshold in np.arange(0.1, 1.0, 0.05):
            thresh_matches = [m for m in normalized_matches if m['similarity'] >= threshold]
            thresh_pairs = set((m['source'], m['target']) for m in thresh_matches)
            
            tp_thresh = len(thresh_pairs & ground_truth)
            fp_thresh = len(thresh_pairs - ground_truth)
            fn_thresh = len(ground_truth - thresh_pairs)
            
            prec_thresh = tp_thresh / (tp_thresh + fp_thresh) if (tp_thresh + fp_thresh) > 0 else 0
            rec_thresh = tp_thresh / (tp_thresh + fn_thresh) if (tp_thresh + fn_thresh) > 0 else 0
            f1_thresh = 2 * (prec_thresh * rec_thresh) / (prec_thresh + rec_thresh) if (prec_thresh + rec_thresh) > 0 else 0
            
            best_f1 = max(best_f1, f1_thresh)
        
        return {
            'coverage': coverage,
            'best_f1': best_f1,
            'precision': precision,
            'recall': recall,
            'mrr': mrr,
            'hits_at_1': hits_1,
            'hits_at_3': hits_3,
            'hits_at_5': hits_5
        }
    
    def print_results_table(self, metrics: Dict[str, float]):
        """Print results in LaTeX table format"""
        print("\\begin{table}[h]")
        print("\\centering")
        print("\\caption{COMA 3.0 schema matching performance.}")
        print("\\label{tab:coma_results}")
        print("\\begin{tabular}{lcccccc}")
        print("\\hline")
        print("Metric & Coverage & Best F1 & Precision & Recall & MRR & Hits@1/3/5 \\\\")
        print("\\hline")
        print(f"Value & {metrics['coverage']:.3f} & {metrics['best_f1']:.3f} & "
              f"{metrics['precision']:.3f} & {metrics['recall']:.3f} & "
              f"{metrics['mrr']:.3f} & {metrics['hits_at_1']:.3f} / "
              f"{metrics['hits_at_3']:.3f} / {metrics['hits_at_5']:.3f} \\\\")
        print("\\hline")
        print("\\end{tabular}")
        print("\\end{table}")
    
    def export_latex_table(self, metrics: Dict[str, float], filename: str = "coma_results.tex"):
        """Export results to LaTeX file"""
        latex_content = f"""\\begin{{table}}[h]
\\centering
\\caption{{COMA 3.0 schema matching performance.}}
\\label{{tab:coma_results}}
\\begin{{tabular}}{{lcccccc}}
\\hline
Metric & Coverage & Best F1 & Precision & Recall & MRR & Hits@1/3/5 \\\\
\\hline
Value & {metrics['coverage']:.3f} & {metrics['best_f1']:.3f} & {metrics['precision']:.3f} & {metrics['recall']:.3f} & {metrics['mrr']:.3f} & {metrics['hits_at_1']:.3f} / {metrics['hits_at_3']:.3f} / {metrics['hits_at_5']:.3f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""
        
        with open(filename, 'w') as f:
            f.write(latex_content)
        print(f"LaTeX table exported to {filename}")

def debug_evaluation():
    """Debug function to check data formats"""
    # Check COMA matches
    matches_file = Path("assets/output/matches.json")
    if matches_file.exists():
        with open(matches_file) as f:
            matches = json.load(f)
        print(f"COMA matches found: {len(matches)}")
        if matches:
            print("Sample COMA match:")
            print(json.dumps(matches[0], indent=2))
            
            # Test normalization
            evaluator = COMAEvaluator("assets/output/matches.json", "assets/expected")
            sample_source = matches[0]['source']
            sample_target = matches[0]['target']
            normalized_source = evaluator.normalize_element_name(sample_source)
            normalized_target = evaluator.normalize_element_name(sample_target)
            print(f"\nNormalization test:")
            print(f"Original source: {sample_source}")
            print(f"Normalized source: {normalized_source}")
            print(f"Original target: {sample_target}")
            print(f"Normalized target: {normalized_target}")
    else:
        print("No COMA matches file found!")
        return
    
    # Check ground truth
    gt_dir = Path("assets/expected")
    if gt_dir.exists():
        gt_files = list(gt_dir.glob("*.json"))
        print(f"\nGround truth files found: {len(gt_files)}")
        
        evaluator = COMAEvaluator("assets/output/matches.json", "assets/expected")
        ground_truth = evaluator.load_ground_truth()
        print(f"\nTotal ground truth pairs: {len(ground_truth)}")
        if ground_truth:
            print("Sample ground truth pairs:")
            for i, gt_pair in enumerate(list(ground_truth)[:5]):
                print(f"  {gt_pair}")
        
        for gt_file in gt_files[:3]:  # Show first 3 files
            print(f"\nFile: {gt_file.name}")
            try:
                with open(gt_file) as f:
                    gt_data = json.load(f)
                print(f"Type: {type(gt_data)}")
                if isinstance(gt_data, dict) and 'matches' in gt_data:
                    print(f"Length: {len(gt_data['matches'])}")
                    if gt_data['matches']:
                        print("Sample item:")
                        print(json.dumps(gt_data['matches'][0], indent=2))
            except Exception as e:
                print(f"Error reading {gt_file}: {e}")
    else:
        print("Ground truth directory not found!")
    
    # Check for format mismatches
    print("\n--- Format Analysis ---")
    if matches:
        match_format = set(matches[0].keys())
        print(f"COMA match keys: {match_format}")
        
        # Sample source/target values
        print(f"Sample source: '{matches[0]['source']}'")
        print(f"Sample target: '{matches[0]['target']}'")

# Usage
if __name__ == "__main__":
    print("Running debug evaluation...")
    debug_evaluation()
    
    print("\n" + "="*50)
    print("Running full evaluation...")
    
    evaluator = COMAEvaluator(
        "assets/output/matches.json",
        "assets/expected"
    )
    
    metrics = evaluator.calculate_metrics()
    evaluator.print_results_table(metrics)
    
    # Export to LaTeX file
    evaluator.export_latex_table(metrics, "coma_results.tex")
    
    # Also print raw metrics for debugging
    print("\nRaw metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")
