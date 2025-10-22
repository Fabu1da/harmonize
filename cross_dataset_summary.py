

import json
import os
from typing import  List
from collections import defaultdict, Counter
from tabulate import tabulate


class CrossDatasetSummaryGenerator:
    """
    Generates comprehensive cross-dataset summary tables for schema matching evaluation.
    """
    
    def __init__(self):
        self.all_results = []
        self.error_patterns = defaultdict(list)
        self.approach_failures = defaultdict(lambda: defaultdict(int))
        self.domain_analysis = defaultdict(list)
        
    def add_dataset_results(self, dataset_name: str, detailed_matches: dict, ground_truth: dict):
        # Convert ground truth to expected format
        # ground_truth is expected to be in format: Dict[target_col: source_col]
        gt_mapping = ground_truth if isinstance(ground_truth, dict) else {}
        
        # Try to infer domain from dataset name or use default
        domain = "unknown"
        if "amazon" in dataset_name.lower() or "walmart" in dataset_name.lower():
            domain = "e-commerce"
        elif "assay" in dataset_name.lower() or "medical" in dataset_name.lower():
            domain = "medical"
        elif "monitoring" in dataset_name.lower():
            domain = "monitoring"
        # Add more domain classifications as needed
        
        # Extract predictions from detailed_matches
        dataset_result = {
            'dataset_name': dataset_name,
            'ground_truth': gt_mapping,
            'predictions': detailed_matches,
            'domain': domain,
        }
        
        self.all_results.append(dataset_result)

    def _categorize_error(self, target_col: str, ground_truth_source: str, 
                         predicted_source: str, confidence: float, dataset_domain: str) -> str:
        """Categorize the type of error for better analysis."""
        if predicted_source == ground_truth_source:
            return "Correct"
            
        if predicted_source == "—" or predicted_source is None:
            return "No Prediction"
            
        if ground_truth_source is None:
            return "No Ground Truth"
            
        # Handle cross-domain scenarios (when no prediction available)
        if predicted_source == "— (No Prediction)":
            return f"Cross-Domain Gap ({dataset_domain})"
            
        # Semantic analysis
        target_words = set(target_col.lower().replace('_', ' ').split())
        gt_words = set(ground_truth_source.lower().replace('_', ' ').split())
        pred_words = set(predicted_source.lower().replace('_', ' ').split())
        
        # Check for semantic incompatibility
        if len(target_words & pred_words) == 0 and len(gt_words & pred_words) == 0:
            return f"Semantic Incompatibility ({dataset_domain})"
            
        # Check for partial semantic match
        if len(target_words & pred_words) > 0 or len(gt_words & pred_words) > 0:
            return "Partial Semantic Match"
            
        # Check for domain-specific errors
        if confidence > 0.8:
            return "High Confidence Error"
        elif confidence < 0.3:
            return "Low Confidence Error"
        else:
            return "Medium Confidence Error"
            
    def generate_cross_dataset_summary(self) -> str:
        """
        Generate the comprehensive cross-dataset summary table.
        
        Returns:
            Formatted table string
        """
        summary_rows = []
        
        for result in self.all_results:
            dataset_name = result['dataset_name']
            ground_truth = result['ground_truth']
            predictions = result['predictions']
            domain = result['domain']
            
            for target_col, gt_source in ground_truth.items():
                row_data = {
                    'dataset': dataset_name,
                    'domain': domain,
                    'target_column': target_col,
                    'ground_truth': gt_source
                }
                
                # Get predictions from different approaches
                approach_results = {}
                error_reasons = []
                
                for approach_name, approach_preds in predictions.items():
                    if target_col in approach_preds:
                        pred_source, confidence, reasoning = approach_preds[target_col]
                        
                        # Format prediction with confidence and correctness
                        correctness = "✅ CORRECT" if pred_source == gt_source else "❌ WRONG"
                        if pred_source == "—" or pred_source is None:
                            correctness = "❌ NO PRED"
                        
                        formatted_pred = f"{pred_source} ({confidence:.2f})\n{correctness}"
                        approach_results[approach_name] = formatted_pred
                        
                        # Analyze error reason
                        if pred_source != gt_source:
                            error_category = self._categorize_error(
                                target_col, gt_source, pred_source, confidence, domain
                            )
                            error_reasons.append(error_category)
                    else:
                        approach_results[approach_name] = "—\n❌ NO PRED"
                        error_reasons.append("Missing Prediction")
                
                # Combine error reasons
                if error_reasons:
                    most_common_error = Counter(error_reasons).most_common(1)[0][0]
                    row_data['error_reason'] = most_common_error
                else:
                    row_data['error_reason'] = "All Correct"
                
                # Add approach results
                row_data.update(approach_results)
                summary_rows.append(row_data)
        
        # Create the table
        if not summary_rows:
            return "No data available for cross-dataset summary."
            
        # Get all unique approach names
        all_approaches = set()
        for row in summary_rows:
            for key in row.keys():
                if key not in ['dataset', 'domain', 'target_column', 'ground_truth', 'error_reason']:
                    all_approaches.add(key)
        
        # Create headers
        headers = ['Dataset', 'Domain', 'Target Column', 'Ground Truth']
        headers.extend(sorted(all_approaches))
        headers.append('Error Analysis')
        
        # Create table rows
        table_rows = []
        for row in summary_rows:
            table_row = [
                row['dataset'],
                row['domain'], 
                row['target_column'],
                row['ground_truth']
            ]
            
            # Add approach predictions
            for approach in sorted(all_approaches):
                table_row.append(row.get(approach, "—\n❌ NO DATA"))
                
            table_row.append(row['error_reason'])
            table_rows.append(table_row)
        
        # Generate the formatted table
        table = tabulate(
            table_rows,
            headers=headers,
            tablefmt="fancy_grid",
        )
        
        return table
        
    def generate_domain_analysis(self) -> str:
        """Generate domain-specific error analysis."""
        domain_stats = defaultdict(lambda: {
            'total_mappings': 0,
            'correct_predictions': defaultdict(int),
            'total_predictions': defaultdict(int),
            'error_types': defaultdict(int),
            'datasets': set()
        })
        
        for result in self.all_results:
            domain = result['domain']
            dataset_name = result['dataset_name']
            ground_truth = result['ground_truth']
            predictions = result['predictions']
            
            domain_stats[domain]['datasets'].add(dataset_name)
            domain_stats[domain]['total_mappings'] += len(ground_truth)
            
            for target_col, gt_source in ground_truth.items():
                for approach_name, approach_preds in predictions.items():
                    if target_col in approach_preds:
                        pred_source, confidence, _ = approach_preds[target_col]
                        domain_stats[domain]['total_predictions'][approach_name] += 1
                        
                        if pred_source == gt_source:
                            domain_stats[domain]['correct_predictions'][approach_name] += 1
                        else:
                            error_type = self._categorize_error(
                                target_col, gt_source, pred_source, confidence, domain
                            )
                            domain_stats[domain]['error_types'][error_type] += 1
        
        # Create domain analysis table
        domain_rows = []
        for domain, stats in domain_stats.items():
            datasets_str = ", ".join(sorted(stats['datasets']))
            
            # Calculate accuracy for each approach
            approach_accuracies = []
            for approach in sorted(stats['total_predictions'].keys()):
                correct = stats['correct_predictions'][approach]
                total = stats['total_predictions'][approach]
                accuracy = correct / total if total > 0 else 0.0
                approach_accuracies.append(f"{approach}: {accuracy:.2f}")
            
            # Top error types
            error_counter = Counter(stats['error_types'])
            top_errors = error_counter.most_common(3)
            error_str = ", ".join([f"{error}: {count}" for error, count in top_errors])
            
            domain_rows.append([
                domain,
                len(stats['datasets']),
                datasets_str,
                stats['total_mappings'],
                "\n".join(approach_accuracies),
                error_str
            ])
        
        domain_headers = [
            "Domain", "# Datasets", "Dataset Names", "Total Mappings", 
            "Approach Accuracies", "Top Error Types"
        ]
        
        domain_table = tabulate(
            domain_rows,
            headers=domain_headers,
            tablefmt="fancy_grid",
        )
        
        return domain_table
        
    def save_comprehensive_report(self, output_path: str = "output/cross_dataset_comprehensive_report.md"):
        """Save a complete markdown report with all analyses."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Cross-Dataset Comprehensive Analysis Report\n\n")
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Datasets Analyzed**: {len(self.all_results)}\n")
            
            total_mappings = sum(len(r['ground_truth']) for r in self.all_results)
            f.write(f"- **Total Schema Mappings**: {total_mappings}\n")
            
            domains = set(r['domain'] for r in self.all_results)
            f.write(f"- **Domains Covered**: {', '.join(sorted(domains))}\n\n")
            
            f.write("## Cross-Dataset Summary Table\n\n")
            f.write("This table provides a comprehensive view of all predictions across all datasets:\n\n")
            f.write("```\n")
            f.write(self.generate_cross_dataset_summary())
            f.write("\n```\n\n")
            
            f.write("## Domain-Specific Analysis\n\n")
            f.write("Analysis of performance patterns by domain:\n\n")
            f.write("```\n")
            f.write(self.generate_domain_analysis())
            f.write("\n```\n\n")
            
            f.write("## Key Findings\n\n")
            f.write("### Error Pattern Analysis\n")
            
            # Analyze error patterns
            all_error_types = Counter()
            for result in self.all_results:
                predictions = result['predictions']
                ground_truth = result['ground_truth']
                domain = result['domain']
                
                for target_col, gt_source in ground_truth.items():
                    for approach_name, approach_preds in predictions.items():
                        if target_col in approach_preds:
                            pred_source, confidence, _ = approach_preds[target_col]
                            if pred_source != gt_source:
                                error_type = self._categorize_error(
                                    target_col, gt_source, pred_source, confidence, domain
                                )
                                all_error_types[error_type] += 1
            
            for error_type, count in all_error_types.most_common(5):
                f.write(f"- **{error_type}**: {count} occurrences\n")
            
            f.write("\n### Recommendations for Improvement\n")
            f.write("Based on the cross-dataset analysis:\n\n")
            f.write("1. **ID/Identifier Handling**: Implement specialized logic for different ID types\n")
            f.write("2. **Domain-Specific Training**: Consider domain-specific fine-tuning\n") 
            f.write("3. **Confidence Calibration**: Improve confidence scoring for edge cases\n")
            f.write("4. **Semantic Enhancement**: Enhance semantic similarity for domain-specific terms\n")
        
        print(f"📄 Comprehensive cross-dataset report saved to: {output_path}")
        return output_path


def run_cross_dataset_analysis(detailed_matches_list: List[dict], 
                              ground_truth_list: List[dict],
                              dataset_names: List[str] = None) -> str:
    """
    Main function to run cross-dataset analysis on multiple datasets.
    """
    generator = CrossDatasetSummaryGenerator()
    
    for i, (matches, gt) in enumerate(zip(detailed_matches_list, ground_truth_list)):
        if gt is not None:
            dataset_name = dataset_names[i] if dataset_names else f"Dataset_{i+1}"
            generator.add_dataset_results(dataset_name, matches, gt)
    
    # Save comprehensive report
    report_path = generator.save_comprehensive_report()
    
    return report_path

