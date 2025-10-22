#!/usr/bin/env python3
"""
Error Analysis Module for Schema Matching Thesis
Comprehensive analysis of prediction errors, failure modes, and error patterns
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import os
from datetime import datetime


class ErrorAnalyzer:
    """Comprehensive error analysis for schema matching approaches"""
    
    def __init__(self):
        self.error_data = []
        self.approach_errors = defaultdict(list)
        self.column_type_errors = defaultdict(list)
        self.confidence_errors = defaultdict(list)
        
    def analyze_predictions(self, detailed_matches: List[Dict], ground_truth: Dict = None) -> Dict:
        """
        Comprehensive error analysis of prediction results
        
        Args:
            detailed_matches: List of prediction results by approach
            ground_truth: Optional ground truth mapping
            
        Returns:
            Dictionary containing comprehensive error analysis
        """
        print("\n🔍 STARTING COMPREHENSIVE ERROR ANALYSIS")
        print("=" * 70)
        
        # Extract ground truth mapping
        gt_mapping = self._extract_ground_truth_mapping(ground_truth)
        
        # Collect all error data
        self._collect_error_data(detailed_matches, gt_mapping)
        
        # Perform different types of analysis
        analysis_results = {
            'error_taxonomy': self._analyze_error_taxonomy(),
            'confidence_analysis': self._analyze_confidence_errors(),
            'column_type_analysis': self._analyze_column_type_errors(),
            'approach_comparison': self._compare_approach_errors(),
            'failure_patterns': self._identify_failure_patterns(),
            'error_correlation': self._analyze_error_correlations(),
            'difficulty_analysis': self._analyze_column_difficulty(),
        }
        
        # Generate visualizations
        self._generate_error_visualizations(analysis_results)
        
        # Save LaTeX tables
        self.save_latex_tables(analysis_results, context="single")
        
        # Print summary
        self._print_error_summary(analysis_results)
        
        return analysis_results
    
    def analyze_aggregated_predictions(self, detailed_matches_list: List[List[Dict]], ground_truth_list: List[Dict] = None) -> Dict:
        """
        Analyze prediction errors across multiple datasets in an aggregated fashion.
        
        Args:
            detailed_matches_list: List of detailed_matches for each dataset
            ground_truth_list: List of ground truth mappings for each dataset
            
        Returns:
            Dictionary containing comprehensive aggregated error analysis results
        """
        print("\n🔍 STARTING AGGREGATED ERROR ANALYSIS...")
        print("=" * 60)
        print(f"📊 Analyzing {len(detailed_matches_list)} datasets")
        
        # Reset error data for fresh analysis
        self.error_data = []
        self.approach_errors = defaultdict(list)
        
        # Process each dataset
        for i, detailed_matches in enumerate(detailed_matches_list):
            print(f"📋 Processing dataset {i+1}/{len(detailed_matches_list)}")
            
            # Get ground truth for this dataset
            gt_data = ground_truth_list[i] if ground_truth_list and i < len(ground_truth_list) else None
            gt_mapping = self._extract_ground_truth_mapping(gt_data) if gt_data else {}
            
            # Collect error data from this dataset
            self._collect_error_data([detailed_matches], gt_mapping)
        
        print(f"✅ Collected error data from {len(detailed_matches_list)} datasets")
        print(f"📊 Total predictions analyzed: {len(self.error_data)}")
        
        # Perform different types of analysis
        analysis_results = {
            'error_taxonomy': self._analyze_error_taxonomy(),
            'confidence_analysis': self._analyze_confidence_errors(),
            'column_type_analysis': self._analyze_column_type_errors(),
            'approach_comparison': self._compare_approach_errors(),
            'failure_patterns': self._identify_failure_patterns(),
            'error_correlation': self._analyze_error_correlations(),
            'difficulty_analysis': self._analyze_column_difficulty(),
        }
        
        # Generate visualizations
        self._generate_error_visualizations(analysis_results)
        
        # Save LaTeX tables with aggregated context
        self.save_latex_tables(analysis_results, context="aggregated")
        
        # Print summary
        self._print_error_summary(analysis_results)
        
        return analysis_results
    
    def _extract_ground_truth_mapping(self, ground_truth: Dict) -> Dict[str, str]:
        """Extract ground truth mapping from various formats"""
        if isinstance(ground_truth, dict):
            if 'matches' in ground_truth:
                # Format: {'matches': [{'target_column': 'x', 'source_column': 'y'}, ...]}
                return {match['target_column']: match['source_column'] 
                       for match in ground_truth['matches']}
            else:
                # Direct mapping format: {'target_col': 'source_col'}
                return ground_truth
        return {}
    
    def _collect_error_data(self, detailed_matches: List[Dict], gt_mapping: Dict[str, str]):
        """Collect detailed error information"""
        print("📊 Collecting error data...")
        
        for dataset_results in detailed_matches:
            for approach_name, predictions in dataset_results.items():
                for target_col, prediction_tuple in predictions.items():
                    
                    # Parse prediction tuple
                    if len(prediction_tuple) >= 3:
                        predicted_source, confidence, reasoning = prediction_tuple[:3]
                    elif len(prediction_tuple) == 2:
                        predicted_source, confidence = prediction_tuple
                        reasoning = None
                    else:
                        continue
                    
                    # Determine if error occurred
                    ground_truth_source = gt_mapping.get(target_col)
                    is_error = predicted_source != ground_truth_source
                    
                    # Classify error type
                    error_type = self._classify_error_type(predicted_source, ground_truth_source)
                    
                    # Store error data
                    error_record = {
                        'approach': approach_name,
                        'target_column': target_col,
                        'predicted_source': predicted_source,
                        'ground_truth_source': ground_truth_source,
                        'confidence': float(confidence),
                        'reasoning': reasoning,
                        'is_error': is_error,
                        'error_type': error_type,
                    }
                    
                    self.error_data.append(error_record)
                    
                    if is_error:
                        self.approach_errors[approach_name].append(error_record)
    
    def _classify_error_type(self, predicted: str, ground_truth: str) -> str:
        """Classify the type of error that occurred"""
        if predicted == ground_truth:
            return "no_error"
        
        # Analyze different error types
        if ground_truth is None:
            return "should_be_null"
        elif predicted is None:
            return "should_not_be_null"
        else:
            return "wrong_column"

    def _analyze_error_taxonomy(self) -> Dict:
        """Analyze types of errors across approaches"""
        print("🔍 Analyzing error taxonomy...")
        
        error_counts = Counter(record['error_type'] for record in self.error_data)
        approach_error_types = defaultdict(lambda: defaultdict(int))
        
        for record in self.error_data:
            if record['is_error']:
                approach_error_types[record['approach']][record['error_type']] += 1
        
        return {
            'overall_error_distribution': dict(error_counts),
            'approach_error_types': dict(approach_error_types)
        }
    
    def _get_confidence_category(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        elif confidence >= 0.4:
            return "low"
        else:
            return "very_low"
    
    def _analyze_confidence_errors(self) -> Dict:
        """Analyze relationship between confidence and errors"""
        print("📈 Analyzing confidence vs errors...")
        
        confidence_vs_accuracy = defaultdict(list)
        overconfident_errors = []
        underconfident_correct = []
        
        for record in self.error_data:
            confidence = record['confidence']
            is_correct = not record['is_error']
            
            # Calculate confidence category on the fly
            confidence_category = self._get_confidence_category(confidence)
            confidence_vs_accuracy[confidence_category].append(is_correct)
            
            # Identify overconfident errors (high confidence, wrong prediction)
            if record['is_error'] and confidence > 0.8:
                overconfident_errors.append(record)
            
            # Identify underconfident correct predictions
            if not record['is_error'] and confidence < 0.5:
                underconfident_correct.append(record)
        
        # Calculate accuracy by confidence level
        accuracy_by_confidence = {}
        for conf_level, correctness_list in confidence_vs_accuracy.items():
            accuracy_by_confidence[conf_level] = np.mean(correctness_list) if correctness_list else 0.0
        
        return {
            'accuracy_by_confidence': accuracy_by_confidence,
            'overconfident_errors': overconfident_errors,
            'underconfident_correct': underconfident_correct,
            'calibration_analysis': self._analyze_calibration()
        }
    
    def _analyze_calibration(self) -> Dict:
        """Analyze confidence calibration for each approach"""
        calibration_data = defaultdict(lambda: {'confidences': [], 'accuracies': []})
        
        for record in self.error_data:
            approach = record['approach']
            confidence = record['confidence']
            accuracy = 1.0 if not record['is_error'] else 0.0
            
            calibration_data[approach]['confidences'].append(confidence)
            calibration_data[approach]['accuracies'].append(accuracy)
        
        # Calculate calibration metrics
        calibration_results = {}
        for approach, data in calibration_data.items():
            if data['confidences']:
                # Bin confidences and calculate average accuracy per bin
                bins = np.linspace(0, 1, 11)
                bin_accuracies = []
                bin_confidences = []
                
                for i in range(len(bins) - 1):
                    bin_mask = (np.array(data['confidences']) >= bins[i]) & \
                              (np.array(data['confidences']) < bins[i + 1])
                    
                    if np.any(bin_mask):
                        bin_accuracy = np.mean(np.array(data['accuracies'])[bin_mask])
                        bin_confidence = np.mean(np.array(data['confidences'])[bin_mask])
                        bin_accuracies.append(bin_accuracy)
                        bin_confidences.append(bin_confidence)
                
                # Calculate Expected Calibration Error (ECE)
                ece = np.mean(np.abs(np.array(bin_confidences) - np.array(bin_accuracies))) if bin_confidences else 0.0
                
                calibration_results[approach] = {
                    'ece': ece,
                    'bin_confidences': bin_confidences,
                    'bin_accuracies': bin_accuracies
                }
        
        return calibration_results
    
    def _analyze_column_type_errors(self) -> Dict:
        """Analyze errors by column semantic type"""
        print("🏷️ Analyzing errors by column type...")
        
        type_error_rates = defaultdict(lambda: {'correct': 0, 'incorrect': 0})
        
        for record in self.error_data:
            # Use a default column type if not available
            col_type = record.get('column_type', 'unknown')
            if record['is_error']:
                type_error_rates[col_type]['incorrect'] += 1
            else:
                type_error_rates[col_type]['correct'] += 1
        
        # Calculate error rates
        type_error_percentages = {}
        for col_type, counts in type_error_rates.items():
            total = counts['correct'] + counts['incorrect']
            error_rate = counts['incorrect'] / total if total > 0 else 0.0
            type_error_percentages[col_type] = {
                'error_rate': error_rate,
                'total_predictions': total,
                'errors': counts['incorrect']
            }
        
        return type_error_percentages
    
    def _compare_approach_errors(self) -> Dict:
        """Compare error patterns across different approaches"""
        print("⚖️ Comparing approach error patterns...")
        
        approach_metrics = {}
        
        for approach in set(record['approach'] for record in self.error_data):
            approach_records = [r for r in self.error_data if r['approach'] == approach]
            
            if approach_records:
                errors = [r for r in approach_records if r['is_error']]
                correct = [r for r in approach_records if not r['is_error']]
                
                approach_metrics[approach] = {
                    'total_predictions': len(approach_records),
                    'errors': len(errors),
                    'accuracy': len(correct) / len(approach_records),
                    'avg_confidence': np.mean([r['confidence'] for r in approach_records]),
                    'avg_error_confidence': np.mean([r['confidence'] for r in errors]) if errors else 0.0,
                    'most_common_error_type': Counter([r['error_type'] for r in errors]).most_common(1)[0][0] if errors else None
                }
        
        return approach_metrics
    
    def _identify_failure_patterns(self) -> Dict:
        """Identify common failure patterns"""
        print("🔍 Identifying failure patterns...")
        
        # Find columns that are consistently difficult
        column_difficulty = defaultdict(lambda: {'approaches_failed': set(), 'total_attempts': 0})
        
        for record in self.error_data:
            target_col = record['target_column']
            approach = record['approach']
            
            column_difficulty[target_col]['total_attempts'] += 1
            if record['is_error']:
                column_difficulty[target_col]['approaches_failed'].add(approach)
        
        # Identify challenging columns
        difficult_columns = {}
        for col, data in column_difficulty.items():
            failure_rate = len(data['approaches_failed']) / data['total_attempts']
            if failure_rate > 0.5:  # More than 50% of approaches failed
                difficult_columns[col] = {
                    'failure_rate': failure_rate,
                    'failed_approaches': list(data['approaches_failed']),
                    'total_attempts': data['total_attempts']
                }
        
        return {
            'difficult_columns': difficult_columns,
            'column_difficulty_distribution': {
                col: len(data['approaches_failed']) / data['total_attempts']
                for col, data in column_difficulty.items()
            }
        }
    
    def _analyze_error_correlations(self) -> Dict:
        """Analyze correlations between different error factors"""
        print("🔗 Analyzing error correlations...")
        
        # Create correlation matrix
        df = pd.DataFrame(self.error_data)
        
        if len(df) > 0:
            # Create numerical features for correlation analysis
            df['is_error_num'] = df['is_error'].astype(int)
            df['confidence_num'] = df['confidence']
            df['approach_encoded'] = pd.Categorical(df['approach']).codes
            df['error_type_encoded'] = pd.Categorical(df['error_type']).codes
            
            # Only include column_type if it exists in the data
            correlation_features = [
                'is_error_num', 'confidence_num', 'approach_encoded', 
                'error_type_encoded'
            ]
            
            if 'column_type' in df.columns:
                df['column_type_encoded'] = pd.Categorical(df['column_type']).codes
                correlation_features.append('column_type_encoded')
            
            correlation_matrix = df[correlation_features].corr()
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'feature_names': correlation_features
            }
        
        return {'correlation_matrix': {}, 'feature_names': []}
    
    def _analyze_column_difficulty(self) -> Dict:
        """Analyze which columns are most difficult to match correctly"""
        print("📊 Analyzing column difficulty...")
        
        column_stats = defaultdict(lambda: {
            'total_predictions': 0,
            'correct_predictions': 0,
            'approaches_succeeded': set(),
            'approaches_failed': set(),
            'avg_confidence_when_correct': [],
            'avg_confidence_when_wrong': []
        })
        
        for record in self.error_data:
            col = record['target_column']
            approach = record['approach']
            
            column_stats[col]['total_predictions'] += 1
            
            if record['is_error']:
                column_stats[col]['approaches_failed'].add(approach)
                column_stats[col]['avg_confidence_when_wrong'].append(record['confidence'])
            else:
                column_stats[col]['correct_predictions'] += 1
                column_stats[col]['approaches_succeeded'].add(approach)
                column_stats[col]['avg_confidence_when_correct'].append(record['confidence'])
        
        # Calculate difficulty metrics
        difficulty_ranking = {}
        for col, stats in column_stats.items():
            accuracy = stats['correct_predictions'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0
            consensus = len(stats['approaches_succeeded']) / (len(stats['approaches_succeeded']) + len(stats['approaches_failed']))
            
            avg_conf_correct = np.mean(stats['avg_confidence_when_correct']) if stats['avg_confidence_when_correct'] else 0
            avg_conf_wrong = np.mean(stats['avg_confidence_when_wrong']) if stats['avg_confidence_when_wrong'] else 0
            
            difficulty_ranking[col] = {
                'accuracy': accuracy,
                'consensus': consensus,
                'difficulty_score': 1 - accuracy,  # Higher = more difficult
                'avg_confidence_correct': avg_conf_correct,
                'avg_confidence_wrong': avg_conf_wrong,
                'total_attempts': stats['total_predictions']
            }
        
        # Sort by difficulty
        sorted_difficulty = sorted(
            difficulty_ranking.items(),
            key=lambda x: x[1]['difficulty_score'],
            reverse=True
        )
        
        return {
            'column_difficulty_ranking': dict(sorted_difficulty),
            'most_difficult_columns': [col for col, _ in sorted_difficulty[:5]],
            'easiest_columns': [col for col, _ in sorted_difficulty[-5:]]
        }
    
    def _generate_error_visualizations(self, analysis_results: Dict):
        """Generate comprehensive error analysis visualizations"""
        print("📊 Generating error analysis visualizations...")
        
        # Create output directory
        os.makedirs("./output/error_analysis", exist_ok=True)
        
        # Set style (with fallback)
        try:
            plt.style.use('seaborn-v0_8')
        except OSError:
            try:
                plt.style.use('seaborn')
            except OSError:
                # Use default style if seaborn is not available
                plt.style.use('default')
        
        # 1. Error Type Distribution
        self._plot_error_distribution(analysis_results['error_taxonomy'])
        
        # 2. Confidence vs Accuracy
        self._plot_confidence_accuracy(analysis_results['confidence_analysis'])
        
        # 3. Approach Comparison
        self._plot_approach_comparison(analysis_results['approach_comparison'])
        
        # 4. Column Type Error Rates
        self._plot_column_type_errors(analysis_results['column_type_analysis'])
        
        # 5. Column Difficulty Heatmap
        self._plot_column_difficulty(analysis_results['difficulty_analysis'])
        
        print(f"📁 Error analysis visualizations saved to ./output/error_analysis/")
    
    def _plot_error_distribution(self, error_taxonomy: Dict):
        """Plot error type distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Overall error distribution
        error_dist = error_taxonomy['overall_error_distribution']
        ax1.pie(error_dist.values(), labels=error_dist.keys(), autopct='%1.1f%%')
        ax1.set_title('Overall Error Type Distribution')
        
        # Error types by approach
        approach_errors = error_taxonomy['approach_error_types']
        if approach_errors:
            df = pd.DataFrame(approach_errors).fillna(0).astype(int)
            sns.heatmap(df, annot=True, fmt='d', ax=ax2, cmap='Reds')
            ax2.set_title('Error Types by Approach')
            ax2.set_xlabel('Error Types')
            ax2.set_ylabel('Approaches')
        
        plt.tight_layout()
        plt.savefig('./output/error_analysis/error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_accuracy(self, confidence_analysis: Dict):
        """Plot confidence vs accuracy analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy by confidence level
        acc_by_conf = confidence_analysis['accuracy_by_confidence']
        ax1.bar(acc_by_conf.keys(), acc_by_conf.values())
        ax1.set_title('Accuracy by Confidence Level')
        ax1.set_xlabel('Confidence Level')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Calibration plot
        calibration = confidence_analysis['calibration_analysis']
        for approach, data in calibration.items():
            if data['bin_confidences']:
                ax2.plot(data['bin_confidences'], data['bin_accuracies'], 'o-', label=f"{approach} (ECE: {data['ece']:.3f})")
        
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
        ax2.set_title('Confidence Calibration by Approach')
        ax2.set_xlabel('Mean Predicted Confidence')
        ax2.set_ylabel('Mean Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./output/error_analysis/confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_approach_comparison(self, approach_comparison: Dict):
        """Plot approach comparison metrics"""
        approaches = list(approach_comparison.keys())
        accuracies = [approach_comparison[app]['accuracy'] for app in approaches]
        avg_confidences = [approach_comparison[app]['avg_confidence'] for app in approaches]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(approaches, accuracies)
        ax1.set_title('Accuracy by Approach')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Average confidence comparison
        bars2 = ax2.bar(approaches, avg_confidences, color='orange')
        ax2.set_title('Average Confidence by Approach')
        ax2.set_ylabel('Average Confidence')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, conf in zip(bars2, avg_confidences):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('./output/error_analysis/approach_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_column_type_errors(self, column_type_analysis: Dict):
        """Plot error rates by column type"""
        if not column_type_analysis:
            return
        
        types = list(column_type_analysis.keys())
        error_rates = [column_type_analysis[t]['error_rate'] for t in types]
        totals = [column_type_analysis[t]['total_predictions'] for t in types]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error rates by column type
        bars1 = ax1.bar(types, error_rates)
        ax1.set_title('Error Rates by Column Type')
        ax1.set_ylabel('Error Rate')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Sample size by column type
        bars2 = ax2.bar(types, totals, color='green', alpha=0.7)
        ax2.set_title('Sample Size by Column Type')
        ax2.set_ylabel('Number of Predictions')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('./output/error_analysis/column_type_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_column_difficulty(self, difficulty_analysis: Dict):
        """Plot column difficulty analysis"""
        difficulty_data = difficulty_analysis['column_difficulty_ranking']
        
        if not difficulty_data:
            return
        
        columns = list(difficulty_data.keys())[:10]  # Top 10 most difficult
        difficulties = [difficulty_data[col]['difficulty_score'] for col in columns]
        accuracies = [difficulty_data[col]['accuracy'] for col in columns]
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        y_pos = np.arange(len(columns))
        bars = ax.barh(y_pos, difficulties, color='red', alpha=0.7, label='Difficulty Score')
        
        # Add accuracy as text
        for i, (diff, acc) in enumerate(zip(difficulties, accuracies)):
            ax.text(diff + 0.01, i, f'Acc: {acc:.2f}', va='center')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(columns)
        ax.set_xlabel('Difficulty Score (1 - Accuracy)')
        ax.set_title('Top 10 Most Difficult Columns to Match')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('./output/error_analysis/column_difficulty.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_error_summary(self, analysis_results: Dict):
        """Print comprehensive error analysis summary"""
        print("\n" + "=" * 70)
        print("📊 ERROR ANALYSIS SUMMARY")
        print("=" * 70)
        
        # Overall statistics
        total_predictions = len(self.error_data)
        total_errors = sum(1 for r in self.error_data if r['is_error'])
        overall_accuracy = (total_predictions - total_errors) / total_predictions if total_predictions > 0 else 0
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   • Total Predictions: {total_predictions}")
        print(f"   • Total Errors: {total_errors}")
        print(f"   • Overall Accuracy: {overall_accuracy:.3f}")
        
        # Error type distribution
        print(f"\n🏷️ ERROR TYPE DISTRIBUTION:")
        error_dist = analysis_results['error_taxonomy']['overall_error_distribution']
        for error_type, count in sorted(error_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
            print(f"   • {error_type}: {count} ({percentage:.1f}%)")
        
        # Approach performance
        print(f"\n⚖️ APPROACH PERFORMANCE:")
        approach_metrics = analysis_results['approach_comparison']
        for approach, metrics in sorted(approach_metrics.items(), 
                                      key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"   • {approach}:")
            print(f"     - Accuracy: {metrics['accuracy']:.3f}")
            print(f"     - Avg Confidence: {metrics['avg_confidence']:.3f}")
            print(f"     - Total Predictions: {metrics['total_predictions']}")
        
        # Most difficult columns
        print(f"\n🎯 MOST DIFFICULT COLUMNS:")
        difficult_cols = analysis_results['difficulty_analysis']['most_difficult_columns']
        for i, col in enumerate(difficult_cols[:5], 1):
            difficulty_score = analysis_results['difficulty_analysis']['column_difficulty_ranking'][col]['difficulty_score']
            print(f"   {i}. {col} (difficulty: {difficulty_score:.3f})")
        
        # Confidence analysis
        print(f"\n📊 CONFIDENCE ANALYSIS:")
        conf_analysis = analysis_results['confidence_analysis']
        print(f"   • Overconfident Errors: {len(conf_analysis['overconfident_errors'])}")
        print(f"   • Underconfident Correct: {len(conf_analysis['underconfident_correct'])}")
        
        acc_by_conf = conf_analysis['accuracy_by_confidence']
        for conf_level, accuracy in acc_by_conf.items():
            print(f"   • {conf_level} confidence accuracy: {accuracy:.3f}")
        
        print("\n" + "=" * 70)
        print("✅ ERROR ANALYSIS COMPLETE")
        print("=" * 70)
        
    def save_latex_tables(self, analysis_results: Dict, context: str = "single"):
        """Save error analysis tables as LaTeX files"""
        print("\n📄 Saving error analysis tables as LaTeX...")
        
        try:
            os.makedirs("output/results", exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Add context to filename prefix
            prefix = f"{timestamp}_error_analysis_{context}"
            
            # Table 1: Approach Performance Comparison
            self._save_approach_comparison_latex(analysis_results['approach_comparison'], prefix)
            
            # Table 2: Error Type Distribution
            self._save_error_taxonomy_latex(analysis_results['error_taxonomy'], prefix)
            
            # Table 3: Column Type Error Analysis
            self._save_column_type_analysis_latex(analysis_results['column_type_analysis'], prefix)
            
            # Table 4: Confidence Analysis
            self._save_confidence_analysis_latex(analysis_results['confidence_analysis'], prefix)
            
            # Table 5: Most Difficult Columns
            self._save_difficulty_analysis_latex(analysis_results['difficulty_analysis'], prefix)
            
            print("✅ All error analysis LaTeX tables saved successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to save error analysis LaTeX tables: {e}")
    
    def _save_approach_comparison_latex(self, approach_comparison: Dict, prefix: str):
        """Save approach performance comparison as LaTeX table"""
        table_data = []
        for approach, metrics in sorted(approach_comparison.items(), 
                                      key=lambda x: x[1]['accuracy'], reverse=True):
            table_data.append([
                approach,
                f"{metrics['accuracy']:.3f}",
                f"{metrics['avg_confidence']:.3f}",
                f"{metrics['avg_error_confidence']:.3f}",
                metrics['total_predictions'],
                metrics['errors'],
                metrics['most_common_error_type'] or "None"
            ])
        
        headers = ['Approach', 'Accuracy', 'Avg Confidence', 'Avg Error Confidence', 
                  'Total Predictions', 'Errors', 'Most Common Error Type']
        
        df = pd.DataFrame(table_data, columns=headers)
        out_path = f"output/results/{prefix}_approach_comparison.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved approach comparison table: {out_path}")
    
    def _save_error_taxonomy_latex(self, error_taxonomy: Dict, prefix: str):
        """Save error type distribution as LaTeX table"""
        error_dist = error_taxonomy['overall_error_distribution']
        total_predictions = len(self.error_data)
        
        table_data = []
        for error_type, count in sorted(error_dist.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_predictions) * 100 if total_predictions > 0 else 0
            table_data.append([
                error_type,
                count,
                f"{percentage:.1f}%"
            ])
        
        headers = ['Error Type', 'Count', 'Percentage']
        df = pd.DataFrame(table_data, columns=headers)
        out_path = f"output/results/{prefix}_taxonomy.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved error taxonomy table: {out_path}")
    
    def _save_column_type_analysis_latex(self, column_type_analysis: Dict, prefix: str):
        """Save column type error analysis as LaTeX table"""
        table_data = []
        for col_type, stats in sorted(column_type_analysis.items(), 
                                    key=lambda x: x[1]['error_rate'], reverse=True):
            table_data.append([
                col_type,
                f"{stats['error_rate']:.3f}",
                stats['total_predictions'],
                stats['errors'],
                f"{(1 - stats['error_rate']):.3f}"
            ])
        
        headers = ['Column Type', 'Error Rate', 'Total Predictions', 'Errors', 'Accuracy']
        df = pd.DataFrame(table_data, columns=headers)
        out_path = f"output/results/{prefix}_column_type_analysis.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved column type analysis table: {out_path}")
    
    def _save_confidence_analysis_latex(self, confidence_analysis: Dict, prefix: str):
        """Save confidence analysis as LaTeX table"""
        acc_by_conf = confidence_analysis['accuracy_by_confidence']
        
        table_data = []
        for conf_level, accuracy in acc_by_conf.items():
            table_data.append([
                conf_level,
                f"{accuracy:.3f}"
            ])
        
        # Add summary statistics
        table_data.append([
            "Overconfident Errors",
            len(confidence_analysis['overconfident_errors'])
        ])
        table_data.append([
            "Underconfident Correct",
            len(confidence_analysis['underconfident_correct'])
        ])
        
        headers = ['Confidence Level/Metric', 'Value']
        df = pd.DataFrame(table_data, columns=headers)
        out_path = f"output/results/{prefix}_confidence_analysis.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved confidence analysis table: {out_path}")
    
    def _save_difficulty_analysis_latex(self, difficulty_analysis: Dict, prefix: str):
        """Save most difficult columns analysis as LaTeX table"""
        difficult_cols = difficulty_analysis['most_difficult_columns']
        col_difficulty = difficulty_analysis['column_difficulty_ranking']
        
        table_data = []
        for i, col in enumerate(difficult_cols[:10], 1):  # Top 10 most difficult
            if col in col_difficulty:
                col_data = col_difficulty[col]
                difficulty_score = col_data['difficulty_score']
                accuracy = col_data['accuracy']
                total_attempts = col_data['total_attempts']
                table_data.append([
                    i,
                    col,
                    f"{difficulty_score:.3f}",
                    f"{accuracy:.3f}",
                    total_attempts,
                    f"{(1-accuracy):.3f}"
                ])
        
        headers = ['Rank', 'Column Name', 'Difficulty Score', 'Accuracy', 'Total Attempts', 'Error Rate']
        df = pd.DataFrame(table_data, columns=headers)
        out_path = f"output/results/{prefix}_difficulty_analysis.tex"
        df.to_latex(out_path, index=False, longtable=True)
        print(f"📄 Saved difficulty analysis table: {out_path}")
    
    def save_detailed_report(self, analysis_results: Dict, filename: str = None):
        """Save detailed error analysis report to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./output/error_analysis/detailed_error_report_{timestamp}.json"
        
        # Prepare serializable data
        serializable_results = {}
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in value.items()
                }
            else:
                serializable_results[key] = value
        
        # Add raw error data
        serializable_results['raw_error_data'] = self.error_data
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"📄 Detailed error report saved to: {filename}")
        return filename


def run_error_analysis(detailed_matches: List[Dict], ground_truth = None) -> Dict:
    """
    Convenience function to run comprehensive error analysis
    
    Args:
        detailed_matches: List of prediction results by approach
        ground_truth: Ground truth mapping(s) - can be:
            - Single dict for one dataset
            - List of dicts for multiple datasets  
            - None for analysis without ground truth
        
    Returns:
        Dictionary containing comprehensive error analysis results
    """
    analyzer = ErrorAnalyzer()
    
    # Handle different ground truth formats
    if isinstance(ground_truth, list):
        # New aggregated format: multiple datasets
        print(f"📊 Running aggregated error analysis on {len(detailed_matches)} datasets")
        results = analyzer.analyze_aggregated_predictions(detailed_matches, ground_truth)
    else:
        # Original format: single dataset
        print("📊 Running single dataset error analysis")
        results = analyzer.analyze_predictions(detailed_matches, ground_truth)
    
    analyzer.save_detailed_report(results)
    return results

