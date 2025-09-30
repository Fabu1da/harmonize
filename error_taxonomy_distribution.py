#!/usr/bin/env python3
"""
Error Taxonomy Distribution Analysis for Harmonize Pipeline

This script analyzes the 877 disagreement cases and categorizes them into
the four-type error taxonomy for thesis documentation.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
import re
from collections import defaultdict, Counter


class ErrorTaxonomyAnalyzer:
    """Analyze and categorize schema matching errors into taxonomy types."""
    
    def __init__(self):
        self.error_patterns = {
            'structural_mismatch': [
                # Type I: Structural Mismatch patterns
                r'id(?:entifier)?.*(?:key|ref|foreign)',
                r'(?:primary|foreign).*key',
                r'.*_id.*_ref',
                r'table.*key.*field',
                r'relation.*structure',
                r'hierarchy.*level'
            ],
            'semantic_ambiguity': [
                # Type II: Semantic Ambiguity patterns
                r'name.*title',
                r'description.*comment',
                r'type.*category',
                r'status.*state',
                r'date.*time',
                r'location.*address',
                r'price.*cost.*value',
                r'amount.*quantity'
            ],
            'technical_variance': [
                # Type III: Technical Variance patterns
                r'varchar.*text',
                r'int.*integer.*number',
                r'datetime.*timestamp',
                r'decimal.*float.*numeric',
                r'boolean.*bit.*flag',
                r'json.*xml.*blob',
                r'enum.*set'
            ],
            'context_dependent': [
                # Type IV: Context-Dependent Terminology patterns
                r'assay.*relationship',
                r'protocol.*procedure',
                r'specimen.*sample',
                r'measurement.*reading',
                r'classification.*taxonomy',
                r'domain.*namespace',
                r'workflow.*process'
            ]
        }
        
        # Domain-specific vocabulary indicators
        self.domain_indicators = {
            'scientific': ['assay', 'specimen', 'protocol', 'measurement', 'classification'],
            'business': ['revenue', 'customer', 'transaction', 'account', 'invoice'],
            'technical': ['endpoint', 'service', 'configuration', 'deployment', 'version'],
            'medical': ['patient', 'diagnosis', 'treatment', 'medication', 'symptom'],
            'geographic': ['location', 'coordinate', 'region', 'boundary', 'elevation']
        }
    
    def categorize_column_pair(self, source_col: str, target_col: str, 
                             matcher_disagreement: List[str]) -> str:
        """
        Categorize a disagreement case into error taxonomy type.
        
        Args:
            source_col: Source column name
            target_col: Target column name  
            matcher_disagreement: List of matchers that disagreed
            
        Returns:
            Error type category
        """
        source_lower = source_col.lower()
        target_lower = target_col.lower()
        combined = f"{source_lower} {target_lower}"
        
        # Check for Type IV: Context-Dependent first (most specific)
        if self._is_context_dependent(source_col, target_col):
            return 'context_dependent'
        
        # Check for Type III: Technical Variance
        if self._matches_patterns(combined, self.error_patterns['technical_variance']):
            return 'technical_variance'
            
        # Check for Type I: Structural Mismatch
        if self._matches_patterns(combined, self.error_patterns['structural_mismatch']):
            return 'structural_mismatch'
            
        # Default to Type II: Semantic Ambiguity
        return 'semantic_ambiguity'
    
    def _is_context_dependent(self, source_col: str, target_col: str) -> bool:
        """Check if column pair represents context-dependent terminology."""
        source_lower = source_col.lower()
        target_lower = target_col.lower()
        
        # Check for domain-specific patterns
        combined = f"{source_lower} {target_lower}"
        if self._matches_patterns(combined, self.error_patterns['context_dependent']):
            return True
        
        # Check for cross-domain terminology
        source_domains = self._get_column_domains(source_col)
        target_domains = self._get_column_domains(target_col)
        
        # If columns belong to different domains, likely context-dependent
        if source_domains and target_domains and not source_domains.intersection(target_domains):
            return True
            
        return False
    
    def _get_column_domains(self, column: str) -> Set[str]:
        """Identify which domains a column belongs to based on vocabulary."""
        domains = set()
        column_lower = column.lower()
        
        for domain, indicators in self.domain_indicators.items():
            if any(indicator in column_lower for indicator in indicators):
                domains.add(domain)
                
        return domains
    
    def _matches_patterns(self, text: str, patterns: List[str]) -> bool:
        """Check if text matches any of the given regex patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False


def load_disagreement_data(data_dir: str) -> pd.DataFrame:
    """
    Load disagreement cases from analysis results.
    
    This is a mock function - you'll need to adapt this to your actual data format.
    """
    # Mock data structure based on your analysis
    # Replace this with your actual disagreement data loading logic
    
    mock_data = []
    
    # Example disagreement cases (replace with your actual data)
    examples = [
        ("musicianID", "artist_identifier", ["clustering", "content_similarity"], "structural_mismatch"),
        ("musicianName", "artist_title", ["gpt4", "embedding"], "semantic_ambiguity"), 
        ("genderType", "gender_category", ["clustering"], "semantic_ambiguity"),
        ("assay_type", "assays_relationship_type", ["embedding", "clustering"], "context_dependent"),
        ("varchar_field", "text_column", ["content_similarity"], "technical_variance"),
        ("patient_id", "medical_record_key", ["gpt4"], "structural_mismatch"),
        ("protocol_name", "procedure_title", ["clustering", "embedding"], "context_dependent"),
        ("timestamp_col", "datetime_field", ["content_similarity"], "technical_variance"),
        ("product_name", "item_description", ["gpt4", "clustering"], "semantic_ambiguity"),
        ("measurement_value", "reading_amount", ["embedding"], "context_dependent")
    ]
    
    # Generate mock disagreement cases to reach 877 total
    base_patterns = [
        ("name", "title", ["gpt4", "embedding"], "semantic_ambiguity"),
        ("id", "key", ["clustering"], "structural_mismatch"),
        ("type", "category", ["content_similarity"], "semantic_ambiguity"),
        ("date", "time", ["gpt4"], "semantic_ambiguity"),
        ("varchar", "text", ["clustering"], "technical_variance"),
        ("int", "number", ["embedding"], "technical_variance"),
        ("assay", "protocol", ["clustering", "embedding"], "context_dependent"),
        ("specimen", "sample", ["gpt4"], "context_dependent")
    ]
    
    # Create 877 cases with realistic distribution
    import random
    random.seed(42)  # For reproducible results
    
    for i in range(877):
        if i < len(examples):
            source, target, disagreeing, category = examples[i]
        else:
            # Generate variations
            base = base_patterns[i % len(base_patterns)]
            source = f"{base[0]}_{i % 10}"
            target = f"{base[1]}_{(i+1) % 10}"
            disagreeing = random.sample(["gpt4", "embedding", "clustering", "content_similarity"], 
                                      random.randint(1, 3))
            category = base[3]
        
        mock_data.append({
            'source_column': source,
            'target_column': target,
            'disagreeing_matchers': disagreeing,
            'dataset': f"dataset_{i % 20}",
            'table_pair': f"table_pair_{i % 100}",
            'error_type': category
        })
    
    return pd.DataFrame(mock_data)


def analyze_error_taxonomy(data_dir: str = "./") -> Dict:
    """
    Analyze error taxonomy distribution from disagreement data.
    
    Args:
        data_dir: Directory containing analysis results
        
    Returns:
        Dictionary with taxonomy analysis results
    """
    # Load disagreement data
    df = load_disagreement_data(data_dir)
    
    # Initialize analyzer
    analyzer = ErrorTaxonomyAnalyzer()
    
    # Categorize if not already done
    if 'error_type' not in df.columns:
        df['error_type'] = df.apply(
            lambda row: analyzer.categorize_column_pair(
                row['source_column'], 
                row['target_column'],
                row['disagreeing_matchers']
            ), axis=1
        )
    
    # Count by category
    error_counts = df['error_type'].value_counts().to_dict()
    total_cases = len(df)
    
    # Calculate percentages
    error_percentages = {
        error_type: (count / total_cases) * 100 
        for error_type, count in error_counts.items()
    }
    
    # Analyze by matcher type
    matcher_errors = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        for matcher in row['disagreeing_matchers']:
            matcher_errors[matcher][row['error_type']] += 1
    
    # Analyze by dataset
    dataset_errors = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        dataset_errors[row['dataset']][row['error_type']] += 1
    
    results = {
        'total_disagreement_cases': total_cases,
        'error_counts': error_counts,
        'error_percentages': error_percentages,
        'matcher_error_distribution': dict(matcher_errors),
        'dataset_error_distribution': dict(dataset_errors),
        'detailed_cases': df.to_dict('records')
    }
    
    return results


def generate_latex_table(results: Dict) -> str:
    """Generate LaTeX table for error taxonomy distribution."""
    
    # Define error type descriptions
    error_descriptions = {
        'structural_mismatch': 'Structural Mismatch (Type I)',
        'semantic_ambiguity': 'Semantic Ambiguity (Type II)', 
        'technical_variance': 'Technical Variance (Type III)',
        'context_dependent': 'Context-Dependent Terminology (Type IV)'
    }
    
    total_cases = results['total_disagreement_cases']
    error_counts = results['error_counts']
    error_percentages = results['error_percentages']
    
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Error Taxonomy Distribution Across 877 Disagreement Cases}
\label{tab:error_taxonomy_distribution}
\begin{tabular}{lrr}
\toprule
\textbf{Error Type} & \textbf{Count} & \textbf{Percentage} \\
\midrule
"""
    
    # Sort by count (descending)
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    for error_type, count in sorted_errors:
        description = error_descriptions.get(error_type, error_type.replace('_', ' ').title())
        percentage = error_percentages[error_type]
        latex_table += f"{description} & {count} & {percentage:.1f}\\% \\\\\n"
    
    latex_table += r"""
\midrule
\textbf{Total Disagreement Cases} & \textbf{""" + str(total_cases) + r"""} & \textbf{100.0\%} \\
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex_table


def generate_detailed_analysis_table(results: Dict) -> str:
    """Generate detailed breakdown by matcher type."""
    
    matcher_errors = results['matcher_error_distribution']
    
    latex_table = r"""
\begin{table}[htbp]
\centering
\caption{Error Type Distribution by Matcher Component}
\label{tab:error_by_matcher}
\begin{tabular}{lrrrr}
\toprule
\textbf{Matcher} & \textbf{Type I} & \textbf{Type II} & \textbf{Type III} & \textbf{Type IV} \\
\midrule
"""
    
    matchers = ['gpt4', 'embedding', 'clustering', 'content_similarity']
    error_types = ['structural_mismatch', 'semantic_ambiguity', 'technical_variance', 'context_dependent']
    
    for matcher in matchers:
        if matcher in matcher_errors:
            counts = [matcher_errors[matcher].get(error_type, 0) for error_type in error_types]
            latex_table += f"{matcher.replace('_', '\\_')} & {' & '.join(map(str, counts))} \\\\\n"
    
    latex_table += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    return latex_table


def main():
    """Main execution function."""
    print("🔍 Analyzing Error Taxonomy Distribution...")
    
    # Analyze error taxonomy
    results = analyze_error_taxonomy()
    
    # Print summary
    print(f"\n📊 Error Taxonomy Analysis Results:")
    print(f"   Total disagreement cases: {results['total_disagreement_cases']}")
    print(f"\n📈 Distribution by Error Type:")
    
    for error_type, count in sorted(results['error_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = results['error_percentages'][error_type]
        print(f"   • {error_type.replace('_', ' ').title()}: {count} cases ({percentage:.1f}%)")
    
    # Generate LaTeX tables
    latex_main_table = generate_latex_table(results)
    latex_detailed_table = generate_detailed_analysis_table(results)
    
    # Save results
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Save JSON results
    with open(output_dir / "error_taxonomy_analysis.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save LaTeX tables
    with open(output_dir / "error_taxonomy_table.tex", 'w') as f:
        f.write(latex_main_table)
    
    with open(output_dir / "error_by_matcher_table.tex", 'w') as f:
        f.write(latex_detailed_table)
    
    # Save detailed cases CSV
    df = pd.DataFrame(results['detailed_cases'])
    df.to_csv(output_dir / "error_taxonomy_detailed_cases.csv", index=False)
    
    print(f"\n💾 Results saved to {output_dir}/")
    print(f"   • error_taxonomy_analysis.json")
    print(f"   • error_taxonomy_table.tex")
    print(f"   • error_by_matcher_table.tex") 
    print(f"   • error_taxonomy_detailed_cases.csv")
    
    print(f"\n📋 LaTeX Table Preview:")
    print("=" * 50)
    print(latex_main_table)


if __name__ == "__main__":
    main()