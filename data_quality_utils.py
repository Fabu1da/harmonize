#!/usr/bin/env python3
"""
Data Quality Utilities for Hamonize
===================================

This module provides utilities to preprocess data and reduce common warnings
during schema matching operations.
"""

import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import logging
from error_handling import error_tracker, TransformationError

def preprocess_data_for_hamonize(df: pd.DataFrame, 
                                min_non_null_ratio: float = 0.1,
                                clean_column_names: bool = True,
                                remove_empty_columns: bool = True) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Preprocess data to reduce warnings during Hamonize processing.
    
    Args:
        df: Input DataFrame
        min_non_null_ratio: Minimum ratio of non-null values to keep column (0.1 = 10%)
        clean_column_names: Whether to clean column names proactively
        remove_empty_columns: Whether to remove columns with all null values
    
    Returns:
        Tuple of (cleaned_dataframe, column_mapping)
    """
    cleaned_df = df.copy()
    column_mapping = {}
    
    # 1. Clean column names proactively
    if clean_column_names:
        original_columns = list(cleaned_df.columns)
        cleaned_columns = []
        
        for col in original_columns:
            # Remove leading/trailing whitespace
            clean_col = col.strip()
            # Replace special characters with underscores
            clean_col = re.sub(r'[^\w\s]', '_', clean_col)
            # Replace multiple spaces/underscores with single underscore
            clean_col = re.sub(r'[\s_]+', '_', clean_col)
            # Remove trailing underscores
            clean_col = clean_col.rstrip('_')
            
            cleaned_columns.append(clean_col)
            if col != clean_col:
                column_mapping[col] = clean_col
        
        cleaned_df.columns = cleaned_columns
        
        if column_mapping:
            print(f"Cleaned {len(column_mapping)} column names:")
            for old, new in column_mapping.items():
                print(f"  '{old}' -> '{new}'")
    
    # 2. Remove columns with insufficient non-null values
    if remove_empty_columns:
        columns_to_remove = []
        column_stats = {}
        
        for col in cleaned_df.columns:
            non_null_count = cleaned_df[col].count()
            total_count = len(cleaned_df)
            non_null_ratio = non_null_count / total_count if total_count > 0 else 0
            
            column_stats[col] = {
                'non_null_count': non_null_count,
                'total_count': total_count,
                'non_null_ratio': non_null_ratio
            }
            
            if non_null_ratio < min_non_null_ratio:
                columns_to_remove.append(col)
        
        if columns_to_remove:
            print(f"Removing {len(columns_to_remove)} columns with insufficient data:")
            for col in columns_to_remove:
                stats = column_stats[col]
                print(f"  '{col}': {stats['non_null_count']}/{stats['total_count']} "
                      f"({stats['non_null_ratio']:.1%}) non-null values")
            
            cleaned_df = cleaned_df.drop(columns=columns_to_remove)
    
    # 3. Report summary
    print(f"\nData preprocessing summary:")
    print(f"  Original shape: {df.shape}")
    print(f"  Cleaned shape: {cleaned_df.shape}")
    print(f"  Columns renamed: {len(column_mapping)}")
    if remove_empty_columns:
        print(f"  Columns removed: {len(columns_to_remove)}")
    
    return cleaned_df, column_mapping

def analyze_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analyze data quality issues that might cause warnings in Hamonize.
    
    Args:
        df: DataFrame to analyze
    
    Returns:
        Dictionary with quality analysis results
    """
    analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'completely_null_columns': [],
        'mostly_null_columns': [],
        'problematic_column_names': [],
        'data_types': {},
        'quality_score': 0.0
    }
    
    # Analyze each column
    for col in df.columns:
        # Check for problematic column names
        if re.search(r'[^\w\s]', col) or col.strip() != col:
            analysis['problematic_column_names'].append(col)
        
        # Check null values
        null_count = df[col].isna().sum()
        null_ratio = null_count / len(df) if len(df) > 0 else 0
        
        if null_ratio == 1.0:
            analysis['completely_null_columns'].append(col)
        elif null_ratio > 0.8:
            analysis['mostly_null_columns'].append({
                'column': col,
                'null_ratio': null_ratio,
                'null_count': null_count
            })
        
        # Data type analysis
        analysis['data_types'][col] = {
            'dtype': str(df[col].dtype),
            'null_count': null_count,
            'null_ratio': null_ratio,
            'unique_values': df[col].nunique()
        }
    
    # Calculate quality score
    issues = len(analysis['completely_null_columns']) + len(analysis['mostly_null_columns']) + len(analysis['problematic_column_names'])
    total_columns = len(df.columns)
    analysis['quality_score'] = max(0, 1 - (issues / total_columns)) if total_columns > 0 else 0
    
    return analysis

def print_data_quality_report(df: pd.DataFrame):
    """Print a comprehensive data quality report."""
    analysis = analyze_data_quality(df)
    
    print("="*60)
    print("DATA QUALITY REPORT")
    print("="*60)
    
    print(f"Dataset shape: {analysis['total_rows']} rows × {analysis['total_columns']} columns")
    print(f"Quality score: {analysis['quality_score']:.1%}")
    
    # Column name issues
    if analysis['problematic_column_names']:
        print(f"\n⚠️  COLUMN NAME ISSUES ({len(analysis['problematic_column_names'])} columns):")
        for col in analysis['problematic_column_names']:
            print(f"  - '{col}' (contains special characters or whitespace)")
    
    # Completely null columns
    if analysis['completely_null_columns']:
        print(f"\n❌ COMPLETELY NULL COLUMNS ({len(analysis['completely_null_columns'])} columns):")
        for col in analysis['completely_null_columns']:
            print(f"  - '{col}' (100% null values)")
    
    # Mostly null columns
    if analysis['mostly_null_columns']:
        print(f"\n⚠️  MOSTLY NULL COLUMNS ({len(analysis['mostly_null_columns'])} columns):")
        for col_info in analysis['mostly_null_columns']:
            print(f"  - '{col_info['column']}' ({col_info['null_ratio']:.1%} null values)")
    
    # Data type summary
    print(f"\n📊 DATA TYPE SUMMARY:")
    type_counts = {}
    for col, info in analysis['data_types'].items():
        dtype = info['dtype']
        type_counts[dtype] = type_counts.get(dtype, 0) + 1
    
    for dtype, count in sorted(type_counts.items()):
        print(f"  - {dtype}: {count} columns")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if analysis['problematic_column_names']:
        print(f"  - Clean column names before processing")
    if analysis['completely_null_columns']:
        print(f"  - Remove completely null columns")
    if analysis['mostly_null_columns']:
        print(f"  - Consider removing or filling mostly null columns")
    if analysis['quality_score'] < 0.7:
        print(f"  - Data quality is below recommended threshold (70%)")
    
    print("="*60)

# Example usage
if __name__ == "__main__":
    # Example with your data structure
    sample_data = {
        'Gptnr. Kunde': [1, 2, 3, None, 5],
        'Fahrzeug-ID': ['A1', 'B2', 'C3', 'D4', 'E5'],
        'Auftragsart': [None, None, None, None, None],  # All null
        'valid_column': ['X', 'Y', 'Z', 'W', 'V'],
        'Positionskz. ': [1, 2, None, None, None],  # Trailing space and mostly null
    }
    
    df = pd.DataFrame(sample_data)
    
    print("Original data:")
    print(df)
    print("\nOriginal columns:", list(df.columns))
    
    # Show quality report
    print_data_quality_report(df)
    
    # Preprocess data
    cleaned_df, mapping = preprocess_data_for_hamonize(df)
    
    print("\nCleaned data:")
    print(cleaned_df)
    print("\nCleaned columns:", list(cleaned_df.columns))
