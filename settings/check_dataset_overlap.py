#!/usr/bin/env python3
"""
Dataset Overlap Checker for Training/Test Data Validation

This script checks for potential data leakage by identifying datasets that appear
in both training and test directories. Ensures proper train/test separation for
unbiased evaluation.

Usage:
    python check_dataset_overlap.py
"""

import os
from pathlib import Path
from collections import defaultdict
import pandas as pd


class DatasetOverlapChecker:
    def __init__(self):
        self.base_dir = Path(".")
        self.training_dirs = [
            "assets/training/source",
            "assets/training/target", 
            "assets/training/expected"
        ]
        self.test_dirs = [
            "assets/test/source",
            "assets/test/target",
            "assets/test/expected"
        ]
    
    def get_dataset_files(self, directory):
        """Get all dataset files from a directory."""
        dir_path = self.base_dir / directory
        if not dir_path.exists():
            print(f"⚠️  Directory {directory} does not exist")
            return []
        
        files = []
        for file in dir_path.glob("*.csv"):
            files.append(file.stem)  # filename without extension
        return files
    
    def check_directory_overlap(self, train_dir, test_dir):
        """Check overlap between specific training and test directories."""
        train_files = set(self.get_dataset_files(train_dir))
        test_files = set(self.get_dataset_files(test_dir))
        
        overlap = train_files.intersection(test_files)
        
        print(f"\n📂 Checking: {train_dir} vs {test_dir}")
        print(f"   Training files: {len(train_files)}")
        print(f"   Test files: {len(test_files)}")
        
        if overlap:
            print(f"   ❌ OVERLAP FOUND: {len(overlap)} files")
            for file in sorted(overlap):
                print(f"      • {file}")
        else:
            print(f"   ✅ No overlap detected")
        
        return overlap
    
    def check_all_overlaps(self):
        """Check all possible training/test directory combinations."""
        print("🔍 DATASET OVERLAP ANALYSIS")
        print("=" * 50)
        
        all_overlaps = []
        
        # Check source directories
        overlap = self.check_directory_overlap("assets/training/source", "assets/test/source")
        if overlap:
            all_overlaps.extend([(f, "source") for f in overlap])
        
        # Check target directories  
        overlap = self.check_directory_overlap("assets/training/target", "assets/test/target")
        if overlap:
            all_overlaps.extend([(f, "target") for f in overlap])
        
        # Check expected directories
        overlap = self.check_directory_overlap("assets/training/expected", "assets/test/expected")
        if overlap:
            all_overlaps.extend([(f, "expected") for f in overlap])
        
        # Cross-directory checks (more comprehensive)
        cross_checks = [
            ("assets/training/source", "assets/test/target"),
            ("assets/training/target", "assets/test/source"),
            ("assets/training/source", "assets/test/expected"),
            ("assets/training/expected", "assets/test/source"),
            ("assets/training/target", "assets/test/expected"),
            ("assets/training/expected", "assets/test/target")
        ]
        
        print(f"\n🔄 CROSS-DIRECTORY CHECKS")
        print("-" * 30)
        
        for train_dir, test_dir in cross_checks:
            overlap = self.check_directory_overlap(train_dir, test_dir)
            if overlap:
                all_overlaps.extend([(f, f"{train_dir.split('/')[-1]}->{test_dir.split('/')[-1]}") for f in overlap])
        
        return all_overlaps
    
    def analyze_file_content_similarity(self, file1_path, file2_path):
        """Check if two CSV files have similar content (same columns, similar data)."""
        try:
            df1 = pd.read_csv(file1_path)
            df2 = pd.read_csv(file2_path)
            
            # Check column similarity
            cols1 = set(df1.columns)
            cols2 = set(df2.columns)
            col_overlap = len(cols1.intersection(cols2)) / len(cols1.union(cols2))
            
            # Check shape similarity
            shape_similar = abs(len(df1) - len(df2)) < 10  # Similar number of rows
            
            return {
                'column_overlap': col_overlap,
                'shape_similar': shape_similar,
                'cols1': len(cols1),
                'cols2': len(cols2),
                'rows1': len(df1),
                'rows2': len(df2)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def deep_content_analysis(self, overlapping_files):
        """Perform deep content analysis on overlapping files."""
        if not overlapping_files:
            return
        
        print(f"\n🔬 DEEP CONTENT ANALYSIS")
        print("=" * 30)
        
        for filename, category in overlapping_files:
            print(f"\n📄 Analyzing: {filename} ({category})")
            
            # Find file paths
            train_paths = []
            test_paths = []
            
            for train_dir in self.training_dirs:
                train_path = self.base_dir / train_dir / f"{filename}.csv"
                if train_path.exists():
                    train_paths.append(train_path)
            
            for test_dir in self.test_dirs:
                test_path = self.base_dir / test_dir / f"{filename}.csv"
                if test_path.exists():
                    test_paths.append(test_path)
            
            # Compare content
            for train_path in train_paths:
                for test_path in test_paths:
                    similarity = self.analyze_file_content_similarity(train_path, test_path)
                    
                    if 'error' in similarity:
                        print(f"   ❌ Error comparing {train_path.name} vs {test_path.name}: {similarity['error']}")
                        continue
                    
                    print(f"   📊 {train_path.parent.name}/{train_path.name} vs {test_path.parent.name}/{test_path.name}")
                    print(f"      Column overlap: {similarity['column_overlap']:.2%}")
                    print(f"      Columns: {similarity['cols1']} vs {similarity['cols2']}")
                    print(f"      Rows: {similarity['rows1']} vs {similarity['rows2']}")
                    
                    if similarity['column_overlap'] > 0.8:
                        print(f"      ⚠️  HIGH CONTENT SIMILARITY - Potential data leakage!")
    
    def generate_report(self):
        """Generate comprehensive overlap report."""
        print("🎯 STARTING COMPREHENSIVE DATASET OVERLAP CHECK")
        print("=" * 60)
        
        # List all directories
        print(f"\n📁 DIRECTORY STRUCTURE")
        for dir_type, dirs in [("Training", self.training_dirs), ("Test", self.test_dirs)]:
            print(f"\n{dir_type} directories:")
            for d in dirs:
                path = self.base_dir / d
                if path.exists():
                    files = list(path.glob("*.csv"))
                    print(f"   {d}: {len(files)} files")
                    for f in sorted(files)[:5]:  # Show first 5 files
                        print(f"      • {f.stem}")
                    if len(files) > 5:
                        print(f"      ... and {len(files) - 5} more")
                else:
                    print(f"   {d}: ❌ Not found")
        
        # Check overlaps
        overlapping_files = self.check_all_overlaps()
        
        # Deep analysis if overlaps found
        if overlapping_files:
            self.deep_content_analysis(overlapping_files)
        
        # Final summary
        print(f"\n🎯 FINAL SUMMARY")
        print("=" * 20)
        
        if overlapping_files:
            print(f"❌ DATA LEAKAGE DETECTED!")
            print(f"   Found {len(overlapping_files)} overlapping files")
            print(f"   Files: {[f[0] for f in overlapping_files]}")
            print(f"\n🚨 RECOMMENDATION: Remove overlapping files from training or test set")
        else:
            print(f"✅ NO DATA LEAKAGE DETECTED")
            print(f"   Training and test sets are properly separated")
        
        return overlapping_files


def main():
    """Main execution function."""
    checker = DatasetOverlapChecker()
    overlaps = checker.generate_report()
    
    # Exit with error code if overlaps found
    if overlaps:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
