#!/usr/bin/env python3
"""
Valentine Dataset Organization Script
Copies Valentine benchmark datasets and organizes them into train/test splits
with proper ground truth placement.

This script:
1. Scans Valentine datasets directory for all schema matching datasets
2. Copies source/target files to appropriate train/test directories
3. Places ground truth mappings in the expected directory
4. Creates a proper academic train/test split for thesis evaluation
"""

import os
import shutil
import json
from pathlib import Path
from typing import List, Dict, Tuple


class ValentineDatasetOrganizer:
    """Organizes Valentine benchmark datasets for harmonize pipeline"""
    
    def __init__(self, valentine_base_dir: str, output_base_dir: str = "assets"):
        self.valentine_dir = Path(valentine_base_dir)
        self.output_dir = Path(output_base_dir)
        
        # Create training and test directories as requested
        self.training_base = self.output_dir / "training"
        self.test_base = self.output_dir / "test"
        
        self.training_source_dir = self.training_base / "source"
        self.training_target_dir = self.training_base / "target"
        self.training_expected_dir = self.training_base / "expected"
        
        self.test_source_dir = self.test_base / "source"
        self.test_target_dir = self.test_base / "target"
        self.test_expected_dir = self.test_base / "expected"
        
        # Create backup directories for the original files we're replacing
        self.backup_dir = self.output_dir / "valentine_backup"
        
        # Ensure all directories exist
        directories = [
            self.training_source_dir, self.training_target_dir, self.training_expected_dir,
            self.test_source_dir, self.test_target_dir, self.test_expected_dir,
            self.backup_dir
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def discover_datasets(self) -> List[Dict]:
        """Discover all Valentine datasets with their components"""
        datasets = []
        
        print("🔍 Discovering Valentine datasets...")
        
        # Walk through all Valentine dataset directories
        for root, dirs, files in os.walk(self.valentine_dir):
            root_path = Path(root)
            
            # Look for datasets (directories with source/target/mapping files)
            source_csv = None
            target_json = None
            mapping_json = None
            
            for file in files:
                if file.endswith('_source.csv'):
                    source_csv = root_path / file
                elif file.endswith('_target.json'):
                    target_json = root_path / file
                elif file.endswith('_mapping.json'):
                    mapping_json = root_path / file
            
            # If we have all components, add to datasets
            if source_csv and target_json and mapping_json:
                dataset_name = source_csv.stem.replace('_source', '')
                
                datasets.append({
                    'name': dataset_name,
                    'source_csv': source_csv,
                    'target_json': target_json,
                    'mapping_json': mapping_json,
                    'category': self._categorize_dataset(root_path)
                })
                
                print(f"   ✅ Found: {dataset_name} ({self._categorize_dataset(root_path)})")
        
        print(f"\n📊 Total datasets discovered: {len(datasets)}")
        return datasets
    
    def _categorize_dataset(self, path: Path) -> str:
        """Categorize dataset based on its path"""
        path_str = str(path).lower()
        
        if 'joinable' in path_str:
            return 'Joinable'
        elif 'unionable' in path_str:
            return 'Unionable'
        elif 'view' in path_str:
            return 'View-Unionable'
        elif 'semantically' in path_str:
            return 'Semantically-Joinable'
        elif 'chembl' in path_str:
            return 'ChEMBL'
        elif 'magellan' in path_str:
            return 'Magellan'
        elif 'wikidata' in path_str:
            return 'Wikidata'
        elif 'tpc' in path_str:
            return 'TPC-DI'
        else:
            return 'Other'
    
    def backup_existing_files(self):
        """Backup any existing files in training and test directories"""
        print("🔄 Backing up existing files...")
        
        backup_count = 0
        
        # Backup training files
        if self.training_source_dir.exists():
            for file in self.training_source_dir.glob("*.csv"):
                backup_dest = self.backup_dir / f"training_{file.name}"
                shutil.copy2(file, backup_dest)
                backup_count += 1
                
        if self.training_target_dir.exists():
            for file in self.training_target_dir.glob("*.json"):
                backup_dest = self.backup_dir / f"training_{file.name}"
                shutil.copy2(file, backup_dest)
                backup_count += 1
        
        # Backup test files
        if self.test_source_dir.exists():
            for file in self.test_source_dir.glob("*.csv"):
                backup_dest = self.backup_dir / f"test_{file.name}"
                shutil.copy2(file, backup_dest)
                backup_count += 1
                
        if self.test_target_dir.exists():
            for file in self.test_target_dir.glob("*.json"):
                backup_dest = self.backup_dir / f"test_{file.name}"
                shutil.copy2(file, backup_dest)
                backup_count += 1
        
        if backup_count > 0:
            print(f"   📦 Backed up {backup_count} existing files to {self.backup_dir}")
        else:
            print(f"   ✅ No existing files to backup")
    
    def copy_dataset(self, dataset: Dict, is_training: bool = True) -> bool:
        """Copy a single dataset to the appropriate training or test directory"""
        
        try:
            if is_training:
                # Copy to training directories
                source_dest = self.training_source_dir / f"{dataset['name']}.csv"
                target_dest = self.training_target_dir / f"{dataset['name']}.json"
                mapping_dest = self.training_expected_dir / f"{dataset['name']}_mapping.json"
            else:
                # Copy to test directories
                source_dest = self.test_source_dir / f"{dataset['name']}.csv"
                target_dest = self.test_target_dir / f"{dataset['name']}.json"
                mapping_dest = self.test_expected_dir / f"{dataset['name']}_mapping.json"
            
            # Copy source CSV
            shutil.copy2(dataset['source_csv'], source_dest)
            
            # Copy target JSON
            shutil.copy2(dataset['target_json'], target_dest)
            
            # Copy ground truth mapping to expected directory
            shutil.copy2(dataset['mapping_json'], mapping_dest)
            
            split_type = "training" if is_training else "test"
            print(f"   ✅ {dataset['name']} → {split_type}")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to copy {dataset['name']}: {e}")
            return False
    
    def organize_datasets(self, limit: int = None):
        """Main method to organize Valentine datasets
        
        Args:
            limit: Maximum number of datasets to organize (None = all datasets)
        """
        
        print("🚀 Valentine Dataset Organization")
        print("=" * 60)
        
        # Step 1: Backup existing files
        self.backup_existing_files()
        
        # Step 2: Discover all datasets
        all_datasets = self.discover_datasets()
        
        if not all_datasets:
            print("❌ No datasets found!")
            return
        
        # Step 3: Apply limit if specified
        if limit is not None:
            if limit <= 0:
                print("❌ Limit must be greater than 0!")
                return
            datasets = all_datasets[:limit]
            print(f"\n🔢 Limiting to first {limit} datasets (out of {len(all_datasets)} available)")
        else:
            datasets = all_datasets
            print(f"\n📊 Processing all {len(datasets)} datasets")
        
        # Step 4: Split datasets into training and test (70/30 split)
        train_size = int(len(datasets) * 0.7)
        training_datasets = datasets[:train_size]
        test_datasets = datasets[train_size:]
        
        print(f"\n📚 Organizing {len(datasets)} datasets:")
        print(f"   🎓 Training: {len(training_datasets)} datasets")
        print(f"   🧪 Test: {len(test_datasets)} datasets")
        
        # Copy training datasets
        print(f"\n📚 Copying training datasets...")
        training_success = 0
        for i, dataset in enumerate(training_datasets, 1):
            print(f"   [{i}/{len(training_datasets)}] ", end="")
            if self.copy_dataset(dataset, is_training=True):
                training_success += 1
        
        # Copy test datasets
        print(f"\n🧪 Copying test datasets...")
        test_success = 0
        for i, dataset in enumerate(test_datasets, 1):
            print(f"   [{i}/{len(test_datasets)}] ", end="")
            if self.copy_dataset(dataset, is_training=False):
                test_success += 1
        
                # Step 5: Generate summary
        self.generate_summary(datasets, training_success, test_success, len(all_datasets))
    
    def generate_summary(self, datasets: List[Dict], training_success: int, test_success: int, total_available: int = None):
        """Generate organization summary"""
        
        print(f"\n📊 ORGANIZATION SUMMARY")
        print("=" * 60)
        print(f"✅ Training datasets: {training_success}/{int(len(datasets) * 0.7)}")
        print(f"✅ Test datasets: {test_success}/{len(datasets) - int(len(datasets) * 0.7)}")
        if total_available and len(datasets) < total_available:
            print(f"📊 Limited processing: {len(datasets)}/{total_available} available datasets")
        
        print(f"\n📂 Output structure:")
        print(f"   🎓 Training: {self.training_base}/")
        print(f"      📁 Source: {training_success} files")
        print(f"      📁 Target: {training_success} files")
        print(f"      🎯 Expected: {training_success} mappings")
        print(f"   🧪 Test: {self.test_base}/")
        print(f"      📁 Source: {test_success} files")
        print(f"      📁 Target: {test_success} files")
        print(f"      🎯 Expected: {test_success} mappings")
        print(f"   📦 Backup: {self.backup_dir} (original files)")
        
        # Category breakdown
        print(f"\n📊 Dataset categories:")
        by_category = {}
        
        for dataset in datasets:
            cat = dataset['category']
            by_category[cat] = by_category.get(cat, 0) + 1
        
        for category in sorted(by_category.keys()):
            count = by_category[category]
            print(f"   📁 {category}: {count} datasets")
        
        print(f"\n🎓 Ready for pipeline execution!")
        print(f"   • Training files: assets/training/source/ and assets/training/target/")
        print(f"   • Test files: assets/test/source/ and assets/test/target/")
        print(f"   • Run training: python3 train_weights.py")
        print(f"   • Run test evaluation: python3 train_weights.py --test")
        if total_available and len(datasets) < total_available:
            print(f"\n💡 To process all {total_available} datasets, run without limit parameter")


def main():
    
    def backup_existing_files(self):
        """Backup any existing files in source/target directories"""
        print("🔄 Backing up existing files...")
        
        backup_count = 0
        
        # Backup source files
        if self.source_dir.exists():
            for file in self.source_dir.glob("*.csv"):
                backup_dest = self.backup_dir / file.name
                shutil.copy2(file, backup_dest)
                backup_count += 1
        
        # Backup target files  
        if self.target_dir.exists():
            for file in self.target_dir.glob("*.json"):
                backup_dest = self.backup_dir / file.name
                shutil.copy2(file, backup_dest)
                backup_count += 1
        
        if backup_count > 0:
            print(f"   📦 Backed up {backup_count} existing files to {self.backup_dir}")
        else:
            print(f"   ✅ No existing files to backup")
    
    def generate_summary(self, datasets: List[Dict], success_count: int, total_available: int = None):
        """Generate organization summary"""
        
        print(f"\n📊 ORGANIZATION SUMMARY")
        print("=" * 60)
        print(f"✅ Datasets copied: {success_count}/{len(datasets)}")
        if total_available and len(datasets) < total_available:
            print(f"📊 Limited processing: {len(datasets)}/{total_available} available datasets")
        print(f"📁 Ground truth mappings: {success_count}")
        
        print(f"\n📂 Output structure:")
        print(f"   � Source: {self.source_dir} ({success_count} files)")
        print(f"   📁 Target: {self.target_dir} ({success_count} files)")
        print(f"   🎯 Expected: {self.expected_dir} ({success_count} mappings)")
        print(f"   📦 Backup: {self.backup_dir} (original files)")
        
        # Category breakdown
        print(f"\n📊 Dataset categories:")
        by_category = {}
        
        for dataset in datasets:
            cat = dataset['category']
            by_category[cat] = by_category.get(cat, 0) + 1
        
        for category in sorted(by_category.keys()):
            count = by_category[category]
            print(f"   📁 {category}: {count} datasets")
        
        print(f"\n🎓 Ready for pipeline execution!")
        print(f"   • All files are now in the locations main.py expects")
        print(f"   • Run individual evaluations: python3 main.py --source-table <name> --target-table <name>")
        print(f"   • Run ensemble training: python3 main.py --training")
        print(f"   • Note: Use train_weights.py for systematic weight optimization")
        if total_available and len(datasets) < total_available:
            print(f"\n💡 To process all {total_available} datasets, run without limit parameter")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Organize Valentine benchmark datasets')
    parser.add_argument('limit', nargs='?', type=int, default=None,
                       help='Maximum number of datasets to organize (default: all)')
    parser.add_argument('--all', action='store_true',
                       help='Process all datasets (override limit)')
    
    args = parser.parse_args()
    
    # Handle limit parameter
    limit = None if args.all else args.limit
    
    # Configuration
    valentine_datasets_dir = "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/harmonize/assets/Monitoring_data/data/Valentine-datasets"
    output_dir = "assets"
    
    # Create organizer and run
    organizer = ValentineDatasetOrganizer(valentine_datasets_dir, output_dir)
    organizer.organize_datasets(limit=limit)


if __name__ == "__main__":
    main()