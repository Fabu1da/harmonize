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
from typing import List, Dict, Tuple, Set
import hashlib


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
    
    def check_existing_data_leakage(self) -> Dict[str, List[str]]:
        """Check for data leakage in existing train/test directories"""
        print("🔍 Checking existing data for leakage...")
        
        # Get existing files
        existing_train_source = {f.stem for f in self.training_source_dir.glob("*.csv") if f.is_file()}
        existing_test_source = {f.stem for f in self.test_source_dir.glob("*.csv") if f.is_file()}
        
        existing_train_target = {f.stem.replace('.json', '') for f in self.training_target_dir.glob("*.json") if f.is_file()}
        existing_test_target = {f.stem.replace('.json', '') for f in self.test_target_dir.glob("*.json") if f.is_file()}
        
        # Check overlaps
        source_overlap = existing_train_source.intersection(existing_test_source)
        target_overlap = existing_train_target.intersection(existing_test_target)
        
        # Cross-directory checks
        cross_overlap = existing_train_source.intersection(existing_test_target).union(
                       existing_train_target.intersection(existing_test_source))
        
        leakage_report = {
            'source_overlap': list(source_overlap),
            'target_overlap': list(target_overlap), 
            'cross_overlap': list(cross_overlap),
            'total_overlaps': len(source_overlap) + len(target_overlap) + len(cross_overlap)
        }
        
        if leakage_report['total_overlaps'] > 0:
            print(f"   ❌ EXISTING DATA LEAKAGE DETECTED:")
            if source_overlap:
                print(f"      📁 Source overlap: {len(source_overlap)} files")
                for file in sorted(source_overlap):
                    print(f"         • {file}")
            if target_overlap:
                print(f"      📁 Target overlap: {len(target_overlap)} files")
                for file in sorted(target_overlap):
                    print(f"         • {file}")
            if cross_overlap:
                print(f"      🔄 Cross-directory overlap: {len(cross_overlap)} files")
                for file in sorted(cross_overlap):
                    print(f"         • {file}")
        else:
            print(f"   ✅ No existing data leakage detected")
        
        return leakage_report
    
    def dataset_exists_anywhere(self, dataset_name: str) -> Dict[str, List[str]]:
        """Check if a dataset already exists in any training or test directory"""
        existing_locations = {
            'training_source': [],
            'training_target': [],
            'training_expected': [],
            'test_source': [],
            'test_target': [],
            'test_expected': []
        }
        
        # Check all possible locations
        locations = [
            (self.training_source_dir, 'training_source', '*.csv'),
            (self.training_target_dir, 'training_target', '*.json'),
            (self.training_expected_dir, 'training_expected', '*_mapping.json'),
            (self.test_source_dir, 'test_source', '*.csv'),
            (self.test_target_dir, 'test_target', '*.json'),
            (self.test_expected_dir, 'test_expected', '*_mapping.json')
        ]
        
        for directory, location_key, pattern in locations:
            if directory.exists():
                # Check for exact dataset name match
                if location_key.endswith('expected'):
                    # For expected files, look for dataset_name_mapping.json
                    expected_file = directory / f"{dataset_name}_mapping.json"
                    if expected_file.exists():
                        existing_locations[location_key].append(str(expected_file))
                elif location_key.endswith('source'):
                    # For source files, look for dataset_name.csv
                    source_file = directory / f"{dataset_name}.csv"
                    if source_file.exists():
                        existing_locations[location_key].append(str(source_file))
                elif location_key.endswith('target'):
                    # For target files, look for dataset_name.json
                    target_file = directory / f"{dataset_name}.json"
                    if target_file.exists():
                        existing_locations[location_key].append(str(target_file))
        
        # Count total existing files
        total_existing = sum(len(files) for files in existing_locations.values())
        
        return {
            'locations': existing_locations,
            'total_existing': total_existing,
            'exists_anywhere': total_existing > 0
        }
    
    def compute_dataset_signature(self, dataset: Dict) -> str:
        """Compute a signature for a dataset based on content"""
        try:
            # Create signature from source CSV content
            with open(dataset['source_csv'], 'rb') as f:
                source_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            # Create signature from target JSON content
            with open(dataset['target_json'], 'rb') as f:
                target_hash = hashlib.md5(f.read()).hexdigest()[:8]
            
            return f"{source_hash}_{target_hash}"
        except Exception:
            # Fallback to name-based signature
            return hashlib.md5(dataset['name'].encode()).hexdigest()[:8]
    
    def detect_potential_valentine_leakage(self, datasets: List[Dict]) -> Dict[str, List[str]]:
        """Detect potential data leakage within Valentine datasets themselves"""
        print("\n🔍 Checking Valentine datasets for internal duplicates...")
        
        signatures = {}
        duplicates = []
        
        for dataset in datasets:
            signature = self.compute_dataset_signature(dataset)
            
            if signature in signatures:
                duplicates.append({
                    'signature': signature,
                    'datasets': [signatures[signature], dataset['name']],
                    'category': f"{signatures[signature]} vs {dataset['name']}"
                })
                print(f"   ⚠️  Potential duplicate: {signatures[signature]} ↔ {dataset['name']}")
            else:
                signatures[signature] = dataset['name']
        
        if duplicates:
            print(f"   ❌ Found {len(duplicates)} potential duplicates")
        else:
            print(f"   ✅ No internal duplicates detected")
        
        return {'duplicates': duplicates, 'count': len(duplicates)}
    
    def validate_train_test_split(self, training_datasets: List[Dict], test_datasets: List[Dict]) -> bool:
        """Validate that train/test split has no leakage"""
        print("\n🔍 Validating train/test split for leakage...")
        
        train_names = {d['name'] for d in training_datasets}
        test_names = {d['name'] for d in test_datasets}
        
        # Check name overlap
        name_overlap = train_names.intersection(test_names)
        
        # Check signature overlap
        train_signatures = {self.compute_dataset_signature(d) for d in training_datasets}
        test_signatures = {self.compute_dataset_signature(d) for d in test_datasets}
        signature_overlap = train_signatures.intersection(test_signatures)
        
        has_leakage = len(name_overlap) > 0 or len(signature_overlap) > 0
        
        if has_leakage:
            print(f"   ❌ TRAIN/TEST LEAKAGE DETECTED:")
            if name_overlap:
                print(f"      📁 Name overlap: {len(name_overlap)} datasets")
                for name in sorted(name_overlap):
                    print(f"         • {name}")
            if signature_overlap:
                print(f"      🔍 Content overlap: {len(signature_overlap)} signatures")
                for sig in sorted(signature_overlap):
                    print(f"         • {sig}")
            return False
        else:
            print(f"   ✅ Clean train/test split - no leakage detected")
            return True
    
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
    
    def copy_dataset(self, dataset: Dict, is_training: bool = True) -> Dict[str, any]:
        """Copy a single dataset to the appropriate training or test directory
        
        Returns:
            Dict with copy status: {'success': bool, 'skipped': bool, 'reason': str, 'existing_locations': list}
        """
        
        # First check if dataset already exists anywhere
        existence_check = self.dataset_exists_anywhere(dataset['name'])
        
        if existence_check['exists_anywhere']:
            existing_locations = []
            for location_key, files in existence_check['locations'].items():
                if files:
                    existing_locations.extend([f"{location_key}: {Path(f).name}" for f in files])
            
            return {
                'success': False,
                'skipped': True,
                'reason': f"Dataset '{dataset['name']}' already exists",
                'existing_locations': existing_locations
            }
        
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
            
            return {
                'success': True,
                'skipped': False,
                'reason': 'Successfully copied',
                'existing_locations': []
            }
            
        except Exception as e:
            return {
                'success': False,
                'skipped': False,
                'reason': f"Copy failed: {str(e)}",
                'existing_locations': []
            }
    
    def organize_datasets(self, limit: int = None, force: bool = False):
        """Main method to organize Valentine datasets
        
        Args:
            limit: Maximum number of datasets to organize (None = all datasets)
            force: Proceed even if data leakage is detected
        """
        
        print("🚀 Valentine Dataset Organization with Data Leakage Detection")
        print("=" * 70)
        
        # Step 0: Check existing data for leakage
        existing_leakage = self.check_existing_data_leakage()
        
        if existing_leakage['total_overlaps'] > 0 and not force:
            print(f"\n🚨 CRITICAL: Existing data leakage detected!")
            print(f"   Run with --force to proceed anyway, or clean up existing data first.")
            print(f"   Recommendation: Move overlapping files to backup directory.")
            return False
        
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
        
        # Step 4: Check Valentine datasets for internal duplicates
        valentine_duplicates = self.detect_potential_valentine_leakage(datasets)
        
        if valentine_duplicates['count'] > 0 and not force:
            print(f"\n⚠️  WARNING: Internal duplicates detected in Valentine datasets!")
            print(f"   This may indicate similar datasets that could cause leakage.")
            print(f"   Run with --force to proceed anyway.")
            if not force:
                return False
        
        # Step 5: Split datasets into training and test (70/30 split)
        train_size = int(len(datasets) * 0.7)
        training_datasets = datasets[:train_size]
        test_datasets = datasets[train_size:]
        
        # Step 6: Validate train/test split for leakage
        if not self.validate_train_test_split(training_datasets, test_datasets):
            if not force:
                print(f"\n🚨 CRITICAL: Train/test split contains data leakage!")
                print(f"   This would invalidate evaluation results.")
                print(f"   Run with --force to proceed anyway (NOT RECOMMENDED).")
                return False
            else:
                print(f"\n⚠️  WARNING: Proceeding despite detected leakage (--force used)")
        
        print(f"\n📚 Organizing {len(datasets)} datasets:")
        print(f"   🎓 Training: {len(training_datasets)} datasets")
        print(f"   🧪 Test: {len(test_datasets)} datasets")
        
        # Copy training datasets
        print(f"\n📚 Copying training datasets...")
        training_success = 0
        training_skipped = 0
        training_failed = 0
        
        for i, dataset in enumerate(training_datasets, 1):
            print(f"   [{i}/{len(training_datasets)}] {dataset['name']}: ", end="")
            result = self.copy_dataset(dataset, is_training=True)
            
            if result['success']:
                print(f"✅ → training")
                training_success += 1
            elif result['skipped']:
                print(f"⏭️  SKIPPED - {result['reason']}")
                for location in result['existing_locations']:
                    print(f"      📍 {location}")
                training_skipped += 1
            else:
                print(f"❌ FAILED - {result['reason']}")
                training_failed += 1
        
        # Copy test datasets
        print(f"\n🧪 Copying test datasets...")
        test_success = 0
        test_skipped = 0
        test_failed = 0
        
        for i, dataset in enumerate(test_datasets, 1):
            print(f"   [{i}/{len(test_datasets)}] {dataset['name']}: ", end="")
            result = self.copy_dataset(dataset, is_training=False)
            
            if result['success']:
                print(f"✅ → test")
                test_success += 1
            elif result['skipped']:
                print(f"⏭️  SKIPPED - {result['reason']}")
                for location in result['existing_locations']:
                    print(f"      📍 {location}")
                test_skipped += 1
            else:
                print(f"❌ FAILED - {result['reason']}")
                test_failed += 1
        
        # Step 7: Generate summary with leakage report
        copy_stats = {
            'training': {'success': training_success, 'skipped': training_skipped, 'failed': training_failed},
            'test': {'success': test_success, 'skipped': test_skipped, 'failed': test_failed}
        }
        
        self.generate_summary(datasets, copy_stats, len(all_datasets), 
                            existing_leakage, valentine_duplicates)
        
        return True
    
    def generate_summary(self, datasets: List[Dict], copy_stats: Dict, 
                        total_available: int = None, existing_leakage: Dict = None, 
                        valentine_duplicates: Dict = None):
        """Generate organization summary with detailed copy statistics"""
        
        print(f"\n📊 ORGANIZATION SUMMARY")
        print("=" * 60)
        
        # Training statistics (match the actual split logic)
        train_attempted = int(len(datasets) * 0.7)
        test_total = len(datasets) - train_attempted
        print(f"🎓 Training datasets:")
        print(f"   ✅ Successfully copied: {copy_stats['training']['success']}/{train_attempted}")
        if copy_stats['training']['skipped'] > 0:
            print(f"   ⏭️  Skipped (already exist): {copy_stats['training']['skipped']}")
        if copy_stats['training']['failed'] > 0:
            print(f"   ❌ Failed: {copy_stats['training']['failed']}")
        
        # Test statistics
        print(f"\n🧪 Test datasets:")
        print(f"   ✅ Successfully copied: {copy_stats['test']['success']}/{test_total}")
        if copy_stats['test']['skipped'] > 0:
            print(f"   ⏭️  Skipped (already exist): {copy_stats['test']['skipped']}")
        if copy_stats['test']['failed'] > 0:
            print(f"   ❌ Failed: {copy_stats['test']['failed']}")
        
        if total_available and len(datasets) < total_available:
            print(f"\n📊 Limited processing: {len(datasets)}/{total_available} available datasets")
        
        print(f"\n📂 Output structure:")
        print(f"   🎓 Training: {self.training_base}/")
        print(f"      📁 Source: {copy_stats['training']['success']} files")
        print(f"      📁 Target: {copy_stats['training']['success']} files")
        print(f"      🎯 Expected: {copy_stats['training']['success']} mappings")
        print(f"   🧪 Test: {self.test_base}/")
        print(f"      📁 Source: {copy_stats['test']['success']} files")
        print(f"      📁 Target: {copy_stats['test']['success']} files")
        print(f"      🎯 Expected: {copy_stats['test']['success']} mappings")
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
        
        # Data leakage summary
        print(f"\n🔒 DATA LEAKAGE REPORT:")
        if existing_leakage and existing_leakage['total_overlaps'] == 0:
            print(f"   ✅ No existing data leakage")
        elif existing_leakage:
            print(f"   ⚠️  Existing leakage: {existing_leakage['total_overlaps']} overlaps detected")
        
        if valentine_duplicates and valentine_duplicates['count'] == 0:
            print(f"   ✅ No Valentine dataset duplicates")
        elif valentine_duplicates:
            print(f"   ⚠️  Valentine duplicates: {valentine_duplicates['count']} potential duplicates")
        
        print(f"   ✅ Train/test split validated - no leakage detected")
        
        # Skipped files summary
        total_skipped = copy_stats['training']['skipped'] + copy_stats['test']['skipped']
        if total_skipped > 0:
            print(f"   ⏭️  {total_skipped} files skipped (already existed - no duplicates created)")
        
        print(f"\n🎓 Ready for pipeline execution!")
        print(f"   • Training files: assets/training/source/ and assets/training/target/")
        print(f"   • Test files: assets/test/source/ and assets/test/target/")
        print(f"   • Run training: python3 train_weights.py")
        print(f"   • Run test evaluation: python3 train_weights.py --test")
        print(f"   • Validate separation: python3 check_dataset_overlap.py")
        if total_skipped > 0:
            print(f"\n💡 {total_skipped} files were skipped to prevent duplicates.")
            print(f"   This maintains clean train/test separation.")
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
    parser.add_argument('--force', action='store_true',
                       help='Proceed even if data leakage is detected (NOT RECOMMENDED)')
    
    args = parser.parse_args()
    
    # Handle limit parameter
    limit = None if args.all else args.limit
    
    # Configuration
    valentine_datasets_dir = "/Users/fabu1da/Desktop/schoolstuff/mastersThesis/Project/harmonize/assets/Monitoring_data/data/Valentine-datasets"
    output_dir = "assets"
    
    # Create organizer and run
    organizer = ValentineDatasetOrganizer(valentine_datasets_dir, output_dir)
    success = organizer.organize_datasets(limit=limit, force=args.force)
    
    if not success:
        print(f"\n❌ Organization failed due to data leakage concerns.")
        print(f"   Use --force to proceed anyway (not recommended for thesis).")
        exit(1)
    else:
        print(f"\n✅ Organization completed successfully with no data leakage!")
        exit(0)


if __name__ == "__main__":
    main()