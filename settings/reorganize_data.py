#!/usr/bin/env python3
"""
Data Reorganization Script
Move files from training to testing to achieve desired split:
- Keep 35 files in training
- Move 35 files from training to testing
"""

import os
import shutil
import random
from pathlib import Path

def get_training_files():
    """Get all training file base names (without extensions)"""
    training_source = Path("assets/training/source")
    files = []
    
    for csv_file in training_source.glob("*.csv"):
        base_name = csv_file.stem
        files.append(base_name)
    
    return sorted(files)

def move_file_set(base_name, from_training=True):
    """Move a complete set of files (source, target, expected) between training and testing"""
    
    if from_training:
        # Move from training to testing
        source_dir = "assets/training"
        target_dir = "assets/test"
        action = "training → testing"
    else:
        # Move from testing to training
        source_dir = "assets/test"
        target_dir = "assets/training"
        action = "testing → training"
    
    moved_files = []
    
    # Files to move
    files_to_move = [
        (f"{source_dir}/source/{base_name}.csv", f"{target_dir}/source/{base_name}.csv"),
        (f"{source_dir}/target/{base_name}.json", f"{target_dir}/target/{base_name}.json"),
        (f"{source_dir}/expected/{base_name}_mapping.json", f"{target_dir}/expected/{base_name}_mapping.json")
    ]
    
    for src_path, dst_path in files_to_move:
        if os.path.exists(src_path):
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            
            # Move the file
            shutil.move(src_path, dst_path)
            moved_files.append(f"{os.path.basename(src_path)}")
        else:
            print(f"⚠️  Warning: {src_path} not found")
    
    if moved_files:
        print(f"✅ Moved {base_name}: {', '.join(moved_files)} ({action})")
    
    return len(moved_files) > 0

def main():
    print("🔄 DATA REORGANIZATION SCRIPT")
    print("=" * 50)
    print("Goal: Keep 35 training files, move 35 to testing")
    
    # Get current state
    training_files = get_training_files()
    current_training_count = len(training_files)
    
    print(f"\n📊 Current State:")
    print(f"   Training files: {current_training_count}")
    
    # Check if we need to move files
    target_training_count = 35
    files_to_move = current_training_count - target_training_count
    
    if files_to_move <= 0:
        print(f"✅ Already have {current_training_count} or fewer training files. No action needed.")
        return
    
    print(f"   Files to move to testing: {files_to_move}")
    print(f"   Files to keep in training: {target_training_count}")
    
    # Randomly select files to move
    random.seed(42)  # For reproducibility
    files_to_move_list = random.sample(training_files, files_to_move)
    files_to_keep = [f for f in training_files if f not in files_to_move_list]
    
    print(f"\n📋 Files selected to move to testing ({len(files_to_move_list)}):")
    for i, file_name in enumerate(sorted(files_to_move_list), 1):
        print(f"   {i:2d}. {file_name}")
    
    print(f"\n📋 Files to keep in training ({len(files_to_keep)}):")
    for i, file_name in enumerate(sorted(files_to_keep), 1):
        print(f"   {i:2d}. {file_name}")
    
    # Confirm before proceeding
    response = input(f"\n❓ Proceed with moving {files_to_move} files from training to testing? (y/N): ")
    if response.lower() != 'y':
        print("❌ Operation cancelled.")
        return
    
    # Move the files
    print(f"\n🔄 Moving files...")
    successful_moves = 0
    
    for base_name in files_to_move_list:
        if move_file_set(base_name, from_training=True):
            successful_moves += 1
    
    # Final summary
    print(f"\n🎉 REORGANIZATION COMPLETE!")
    print(f"✅ Successfully moved {successful_moves}/{len(files_to_move_list)} file sets")
    
    # Verify final counts
    final_training_files = get_training_files()
    print(f"📊 Final counts:")
    print(f"   Training files: {len(final_training_files)}")
    print(f"   Expected training: {target_training_count}")
    
    if len(final_training_files) == target_training_count:
        print("✅ Perfect! Target achieved.")
    else:
        print(f"⚠️  Warning: Expected {target_training_count}, got {len(final_training_files)}")
    
    print(f"\n📄 You can now run your training/testing pipelines with the new data split!")

if __name__ == "__main__":
    main()