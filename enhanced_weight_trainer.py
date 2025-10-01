#!/usr/bin/env python3
"""
Enhanced Ensemble Weight Trainer with Explicit Train/Validation/Test Split
Supports the academic three-way split methodology for robust evaluation.

Directory Structure:
assets/
├── training/           # 60% of data - used to run all 8 configs
│   ├── source/
│   └── target/
├── validation/         # 20% of data - used to SELECT best config
│   ├── source/
│   └── target/
└── test/              # 20% of data - used ONCE for Chapter 6 results
    ├── source/
    └── target/
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, Tuple, List, Optional


class EnhancedEnsembleWeightTrainer:
    """
    Enhanced training pipeline supporting explicit train/validation/test splits.
    """
    
    def __init__(self, base_dir: str = "assets"):
        self.base_dir = Path(base_dir)
        self.training_dir = self.base_dir / "training"
        self.validation_dir = self.base_dir / "validation" 
        self.test_dir = self.base_dir / "test"
        self.results_dir = self.base_dir / "training" / "training" / "results"
        self.configs_dir = self.base_dir / "training" / "training" / "configs"
        
        # Ensure directories exist
        for dir_path in [self.results_dir, self.configs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def discover_dataset_pairs(self, source_dir: Path, target_dir: Path) -> Dict[str, Dict]:
        """
        Discover all valid source-target pairs in a directory.
        
        Returns:
            Dict of {pair_name: {'source': path, 'target': path}}
        """
        pairs = {}
        
        if not source_dir.exists() or not target_dir.exists():
            return pairs
            
        source_files = list(source_dir.glob("*.csv"))
        target_files = list(target_dir.glob("*.json"))
        
        # Create all combinations of source × target
        for source_file in source_files:
            for target_file in target_files:
                source_stem = source_file.stem.replace("_source", "")
                target_stem = target_file.stem.replace("_target", "")
                
                pair_name = f"{source_file.stem}_to_{target_file.stem}"
                pairs[pair_name] = {
                    'source': source_file,
                    'target': target_file,
                    'source_stem': source_stem,
                    'target_stem': target_stem
                }
        
        return pairs
    
    def load_datasets(self) -> Tuple[Dict, Dict, Dict]:
        """
        Load training, validation, and test datasets.
        
        Returns:
            Tuple of (training_pairs, validation_pairs, test_pairs)
        """
        print("🔍 Discovering datasets...")
        
        training_pairs = self.discover_dataset_pairs(
            self.training_dir / "source", 
            self.training_dir / "target"
        )
        
        validation_pairs = self.discover_dataset_pairs(
            self.validation_dir / "source",
            self.validation_dir / "target" 
        )
        
        test_pairs = self.discover_dataset_pairs(
            self.test_dir / "source",
            self.test_dir / "target"
        )
        
        print(f"📊 Training pairs: {len(training_pairs)}")
        print(f"📊 Validation pairs: {len(validation_pairs)}")
        print(f"📊 Test pairs: {len(test_pairs)}")
        
        return training_pairs, validation_pairs, test_pairs
    
    def train_weights_explicit_split(self) -> Dict:
        """
        Train optimal ensemble weights using explicit train/validation/test split.
        
        Training data: Used to run all 8 weight configurations
        Validation data: Used to select the best configuration
        Test data: Reserved for final Chapter 6 evaluation
        """
        print("🚀 Starting explicit split weight training...")
        
        # Load all datasets
        training_pairs, validation_pairs, test_pairs = self.load_datasets()
        
        if not training_pairs:
            raise ValueError("No training pairs found! Add datasets to assets/training/source/ and assets/training/target/")
        
        if not validation_pairs:
            raise ValueError("No validation pairs found! Add datasets to assets/validation/source/ and assets/validation/target/")
        
        # Run training pipeline on training data
        print("\n🏋️ Training phase: Running all configurations on training data...")
        training_results = self._run_pipeline_on_pairs(training_pairs, "training")
        
        # Evaluate configurations on validation data  
        print("\n🎯 Validation phase: Evaluating configurations on validation data...")
        validation_results = self._run_pipeline_on_pairs(validation_pairs, "validation")
        
        # Weight configurations to test
        configs = self._get_weight_configurations()
        
        # Calculate scores for each configuration
        config_performance = []
        
        for config in configs:
            # Training score (for reference)
            train_score = self._calculate_config_score(config, training_results)
            
            # Validation score (for selection)
            val_score = self._calculate_config_score(config, validation_results)
            
            config_performance.append({
                'name': config['name'],
                'weights': config['weights'],
                'train_score': train_score,
                'val_score': val_score,
                'config': config
            })
            
            print(f"   {config['name']}: train={train_score:.3f}, val={val_score:.3f}")
        
        # Select best configuration based on validation score
        best_config = max(config_performance, key=lambda x: x['val_score'])
        
        print(f"\n🏆 Best configuration: {best_config['name']}")
        print(f"   Weights: {best_config['weights']}")
        print(f"   Training score: {best_config['train_score']:.3f}")
        print(f"   Validation score: {best_config['val_score']:.3f}")
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'explicit_train_validation_test_split',
            'data_split': {
                'training_pairs': len(training_pairs),
                'validation_pairs': len(validation_pairs), 
                'test_pairs': len(test_pairs),
                'total_pairs': len(training_pairs) + len(validation_pairs) + len(test_pairs)
            },
            'configurations': config_performance,
            'selected_config': best_config,
            'training_pairs_list': list(training_pairs.keys()),
            'validation_pairs_list': list(validation_pairs.keys()),
            'test_pairs_list': list(test_pairs.keys()) if test_pairs else []
        }
        
        # Save detailed results
        results_file = self.results_dir / f"explicit_split_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save optimal weights config
        weights_config = {
            'gpt_weight': best_config['weights'][0],
            'embed_weight': best_config['weights'][1], 
            'cluster_weight': best_config['weights'][2],
            'config_name': best_config['name'],
            'train_score': best_config['train_score'],
            'val_score': best_config['val_score'],
            'methodology': 'explicit_train_validation_test_split',
            'trained_on': datetime.now().isoformat()
        }
        
        config_file = self.configs_dir / "optimal_weights.json"
        with open(config_file, 'w') as f:
            json.dump(weights_config, f, indent=2)
        
        print(f"\n📄 Results saved:")
        print(f"  • Training details: {results_file}")
        print(f"  • Optimal weights: {config_file}")
        
        return results
    
    def evaluate_on_test_set(self) -> Dict:
        """
        Evaluate the selected configuration on the test set.
        This should only be called ONCE for final Chapter 6 results.
        """
        print("🧪 FINAL EVALUATION: Running on test set...")
        print("⚠️  This should only be done ONCE for Chapter 6!")
        
        # Load test data
        _, _, test_pairs = self.load_datasets()
        
        if not test_pairs:
            raise ValueError("No test pairs found! Add datasets to assets/test/source/ and assets/test/target/")
        
        # Load optimal weights
        config_file = self.configs_dir / "optimal_weights.json"
        if not config_file.exists():
            raise ValueError("No optimal weights found! Run training first.")
        
        with open(config_file, 'r') as f:
            optimal_config = json.load(f)
        
        print(f"🎯 Using configuration: {optimal_config['config_name']}")
        print(f"   Weights: ({optimal_config['gpt_weight']}, {optimal_config['embed_weight']}, {optimal_config['cluster_weight']})")
        
        # Run evaluation on test set
        test_results = self._run_pipeline_on_pairs(test_pairs, "test")
        
        # Calculate final test score
        config = {
            'weights': [optimal_config['gpt_weight'], optimal_config['embed_weight'], optimal_config['cluster_weight']]
        }
        test_score = self._calculate_config_score(config, test_results)
        
        print(f"\n🏆 FINAL TEST SCORE: {test_score:.3f}")
        
        # Save test results
        final_results = {
            'timestamp': datetime.now().isoformat(),
            'methodology': 'final_test_evaluation',
            'optimal_config': optimal_config,
            'test_pairs': len(test_pairs),
            'test_score': test_score,
            'test_pairs_list': list(test_pairs.keys())
        }
        
        results_file = self.results_dir / f"final_test_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2, default=str)
        
        print(f"📄 Final results saved: {results_file}")
        
        return final_results
    
    def _run_pipeline_on_pairs(self, pairs: Dict, phase: str) -> Dict:
        """Run the matching pipeline on a set of pairs."""
        import random
        
        print(f"   Running {phase} on {len(pairs)} pairs...")
        
        # For now, generate realistic mock scores
        # TODO: Replace with actual pipeline integration
        results = {}
        random.seed(42)  # For reproducible results
        
        for pair_name, pair_info in pairs.items():
            # Generate realistic scores with some variance
            base_gpt = 0.25 + random.uniform(0, 0.3)
            base_embed = 0.35 + random.uniform(0, 0.3) 
            base_cluster = 0.20 + random.uniform(0, 0.3)
            
            results[pair_name] = {
                'gpt_score': round(base_gpt, 3),
                'embed_score': round(base_embed, 3),
                'cluster_score': round(base_cluster, 3),
                'source': str(pair_info['source']),
                'target': str(pair_info['target'])
            }
            
        print(f"   ✓ Generated scores for {len(results)} pairs")
        return results
    
    def _calculate_config_score(self, config: Dict, results: Dict) -> float:
        """Calculate weighted score for a configuration."""
        total_score = 0.0
        weights = config['weights']
        
        for pair_results in results.values():
            weighted_score = (
                weights[0] * pair_results['gpt_score'] +
                weights[1] * pair_results['embed_score'] + 
                weights[2] * pair_results['cluster_score']
            )
            total_score += weighted_score
        
        return total_score / len(results) if results else 0.0
    
    def _get_weight_configurations(self) -> List[Dict]:
        """Get the 8 weight configurations to test."""
        return [
            {"name": "Equal Weights", "weights": [0.33, 0.33, 0.34]},
            {"name": "GPT Dominant", "weights": [0.6, 0.3, 0.1]},
            {"name": "Embedding Dominant", "weights": [0.2, 0.6, 0.2]},
            {"name": "Clustering Dominant", "weights": [0.1, 0.3, 0.6]},
            {"name": "Balanced Optimized", "weights": [0.2, 0.5, 0.3]},
            {"name": "Conservative", "weights": [0.25, 0.45, 0.3]},
            {"name": "No Clustering", "weights": [0.6, 0.4, 0.0]},
            {"name": "Embed-Cluster Focus", "weights": [0.15, 0.55, 0.3]}
        ]


if __name__ == "__main__":
    trainer = EnhancedEnsembleWeightTrainer()
    
    # Run training with explicit splits
    results = trainer.train_weights_explicit_split()
    
    # Optionally run final test evaluation (only for Chapter 6!)
    # final_results = trainer.evaluate_on_test_set()