#!/usr/bin/env python
"""
Dedicated script for running Step 2 pairwise comparisons from your sketch.
This focuses specifically on GPT vs Embedding, GPT vs Clustering, Embedding vs Clustering
with P, R, F1 score calculations.
"""

from dotenv import load_dotenv
load_dotenv(override=True)

import asyncio
import glob
import json
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from json_schema import ObjectSchema
from schema_inference import infer_schema
from gpt_utils import gpt_column_mapping
from embedding_utils import embedding_column_mapping
from clustering_matcher import clustering_matcher

async def run_pairwise_analysis():
    """
    Run comprehensive pairwise analysis across all datasets
    """
    print("🔄 Starting Step 2: Pairwise Matcher Analysis")
    print("=" * 60)
    print("Comparing: GPT vs Embedding, GPT vs Clustering, Embedding vs Clustering")
    print("Metrics: Precision, Recall, F1 Score, Agreement Rate")
    print("=" * 60)
    
    # Process all source-target combinations
    for source_csv_path in sorted(glob.glob("./assets/source/*.csv")):
        source_table = Path(source_csv_path).stem
        source_data = pd.read_csv(source_csv_path)
        source_schema_path = f"./assets/source/{source_table}.json"
        
        # Load or infer source schema
        if os.path.exists(source_schema_path):
            with open(source_schema_path) as f:
                source_schema = ObjectSchema.model_validate_json(f.read())
            print(f"✅ Loaded source schema: {len(source_schema.properties)} columns")
        else:
            source_schema = await infer_schema(source_data)
            print(f"🔮 Inferred source schema: {len(source_schema.properties)} columns")
        
        print(f"🔍 Source columns: {list(source_schema.properties.keys())[:5]}...")  # Show first 5
        
        for target_path in tqdm(sorted(glob.glob("./assets/target/*.json", recursive=True))):
            target_table, _ = os.path.splitext(os.path.basename(target_path))
            
            with open(target_path) as f:
                target_schema = ObjectSchema.model_validate_json(f.read())
            
            print(f"🔍 Target columns: {list(target_schema.properties.keys())[:5]}...")  # Show first 5
            
            print(f"\n🎯 Analyzing: {source_table} → {target_table}")
            
            # Load ground truth if available
            target_prefix = "_".join(target_table.split("_")[:2])
            real_gt_path = f"./assets/expected/{target_prefix}_mapping.json"
            ground_truth = None
            
            # Check for ground truth file
            print(f"🔍 Looking for ground truth: {real_gt_path}")
            
            try:
                with open(real_gt_path) as f:
                    gt_data = json.load(f)
                    
                # Fix: Handle different ground truth formats
                if "mappings" in gt_data:
                    # Original format: {"mappings": [{"target_column": "...", "source_column": "..."}]}
                    ground_truth = {
                        mapping["target_column"]: mapping["source_column"]
                        for mapping in gt_data["mappings"]
                    }
                elif "matches" in gt_data:
                    # New format: {"matches": [{"target_column": "...", "source_column": "..."}]}
                    ground_truth = {
                        mapping["target_column"]: mapping["source_column"]
                        for mapping in gt_data["matches"]
                    }
                elif isinstance(gt_data, dict) and "target_column" not in gt_data:
                    # Direct mapping format: {"target_col": "source_col"}
                    ground_truth = gt_data
                else:
                    ground_truth = None
                    
                if ground_truth:
                    print(f"✅ Using ground truth with {len(ground_truth)} mappings")
                else:
                    print(f"⚠️ Could not parse ground truth format")
                    
            except Exception as e:
                print(f"⚠️ No ground truth available - using matcher agreement analysis")
                ground_truth = None
            
            # Run all three matchers for pairwise comparison
            print(f"🤖 Running GPT matcher...")
            gpt_predictions = await gpt_column_mapping(source_schema, target_schema, seed=42)
            print(f"   ✅ GPT: {len(gpt_predictions)} predictions")
            
            print(f"🔢 Running Embedding matcher...")
            embedding_predictions = embedding_column_mapping(
                source_columns=list(source_schema.properties.keys()),
                target_columns=list(target_schema.properties.keys()),
                threshold=0
            )
            print(f"   ✅ Embedding: {len(embedding_predictions)} predictions")
            
            print(f"🔗 Running Clustering matcher...")
            clustering_predictions = clustering_matcher(source_schema, target_schema)
            print(f"   ✅ Clustering: {len(clustering_predictions)} predictions")
            
            # Check if any predictions exist
            total_predictions = len(gpt_predictions) + len(embedding_predictions) + len(clustering_predictions)
            if total_predictions == 0:
                print("❌ NO PREDICTIONS FROM ANY MATCHER! Skipping...")
                continue
            
            # Perform pairwise comparisons between all three matchers
            print(f"🔄 Performing pairwise comparisons...")
            pairwise_results = run_all_pairwise_comparisons(
                gpt_predictions=gpt_predictions,
                embedding_predictions=embedding_predictions,
                clustering_predictions=clustering_predictions,
                ground_truth=ground_truth,
                source_table=source_table,
                target_table=target_table
            )
            
            # Show summary of comparisons
            print(f"📊 Pairwise Comparison Results:")
            for comparison_name, metrics in pairwise_results.items():
                comparison_display = comparison_name.replace("_", " ")
                print(f"   {comparison_display}: P={metrics.precision:.3f}, R={metrics.recall:.3f}, F1={metrics.f1_score:.3f}")
            
            # Display detailed table
            comparator.print_pairwise_comparison_table(
                pairwise_results, source_table, target_table
            )
    
    # Export final comprehensive results
    print(f"\n🎯 EXPORTING COMPLETE PAIRWISE ANALYSIS RESULTS")
    comparator.export_pairwise_results("step2_pairwise_complete")
    
    print(f"\n✅ Step 2 Pairwise Analysis Complete!")
    print(f"📁 Results saved to output/step2_pairwise_complete.json")
    print(f"📊 LaTeX table saved to output/step2_pairwise_complete_summary.tex")

