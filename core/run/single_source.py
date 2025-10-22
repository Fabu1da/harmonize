import argparse

from core.run.matchers import run_all_matchers
from core.utils.data import load_real_ground_truth, load_source_data, load_target_schema

import logging


async def process_single_source_target_pair(source_csv_path: str, target_path: str, 
                                        args: argparse.Namespace,
                                        source_dir: str = "./assets/test/source"):
    """Process a single source-target schema pair"""
    
    
    
    # Load data and schemas
    source_table, source_data, source_schema = await load_source_data(source_csv_path, source_dir)
    target_table, target_schema = load_target_schema(target_path)
    
    
    logging.info(f"🎯 Matching {source_table} → {target_table}")
    
    # Load real ground truth
    real_gt_mapping = load_real_ground_truth(target_table)
    
    # Run all matchers
    predictions_by_approach = await run_all_matchers(
        source_schema, target_schema, args.seed, real_gt_mapping,
        source_table, target_table
    )

    return predictions_by_approach
