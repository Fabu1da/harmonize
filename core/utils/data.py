from typing import Dict, List, Tuple, Optional, Any
import os
from pathlib import Path
import pandas as pd
from gpt_calibration import GPTConfidenceCalibrator
from json_schema import ObjectSchema
from schema_inference import infer_schema
import json
import logging


async def load_source_data(source_csv_path: str, source_dir: str = "./assets/test/source") -> Tuple[str, pd.DataFrame, Any]:
    """Load source data and schema"""
    source_table = Path(source_csv_path).stem
    source_path = f"{source_dir}/{source_table}.csv"
    source_data = pd.read_csv(source_path)
    source_schema_path = f"{source_dir}/{source_table}.json"

    if os.path.exists(source_schema_path):
        with open(source_schema_path) as f:
            source_schema = ObjectSchema.model_validate_json(f.read())
    else:
        source_schema = await infer_schema(source_data)
    
    return source_table, source_data, source_schema


def load_target_schema(target_path: str) -> Tuple[str, Any]:
    """Load target schema from JSON file"""
    target_table, _ = os.path.splitext(os.path.basename(target_path))
    with open(target_path) as f:
        target_schema = ObjectSchema.model_validate_json(f.read())
    return target_table, target_schema



def load_real_ground_truth(target_table: str, expected_dir: str = "./assets/test/expected") -> Optional[Dict]:
    """Load real ground truth mapping if available"""
    # Try multiple strategies to find the ground truth file
    possible_paths = [
        f"{expected_dir}/{target_table}_mapping.json",  # Direct name match
        f"{expected_dir}/{target_table}.json",          # Without _mapping suffix
    ]
    
    # Also try extracting base name patterns (e.g., musicians_joinable from musicians_joinable_target)
    if "_target" in target_table:
        base_name = target_table.replace("_target", "")
        possible_paths.extend([
            f"{expected_dir}/{base_name}_mapping.json",
            f"{expected_dir}/{base_name}.json",
        ])
    
    # Try using first two parts of the target name
    target_prefix = "_".join(target_table.split("_")[:2])
    if target_prefix != target_table:  # Only add if different
        possible_paths.extend([
            f"{expected_dir}/{target_prefix}_mapping.json",
            f"{expected_dir}/{target_prefix}.json",
        ])
    
    for real_gt_path in possible_paths:
        logging.info(f"🔍 Trying to load expected mapping from: {real_gt_path}")
        
        try:
            with open(real_gt_path) as f:
                real_gt_data = json.load(f)
            
            # Extract the actual mapping from the loaded data
            real_gt_mapping = None
            
            if "mappings" in real_gt_data:
                # Format: {"mappings": [{"target_column": "...", "source_column": "..."}]}
                real_gt_mapping = {
                    mapping["target_column"]: mapping["source_column"]
                    for mapping in real_gt_data["mappings"]
                }
            elif "matches" in real_gt_data:
                # Format: {"matches": [{"target_column": "...", "source_column": "..."}]}
                real_gt_mapping = {
                    mapping["target_column"]: mapping["source_column"]
                    for mapping in real_gt_data["matches"]
                }
            elif isinstance(real_gt_data, dict) and all(isinstance(v, str) for v in real_gt_data.values()):
                # Format: {"target_col": "source_col", ...}
                real_gt_mapping = real_gt_data
            else:
                print(f"⚠️ Unknown real ground truth format in {real_gt_path}")
                continue
                
            print(f"✅ Loaded real ground truth from {real_gt_path} with {len(real_gt_mapping)} mappings")
            logging.info(f"✅ Real ground truth mappings: {real_gt_mapping}")
            return real_gt_mapping
            
        except FileNotFoundError:
            logging.debug(f"⚠️ Real ground truth file not found: {real_gt_path}")
            continue
        except Exception as e:
            print(f"⚠️ Error loading real ground truth from {real_gt_path}: {e}")
            continue
    
    print(f"⚠️ No real ground truth file found for target: {target_table}")
    return None
    
    
    
async def load_gpt_calibrator():
    """Load or initialize the GPT calibrator"""
    calibrator_path = "./models/gpt_isotonic_calibrator.pkl"
    if os.path.exists(calibrator_path):
        global gpt_calibrator
        gpt_calibrator = GPTConfidenceCalibrator.load(calibrator_path)
        logging.info("✅ Loaded pre-trained GPT isotonic calibrator")
        return gpt_calibrator
    else:
        logging.warning("⚠️ No pre-trained GPT calibrator found, starting from scratch")
        return None