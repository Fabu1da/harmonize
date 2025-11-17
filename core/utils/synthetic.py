from typing import Any, Dict, Tuple

from synthetic_data import apply_perturbations
import logging

logging.basicConfig(level=logging.INFO)

async def generate_synthetic_ground_truth(target_schema: Any, seed: int) -> Tuple[Any, Dict]:
    """Generate synthetic ground truth for evaluation"""
    synthetic_source_schema, expected_mapping = await apply_perturbations(target_schema, seed=seed)
    logging.info(f"📋 Generated {len(expected_mapping)} synthetic mappings as ground truth")
    return synthetic_source_schema, expected_mapping