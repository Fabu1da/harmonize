import json
from typing import Union, Dict
def load_ground_truth_from_matches(gt_data: Union[Dict, str]) -> Dict[str, str]:
    """
    Convert your ground truth format to a simple target->source mapping.
    
    """
    if isinstance(gt_data, str):
        # Load from file path
        with open(gt_data, 'r') as f:
            gt_data = json.load(f)
    
    if 'matches' not in gt_data:
        raise ValueError("Ground truth data must contain 'matches' key")
    
    # Convert to simple mapping
    mapping = {}
    for match in gt_data['matches']:
        target_col = match['target_column']
        source_col = match['source_column']
        mapping[target_col] = source_col
    
    return mapping  