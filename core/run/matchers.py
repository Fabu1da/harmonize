from typing import Any, Dict, Optional, Tuple
import clustering_matcher
from embedding_utils import embedding_column_mapping
from gpt_utils import gpt_column_mapping


async def run_all_matchers(source_schema: Any, target_schema: Any, seed: int, 
                          real_gt_mapping: Optional[Dict], gpt_calibrator: Any) -> Tuple[Dict, Dict, Dict]:
    """Run all matching algorithms and return their predictions"""
    # Run GPT matcher
    raw_gpt_predictions = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    # Collect training data if we have ground truth
    if real_gt_mapping and gpt_calibrator:
        gpt_calibrator.collect_training_data(raw_gpt_predictions, real_gt_mapping)
    
    # Apply calibration if calibrator is fitted
    if gpt_calibrator and gpt_calibrator.is_fitted:
        predicted_mapping = gpt_calibrator.calibrate_predictions(raw_gpt_predictions)
        print("🎯 Applied isotonic calibration to GPT-4 confidences")
    else:
        predicted_mapping = raw_gpt_predictions
        print("⚠️ Using raw GPT-4 confidences (calibrator not fitted)")

    # Run other matchers
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )
    
    cluster_predicted, cluster_info = clustering_matcher.clustering_matcher(source_schema, target_schema, return_cluster_info=True)
    
    # Note: cluster_info can be used here if needed for logging or stats collection
    
    return predicted_mapping, embed_predicted, cluster_predicted
