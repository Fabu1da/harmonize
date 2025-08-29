from typing import Any, Dict, Tuple

from ensemble_matchers import create_ensemble_matchers


def create_ensembles(gpt_predictions: Dict, embed_predictions: Dict, 
                    cluster_predictions: Dict, source_schema: Any, target_schema: Any) -> Tuple[Any, Any, Dict, Dict]:
    """Create and run ensemble matchers"""
    majority_ensemble, weighted_ensemble = create_ensemble_matchers(
        gpt_predictions,
        embed_predictions,
        cluster_predictions,
    )

    majority_predicted = majority_ensemble.predict(source_schema, target_schema)
    weighted_predicted = weighted_ensemble.predict(source_schema, target_schema)
    
    return majority_ensemble, weighted_ensemble, majority_predicted, weighted_predicted
