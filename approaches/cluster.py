from .base import BaseApproach
from json_schema import ObjectSchema
from typing import Optional
from clustering_matcher import clustering_matcher

class ClusteringApproach(BaseApproach):
    def __init__(self, model: str):
        self.model = model

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        try:
            # Filter out parameters that clustering_matcher doesn't accept
            valid_kwargs = {k: v for k, v in kwargs.items() if k in ['n_clusters', 'return_cluster_info']}
            
            # Always set return_cluster_info=True to get the second return value
            valid_kwargs['return_cluster_info'] = True
            
            # Note: clustering_matcher currently doesn't support model parameter
            # This is a known limitation that needs to be fixed in clustering_matcher.py
            result, cluster_info = clustering_matcher(source_schema, target_schema, n_clusters=5, model=self.model, return_cluster_info=True)
            return result
        except Exception as e:
            # Return empty predictions if clustering fails
            target_columns = list(target_schema.properties.keys())
            return {col: (None, 0.0, f"Clustering failed: {str(e)}") for col in target_columns}
