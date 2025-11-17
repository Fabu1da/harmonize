from .base import BaseApproach
from json_schema import ObjectSchema
from typing import Optional
from clustering_matcher import clustering_matcher

class ClusteringApproach(BaseApproach):
    def __init__(self, model: str):
        self.model = model

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        return await clustering_matcher(source_schema, target_schema, model=self.model)
