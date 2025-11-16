
from typing import Literal, Optional, Union

from .base import BaseApproach
from embedding_utils import embedding_column_mapping
from json_schema import ObjectSchema


class EmbeddingApproach(BaseApproach):
    model: str
    threshold: float
    method: Union[Literal["hungarian"], Literal["greedy"]]

    def __init__(self, model: str, threshold: float, method: Union[Literal["hungarian"], Literal["greedy"]]):
        self.model = model
        self.threshold = threshold
        self.method = method

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        return await embedding_column_mapping(source_schema, target_schema, model=self.model, method=self.method, threshold=self.threshold)
