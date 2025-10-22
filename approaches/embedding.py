
from typing import Literal, Optional, Union

from approaches.base import BaseApproach
from embedding_utils import embedding_column_mapping
from json_schema import ObjectSchema


class EmbeddingApproach(BaseApproach):
    model: str
    threshold: float
    method: Union[Literal["hungarian", "greedy"]]

    def __init__(self, model: str, threshold: float, method: Union[Literal["hungarian", "greedy"]]):
        self.model = model
        self.threshold = threshold
        self.method = method

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema,  **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        # Extract column names from schemas
        target_columns = list(target_schema.properties.keys())

        # Call embedding function with column lists and model
        result = embedding_column_mapping(source_schema, target_schema, model=self.model, method=self.method, threshold=self.threshold)

        # Convert to expected format with enhanced reasoning
        formatted_result = {}
        for target_col in target_columns:
            if target_col in result:
                source_col, confidence, reasoning = result[target_col]
                formatted_result[target_col] = (source_col, confidence, reasoning)
            else:
                formatted_result[target_col] = (None, 0.0, "No embedding match found")

        return formatted_result