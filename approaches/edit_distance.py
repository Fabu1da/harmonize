from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema
from Levenshtein import ratio

class EditDistanceApproach(BaseApproach):
    def __init__(self, threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        assert source_schema.properties is not None
        assert target_schema.properties is not None

        source_columns = list(source_schema.properties.keys())
        target_columns = list(target_schema.properties.keys())

        result: dict[str, tuple[Optional[str], float, Optional[str]]] = {
            target_column: (None, self.threshold, None) for target_column in target_columns
        }
        for target_column in target_columns:
            for source_column in source_columns:
                confidence = ratio(target_column, source_column)
                if confidence > result[target_column][1]:
                    result[target_column] = (source_column, confidence, None)
        return result
