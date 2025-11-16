from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema

class NullApproach(BaseApproach):
    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        assert target_schema.properties is not None

        target_columns = list(target_schema.properties.keys())

        return {
            target_column: (None, 0.0, None)
            for target_column in target_columns
        }
