import random
from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema

class RandomApproach(BaseApproach):
    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        assert source_schema.properties is not None
        assert target_schema.properties is not None
        source_columns = list(source_schema.properties.keys())
        target_columns = list(target_schema.properties.keys())
        options = source_columns + [None]
        return {
            target_column: (
                random.choice(options),
                1.0 / len(options),
                "Randomly selected",
            )
            for target_column in target_columns
        }
