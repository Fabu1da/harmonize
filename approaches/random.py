import random
from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema

class RandomApproach(BaseApproach):
    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        source_columns = list(source_schema.properties.keys())
        target_columns = list(target_schema.properties.keys())
        options = source_columns + [None]
        return {
            target_col: (
                random.choice(options),
                1.0 / len(options),
                "Randomly selected",
            )
            for target_col in target_columns
        }
