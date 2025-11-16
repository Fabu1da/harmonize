from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema

class IdentityApproach(BaseApproach):
    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        assert source_schema.properties is not None
        assert target_schema.properties is not None

        source_columns = list(source_schema.properties.keys())
        target_columns = list(target_schema.properties.keys())

        return {
            target_column: (
                target_column if target_column in source_columns else None,
                1.0 if target_column in source_columns else 0.0,
                "Identity mapping" if target_column in source_columns else "No mapping found",
            )
            for target_column in target_columns
        }
