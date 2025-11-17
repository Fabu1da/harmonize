from typing import Optional

from json_schema import ObjectSchema

class BaseApproach:
    """returns dict from target column name to (source column name, confidence, reasoning)"""
    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        raise NotImplementedError("Subclasses should implement this method.")
