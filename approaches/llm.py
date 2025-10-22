from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema
from gpt_utils import gpt_column_mapping

class LLMApproach(BaseApproach):
    model: str

    def __init__(self, model: str):
        self.model = model

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        return await gpt_column_mapping(source_schema, target_schema, model=self.model, **kwargs)
