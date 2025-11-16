from typing import Optional

from .base import BaseApproach
from json_schema import ObjectSchema
from gpt_utils import gpt_column_mapping

class LLMApproach(BaseApproach):
    model: str
    reasoning_effort: str

    def __init__(self, model: str, reasoning_effort: str = "medium"):
        self.model = model
        self.reasoning_effort = reasoning_effort

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        return await gpt_column_mapping(source_schema, target_schema, model=self.model, reasoning_effort=self.reasoning_effort)
