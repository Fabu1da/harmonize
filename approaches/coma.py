import logging
import random
import tempfile
import os
import pandas as pd
from typing import Optional
from uuid import uuid4

from .base import BaseApproach
import asyncio
from json_schema import ObjectSchema
from run_coma_example import coma_matching

semaphore = asyncio.Semaphore(2)
class ComaApproach(BaseApproach):
    def __init__(self, use_instances: bool, **kwargs):
        super().__init__(**kwargs)
        self.use_instances = use_instances

    def _schema_to_csv_path(self, schema: ObjectSchema, n: int = 10) -> str:
        """Convert an ObjectSchema to a temporary CSV file for COMA"""
        assert schema.properties is not None
        rows = []

        for _ in range(n):
            row = {}
            for prop_name, prop_info in schema.properties.items():
                if prop_info.examples:
                    row[prop_name] = random.choice(prop_info.examples)
                    continue
                match prop_info.type:
                    case 'string':
                        row[prop_name] = f"sample_{prop_name}"
                    case 'integer':
                        row[prop_name] = random.randint(1, 100)
                    case 'number':
                        row[prop_name] = random.uniform(1, 100)
                    case 'boolean':
                        row[prop_name] = random.choice([True, False])
                    case _:
                        logging.error(f"Unsupported type {prop_info.type} for property {prop_name}, setting to None")
                        row[prop_name] = None
            rows.append(row)

        # Create DataFrame with sample row
        df = pd.DataFrame(rows)

        # Create temporary CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        df.to_csv(temp_file.name, index=False)
        temp_file.close()

        return temp_file.name

    async def predict(self, source_schema: ObjectSchema, target_schema: ObjectSchema, **kwargs) -> dict[str, tuple[Optional[str], float, Optional[str]]]:
        temp_files = []

        try:
            # Extract schema names - check multiple possible attributes
            source_name = uuid4().hex
            target_name = uuid4().hex

            source_csv_path = kwargs.get("source_csv_path")
            target_csv_path = kwargs.get("target_csv_path")

            if source_csv_path is None or not self.use_instances:
                source_csv_path = self._schema_to_csv_path(source_schema)
                temp_files.append(source_csv_path)

            if target_csv_path is None or not self.use_instances:
                target_csv_path = self._schema_to_csv_path(target_schema)
                temp_files.append(target_csv_path)

            # Run COMA matching with file paths
            result = coma_matching(source_csv_path, target_csv_path, source_name, target_name, use_instances=self.use_instances, max_rows=500)

            # async with semaphore:
                # result = await asyncio.to_thread(
                    # coma_matching,
                    # source_csv_path,
                    # target_csv_path,
                    # source_name,
                    # target_name,
                    # self.use_instances,
                # )
            return result
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"   ⚠️ Failed to clean up {temp_file}: {e}")
