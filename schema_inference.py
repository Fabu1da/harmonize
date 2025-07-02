import random
import pandas as pd
import numpy as np

from json_schema import ObjectSchema, Schema

# 🔍 **Step 1: Infer Schema from Source Dataset**
async def infer_schema(dataset: pd.DataFrame) -> ObjectSchema:
    schema = ObjectSchema()

    for column in dataset.columns:
        non_null_values = dataset[column].dropna()  # Remove null values

        # Determine Python native datatype
        inferred_type = type(non_null_values.iloc[0]) if not non_null_values.empty else str

        # Convert NumPy types to standard Python types
        type_mapping = {
            np.int64: "integer",
            np.float64: "number",
            np.bool_: "boolean",
            np.str_: "string",
            str: "string",
        }
        inferred_type = type_mapping.get(inferred_type, inferred_type)
        required = bool(dataset[column].isnull().sum() > 0),

        schema.properties[column] = Schema(
            type=inferred_type,
            examples=random.sample(non_null_values.tolist(), min(3, len(non_null_values))),
        )

        if required:
            schema.required.append(column)

    return schema  