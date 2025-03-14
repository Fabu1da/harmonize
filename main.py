import argparse
import asyncio
import json
import pandas as pd
import logging
import random
import numpy as np
import openai
from functools import partial
from typing import Any, Callable
import logging

# Configure logging
logging.basicConfig(filename="harmonization.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
INPUT_FILE = "./files/vehiclesRawData.csv"
OUTPUT_FILE = "standardized_data.csv"
TARGET_SCHEMA_FILE = "./files/target/target_schema.json"

# 🔍 **Step 1: Infer Schema from Source Dataset**
async def infer_schema(dataset: pd.DataFrame):
    schema = {}

    for column in dataset.columns:
        non_null_values = dataset[column].dropna()  # Remove null values

        # Determine Python native datatype
        inferred_type = type(non_null_values.iloc[0]) if not non_null_values.empty else str

        # Convert NumPy types to standard Python types
        type_mapping = {
            np.int64: int,
            np.float64: float,
            np.bool_: bool,
            np.str_: str
        }
        inferred_type = type_mapping.get(inferred_type, inferred_type)

        schema[column] = {
            "type": inferred_type.__name__,
            "nullable": bool(dataset[column].isnull().sum() > 0),
            "example": random.choice(non_null_values.tolist()) if not non_null_values.empty else None,
        }
    
    return schema  

# 🔍 **Step 2: Parse Target Schema**
def parse_schema(schema_file):
    with open(schema_file, 'r') as f:
        schema = json.load(f)

    mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "datetime": "str",
    }

    if "properties" not in schema:
        raise ValueError("Invalid target schema: missing 'properties' key")

    schema_properties = schema["properties"]

    return {
        column: {
            "type": mapping.get(value.get("type", "string"), "str"),
            "nullable": False,
            "example": None
        }
        for column, value in schema_properties.items()
    }

# 🔍 **Step 3: Infer Mapping Rules with GPT**
async def gpt_column_mapping(source_schema: dict, target_schema: dict) -> dict[str, str]:
    print("🔄 Mapping columns...")

    system_message = (
        "You are an expert in schema matching and data integration. Your task is to match columns from a given source schema "
        "to the most semantically similar columns in a target schema. Use both lexical similarity (string similarity) and semantic understanding. "
        "Provide a JSON object where each target column is mapped to its most relevant source column.\n"
    )

    user_message = (
        f"### Source Schema:\n{json.dumps(source_schema, indent=2)}\n\n"
        f"### Target Schema:\n{json.dumps(target_schema, indent=2)}\n\n"
    )

    try:
        response =  openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            functions=[
                {
                    "name": "map_schema",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mappings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target_column": {"type": "string"},
                                        "source_column": {"type": "string"},
                                    },
                                    "required": ["target_column", "source_column"],
                                }
                            }
                        },
                        "required": ["mappings"]
                    }
                }
            ],
            function_call={"name": "map_schema"},
            temperature=0,
        )

        print("GPT Response:", response)  # Debugging output

        if response.choices and response.choices[0].message.function_call:
            content = response.choices[0].message.function_call.arguments
            mappings = json.loads(content).get("mappings", [])

            return {item["target_column"]: item["source_column"] for item in mappings}
        
    except Exception as e:
        logging.error(f"Error during GPT request: {e}")

    return {target_column: None for target_column in target_schema}

# 🔍 **Step 4: Apply Rules**
def apply_rules(dataset: pd.DataFrame, rules: dict[str, Callable[[dict], Any]], target_schema: dict) -> pd.DataFrame:
    columns = rules.keys()
    dataset2 = pd.DataFrame(columns=columns)

    type_mapping = {
        "string": str,
        "str": str,
        "integer": int,
        "int": int,
        "number": float,
        "float": float,
        "boolean": bool,
        "bool": bool,
        "datetime": str,
    }

    for row in dataset.itertuples():
        index = row.Index
        data = row._asdict()
        row_values = {}

        for column, rule in rules.items():
            value = rule(data)
            if value is None:
                dataset2.loc[index, column] = None
                row_values[column] = None
                continue

            try:
                expected_type = target_schema.get(column, {}).get("type", "str")

                if expected_type in ["int", "integer"] and isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
                    value = int(value)
                elif expected_type in ["float", "number"]:
                    value = float(value)
                elif expected_type in ["boolean", "bool"]:
                    value = bool(value)

                dataset2.loc[index, column] = value
                row_values[column] = value  # Store processed row values

            except (ValueError, TypeError) as e:
                logging.error(f"❌ Value conversion error for column {column} at row {index}: {e}")
                dataset2.loc[index, column] = None
                row_values[column] = None

        # Log row processing
        logging.info(f"✅ Row {index} added to standardized dataset: {row_values}")

    return dataset2

# 🚀 **Main Execution**
async def main(args):
    dataset = pd.read_csv(args.input_file)
    target_schema = parse_schema(args.target_schema_file)
    schema = await infer_schema(dataset)
    column_map = await gpt_column_mapping(schema, target_schema)
    
    print(column_map)

    rules = {
        col: partial(lambda data, target_col=col: data.get(column_map.get(target_col, None), None), target_col=col)
        for col in target_schema
    }
    
    standardized_dataset = apply_rules(dataset, rules, target_schema)
    standardized_dataset.to_csv(args.output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize a dataset to a target schema.")
    parser.add_argument("--input_file", default=INPUT_FILE, type=str)
    parser.add_argument("--output_file", default=OUTPUT_FILE, type=str)
    parser.add_argument("--target_schema_file", default=TARGET_SCHEMA_FILE, type=str)
    args = parser.parse_args()

    asyncio.run(main(args))
