#!/usr/bin/env python
import argparse
import asyncio
from functools import partial
import json
from collections.abc import Callable
from typing import Any
import logging
import glob
from pathlib import Path
import os

import pandas as pd
from tqdm import tqdm

from gpt_utils import gpt_column_mapping
from json_schema import ObjectSchema
from schema_inference import infer_schema
from synthetic_data import apply_perturbations, score_mapping

# File paths
INPUT_FILE = "./files/source/partner_1.csv"
OUTPUT_FILE = "./files/standardized_data.csv"
TARGET_SCHEMA_FILE = "./files/target"


async def infer_rules(column_map, target_schema: ObjectSchema) -> dict[str, Callable[[dict], Any]]:
    # Ensure column_map contains only strings
    column_map = {
        target: source
        for target, source in column_map.items()
    }

    print("🔍 Processed column_map:", column_map)  # Debugging output

    def rule(data: dict, target_column: str) -> Any:
        """
        Transformation rule: Maps a source column to a target column and applies conversion.
        """
        source_column = column_map.get(target_column)
        if not source_column:
            return None  # No source column mapped

        value = data.get(source_column, None)  # Fetch source data
        if value is None:
            return None  # No value found

        try:
            # Convert value to target schema's expected type
            jsontype = target_schema.properties[target_column].type
            datatype = {
                "string": str,
                "number": float,
                "integer": int,
                "boolean": bool,
            }.get(jsontype, str)
            value = datatype(value)
        except ValueError as e:
            logging.error(f"❌ Value conversion error for column {target_column}: {e}")
            value = None  # Handle conversion errors gracefully

        return value

    # ✅ Return transformation rules dictionary
    return {
        column: partial(rule, target_column=column)
        for column in target_schema.properties.keys()
    }

# TODO: improve runtime complexity from O(R*C) to O(C)
def apply_rules(dataset: pd.DataFrame, rules: dict[str, Callable[[dict], Any]]) -> pd.DataFrame:
    columns = rules.keys()
    dataset2 = pd.DataFrame(columns=columns)
    for row in dataset.itertuples():
        index = row.Index
        data = row._asdict()
        for column, rule in rules.items():
            value = rule(data)
            dataset2.loc[index, column] = value
    return dataset2

async def main(args: argparse.ArgumentParser):
    await main1(args)
    await main2(args)

async def main1(args: argparse.ArgumentParser):
    dataset = pd.read_csv(args.input_file)

    source_schema = await infer_schema(dataset)
    print("Source Schema:", source_schema.model_dump_json())

    schema_files = sorted(glob.glob(os.path.join(args.target_schema_file, "*.json")))

    if not schema_files:
        print(f"No schema files found in {args.target_schema_file}")

    for schema_path in tqdm(schema_files):
        schema_name = Path(schema_path).stem
        print(f"\n📁 Processing schema: {schema_name}")

        with open(schema_path, 'r') as f:
            target_schema = ObjectSchema.model_validate_json(f.read())
            print("Target Schema:", target_schema.model_dump_json())

            predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=args.seed)
            print("Predicted:", predicted_mapping)

            rules = await infer_rules(predicted_mapping, target_schema)
            standardized_dataset = apply_rules(dataset, rules)
            standardized_dataset.to_csv(args.output_file, index=False)


async def main2(args: argparse.ArgumentParser):
    schema_files = sorted(glob.glob(os.path.join(args.target_schema_file, "*.json")))

    if not schema_files:
        print(f"No schema files found in {args.target_schema_file}")

    score_sum = 0
    weight_sum = 0

    for schema_path in tqdm(schema_files):
        schema_name = Path(schema_path).stem
        print(f"\n📁 Processing schema: {schema_name}")

        with open(schema_path, 'r') as f:
            target_schema = ObjectSchema.model_validate_json(f.read())
            print("Target Schema:", target_schema.model_dump_json())
            
            # TODO: generate synthetic expected dataset for target schema

            source_schema, expected_mapping = await apply_perturbations(target_schema, seed=args.seed)
            print("Source Schema:", source_schema.model_dump_json())
            print("Expected:", expected_mapping)

            predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=args.seed)
            print("Predicted:", predicted_mapping)
            
            for k, v in expected_mapping.items():
                if predicted_mapping.get(k) != v:
                    print(f"Expected: {k} -> {v}, Actual: {k} -> {predicted_mapping.get(k)}")

            score = score_mapping(predicted_mapping, expected_mapping)
            # weight = len(target_schema.properties.keys())
            # score_sum += score[0] * weight
            # weight_sum += weight
            print(score)

            rules = await infer_rules(predicted_mapping, target_schema)
            # standardized_dataset = apply_rules(dataset, rules)
            # standardized_dataset.to_csv(args.output_file, index=False)

    overall_score = score_sum / weight_sum
    print("Overall Score", overall_score)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize a dataset to a target schema.")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--input_file", help="Path to the input file", default=INPUT_FILE, type=str)
    parser.add_argument("--output_file", help="Path to save the standardized dataset", default=OUTPUT_FILE, type=str)
    parser.add_argument("--target_schema_file", help="Path to the target schema file", default=TARGET_SCHEMA_FILE, type=str)
    args = parser.parse_args()
    asyncio.run(main(args))
