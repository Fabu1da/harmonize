#!/usr/bin/env python
"""
harmonize: Schema Matching and Data Harmonization System
Main entry point for the system providing schema matching capabilities.

IMPORTANT: main_test() uses CARTESIAN PRODUCT matching:
- Each source file is matched against EVERY target file
- With N sources and M targets, you get N×M combinations
- Example: 2 sources × 3 targets = 6 total matches
"""

from dotenv import load_dotenv
load_dotenv(override=True)

from core.utils.save_calibrator import save_calibrator

from config import APPROACH_NAMES, APPROACHES

import argparse
import asyncio

import json
import logging
import glob
import os
import sys
from typing import Union, Literal


import pandas as pd
from tqdm import tqdm

from json_schema import ObjectSchema
from schema_inference import infer_schema
from synthetic_data import apply_perturbations, score_mapping 

from confidence_calibration import ConfidenceCalibrator


def load_schema(type: Union[Literal["source"], Literal["target"]], name: str, use_cache: bool = False) -> ObjectSchema:
    json_path = f"./assets/{type}/{name}.json"
    csv_path = f"./assets/{type}/{name}.csv"
    csv_json_path = f"./assets/{type}/{name}.csv.json"

    if os.path.exists(json_path):
        with open(json_path) as f:
            schema = ObjectSchema.model_validate_json(f.read())
    elif use_cache and os.path.exists(csv_json_path):
        with open(csv_json_path) as f:
            schema = ObjectSchema.model_validate_json(f.read())
    else:
        data = pd.read_csv(csv_path)
        schema = infer_schema(data)

        with open(csv_json_path, "w") as f:
            f.write(schema.model_dump_json(indent=4))

    return schema


async def main(args: argparse.Namespace):
    """Main entry point for the harmonize system."""
    os.chdir(os.path.dirname(__file__))

    if args.generate_synthetic:
        await main_generate_synthetic(args)

    if args.predict:
        await main_predict(args)

    if args.train:
        main_train(args)

    if args.calibrate:
        main_calibrate(args)

    if args.evaluate:
        main_evaluate(args)


async def main_predict(args: argparse.Namespace):
    """
    Process all expected files and compute overall score.
    """
    print("Phase: Prediction")

    expected_paths = sorted(glob.glob("**/*.json", root_dir="./assets/expected", recursive=True))

    for approach, approach_name in tqdm(APPROACHES, desc="Approaches"):
        sys.stdout.flush()

        async def process(expected_path: str):
            expected_name, _ = os.path.splitext(expected_path)
            predicted_path = f"./assets/predicted/{approach_name}/{expected_name}.json"

            if not args.no_cache and os.path.exists(predicted_path):
                return

            with open(f"./assets/expected/{expected_name}.json") as f:
                expectation = json.load(f)

            source_table = expectation["source_table"]
            target_table = expectation["target_table"]

            source_schema = load_schema("source", source_table, use_cache=not args.no_cache)
            target_schema = load_schema("target", target_table, use_cache=not args.no_cache)

            assert source_schema.properties is not None
            assert target_schema.properties is not None

            # needed for COMA
            source_csv_path=f"./assets/source/{source_table}.csv"
            target_csv_path=f"./assets/target/{target_table}.csv"

            predictions = await approach.predict(
                source_schema=source_schema,
                target_schema=target_schema,
                seed=args.seed,
                # needed for ensemble
                expected_name=expected_name,
                # needed for COMA
                source_csv_path=source_csv_path if os.path.exists(source_csv_path) else None,
                target_csv_path=target_csv_path if os.path.exists(target_csv_path) else None,
                source_table=source_table,
                target_table=target_table,
            )
            if len(predictions) != len(target_schema.properties):
                logging.error(f"Number of predictions ({len(predictions)}) does not match number of target columns ({len(target_schema.properties)}) for {expected_name} using approach {approach_name}")

            os.makedirs(os.path.dirname(predicted_path), exist_ok=True)

            with open(predicted_path, "w") as f:
                json.dump(predictions, f, indent=4)

        tasks = [process(expected_path) for expected_path in expected_paths]

        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"[{approach_name}] Expected Paths", leave=False, miniters=1):
            sys.stdout.flush()
            await task


def main_train(args: argparse.Namespace):
    """
    Train and save confidence calibrators for all approaches.
    """
    print("Phase: Training")

    expected_paths = sorted(glob.glob("**/*.json", root_dir="./assets/expected", recursive=True))

    for approach_name in tqdm(APPROACH_NAMES, desc="Approaches"):
        sys.stdout.flush()
        confidence_calibrator = ConfidenceCalibrator()

        for expected_path in tqdm(expected_paths, desc="Expected Paths", leave=False):
            sys.stdout.flush()
            expected_name, _ = os.path.splitext(expected_path)

            with open(f"./assets/expected/{expected_name}.json") as f:
                expectation = json.load(f)

            expected_mappings = expectation["mappings"]
            expected_mapping = {
                mapping["target_column"]: mapping["source_column"]
                for mapping in expected_mappings
            }

            predicted_path = f"./assets/predicted/{approach_name}/{expected_name}.json"

            if not os.path.exists(predicted_path):
                continue

            with open(predicted_path) as f:
                uncalibrated_predictions = json.load(f)

            confidence_calibrator.collect_training_data(uncalibrated_predictions, expected_mapping)

        confidence_calibrator.fit()
        save_calibrator(confidence_calibrator, approach_name)


def main_calibrate(args: argparse.Namespace):
    print("Phase: Calibration")

    expected_paths = sorted(glob.glob("**/*.json", root_dir="./assets/expected", recursive=True))

    for approach_name in tqdm(APPROACH_NAMES, desc="Approaches"):
        sys.stdout.flush()

        confidence_calibrator_path = f"./models/confidence_calibrators/{approach_name}.pkl"
        
        if not os.path.exists(confidence_calibrator_path):
            logging.warning(f"No confidence calibrator found at {confidence_calibrator_path}, skipping calibration for {approach_name}")
            continue

        confidence_calibrator = ConfidenceCalibrator.load(confidence_calibrator_path)

        for expected_path in tqdm(expected_paths, desc="Expected Paths", leave=False):
            sys.stdout.flush()
            expected_name, _ = os.path.splitext(expected_path)

            predicted_path = f"./assets/predicted/{approach_name}/{expected_name}.json"

            if not os.path.exists(predicted_path):
                continue

            with open(predicted_path) as f:
                uncalibrated_predictions = json.load(f)

            calibrated_predictions = confidence_calibrator.calibrate_predictions(uncalibrated_predictions)
            predicted_calibrated_path = f"./assets/predicted/calibrated/{approach_name}/{expected_name}.json"

            os.makedirs(os.path.dirname(predicted_calibrated_path), exist_ok=True)

            with open(predicted_calibrated_path, "w") as f:
                json.dump(calibrated_predictions, f, indent=4)


def main_evaluate(args: argparse.Namespace):
    """
    Process all expected files and compute overall score.
    """
    print("Phase: Evaluation")

    expected_paths = sorted(glob.glob("**/*.json", root_dir="./assets/expected", recursive=True))
    approach_names = APPROACH_NAMES + ["calibrated/" + name for name in APPROACH_NAMES]

    scores = {}

    for approach_name in tqdm(approach_names, desc="Approaches"):
        sys.stdout.flush()
        unweighted_accuracy_sum = 0
        confidence_weighted_accuracy_sum = 0
        table_weight_sum = 0

        scores[approach_name] = {
            "per_table": {},
            "per_column": {},
        }

        for expected_path in tqdm(expected_paths, desc="Expected Paths", leave=False):
            sys.stdout.flush()
            expected_name, _ = os.path.splitext(expected_path)

            with open(f"./assets/expected/{expected_name}.json") as f:
                expectation = json.load(f)

            expected_mappings = expectation["mappings"]
            expected_mapping = {
                mapping["target_column"]: mapping["source_column"]
                for mapping in expected_mappings
            }

            predicted_path = f"./assets/predicted/{approach_name}/{expected_name}.json"

            if not os.path.exists(predicted_path):
                continue

            with open(predicted_path) as f:
                predictions = json.load(f)

            unweighted_accuracy, confidence_weighted_accuracy = score_mapping(predictions, expected_mapping)
            # Assumes that expected mapping contains all target columns, including to null/None
            table_weight = len(expected_mapping)
            unweighted_accuracy_sum += unweighted_accuracy * table_weight
            confidence_weighted_accuracy_sum += confidence_weighted_accuracy * table_weight
            table_weight_sum += table_weight

            for target_column in expected_mapping.keys():
                scores[approach_name]["per_column"][f"{expected_name}/{target_column}"] = {
                    "source_column": predictions[target_column][0],
                    "unweighted_accuracy": int(predictions[target_column][0] == expected_mapping[target_column]),
                }

            scores[approach_name]["per_table"][expected_name] = {
                "expected_path": expected_path,
                "unweighted_accuracy": unweighted_accuracy,
                "confidence_weighted_accuracy": confidence_weighted_accuracy,
                "table_weight": table_weight
            }

        # TODO: calculate per dataset scores

        unweighted_accuracy = unweighted_accuracy_sum / table_weight_sum
        confidence_weighted_accuracy = confidence_weighted_accuracy_sum / table_weight_sum
        scores[approach_name]["overall"] = {
            "unweighted_accuracy": unweighted_accuracy,
            "confidence_weighted_accuracy": confidence_weighted_accuracy
        }

    with open("./assets/predicted/scores.json", "w") as f:
        json.dump(scores, f, indent=4)


async def main_generate_synthetic(args: argparse.Namespace):
    """
    Generate and process synthetic test data.
    """

    async def process(target_path: str):
        target_name, _ = os.path.splitext(target_path)

        source_schema_path = f"./assets/source/synthetic/{target_name}.json"
        expected_path = f"./assets/expected/synthetic/{target_name}.json"

        if not args.no_cache and os.path.exists(source_schema_path) and os.path.exists(expected_path):
            return

        with open(f"./assets/target/{target_name}.json", 'r') as f:
            content = f.read()
            target_schema = ObjectSchema.model_validate_json(content)

        source_schema, expected_mapping = await apply_perturbations(target_schema, model="gpt-5-nano", seed=args.seed)

        os.makedirs(os.path.dirname(source_schema_path), exist_ok=True)
        with open(source_schema_path, "w") as f:
            f.write(source_schema.model_dump_json(indent=4))

        os.makedirs(os.path.dirname(expected_path), exist_ok=True)
        with open(expected_path, "w") as f:
            json.dump({
                "source_table": f"synthetic/{target_name}",
                "target_table": target_name,
                "synthetic": True,
                "mappings": [
                    {"source_column": src, "target_column": tgt}
                    for tgt, src in expected_mapping.items()
            ]}, f, indent=4)

    target_files = sorted(glob.glob("**/*.json", root_dir="./assets/target", recursive=True))

    tasks = [process(target_path) for target_path in target_files]

    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Target Schemas"):
        sys.stdout.flush()
        await task


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize a dataset to a target schema.")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--generate-synthetic", action="store_true", help="Use calibrated predictions where applicable")
    
    parser.add_argument("-p", "--predict", action="store_true", help="Run prediction on all expected files")
    parser.add_argument("-t", "--train", action="store_true", help="Train and save confidence calibrators for all approaches")
    parser.add_argument("-c", "--calibrate", action="store_true", help="Calibrate predictions using trained confidence calibrators")
    parser.add_argument("-e", "--evaluate", action="store_true", help="Run evaluation on all expected files")
    parser.add_argument("--ptce", action="store_true", help="Run all steps: predict, train, calibrate, evaluate")
    
    parser.add_argument("--no-cache", action="store_true", help="Disable schema inference caching")

    args = parser.parse_args()

    if args.ptce:
        args.predict = True
        args.train = True
        args.calibrate = True
        args.evaluate = True

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    asyncio.run(main(args))