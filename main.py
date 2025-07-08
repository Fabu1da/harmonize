#!/usr/bin/env python
import dotenv
dotenv.load_dotenv(override=True)
import sys
import argparse
import asyncio
from functools import partial
from collections.abc import Callable
import json
from typing import Any, Optional
import logging
import glob
from pathlib import Path
import os
from tabulate import tabulate

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from gpt_utils import gpt_column_mapping
from json_schema import ObjectSchema
from schema_inference import infer_schema
from synthetic_data import apply_perturbations, score_mapping
from embedding_utils import embedding_column_mapping
from clustering_matcher import clustering_matcher 

from pydantic import BaseModel, ValidationError, field_validator
from logging_config import setup_logging

class CLIArgs(BaseModel):
    source_table: str | None
    target_table: str | None
    seed: int | None
    weight_gpt: float
    weight_embed: float
    weight_cluster: float
    output_name: str | None

    @field_validator("weight_gpt", "weight_embed", "weight_cluster")
    def weights_must_be_0_to_1(cls, v, field):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{field.name} must be between 0.0 and 1.0")
        return v

    @field_validator("source_table", "target_table")
    def names_must_be_nonempty(cls, v):
        if not v.strip():
            raise ValueError("table names must be non-empty strings")
        return v



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
            logging.error(f" Value conversion error for column {target_column}: {e}")
            value = None  # Handle conversion errors gracefully

        return value

    #  Return transformation rules dictionary
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
    os.chdir(os.path.dirname(__file__))
    # await main_all_expected(args)
    await main_test(args)
    # await main_synthetic(args)



def show_mapping_with_examples(predicted_mapping: dict, source_data: pd.DataFrame, num_examples: int = 1):
    table = []

    for target, (src, conf) in predicted_mapping.items():
        if src and src in source_data.columns:
            examples = source_data[src].dropna().astype(str).unique()[:num_examples]
            example_val = ", ".join(examples) if examples.any() else "—"
        else:
            example_val = "—"
        
        table.append([target, src if src else "—", f"{conf:.2f}", example_val])

    print(tabulate(table, headers=["Target Column", "Predicted Source", "Confidence", "Example"], tablefmt="fancy_grid"))




def export_table_as_image(data, headers, filename):
    df = pd.DataFrame(data, columns=headers)

    fig, ax = plt.subplots(figsize=(len(headers) * 2, len(data) * 0.6 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    os.makedirs("output", exist_ok=True)
    filepath = os.path.join("output", filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close(fig)
    print(f"✅ Table saved to {filepath}")



async def main_test(args: argparse.Namespace):
    results = []
    detailed_matches = []

    # Iterate over all source CSV files
    for source_csv_path in sorted(glob.glob("./assets/source/*.csv")):
        source_table = Path(source_csv_path).stem
        source_data = pd.read_csv(source_csv_path)
        source_schema_path = f"./assets/source/{source_table}.json"

        # Load or infer source schema
        if os.path.exists(source_schema_path):
            with open(source_schema_path) as f:
                source_schema = ObjectSchema.model_validate_json(f.read())
        else:
            source_schema = await infer_schema(source_data)

        # Iterate over all target JSON schemas
        for target_path in tqdm(sorted(glob.glob("./assets/target/*.json", recursive=True))):
            target_table = Path(target_path).stem

            with open(target_path) as f:
                target_schema = ObjectSchema.model_validate_json(f.read())

            predicted_mapping, score, weight = await main_core(
                source_table,
                target_table,
                seed=args.seed,
                output_name=args.output_name
            )

            show_mapping_with_examples(predicted_mapping, source_data)

            embed_predicted = embedding_column_mapping(
                source_columns=list(source_schema.properties.keys()),
                target_columns=list(target_schema.properties.keys()),
                threshold=0
            )
            cluster_predicted = clustering_matcher(source_schema, target_schema)

            print(f"\n📊 Matcher Comparison for {source_table} → {target_table}")
            weights = {"gpt": 0.5, "embed": 0.3, "cluster": 0.2}
            comparison_table = []
            headers = [
                "Target",
                "GPT Match", "GPT Score",
                "Embed Match", "Embed Score",
                "Cluster Match", "Cluster Score",
                "Avg Score", "Weighted Score",
                "Potential Match", "Suggested Match"
            ]

            for col in target_schema.properties.keys():
                gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
                emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
                cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))

                gpt_score = float(gpt_score or 0.0)
                emb_score = float(emb_score or 0.0)
                cluster_score = float(cluster_score or 0.0)

                avg_score = round((gpt_score + emb_score + cluster_score) / 3, 2)
                weighted_score = round(
                    gpt_score * weights["gpt"] +
                    emb_score * weights["embed"] +
                    cluster_score * weights["cluster"], 2
                )
                high_potential = "✅" if weighted_score >= 0.5 else "—"

                suggested = "—"
                if avg_score >= 0.5:
                    preds = {
                        gpt_match: gpt_score,
                        emb_match: emb_score,
                        cluster_match: cluster_score
                    }
                    max_score = max(preds.values())
                    best = [k for k, v in preds.items() if v == max_score and k not in ("—", None)]
                    suggested = " / ".join(sorted(set(best))) if best else "—"

                comparison_table.append([
                    col,
                    gpt_match, f"{gpt_score:.2f}",
                    emb_match, f"{emb_score:.2f}",
                    cluster_match, f"{cluster_score:.2f}",
                    f"{avg_score:.2f}",
                    f"{weighted_score:.2f}",
                    high_potential,
                    suggested
                ])

                # Collect all suggested matches for this source/target pair
                match_entry = next(
                    (item for item in detailed_matches if item["source_table"] == source_table and item["target_table"] == target_table),
                    None
                )
                if suggested not in ("—", None):
                    mapping_obj = {
                        "source_column": suggested,
                        "target_column": col,
                        "similarity": round(weighted_score, 4)
                    }
                    if match_entry:
                        match_entry["mapping"].append(mapping_obj)
                    else:
                        detailed_matches.append(
                            {
                                "source_table": source_table,
                                "target_table": target_table,
                                "synthetic": False,
                                "generated_with": "hamonize/main.py",
                                "runtime": 0.0,
                                "mapping": [mapping_obj]
                            }
                        )

            print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))
            export_table_as_image(comparison_table, headers, f"Comparison_{source_table}_to_{target_table}.png")

            score_val = score[0] if score and isinstance(score, tuple) else None
            score_display = f"{score_val:.2f}" if score_val is not None else "—"
            weight_display = str(weight) if weight is not None else "—"

            results.append([
                source_table,
                target_table,
                score_display,
                weight_display
            ])

    print("\n📊 Harmonization Summary Table")
    summary_headers = ["Source Table", "Target Table", "Score", "Weight"]
    print(tabulate(results, headers=summary_headers, tablefmt="fancy_grid"))
    export_table_as_image(results, summary_headers, "Harmonization_Summary.png")

    with open("output/detailed_matches.json", "w") as f:
        json.dump(detailed_matches, f, indent=2)
    print("✅ Detailed matcher results saved to output/detailed_matches.json")

    



async def main_all_expected(args: argparse.Namespace):
    score_sum = 0
    weight_sum = 0

    for expected_path in tqdm(sorted(glob.glob("**/*.json", root_dir="./assets/expected", recursive=True))):
        print(flush=True)
        expected_name, _ = os.path.splitext(expected_path)

        with open(f"./assets/expected/{expected_name}.json") as f:
            expectation = json.load(f)

        source_table = expectation["source_table"]
        target_table = expectation["target_table"]
        expected_mappings = expectation["mappings"]
        expected_mapping = {
            mapping["target_column"]: mapping["source_column"]
            for mapping in expected_mappings
        }

        print(" ")
        print("[INFO]", expected_name, source_table, target_table)
        print("Expected Mapping:", expected_mapping)
        
        predicted_mapping, score, weight = await main_core(source_table, target_table, expected_mapping, seed=args.seed, output_name=args.output_name)
        score_sum += score[0] * weight
        weight_sum += weight

    overall_score = score_sum / weight_sum
    print("Overall Score", overall_score)


async def main_synthetic(args: argparse.ArgumentParser):
    score_sum = 0
    weight_sum = 0

    for target_path in tqdm(sorted(glob.glob("**/*.json", root_dir="./assets/target", recursive=True))):
        print(flush=True)
        target_table, _ = os.path.splitext(target_path)

        with open(f"./assets/target/{target_table}.json", 'r') as f:
            target_schema = ObjectSchema.model_validate_json(f.read())
            # print("Target Schema:", target_schema.model_dump_json())
        

        source_schema, expected_mapping =  await apply_perturbations(target_schema, seed=args.seed)
        # print("Source Schema:", source_schema.model_dump_json())
        # print("Expected Mapping:", expected_mapping)
        
        out_name = f"{target_table}__synthetic.json"
        with open(f"./assets/expected/{out_name}", "w") as f:
            json.dump({
                "source_table": target_table + "_synthetic",
                "target_table": target_table,
                "synthetic": True,
                "generated_with": f"main_synthetic('{target_table}')",
                "mappings": [
                    {"source_column": src, "target_column": tgt}
                    for tgt, src in expected_mapping.items()
                    if src is not None
            ]
        }, f, indent=2)
        

        predicted_mapping, score, weight = await main_core_inner(None, source_schema, target_schema, expected_mapping, seed=args.seed, output_name=args.output_name)
        score_sum += score[0] * weight
        weight_sum += weight

    overall_score = score_sum / weight_sum
    print("Overall Score", overall_score)


async def main_core(source_table: str, target_table: str, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    try:
        source_data = pd.read_csv(f"./assets/source/{source_table}.csv")
    except FileNotFoundError as e:
        logging.error(f"Source CSV not found: {e}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        logging.error(f"Failed parsing source CSV: {e}")
        sys.exit(1)
    source_schema_path = f"./assets/source/{source_table}.json"
    target_schema_path = f"./assets/target/{target_table}.json"

    if os.path.exists(source_schema_path):
        try:
            raw = Path(source_schema_path).read_text()
            source_schema = ObjectSchema.model_validate_json(raw)
        except (ValidationError, ValueError, json.JSONDecodeError) as e:
            logging.error(f"Invalid JSON schema at {source_schema_path}: {e}")
            sys.exit(1)
    else:
        source_schema = await infer_schema(source_data)
        print("Source Schema:", source_schema.model_dump_json())

    try:
        raw_t = Path(target_schema_path).read_text()
        target_schema = ObjectSchema.model_validate_json(raw_t)
    except FileNotFoundError:
        logging.error(f"Target schema file missing: {target_schema_path}")
        sys.exit(1)
    except (ValidationError, ValueError, json.JSONDecodeError) as e:
        logging.error(f"Invalid JSON schema at {target_schema_path}: {e}")
        sys.exit(1)
    
    return await main_core_inner(source_data, source_schema, target_schema, expected_mapping, seed=seed, output_name=output_name)


def compare_mappings(old_mapping, new_mapping):
    """
    Compares two dictionaries and identifies:
    - Unchanged mappings
    - Changed mappings
    - Newly added mappings
    - Removed mappings
    """
    unchanged = {}
    changed = {}
    added = {}
    removed = {}

    for key in old_mapping:
        if key in new_mapping:
            if old_mapping[key] == new_mapping[key]:
                unchanged[key] = old_mapping[key]
            else:
                changed[key] = (old_mapping[key], new_mapping[key])
        else:
            removed[key] = old_mapping[key]

    for key in new_mapping:
        if key not in old_mapping:
            added[key] = new_mapping[key]

    return {"unchanged": unchanged, "changed": changed, "added": added, "removed": removed}



async def main_core_inner(source_data: Optional[pd.DataFrame], source_schema: ObjectSchema, target_schema: ObjectSchema, expected_mapping: Optional[dict[str, Optional[str]]] = None, seed: Optional[int] = None, output_name: Optional[str] = None):
    predicted_mapping = await gpt_column_mapping(source_schema, target_schema, seed=seed)
    
    embed_predicted = embedding_column_mapping(
        source_columns=list(source_schema.properties.keys()),
        target_columns=list(target_schema.properties.keys()),
        threshold=0
    )

    cluster_predicted = clustering_matcher(source_schema, target_schema)
    
    if expected_mapping is None:
        conf_sum = sum(conf for _, conf in predicted_mapping.values())
        total = len(target_schema.properties)
        fallback_score = conf_sum / total if total > 0 else 0.0
        score = (fallback_score, {"note": "proxy score based on high-confidence matches"})
        weight = total
    else:
        comparison_table = []
        headers = [
            "Target", "Expected",
            "GPT Match", "GPT Score",
            "Embed Match", "Embed Score",
            "Cluster Match", "Cluster Score"
        ]

        for col in target_schema.properties.keys():
            expected = expected_mapping.get(col, "—")


            gpt_match, gpt_score = predicted_mapping.get(col, ("—", 0.0))
            emb_match, emb_score = embed_predicted.get(col, ("—", 0.0))
            cluster_match, cluster_score = cluster_predicted.get(col, ("—", 0.0))
            
            comparison_table.append([
                col, expected,
                gpt_match, f"{gpt_score:.2f}",
                emb_match, f"{emb_score:.2f}",
                cluster_match, f"{cluster_score:.2f}"
            ])

        print("\n📊 Combined Matcher Comparison (Including GitTables)")
        print(tabulate(comparison_table, headers=headers, tablefmt="fancy_grid"))

        score = score_mapping(predicted_mapping, expected_mapping)
        weight = len(target_schema.properties.keys())
        print("Score:", score)

    if source_data is not None and output_name:
        rules = await infer_rules(predicted_mapping, target_schema)
        predicted_data = apply_rules(source_data, rules)
        predicted_data.to_csv(output_name, index=False)

    return predicted_mapping, score, weight





if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Harmonize a dataset to a target schema.")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--source_table", help="Name of the source schema/data", type=str)
    parser.add_argument("--target_table", help="Name of the target schema/data", type=str)
    parser.add_argument("--output_name", help="Name of the output files", type=str)
    parser.add_argument("--weight_gpt", default=0, type=int)
    parser.add_argument("--weight_embed", default=0, type=int)
    parser.add_argument("--weight_cluster", default=0, type=int)
    

    
    args = parser.parse_args()

    try:
        cfg = CLIArgs(
            source_table   = args.source_table,
            target_table   = args.target_table,
            seed           = args.seed,
            weight_gpt     = args.weight_gpt,
            weight_embed   = args.weight_embed,
            weight_cluster = args.weight_cluster,
            output_name    = args.output_name,
        )
    except ValidationError as e:
        logging.error(f"Invalid arguments: {e}")
        parser.error(e.errors())
        sys.exit(1)
        
    
    # unpack validated values here
    source_table = cfg.source_table
    target_table = cfg.target_table

    try:
        asyncio.run(main(args))
    except Exception as e:
        import logging
        from metrics import PIPELINE_ERRORS

        PIPELINE_ERRORS.inc()
        logging.exception("Unhandled exception in pipeline")
        sys.exit(1)
