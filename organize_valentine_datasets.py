#!/usr/bin/env python3
"""
Valentine Dataset Organization Script
Copies Valentine benchmark datasets and organizes them into train/test splits
with proper ground truth placement.

This script:
1. Scans Valentine datasets directory for all schema matching datasets
2. Copies source/target files to appropriate train/test directories
3. Places ground truth mappings in the expected directory
4. Creates a proper academic train/test split for thesis evaluation
"""

import json
import logging
import os
from pathlib import Path
from typing import Generator, List, Dict

from tqdm import tqdm


class ValentineDatasetOrganizer:
    """Organizes Valentine benchmark datasets for harmonize pipeline"""
    
    def __init__(self, valentine_base_dir: str):
        self.valentine_dir = Path(valentine_base_dir)
        self.assets_dir = Path("./assets")
        
        self.suffix = "valentine"
        
        self.source_dir = self.assets_dir / "source" / self.suffix
        self.target_dir = self.assets_dir / "target" / self.suffix
        self.expected_dir = self.assets_dir / "expected" / self.suffix

        # Ensure all directories exist
        directories = [self.source_dir, self.target_dir, self.expected_dir]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def discover_datasets(self) -> Generator[Dict]:
        # Walk through all Valentine dataset directories
        for root, dirs, files in os.walk(self.valentine_dir):
            root_path = Path(root)
            name = root_path.relative_to(self.valentine_dir)
            basename = root_path.name
            
            # Look for datasets (directories with source/target/mapping files)
            mapping_json = root_path / f"{basename}_mapping.json"
            source_csv = root_path / f"{basename}_source.csv"
            source_json = root_path / f"{basename}_source.json"
            target_csv = root_path / f"{basename}_target.csv"
            target_json = root_path / f"{basename}_target.json"

            # If we have all components, add to datasets
            if mapping_json.exists() and (source_csv.exists() or source_json.exists()) and (target_csv.exists() or target_json.exists()):
                dataset = {
                    "name": name,
                    "mapping_json": mapping_json,
                    "source_csv": source_csv if source_csv.exists() else None,
                    "source_json": source_json if source_json.exists() else None,
                    "target_csv": target_csv if target_csv.exists() else None,
                    "target_json": target_json if target_json.exists() else None,
                }
                yield dataset

    def copy_dataset(self, dataset: Dict):
        """Copy a single dataset to the appropriate directory
        """
        
        mapping_json_dest = self.expected_dir / f"{dataset['name']}.json"
        source_csv_dest = self.source_dir / f"{dataset['name']}.csv"
        source_json_dest = self.source_dir / f"{dataset['name']}.json"
        target_csv_dest = self.target_dir / f"{dataset['name']}.csv"
        target_json_dest = self.target_dir / f"{dataset['name']}.json"

        self.import_mapping(dataset['mapping_json'], mapping_json_dest, self.suffix + "/" + str(dataset['name']))
        if dataset['source_csv']:
            self.import_csv(dataset['source_csv'], source_csv_dest)
        if dataset['source_json']:
            self.import_schema(dataset['source_json'], source_json_dest)
        if dataset['target_csv']:
            self.import_csv(dataset['target_csv'], target_csv_dest)
        if dataset['target_json']:
            self.import_schema(dataset['target_json'], target_json_dest)

    def import_mapping(self, inpath: str, outpath: str, name: str):
        with open(inpath, "r") as f:
            external = json.load(f)

        expected = {
            "mappings": [],
        }
        
        for match in external["matches"]:
            expected["source_table"] = name
            expected["target_table"] = name
            expected["mappings"].append({
                "source_column": match["source_column"],
                "target_column": match["target_column"],
            })

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        with open(outpath, "w") as f:
            f.write(json.dumps(expected, indent=4))


    def import_csv(self, inpath: str, outpath: str):
        with open(inpath, "r") as f:
            content = f.read()
        lines = content.splitlines()
        if len(lines) > 1 and lines[0] == lines[1]:
            lines = lines[1:]

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        with open(outpath, "w") as f:
            f.write("\n".join(lines) + "\n")


    def import_schema(self, inpath: str, outpath: str):
        with open(inpath, "r") as f:
            external = json.load(f)

        schema = {
            "type": "object",
            "properties": {},
        }
        
        for key, value in external.items():
            raw_type = value["type"].strip()

            match raw_type:
                case "text":
                    type = "string"
                case "int" | "integer" | "integer unsigned" | "bigint" | "bigint unsigned" | "smallint" | "smallint unsigned":
                    type = "integer"
                case "real":
                    type = "number"
                case _ if raw_type.startswith("varchar("):
                    type = "string"
                case _ if raw_type.startswith("numeric("):
                    type = "number"
                case _:
                    logging.warning(f"Unknown type {raw_type}, defaulting to string, path: {path}")
                    type = "string"

            schema["properties"][key] = {
                "type": type,
            }

        os.makedirs(os.path.dirname(outpath), exist_ok=True)

        with open(outpath, "w") as f:
            f.write(json.dumps(schema, indent=4))

    def organize_datasets(self):
        # Step 2: Discover all datasets
        datasets = list(self.discover_datasets())

        for dataset in tqdm(datasets):
            self.copy_dataset(dataset)

def main():
    # Configuration
    valentine_datasets_dir = "/Users/fabu1da/Downloads/Valentine-datasets"

    # Create organizer and run
    organizer = ValentineDatasetOrganizer(valentine_datasets_dir)
    organizer.organize_datasets()


if __name__ == "__main__":
    main()