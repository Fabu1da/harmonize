import os
import json
import logging
from collections import defaultdict
import pyarrow.parquet as pq

# ------------------ Logging Setup ------------------ #
logging.basicConfig(
    filename="gittables_extraction.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ------------------ Metadata Extractor ------------------ #
def extract_gittables_metadata(parquet_path):
    try:
        pf = pq.ParquetFile(parquet_path)
        meta = pf.metadata.metadata

        if b'gittables' not in meta:
            logging.warning(f"Missing 'gittables' metadata in: {parquet_path}")
            return None

        gittables_meta = json.loads(meta[b'gittables'])

        table_id = gittables_meta.get("table_id")
        columns = gittables_meta.get("dtypes", {}).keys()

        schema_sem = gittables_meta.get("schema_semantic_column_types", {})
        dbpedia_sem = gittables_meta.get("dbpedia_semantic_column_types", {})

        column_data = []
        for col in columns:
            column_data.append({
                "file": os.path.basename(parquet_path),
                "table_id": table_id,
                "column": col,
                "schema_type": schema_sem.get(col, {}).get("id"),
                "schema_label": schema_sem.get(col, {}).get("cleaned_label"),
                "dbpedia_type": dbpedia_sem.get(col, {}).get("id"),
                "dbpedia_label": dbpedia_sem.get(col, {}).get("cleaned_label"),
            })

        logging.info(f"Extracted metadata from {parquet_path}")
        return column_data

    except Exception as e:
        logging.error(f"Error reading {parquet_path}: {e}")
        return None

# ------------------ Folder Scanner ------------------ #
def scan_gittables_folder(root_path):
    result = defaultdict(list)
    total_files = 0
    processed_tables = 0

    logging.info(f"Starting scan at: {root_path}")

    for folder, _, files in os.walk(root_path):
        folder_name = os.path.basename(folder)
        logging.info(f"Scanning folder: {folder_name}")

        for file in files:
            if file.endswith(".parquet"):
                total_files += 1
                full_path = os.path.join(folder, file)
                metadata = extract_gittables_metadata(full_path)

                if metadata:
                    result[folder_name].extend(metadata)
                    processed_tables += 1

    logging.info(f"Scanned {total_files} files across {len(result)} folders.")
    logging.info(f"Successfully extracted from {processed_tables} tables.")
    return [{folder: data} for folder, data in result.items()]

# ------------------ Script Entrypoint ------------------ #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract GitTables metadata from parquet files")
    parser.add_argument("input_dir", help="./assets/6517052")
    parser.add_argument("--output", default="./assets/gittables_metadata.json", help="Output JSON file")
    args = parser.parse_args()

    output_data = scan_gittables_folder(args.input_dir)

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    logging.info(f"Metadata saved to {args.output}")
    print(f"✅ Metadata extraction complete. Output written to {args.output}")
