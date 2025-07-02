import pandas as pd
import json
import logging
from fuzzywuzzy import process
from gpt_utils import gpt_column_mapping
from schema_inference import convert_data_types

logging.basicConfig(level=logging.INFO)


def check_data_type(dataset, column, target_dtype):
    print("""Check if the data type of a dataset column matches the target schema.""")
    actual_dtype = dataset[column].dtype
    expected_dtype = target_dtype.lower()

    dtype_map = {
        "int": "int64",
        "float": "float64",
        "datetime": "datetime64[ns]",
        "string": "object"
    }
    
    if actual_dtype == dtype_map.get(expected_dtype, "object"):
        return True
    else:
        logging.warning(f"Column '{column}' has mismatched data type: expected {expected_dtype}, found {actual_dtype}")
        return False


def check_uniqueness(dataset, column):
    """Check if a column contains unique values or duplicates."""
    if dataset[column].is_unique:
        logging.info(f"Column '{column}' has unique values.")
        return True
    else:
        duplicate_count = dataset[column].duplicated().sum()
        logging.warning(f"Column '{column}' has {duplicate_count} duplicate values.")
        return False


def align_schema(dataset, target_schema_file):
    with open(target_schema_file, 'r') as schema_file:
        target_schema = json.load(schema_file)

    dataset_columns_lower = {col.lower(): col for col in dataset.columns}
    print(f"Dataset with schema : {dataset_columns_lower}")
    mapped_columns = {}
    remaining_columns = set(dataset.columns)

    for target_col in target_schema:
        matched_col = dataset_columns_lower.get(target_col.lower()) or find_best_match(target_col, dataset.columns)
        if matched_col:
            # Check data type consistency before mapping
            if check_data_type(dataset, matched_col, target_schema[target_col]):
                mapped_columns[target_col] = matched_col
                remaining_columns.discard(matched_col)

                # Check uniqueness
                check_uniqueness(dataset, matched_col)
            else:
                logging.warning(f"Skipping column '{matched_col}' due to data type mismatch.")

    logging.info(f"Fuzzy matched columns: {mapped_columns}")

    for target_col in target_schema:
        logging.info(f"Processing target column for GPT: {target_col}")
        if target_col not in mapped_columns:
            gpt_results = gpt_column_mapping(target_col, tuple(remaining_columns))
            print(f"gpt_results::{gpt_results}")
            if gpt_results:
                best_match, score = gpt_results
                if best_match and score > 0.8:  # Ensure similarity threshold
                    mapped_columns[target_col] = best_match
                    remaining_columns.discard(best_match)

    logging.info(f"Final mapped columns after GPT matching: {mapped_columns}")
    unmatched_target_columns = [col for col in target_schema if col not in mapped_columns]
    unmatched_dataset_columns = list(remaining_columns)

    logging.info(f"Remaining unmatched columns in dataset: {unmatched_dataset_columns}")
    logging.info(f"Remaining unmatched columns in target schema: {unmatched_target_columns}")

    dataset.rename(columns=mapped_columns, inplace=True)

    for col in target_schema:
        if col not in dataset.columns:
            dataset[col] = pd.NA

    # Convert data types after ensuring schema alignment
    dataset = convert_data_types(dataset, target_schema)
    return mapped_columns, unmatched_dataset_columns, unmatched_target_columns


def find_best_match(target_col, dataset_columns):
    logging.info(f"Finding best match for column '{target_col}'...")
    match, score = process.extractOne(target_col, dataset_columns)
    logging.info(f"Best fuzzy match: {match} with score: {score}")
    return match if score > 80 else None
