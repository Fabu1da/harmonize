
#!/usr/bin/env python3
import os
import sys
# Ensure project root is on PYTHONPATH so local modules can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, PROJECT_ROOT)
import glob
import json
import time
import asyncio
import csv

from schema_matching import match_schema        # embedding-based matcher
from clustering_matcher import clustering_matcher
from synthetic_data import score_mapping        # score function for your model
from main import main_core_inner                # core mapping function without CSV I/O
from json_schema import ObjectSchema            # schema loader

# Directory paths (absolute, based on project root)
SRC_DIR = os.path.join(PROJECT_ROOT, "../assets/source")
TRG_DIR = os.path.join(PROJECT_ROOT, "../assets/target")
EXP_DIR = os.path.join(PROJECT_ROOT, "../assets/expected")
COMA_OUTPUT = os.path.join(PROJECT_ROOT, "../assets/output/matches.json")  # COMA++ JSON dump file
COMA_THRESHOLD = 0.6

async def run_python_models(source_table: str, target_table: str, expected: dict):
    """
    Runs GPT, embed, and cluster matchers using CSV source and JSON target schemas.
    Skips if source CSV or target JSON is missing.
    Returns: dict with runtimes and precision/recall/F1 for each variant.
    """
    results = {}

    # Paths for source CSV and target JSON
    src_csv_path = os.path.join(SRC_DIR, f"{source_table}.csv")
    trg_json_path = os.path.join(TRG_DIR, f"{target_table}.json")
    if not os.path.exists(src_csv_path) or not os.path.exists(trg_json_path):
        return results  # skip missing

    # 1. GPT-based mapping (async) uses main_core to read CSV & JSON
    t0 = time.time()
    predicted_gpt, *_ = await main_core(source_table, target_table, expected_mapping=expected)
    dt = time.time() - t0
    prec_gpt, rec_gpt, f1_gpt = score_mapping(predicted_gpt, expected)
    results['gpt'] = {'time': dt, 'precision': prec_gpt, 'recall': rec_gpt, 'f1': f1_gpt}

    # 2. Embedding FAISS matcher: use CSV headers as source schema
    t0 = time.time()
    import csv as _csv
    with open(src_csv_path, newline='') as f:
        reader = _csv.reader(f)
        headers = next(reader)
    with open(trg_json_path) as f:
        trg_schema = json.load(f)
    embed_map = match_schema(
        source_schema={'properties': {col: {} for col in headers}},
        target_schema=trg_schema,
        threshold=0.0
    )
    dt = time.time() - t0
    prec_emb, rec_emb, f1_emb = score_mapping(embed_map, expected)
    results['embed'] = {'time': dt, 'precision': prec_emb, 'recall': rec_emb, 'f1': f1_emb}

    # 3. Clustering matcher: same source headers + pydantic target schema
    t0 = time.time()
    from json_schema import ObjectSchema
    target_obj_schema = ObjectSchema.model_validate_json(json.dumps(trg_schema))
    # Convert headers into minimal ObjectSchema for source
    source_obj_schema = ObjectSchema(properties={col: {} for col in headers})
    cluster_map = clustering_matcher(source_obj_schema, target_obj_schema)
    dt = time.time() - t0
    prec_clu, rec_clu, f1_clu = score_mapping(cluster_map, expected)
    results['cluster'] = {'time': dt, 'precision': prec_clu, 'recall': rec_clu, 'f1': f1_clu}

    return results

    # Load schemas
    with open(src_schema_path) as f:
        src_schema = ObjectSchema.model_validate_json(f.read())
    with open(trg_schema_path) as f:
        trg_schema = ObjectSchema.model_validate_json(f.read())

    # 1. GPT-based mapping (async)
    t0 = time.time()
    predicted_gpt, *_ = await main_core_inner(None, src_schema, trg_schema, expected_mapping=expected)
    dt = time.time() - t0
    prec_gpt, rec_gpt, f1_gpt = score_mapping(predicted_gpt, expected)
    results['gpt'] = {'time': dt, 'precision': prec_gpt, 'recall': rec_gpt, 'f1': f1_gpt}

    # 2. Embedding FAISS matcher
    t0 = time.time()
    embed_map = match_schema(
        source_schema=src_schema.model_dump(),
        target_schema=trg_schema.model_dump(),
        threshold=0.0
    )
    dt = time.time() - t0
    prec_emb, rec_emb, f1_emb = score_mapping(embed_map, expected)
    results['embed'] = {'time': dt, 'precision': prec_emb, 'recall': rec_emb, 'f1': f1_emb}

    # 3. Clustering matcher
    t0 = time.time()
    cluster_map = clustering_matcher(src_schema, trg_schema)
    dt = time.time() - t0
    prec_clu, rec_clu, f1_clu = score_mapping(cluster_map, expected)
    results['cluster'] = {'time': dt, 'precision': prec_clu, 'recall': rec_clu, 'f1': f1_clu}

    return results


def run_coma_from_json(source_table: str, target_table: str, expected: dict):
    """
    Reads COMA++ matches from JSON dump and evaluates against expected mapping.
    Returns: dict with precision/recall/F1 (no runtime here).
    """
    if not os.path.exists(COMA_OUTPUT):
        return {'precision': None, 'recall': None, 'f1': None}

    data = json.load(open(COMA_OUTPUT))
    coma_map = {}
    for entry in data:
        src_file = os.path.splitext(entry['src_file'])[0]
        trg_file = os.path.splitext(entry['trg_file'])[0]
        if src_file == source_table and trg_file == target_table and entry.get('similarity',0) >= COMA_THRESHOLD:
            coma_map[entry['source']] = entry['target']
    if coma_map:
        prec, rec, f1 = score_mapping(coma_map, expected)
    else:
        prec = rec = f1 = 0.0
    return {'precision': prec, 'recall': rec, 'f1': f1}


def load_expected_mapping(path: str) -> dict:
    with open(path) as f:
        js = json.load(f)
    return {m['source_column']: m['target_column'] for m in js['mappings']}


def main():
    records = []
    for exp_file in glob.glob(os.path.join(EXP_DIR, '*.json')):
        js = json.load(open(exp_file))
        # Use only target_table to locate files, ignore synthetic source_table naming
        target_table = js['target_table']
        source_table = target_table
        expected = load_expected_mapping(exp_file)
        target_table = js['target_table']
        expected = load_expected_mapping(exp_file)

        # Run matchers
        py_results = asyncio.run(run_python_models(source_table, target_table, expected))
        if not py_results:
            continue
        coma_results = run_coma_from_json(source_table, target_table, expected)

        rec = {
            'source': source_table,
            'target': target_table,
            'coma_precision': coma_results['precision'],
            'coma_recall': coma_results['recall'],
            'coma_f1': coma_results['f1']
        }
        for key, vals in py_results.items():
            rec[f'{key}_time'] = vals['time']
            rec[f'{key}_precision'] = vals['precision']
            rec[f'{key}_recall'] = vals['recall']
            rec[f'{key}_f1'] = vals['f1']

        records.append(rec)

    # Write CSV
    if records:
        fieldnames = list(records[0].keys())
        with open('benchmark_results.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        print("Benchmark results written to benchmark_results.csv")
    else:
        print("No benchmark records to write.")

if __name__ == '__main__':
    main()

