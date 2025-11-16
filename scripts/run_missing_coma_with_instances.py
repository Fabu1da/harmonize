#!/usr/bin/env python3
"""
Run COMA (with instances) only for expected files that do not yet have a prediction.
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Ensure repo root is on sys.path so imports like `from main import ...` work when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))



# Import helpers from the repo
from main import load_schema  # safe because main.py protects running via __name__ == "__main__"
from approaches.coma import ComaApproach

# Configuration
APPROACH_NAME = "COMA (with instances)"
PRED_ROOT = Path("assets/predicted") / APPROACH_NAME
EXPECTED_ROOT = Path("assets/expected")
INPROGRESS_AGE_MINUTES = 60  # treat .inprogress older than this as stale (optional)

MAX_CONCURRENT_COMA = 1
coma_semaphore = asyncio.Semaphore(MAX_CONCURRENT_COMA)


async def process_expected(expected_rel_path: str):
    async with coma_semaphore:
        expected_name, _ = os.path.splitext(expected_rel_path)
        predicted_path = PRED_ROOT / f"{expected_name}.json"
        inprogress_path = predicted_path.with_suffix(predicted_path.suffix + ".inprogress")

        # Skip if already predicted
        if predicted_path.exists():
            print("SKIP (exists):", predicted_path)
            return

        # If another worker left an inprogress file, consider its age
        if inprogress_path.exists():
            mtime = datetime.fromtimestamp(inprogress_path.stat().st_mtime)
            if datetime.now() - mtime < timedelta(minutes=INPROGRESS_AGE_MINUTES):
                print("SKIP (inprogress):", predicted_path)
                return
            else:
                print("STALE inprogress found, removing:", inprogress_path)
                try:
                    inprogress_path.unlink()
                except Exception:
                    pass

        # create parent dir
        os.makedirs(predicted_path.parent, exist_ok=True)

        # create inprogress marker
        inprogress_path.parent.mkdir(parents=True, exist_ok=True)
        inprogress_path.write_text(str(datetime.now()))

        try:
            # load expectation
            with open(EXPECTED_ROOT / expected_rel_path, "r") as f:
                expectation = json.load(f)

            source_table = expectation["source_table"]
            target_table = expectation["target_table"]

            # load schemas (use cache unless you want fresh)
            source_schema = load_schema("source", source_table, use_cache=True)
            target_schema = load_schema("target", target_table, use_cache=True)

            # optional csv paths if available
            source_csv_path = Path(f"./assets/source/{source_table}.csv")
            target_csv_path = Path(f"./assets/target/{target_table}.csv")
            source_csv = str(source_csv_path) if source_csv_path.exists() else None
            target_csv = str(target_csv_path) if target_csv_path.exists() else None

            # run COMA approach
            approach = ComaApproach(use_instances=True)
            predictions = await approach.predict(
                source_schema=source_schema,
                target_schema=target_schema,
                source_csv_path=source_csv,
                target_csv_path=target_csv,
                expected_name=expected_name
            )

            # atomic write: write to tmp then replace
            tmp_path = predicted_path.with_suffix(predicted_path.suffix + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(predictions, f, indent=4)
            os.replace(tmp_path, predicted_path)
            print("WROTE:", predicted_path)
        except Exception as e:
            print("ERROR processing", expected_rel_path, "->", repr(e))
        finally:
            # cleanup marker
            try:
                if inprogress_path.exists():
                    inprogress_path.unlink()
            except Exception:
                pass


async def main():
    expected_paths = sorted([str(p.relative_to(EXPECTED_ROOT)) for p in EXPECTED_ROOT.rglob("*.json")])

    # Filter only those missing predicted files
    missing = []
    for ep in expected_paths:
        expected_name, _ = os.path.splitext(ep)
        predicted_path = PRED_ROOT / f"{expected_name}.json"
        if not predicted_path.exists():
            missing.append(ep)

    print("Total expected:", len(expected_paths), "Missing predictions to run:", len(missing))

    # Run sequentially to avoid COMA resource contention.
    DRY = '--dry-run' in sys.argv
    if DRY:
        print('Dry-run: would process', len(missing), 'missing files (not executing).')
        for ep in missing:
            print(ep)
        return

    for ep in missing:
        await process_expected(ep)

if __name__ == "__main__":
    asyncio.run(main())