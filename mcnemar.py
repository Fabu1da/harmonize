from config import APPROACHES
from scipy.stats import chi2
from tabulate import tabulate
import json
from typing import Union, List, Dict

import os, glob, json
import re


from scipy.stats import chi2
from typing import Union, List, Dict, Tuple, Any

import os, glob, json
import re


def natural_key(path: str):
    # natural sort so "..._2.json" comes before "..._10.json"
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(path))]


def load_expected_ground_truth(expected_dir: str):
    """
    Finds JSON files in expected_dir.
    Returns either a single file path (string) or a list of file paths (strings).
    Both are accepted by format_ground_truth_for_mcnemar.
    """
    files = sorted(glob.glob(os.path.join(expected_dir, "*.json")), key=natural_key)
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {expected_dir}")
    return files[0] if len(files) == 1 else files


def load_ground_truth_from_expected_folder(expected_dir: str) -> list:
    """
    (Optional helper) Eagerly loads and converts each file to target->source dict.
    Not required by mcnemar_analysis, but kept for convenience/debug.
    """
    def natural_sort_key(path: str):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', os.path.basename(path))]

    def convert_mapping_file(filepath: str) -> dict:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Accept both {"matches":[...]} or plain dict mapping
        if isinstance(data, dict) and "matches" in data:
            mapping = {}
            for match in data["matches"]:
                t = match.get("target_column")
                s = match.get("source_column")
                if t and s:
                    mapping[t] = s
            return mapping
        elif isinstance(data, dict):
            # assume already target->source
            return {k: v for k, v in data.items() if v is not None}
        else:
            raise ValueError(f"Unsupported ground truth file format in {filepath}")

    files = sorted(glob.glob(os.path.join(expected_dir, "*.json")), key=natural_sort_key)
    if not files:
        raise FileNotFoundError(f"No JSON files found in: {expected_dir}")

    gt_list = []
    for filepath in files:
        try:
            mapping = convert_mapping_file(filepath)
            gt_list.append(mapping)
            print(f"✓ Loaded: {os.path.basename(filepath)} ({len(mapping)} mappings)")
        except Exception as e:
            print(f"✗ Error loading {os.path.basename(filepath)}: {e}")
    return gt_list


# ---------- NEW: normalize ground truth in any supported format ----------
def format_ground_truth_for_mcnemar(
    ground_truth_data: Union[
        Dict[str, str],
        List[Union[Dict[str, str], str]],
        str
    ]
) -> List[Dict[str, str]]:
    """
    Normalize ground truth into a list[dict[target_col -> source_col]].
    """
    def _convert_one(item: Any) -> Dict[str, str]:
        # If item is a file path, load it
        if isinstance(item, str):
            if not os.path.exists(item):
                raise FileNotFoundError(f"Ground truth file not found: {item}")
            with open(item, "r") as f:
                item = json.load(f)

        # If item is already a dict, convert if needed
        if isinstance(item, dict):
            if "matches" in item and isinstance(item["matches"], list):
                mapping = {}
                for m in item["matches"]:
                    t = m.get("target_column")
                    s = m.get("source_column")
                    if t is not None and s is not None:
                        mapping[t] = s
                return mapping
            # assume plain mapping already
            return {k: v for k, v in item.items() if v is not None}

        # If item is a list (combined file with multiple datasets)
        if isinstance(item, list):
            # Recurse for each element and flatten
            return None  # handled at outer level

        raise ValueError(f"Unsupported ground truth element type: {type(item)}")

    # list input (could be list of dicts or file paths)
    if isinstance(ground_truth_data, list):
        out: List[Dict[str, str]] = []
        for elem in ground_truth_data:
            if isinstance(elem, list):
                # a nested list => expand
                inner = format_ground_truth_for_mcnemar(elem)
                out.extend(inner)
            else:
                converted = _convert_one(elem)
                if converted is None:
                    # elem was a list inside; already expanded
                    continue
                out.append(converted)
        return out

    # single string path or single dict
    if isinstance(ground_truth_data, str):
        with open(ground_truth_data, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return format_ground_truth_for_mcnemar(data)
        return [ _convert_one(data) ]

    if isinstance(ground_truth_data, dict):
        return [{k: v for k, v in ground_truth_data.items() if v is not None}]

    raise ValueError(f"Unsupported ground_truth type: {type(ground_truth_data)}")




def mcnemar_analysis(detailed_matches_all_datasets, expected_dir: str, ground_truth=None):
    """
    CORRECTED McNemar's Test with proper alignment and flexible ground truth input.
    """

    # ---- Ground truth intake ----
    if ground_truth is not None:
        formatted_ground_truth = format_ground_truth_for_mcnemar(ground_truth)
    elif expected_dir is not None:
        gt_from_dir = load_expected_ground_truth(expected_dir)
        formatted_ground_truth = format_ground_truth_for_mcnemar(gt_from_dir)
    else:
        print("⚠️ No ground truth provided (neither 'ground_truth' nor 'expected_dir').")
        print("💡 McNemar's test requires ground truth to determine correctness.")
        print("🔍 Skipping McNemar's analysis...")
        return None

    # Use dict with (dataset_idx, column) as keys
    data = {approach_name: {} for approach, approach_name in APPROACHES}
    
    for dataset_idx, (dataset_matches, dataset_gt) in enumerate(
        zip(detailed_matches_all_datasets, formatted_ground_truth)
    ):
        print(f"  📊 Dataset {dataset_idx + 1}/{len(detailed_matches_all_datasets)}")
        
        if dataset_gt is None:
            print(f"     ⚠️  No ground truth available (cross-domain dataset) - skipping")
            continue
        
        valid_mappings = sum(1 for v in dataset_gt.values() if v is not None)
        print(f"     Ground truth has {len(dataset_gt)} mappings ({valid_mappings} valid)")
        
        for approach, approach_name in APPROACHES:
            if approach_name not in dataset_matches:
                print(f"    ⚠️  {approach_name}: Not available in dataset {dataset_idx + 1}")
                continue
            
            predictions = dataset_matches[approach_name]
            if not predictions:
                print(f"    ⚠️  {approach_name}: No predictions")
                continue
                
            correct_count = 0
            total_count = 0
            
            for target_col, (predicted_col, confidence, explanation) in predictions.items():
                #  CHECK: Is this column in ground truth?
                if target_col in dataset_gt:
                    correct_col = dataset_gt[target_col]
                    
                    # CRITICAL: Skip if ground truth is None (no valid match)
                    if correct_col is None:
                        continue
                    
                    is_correct = 1 if predicted_col == correct_col else 0
                    
                    # KEY FIX: Store with (dataset, column) as key
                    key = (dataset_idx, target_col)
                    data[approach_name][key] = is_correct
                    
                    if is_correct:
                        correct_count += 1
                    total_count += 1
            
            # if total_count > 0:
            #     print(f"    ✓ {method}: {correct_count}/{total_count} correct")
            # else:
            #     print(f"    ⚠️  {method}: 0 predictions with valid ground truth")
    
    for approach, approach_name in APPROACHES:
        total = len(data[approach_name])
        correct = sum(data[approach_name].values())
        if total > 0:
            accuracy = (correct / total) * 100
            print(f"  {approach_name}: {correct}/{total} correct ({accuracy:.1f}%)")
        else:
            print(f"  {approach_name}: No data")
    print()
    
    
    results = []
    
    for i, (approachA, approachA_name) in enumerate(APPROACHES):
        for j, (approachB, approachB_name) in enumerate(APPROACHES):
            if i >= j:
                continue
            
            # KEY FIX: Only compare where BOTH methods have predictions
            common_keys = set(data[approachA_name].keys()) & set(data[approachB_name].keys())
            
            if len(common_keys) == 0:
                print(f"⚠️  Skipping {approachA_name} vs {approachB_name}: No overlapping predictions")
                continue
            
            # Build contingency table from ALIGNED predictions only
            b = 0  # A correct, B wrong
            c = 0  # A wrong, B correct
            both_correct = 0
            both_wrong = 0
            
            for key in common_keys:
                A_val = data[approachA_name][key]
                B_val = data[approachB_name][key]

                if A_val == 1 and B_val == 1:
                    both_correct += 1
                elif A_val == 0 and B_val == 0:
                    both_wrong += 1
                elif A_val == 1 and B_val == 0:
                    b += 1
                elif A_val == 0 and B_val == 1:
                    c += 1
            
            # McNemar statistic
            if (b + c) == 0:
                statistic = 0
                p_value = 1.0
            else:
                statistic = ((abs(b - c) - 1) ** 2) / (b + c)
                p_value = 1 - chi2.cdf(statistic, df=1)
            
            # Determine winner
            if b > c:
                winner = approachA_name
                advantage = b - c
            elif c > b:
                winner = approachB_name
                advantage = c - b
            else:
                winner = "Tie"
                advantage = 0
            
            results.append({
                'Comparison': f"{approachA_name}\nvs\n{approachB_name}",
                'χ² Statistic': f"{statistic:.4f}",
                'p-value': f"{p_value:.4f}",
                'Significant\n(α=0.05)': "✓ YES" if p_value < 0.05 else "✗ NO",
                'Winner': winner if winner != "Tie" else "---",
                'Advantage': f"+{advantage}" if advantage > 0 else "0",
                'Both Correct': both_correct,
                'Both Wrong': both_wrong,
                'A Only': b,
                'B Only': c,
                'n (overlapping)': len(common_keys)
            })
    

    
    sig_count = sum(1 for r in results if r['Significant\n(α=0.05)'] == "✓ YES")
    total = len(results)
    
    
    return results

