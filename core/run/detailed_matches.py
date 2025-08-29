from typing import Any, Dict, List


def collect_detailed_matches(detailed_matches: List, source_table: str, target_table: str,
                           target_schema: Any, predicted_mapping: Dict, embed_predicted: Dict,
                           cluster_predicted: Dict, majority_predicted: Dict, 
                           weighted_predicted: Dict, evaluation_mapping: Dict):
    """Collect detailed match information for each matcher"""
    for col in target_schema.properties.keys():
        ground_truth_for_col = evaluation_mapping.get(col, "—")
        
        for matcher_name, predictions in [
            ("gpt", predicted_mapping),
            ("embed", embed_predicted),
            ("cluster", cluster_predicted),
            ("majority", majority_predicted),
            ("weighted", weighted_predicted)
        ]:
            match, sim = predictions.get(col, ("—", 0.0))
            if match not in ("—", None):
                detailed_matches.append({
                    "source": f"real_{source_table}.{match}",
                    "target": f"{target_table}.{col}",
                    "similarity": round(sim, 4),
                    "src_file": f"{source_table}.csv",
                    "trg_file": f"{target_table}.json",
                    "matcher": matcher_name,
                    "ground_truth": ground_truth_for_col
                })
