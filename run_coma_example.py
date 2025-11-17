# COMA Schema Matching Function

import pathlib
import pandas as pd
from valentine.algorithms import Coma
from valentine import valentine_match
from typing import Dict, Tuple, Optional

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file with proper error handling"""
    return pd.read_csv(path, dtype=str, low_memory=False)


def coma_matching(source_path: str, target_path: str, 
                    source_name: Optional[str] = None, 
                    target_name: Optional[str] = None,
                    use_instances: bool = False,
                    source_cols: Optional[list[str]] = None,
                    target_cols: Optional[list[str]] = None,
                    max_rows: Optional[int]=None
    ) -> Dict[str, Tuple[Optional[str], float, Optional[str]]]:
    
    """
    Run COMA schema matching between source and target files.
    
    Args:
        source_path: Path to source CSV file
        target_path: Path to target CSV file  
        source_name: Optional name for source (defaults to filename)
        target_name: Optional name for target (defaults to filename)
        
    Returns:
        Dict in format: {target_column: (predicted_source_column, confidence, reasoning)}
    """
    
    # Get names from file paths if not provided
    if source_name is None:
        source_name = pathlib.Path(source_path).stem
    if target_name is None:
        target_name = pathlib.Path(target_path).stem

    # Load data
    source_df = load_csv(source_path)
    target_df = load_csv(target_path)


    if source_cols is not None:
        source_df = source_df[source_cols]
    if target_cols is not None:
        target_df = target_df[target_cols]

    if max_rows is not None:
        # now the type-checker knows this is an int
        source_df = source_df.head(int(max_rows))
        target_df = target_df.head(int(max_rows))

    # Initialize COMA matcher
    # matcher = Coma(java_xmx="8192m", use_instances=use_instances)
    matcher = Coma(java_xmx="8192m", use_instances=use_instances)

    # Run matching
    matches = valentine_match(source_df, target_df, matcher, 
                              df1_name=source_name, df2_name=target_name)

    # Convert to our standard format: {target_column: (source_column, confidence, reasoning)}
    predictions = {}

    for (col1, col2), score in matches.items():
        # Extract column names (handle tuple format)
        source_col = col1[1] if isinstance(col1, tuple) else col1
        target_col = col2[1] if isinstance(col2, tuple) else col2

        # COMA returns source->target matches, we need target->source format
        # So we swap: target_col gets mapped to source_col
        confidence = float(score)
        reasoning = f"COMA similarity: {confidence:.3f}"

        if target_col not in predictions or confidence > predictions[target_col][1]:
            predictions[target_col] = (source_col, confidence, reasoning)

    for target_col in target_df.columns:
        if target_col not in predictions:
            predictions[target_col] = (None, 0.0, "No match found by COMA")

    return predictions
