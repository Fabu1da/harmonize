import pandas as pd
from collections.abc import Callable
from typing import Any


# TODO: improve runtime complexity from O(R*C) to O(C)
def apply_rules(dataset: pd.DataFrame, rules: dict[str, Callable[[dict], Any]]) -> pd.DataFrame:
    """
    Apply transformation rules to convert source dataset to target format.
    
    Args:
        dataset: Source dataframe to transform
        rules: Dictionary mapping target columns to transformation functions
        
    Returns:
        Transformed dataframe with target schema
        
    Note:
        Current implementation has O(R*C) complexity where R=rows, C=columns.
        Could be optimized to O(C) by vectorizing operations.
    """
    columns = rules.keys()
    transformed_df = pd.DataFrame(columns=columns)
    
    for row in dataset.itertuples():
        index = row.Index
        data = row._asdict()
        for column, rule in rules.items():
            value = rule(data)
            transformed_df.loc[index, column] = value
            
    return transformed_df
