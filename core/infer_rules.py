from json_schema import ObjectSchema
from collections.abc import Callable
import logging
from typing import Any
from functools import partial

async def infer_rules(column_map, target_schema: ObjectSchema) -> dict[str, Callable[[dict], Any]]:
    # Ensure column_map contains only strings
    column_map = {
        target: source
        for target, source in column_map.items()
    }

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