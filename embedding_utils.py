import math
from typing import Dict, Optional, Tuple, List, Literal, Union

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np

from get_embedding import get_embeddings_batch
from json_schema import ObjectSchema

def enrich_column_text(column_name: str, schema_properties: dict) -> str:
    """
    Enrich column name with additional context for better embeddings.
    """
    if schema_properties and column_name in schema_properties:
        prop = schema_properties[column_name]
        description = getattr(prop, 'description', '') or ''
        type_info = getattr(prop, 'type', '') or ''
        
        # Create enriched text
        enriched = f"{column_name}"
        if type_info:
            enriched += f" ({type_info})"
        if description:
            enriched += f" - {description}"
        return enriched
    return column_name

async def embedding_column_mapping(source_schema: ObjectSchema, target_schema: ObjectSchema,
                           model: str, threshold: float,
                           method: Union[Literal["hungarian"], Literal["greedy"]]) -> Dict[str, Tuple[Optional[str], float, Optional[str]]]:
    """
    Maps target columns to source columns using cosine similarity on embeddings.
    
    TODO:
    - Batch embedding calls for better performance
    - Optional Hungarian algorithm for optimal 1:1 assignment
    - Text enrichment with schema information
    - Better similarity analysis and reasoning
    
    Returns: dict of {target_column: (matched_source_column, confidence, reasoning)}
    """
    assert source_schema.properties is not None
    assert target_schema.properties is not None
    
    # Extract column names from schemas
    source_columns = list(source_schema.properties.keys())
    target_columns = list(target_schema.properties.keys())

    # Enrich column texts for better embeddings
    enriched_source = [enrich_column_text(col, source_schema.properties) for col in source_columns]
    enriched_target = [enrich_column_text(col, target_schema.properties) for col in target_columns]

    # Get all embeddings in batch (much faster!)
    all_texts = enriched_source + enriched_target
    all_embeddings = await get_embeddings_batch(all_texts, model)
    all_embeddings = {text: all_embeddings[i] for i, text in enumerate(all_texts)}

    # Split embeddings back
    source_embeddings = {source_columns[i]: all_embeddings[enriched_source[i]] 
                        for i in range(len(source_columns))}
    target_embeddings = {target_columns[i]: all_embeddings[enriched_target[i]] 
                        for i in range(len(target_columns))}

    # Create similarity matrix
    similarity_matrix = np.zeros((len(target_columns), len(source_columns)))

    similarity_matrix = cosine_similarity(
        np.array([target_embeddings[tgt_col] for tgt_col in target_columns]),
        np.array([source_embeddings[src_col] for src_col in source_columns])
    )

    if method == "hungarian" and len(source_columns) > 1 and len(target_columns) > 1:
        return _hungarian_mapping(source_columns, target_columns, similarity_matrix, threshold)
    else:
        return _greedy_mapping(source_columns, target_columns, similarity_matrix, threshold)

def _hungarian_mapping(source_columns: List[str], target_columns: List[str], similarity_matrix: np.ndarray,
                       threshold: float) -> Dict[str, Tuple[Optional[str], float, Optional[str]]]:
    """
    Use Hungarian algorithm for optimal 1:1 assignment.
    """
    # Hungarian algorithm (minimizes cost, so we use negative similarity)
    row_indices, col_indices = linear_sum_assignment(similarity_matrix, maximize=True)
    
    mapping: dict[str, tuple[Optional[str], float, Optional[str]]] = {
        target_column: (None, 0.0, None)
        for target_column in target_columns
    }

    for i, j in zip(row_indices, col_indices, strict=True):
        tgt_col = target_columns[i]
        src_col = source_columns[j]
        similarity = similarity_matrix[i, j]

        confidence = math.exp(similarity) / sum(math.exp(similarity_matrix[i, k]) for k in range(len(source_columns)))
        
        if similarity >= threshold:
            mapping[tgt_col] = (src_col, confidence, f"Embedding match: {tgt_col} <- {src_col} (similarity: {similarity:.3f})")
        else:
            mapping[tgt_col] = (None, 1 - confidence, f"No suitable match found (best similarity: {similarity:.3f}, below threshold {threshold})")

    return mapping

def _greedy_mapping(source_columns: List[str], target_columns: List[str], similarity_matrix: np.ndarray,
                    threshold: float) -> Dict[str, Tuple[Optional[str], float, Optional[str]]]:
    """
    Greedy mapping (original algorithm but with improvements).
    """
    mapping = {}
    
    for i, tgt_col in enumerate(target_columns):
        best_match = None
        best_score = -1.0
        sum_score = 0.0
        
        for j, src_col in enumerate(source_columns):
            sim = similarity_matrix[i, j]
            sum_score += math.exp(sim)
            if sim > best_score:
                best_match = src_col
                best_score = sim

        confidence = math.exp(best_score) / sum_score if sum_score > 0 else 0.0

        if best_score >= threshold:
            mapping[tgt_col] = (best_match, confidence, f"Best match: {best_match} (similarity: {best_score:.3f})")
        else:
            mapping[tgt_col] = (None, 1 - confidence, f"No suitable match found (best similarity: {best_score:.3f}, below threshold {threshold})")

    return mapping
