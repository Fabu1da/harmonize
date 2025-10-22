import math
from typing import Dict, Tuple, List, Literal, Union

from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np

from get_embedding import get_embeddings_batch
from json_schema import ObjectSchema

def enrich_column_text(column_name: str, schema_properties: dict = None) -> str:
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

def embedding_column_mapping(source_schema: ObjectSchema, target_schema: ObjectSchema,
                           model: str, threshold: float,
                           method: Union[Literal["hungarian", "greedy"]]) -> Dict[str, Tuple[str, float, str]]:
    """
    Maps target columns to source columns using cosine similarity on embeddings.
    
    TODO:
    - Batch embedding calls for better performance
    - Optional Hungarian algorithm for optimal 1:1 assignment
    - Text enrichment with schema information
    - Better similarity analysis and reasoning
    
    Returns: dict of {target_column: (matched_source_column, confidence, reasoning)}
    """
    
    # Extract column names from schemas
    source_columns = list(source_schema.properties.keys())
    target_columns = list(target_schema.properties.keys())

    print(f"🎯 Starting embedding-based column mapping")
    print(f"   Source columns: {len(source_columns)}")
    print(f"   Target columns: {len(target_columns)}")
    print(f"   Threshold: {threshold}")
    print(f"   Method: {method}")
    
    # Enrich column texts for better embeddings
    enriched_source = [enrich_column_text(col, source_schema.properties) for col in source_columns]
    enriched_target = [enrich_column_text(col, target_schema.properties) for col in target_columns]
    
    # Get all embeddings in batch (much faster!)
    all_texts = enriched_source + enriched_target
    all_embeddings = get_embeddings_batch(all_texts, model)
    
    # Split embeddings back
    source_embeddings = {source_columns[i]: all_embeddings[enriched_source[i]] 
                        for i in range(len(source_columns))}
    target_embeddings = {target_columns[i]: all_embeddings[enriched_target[i]] 
                        for i in range(len(target_columns))}
    
    # Create similarity matrix
    similarity_matrix = np.zeros((len(target_columns), len(source_columns)))
    
    for i, tgt_col in enumerate(target_columns):
        for j, src_col in enumerate(source_columns):
            similarity_matrix[i, j] = cosine_similarity([target_embeddings[tgt_col]], [source_embeddings[src_col]])[0][0]

    if method == "hungarian" and len(source_columns) > 1 and len(target_columns) > 1:
        return _hungarian_mapping(source_columns, target_columns, similarity_matrix, threshold)
    else:
        return _greedy_mapping(source_columns, target_columns, similarity_matrix, threshold)

def _hungarian_mapping(source_columns: List[str], target_columns: List[str], similarity_matrix: np.ndarray,
                       threshold: float) -> Dict[str, Tuple[str, float, str]]:
    """
    Use Hungarian algorithm for optimal 1:1 assignment.
    """
    print("🧮 Using optimal assignment (Hungarian algorithm)")
    
    # Hungarian algorithm (minimizes cost, so we use negative similarity)
    row_indices, col_indices = linear_sum_assignment(-similarity_matrix)
    
    mapping = {}
    for i, j in zip(row_indices, col_indices):
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
                    threshold: float) -> Dict[str, Tuple[str, float, str]]:
    """
    Greedy mapping (original algorithm but with improvements).
    """
    print("🎯 Using greedy assignment")
    
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
