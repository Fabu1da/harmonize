from sklearn.metrics.pairwise import cosine_similarity
from get_embedding import get_embedding

def embedding_column_mapping(source_columns: list[str], target_columns: list[str], threshold: float = 0) -> dict[str, tuple[str, float]]:
    """
    Maps target columns to source columns using cosine similarity on embeddings.
    Returns: dict of {target_column: (matched_source_column, similarity)}
    """
    source_embeddings = {col: get_embedding(col) for col in source_columns}
    target_embeddings = {col: get_embedding(col) for col in target_columns}
    
    mapping = {}
    
    for tgt_col, tgt_emb in target_embeddings.items():
        best_match = None
        best_score = -1.0
        
        for src_col, src_emb in source_embeddings.items():
            sim = cosine_similarity([tgt_emb], [src_emb])[0][0]
            if sim > best_score:
                best_match = src_col
                best_score = sim

        if best_score >= threshold:
            mapping[tgt_col] = (best_match, round(float(best_score), 4))
        else:
            mapping[tgt_col] = ("null", round(float(best_score), 4))

    return mapping
