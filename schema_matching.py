import faiss
import numpy as np
from get_embedding import get_embedding

def cosine_similarity_from_l2(l2_distance):
    """Convert L2 distance between normalized vectors to cosine similarity."""
    return 1 - (l2_distance ** 2) / 2

def embed_columns(schema: dict) -> np.ndarray:
    """
    Embed columns in schema using a pre-trained model.
    """
    cols = list(schema.keys())
    texts = [
        f"{col} (type: {schema[col].get('type', '')}): {schema[col].get('description', '')}"
        for col in cols
    ]
    embeddings = np.array([get_embedding(text) for text in texts], dtype=np.float32)
    return cols, embeddings

def match_schema(source_schema: dict, target_schema: dict, threshold: float = 0.85) -> dict:
    """
    Match columns from target_schema to source_schema by embedding similarity.
    """

    source_cols, source_embeddings = embed_columns(source_schema)
    target_cols, target_embeddings = embed_columns(target_schema)

    # Build FAISS index and search
    dim = source_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(source_embeddings)

    distances, indices = index.search(target_embeddings, k=1)  # k=1 for top-1 match

    # Build mapping output
    column_mapping = {}
    for i, target_col in enumerate(target_cols):
        best_match_idx = indices[i][0]
        best_source_col = source_cols[best_match_idx]
        similarity = cosine_similarity_from_l2(distances[i][0])

        print(f"[Match Attempt] Target: {target_col} → Source: {best_source_col} | Similarity: {similarity:.4f}")

        if similarity >= threshold:
            column_mapping[target_col] = {
                "source_column": best_source_col,
                "similarity": float(similarity)
            }
        else:
            column_mapping[target_col] = {
                "source_column": None,
                "similarity": float(similarity),
                "note": "No sufficient match found"
            }

    return column_mapping
