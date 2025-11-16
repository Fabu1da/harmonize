from typing import Optional
import numpy as np
from sklearn.cluster import KMeans
from get_embedding import get_embeddings_batch
from json_schema import ObjectSchema

def enrich_column(col, schema):
    """
    For better cache utilization, use just the column name.
    The examples don't add significant value for clustering/similarity
    but prevent cache hits, causing unnecessary API calls.
    """
    return col  # Use simple column name for cache efficiency


async def clustering_matcher(source_schema: ObjectSchema, target_schema: ObjectSchema, model: str):
    assert source_schema.properties is not None
    assert target_schema.properties is not None

    source_cols = list(source_schema.properties.keys())
    target_cols = list(target_schema.properties.keys())
    
    # Enriched text for better embedding
    enriched_source = [enrich_column(col, source_schema) for col in source_cols]
    enriched_target = [enrich_column(col, target_schema) for col in target_cols]
    enriched_texts = enriched_source + enriched_target

    # Get embeddings
    embedding_matrix = np.array(await get_embeddings_batch(enriched_texts, model=model))

    # Heuristic for cluster count
    n_clusters = len(target_cols)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding_matrix)
    labels = kmeans.labels_

    # Create index maps
    cluster_map = {col: labels[i] for i, col in enumerate(enriched_texts)}
    column_embeddings = {col: embedding_matrix[i] for i, col in enumerate(enriched_texts)}

    # Match each target to the closest source in the same cluster
    predicted_mapping = {}

    for tgt in target_cols:
        tgt_cluster = cluster_map[tgt]
        tgt_emb = column_embeddings[tgt]

        candidates = [
            (src, column_embeddings[src])
            for src in source_cols
            if cluster_map[src] == tgt_cluster
        ]

        if not candidates:
            similarities = [
                (src, float(np.dot(tgt_emb, column_embeddings[src])))
                for src in source_cols
            ]
            best_match, best_score = max(similarities, key=lambda x: x[1])
            predicted_mapping[tgt] = (None, 1.0, f"Clustering similarity: {round(best_score, 4)}")
            continue

        # Cosine similarity
        similarities = [
            (src, float(np.dot(tgt_emb, src_emb)))
            for src, src_emb in candidates
        ]
        best_match, best_score = max(similarities, key=lambda x: x[1])
        predicted_mapping[tgt] = (best_match, 1 / len(candidates), f"Clustering similarity: {round(best_score, 4)}")

    return predicted_mapping
