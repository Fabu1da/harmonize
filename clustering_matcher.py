import numpy as np
from sklearn.cluster import KMeans
from get_embedding import get_embedding

def enrich_column(col, schema):
    """
    For better cache utilization, use just the column name.
    The examples don't add significant value for clustering/similarity
    but prevent cache hits, causing unnecessary API calls.
    """
    return col  # Use simple column name for cache efficiency


def clustering_matcher(source_schema, target_schema, n_clusters=None, return_cluster_info=False):
    
    source_cols = list(source_schema.properties.keys())
    target_cols = list(target_schema.properties.keys())
    all_columns = source_cols + target_cols
    
    # Enriched text for better embedding
    enriched_texts = [enrich_column(col, source_schema) if col in source_cols else enrich_column(col, target_schema) for col in all_columns]

    # Get embeddings
    embedding_matrix = np.array([get_embedding(text) for text in enriched_texts])
   
    # Heuristic for cluster count
    if n_clusters is None:
        n_clusters = max(2, int(len(all_columns) / 2))

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding_matrix)
    labels = kmeans.labels_
    
    # Create index maps
    cluster_map = {col: labels[i] for i, col in enumerate(all_columns)}
    column_embeddings = {col: embedding_matrix[i] for i, col in enumerate(all_columns)}
    
    # Calculate actual clusters used
    actual_clusters_used = len(set(labels))

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
            # Fall back to best source overall
            similarities = [
                (src, float(np.dot(tgt_emb, column_embeddings[src]) / 
                            (np.linalg.norm(tgt_emb) * np.linalg.norm(column_embeddings[src]) + 1e-9)))
                for src in source_cols
            ]
            best_match, best_score = max(similarities, key=lambda x: x[1])
            predicted_mapping[tgt] = (best_match, round(best_score, 4))
            continue


        # Cosine similarity
        similarities = [
            (src, float(np.dot(tgt_emb, src_emb) / (np.linalg.norm(tgt_emb) * np.linalg.norm(src_emb) + 1e-9)))
            for src, src_emb in candidates
        ]
        best_match, best_score = max(similarities, key=lambda x: x[1])
        predicted_mapping[tgt] = (best_match, round(best_score, 4))

    if return_cluster_info:
        cluster_info = {
            'n_clusters_requested': n_clusters,
            'n_clusters_actual': actual_clusters_used,
            'total_columns': len(all_columns),
            'source_columns': len(source_cols),
            'target_columns': len(target_cols),
            'cluster_assignments': cluster_map
        }
        return predicted_mapping, cluster_info
    
    return predicted_mapping
