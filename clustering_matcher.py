import numpy as np
from sklearn.cluster import KMeans
from get_embedding import get_embedding

def enrich_column(col, schema):
    try:
        prop = schema.properties.get(col)
        if prop and hasattr(prop, "examples") and prop.examples:
            example_str = ", ".join(map(str, prop.examples[:3]))
        else:
            example_str = ""
        return f"{col}: {example_str}"
    except Exception as e:
        return col  # fallback


def clustering_matcher(source_schema, target_schema, n_clusters=None):
    
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

    return predicted_mapping
