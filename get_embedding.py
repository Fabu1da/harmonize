
import logging
import os
import json
import openai
import numpy as np


EMBEDDING_CACHE_FILE = "./cache_storage/embedding_cache.json"

# Load cached embeddings (to avoid redundant API calls)
if os.path.exists(EMBEDDING_CACHE_FILE):
    try:
        with open(EMBEDDING_CACHE_FILE, "r") as f:
            EMBEDDING_CACHE = json.load(f)
    except json.JSONDecodeError:
        logging.warning(f"Corrupted cache file '{EMBEDDING_CACHE_FILE}', resetting cache.")
        EMBEDDING_CACHE = {}
else:
    EMBEDDING_CACHE = {}
def get_embedding(text):
    print("------>", text)
    """Generate or retrieve cached embedding for a column name."""
    if text in EMBEDDING_CACHE:
        
        return np.array(EMBEDDING_CACHE[text])
    
    try:
        response = openai.embeddings.create(input=[text], model="text-embedding-3-small")
        embedding = response.data[0].embedding  
        EMBEDDING_CACHE[text] = embedding
        
        with open(EMBEDDING_CACHE_FILE, "w") as f:
            json.dump(EMBEDDING_CACHE, f)

        return np.array(embedding)

    except Exception as e:
        logging.error(f"Error generating embedding for '{text}': {e}")
        return np.zeros(1536) 
