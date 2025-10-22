
import logging
import os
import json
import openai
import numpy as np
from typing import List, Dict


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

def get_embedding(text: str, model: str) -> np.ndarray:
    """Generate or retrieve cached embedding for a single text."""
    print("------>", text)
    if text in EMBEDDING_CACHE:
        return np.array(EMBEDDING_CACHE[text])
    
    try:
        response = openai.embeddings.create(input=[text], model=model)
        embedding = response.data[0].embedding  
        EMBEDDING_CACHE[text] = embedding
        
        # Save cache after each addition
        os.makedirs(os.path.dirname(EMBEDDING_CACHE_FILE), exist_ok=True)
        with open(EMBEDDING_CACHE_FILE, "w") as f:
            json.dump(EMBEDDING_CACHE, f)

        return np.array(embedding)

    except Exception as e:
        logging.error(f"Error generating embedding for '{text}': {e}")
        # Get dimension from model (fallback to common dimensions)
        model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072
        }
        dim = model_dimensions.get(model, 1536)
        return np.zeros(dim)


def get_embeddings_batch(texts: List[str], model: str) -> Dict[str, np.ndarray]:
    """
    Efficiently get embeddings for multiple texts using batch API calls.
    Significantly faster and more cost-effective than individual calls.
    """
    print(f"🔄 Getting embeddings for {len(texts)} texts using model {model}")
    
    # Check cache first
    cached_embeddings = {}
    uncached_texts = []
    
    for text in texts:
        if text in EMBEDDING_CACHE:
            cached_embeddings[text] = np.array(EMBEDDING_CACHE[text])
        else:
            uncached_texts.append(text)
    
    print(f"📁 Found {len(cached_embeddings)} cached, need {len(uncached_texts)} new embeddings")
    
    # Get new embeddings in batch if needed
    if uncached_texts:
        try:
            response = openai.embeddings.create(input=uncached_texts, model=model)
            
            # Process batch response
            for i, embedding_obj in enumerate(response.data):
                text = uncached_texts[i]
                embedding = embedding_obj.embedding
                EMBEDDING_CACHE[text] = embedding
                cached_embeddings[text] = np.array(embedding)
            
            # Save updated cache
            os.makedirs(os.path.dirname(EMBEDDING_CACHE_FILE), exist_ok=True)
            with open(EMBEDDING_CACHE_FILE, "w") as f:
                json.dump(EMBEDDING_CACHE, f)
                
            print(f"✅ Retrieved {len(uncached_texts)} new embeddings")
            
        except Exception as e:
            logging.error(f"Error in batch embedding request: {e}")
            # Fallback for failed texts
            model_dimensions = {
                "text-embedding-ada-002": 1536,
                "text-embedding-3-small": 1536,
                "text-embedding-3-large": 3072
            }
            dim = model_dimensions.get(model, 1536)
            for text in uncached_texts:
                cached_embeddings[text] = np.zeros(dim)
    
    return cached_embeddings 
