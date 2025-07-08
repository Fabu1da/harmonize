
import logging
import os
import json
import openai
import numpy as np

from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from metrics import OPENAI_CALLS, OPENAI_LATENCY



EMBEDDING_CACHE_FILE = "./embedding_cache.json"

# this helper will retry on 429s, 5xx’s or timeouts, with exponential backoff + jitter
@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_random_exponential(min=1, max=60),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIError, openai.Timeout))
)
def _call_embedding_api(*args, **kwargs):
    """Pass-through to openai.embeddings.create, retrying on transient errors."""
    OPENAI_CALLS.labels(api_type="embeddings").inc()
    with OPENAI_LATENCY.labels(api_type="embeddings").time():
        return openai.embeddings.create(*args, **kwargs)


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
    """Generate or retrieve cached embedding for a column name."""
    if text in EMBEDDING_CACHE:
        return np.array(EMBEDDING_CACHE[text])
    
    try:
        response = _call_embedding_api(input=[text], model="text-embedding-3-small")
        embedding = response.data[0].embedding  
        EMBEDDING_CACHE[text] = embedding
        
        with open(EMBEDDING_CACHE_FILE, "w") as f:
            json.dump(EMBEDDING_CACHE, f)

        return np.array(embedding)

    except Exception as e:
        logging.error(f"Error generating embedding for '{text}': {e}")
        return np.zeros(1536) 
