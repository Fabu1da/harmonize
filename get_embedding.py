import logging
import os
import pickle
from typing import List

import numpy as np
import openai
from tenacity import retry, retry_if_exception_type, stop_after_attempt, stop_after_delay, wait_fixed, wait_random_exponential
import tiktoken

from batch_util import batched
from gpt_utils import client


logger = logging.getLogger(__name__)

EMBEDDING_CACHE_FILE = "./cache_storage/embedding_cache.pkl"
os.makedirs(os.path.dirname(EMBEDDING_CACHE_FILE), exist_ok=True)

# Load cached embeddings (to avoid redundant API calls)
if os.path.exists(EMBEDDING_CACHE_FILE):
    try:
        with open(EMBEDDING_CACHE_FILE, "rb") as f:
            EMBEDDING_CACHE = pickle.load(f)
    except (pickle.UnpicklingError, EOFError):
        logging.warning(f"Corrupted cache file '{EMBEDDING_CACHE_FILE}', resetting cache.")
        EMBEDDING_CACHE = {}
else:
    EMBEDDING_CACHE = {}


@retry(retry=retry_if_exception_type(openai.APIError), wait=wait_fixed(1), stop=stop_after_attempt(5))
@retry(retry=retry_if_exception_type(openai.RateLimitError), wait=wait_random_exponential(min=1, max=60), stop=stop_after_delay(300))
async def create(*args, **kwargs):
    try:
       return await client.embeddings.create(*args, **kwargs)
    except Exception as e:
        # logger.warning(args, kwargs, e)
        raise e


async def get_embeddings_batch(texts: List[str], model: str) -> List[np.ndarray]:
    """
    Efficiently get embeddings for multiple texts using batch API calls.
    Significantly faster and more cost-effective than individual calls.
    """

    # Check cache first
    cached_embeddings = {}
    uncached_texts = []

    cache = EMBEDDING_CACHE.setdefault(model, {})

    for text in texts:
        if text in cache:
            cached_embeddings[text] = np.array(cache[text])
        else:
            uncached_texts.append(text)

    uncached_texts = list(set(uncached_texts))  # Remove duplicates

    # Get new embeddings in batch if needed
    if uncached_texts:
        encoding = tiktoken.encoding_for_model(model)
        sizes: dict[str, int] = dict(zip(uncached_texts, map(len, encoding.encode_ordinary_batch(uncached_texts))))

        if any(size > 8191 for size in sizes.values()):
            raise ValueError("One or more texts exceed the token limit for the embedding model.")

        for batch in batched(uncached_texts, max_count=2048, max_size=300000, size_fn=lambda text: sizes[text] + 1):
            response = await create(input=batch, model=model)

            # Process batch response
            for i, embedding_obj in enumerate(response.data):
                text = batch[i]
                embedding = embedding_obj.embedding
                cache[text] = embedding
                cached_embeddings[text] = np.array(embedding)

        # Save updated cache
        with open(EMBEDDING_CACHE_FILE, "wb") as f:
            pickle.dump(EMBEDDING_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)

    return [cached_embeddings[text] for text in texts]
