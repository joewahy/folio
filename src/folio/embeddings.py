"""Wrapper around the OpenAI embeddings API.

Mechanical only: given text, return a vector. What you do with the vector
(chunking strategy, storage, similarity search) is not this module's job --
see chunking.py and search.py.
"""

from __future__ import annotations

import os

from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _model() -> str:
    return os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def embed_text(text: str) -> list[float]:
    """Embed a single string, returning its vector."""
    response = _get_client().embeddings.create(model=_model(), input=text)
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings in one API call.

    Batching is a straightforward efficiency win (one request instead of N)
    rather than a design decision, which is why it lives here alongside
    embed_text() instead of being left as a TODO.
    """
    response = _get_client().embeddings.create(model=_model(), input=texts)
    return [item.embedding for item in response.data]
