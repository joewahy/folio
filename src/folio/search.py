"""The search_pdf tool: its schema, and the similarity search behind it.

SEARCH_PDF_TOOL is the schema Claude sees; search_pdf() is the cosine
similarity search that backs it up once Claude decides to call it. Worth
revisiting SEARCH_PDF_TOOL's description text once you can watch, against
real questions, whether Claude actually calls it when it should.
"""

from __future__ import annotations

import numpy as np

from folio import embeddings
from folio.chunking import Chunk

# ---------------------------------------------------------------------------
# The tool schema
# ---------------------------------------------------------------------------
# This dict is the ONLY thing Claude ever sees about this tool -- it never
# reads the Python function below. The two description fields have
# different jobs:
#   - "description" (top level): WHEN should Claude call this, and what's
#     it for?
#   - "query"'s "description": WHAT should Claude put in this string, and
#     in what format (verbatim question vs. rephrased keywords)?
SEARCH_PDF_TOOL = {
    "name": "search_pdf",
    "description": "This tool indexes the PDF. Call this function to retrieve the passages most relevant to a question before answering.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Rephrase the question into keywords first.",
            }
        },
        "required": ["query"],
    },
}


# ---------------------------------------------------------------------------
# The actual search
# ---------------------------------------------------------------------------
# `chunks` and `vectors` are what build_index() in cli.py already produces:
# vectors[i] is the embedding of chunks[i].text, lined up by position, and
# vectors is already normalized to unit length by build_index() -- this
# function does NOT rebuild or re-normalize the corpus, just uses what it
# was handed. Claude never sees `chunks`/`vectors` either -- the agent loop
# supplies them when it executes this tool call (see agent.py), alongside
# whatever `query` string Claude decided on.
def search_pdf(
    query: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    top_k: int = 5,
) -> str:
    # Embed the query -- a single string, so embed_text() (not the plural
    # embed_texts()).
    array = np.array(embeddings.embed_text(query))

    # Normalize the query vector to unit length, then score every chunk
    # against it in one shot. Cosine similarity between two UNIT vectors is
    # just their dot product; `vectors` is already normalized (build_index()
    # did that once, in cli.py), so the only normalizing left to do is on
    # the query vector just built above.
    normalized_query_vector = array / np.linalg.norm(array)
    # This machine's numpy is built against Apple's Accelerate BLAS backend,
    # which can emit spurious divide-by-zero/overflow/invalid-value warnings
    # on this matmul even when the result is correct (verified: no NaN/Inf
    # in the inputs or the output, scores land in a sane cosine-similarity
    # range). Suppressed here rather than left to print on every query.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        scores = vectors @ normalized_query_vector

    # Pick the top_k highest-scoring chunks. np.argsort(scores) sorts
    # ascending (lowest first), so reverse it to get highest-scoring first.
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Map those indices back to the actual Chunk objects, formatting each
    # with its page number so the citation survives once this is embedded
    # in Claude's context. tool_result content must be a single string, so
    # join the formatted chunks together before returning.
    strings = []
    for i in top_indices:
        strings.append(f"[p. {chunks[i].page_number}] {chunks[i].text}")

    result = "\n\n".join(strings)
    return result
