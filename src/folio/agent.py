"""The agent loop: this is the core of the "agentic" in agentic PDF Q&A.

Unlike a fixed retrieve-then-generate pipeline, Claude here decides
*whether* and *how many times* to call search_pdf before answering. Your
job is to drive that loop -- call the model, check what it wants to do,
act on it, repeat -- not to decide the answer yourself.

The "how" (branching on stop_reason, building tool_result blocks, message
shapes) is API mechanics; the "what" (system prompt wording, citation
enforcement, whether questions share history) is the design work.
"""

from __future__ import annotations

import numpy as np

from folio import search
from folio.chunking import Chunk
from folio.llm import call_llm

# Hard cap on tool-call round trips for a single question. Without one, a
# model that keeps deciding to search never terminates the loop -- a real
# failure mode, not a hypothetical one. Adjust as you like; the point is
# that *some* cap exists.
MAX_TOOL_CALLS = 5

# Decisions made below: always search (rather than trust judgment about
# whether a question needs the document), and explicitly require citing
# pages -- since search_pdf's results carry "[p. N]" markers, but nothing
# forces Claude to repeat them in its answer unless told to.
SYSTEM_PROMPT = """
Always call search_pdf before answering. Use output from search_pdf to answer question. Cite pages where answer was found in the answer. If the retrieved passages don't answer the question, say so instead of guessing.
"""

FALLBACK_PROMPT = """
Answer using whatever was already retrieved above. Cite pages only for content that actually came from a search result, don't invent a citation. Tell the user the answer may be incomplete since search was cut short.
"""

def extraction(call):
    result = ""
    for block in call.content:
        if block.type == "text":
            result = block.text
    return result

def ask(question: str, chunks: list[Chunk], vectors: np.ndarray) -> str:
    """Answer one question, letting Claude call search_pdf as needed.

    `chunks`/`vectors` are the index cli.py built once at startup -- this
    function threads them through to search.search_pdf() every time
    Claude asks to search. Claude itself never sees them, only the
    `query` string it decides to send (per SEARCH_PDF_TOOL's schema in
    search.py).
    """
    # A message's "content" can be a plain string (as here) or a list of
    # content blocks (as you'll need below for tool_result). Both are
    # valid -- which one to use depends on what you're sending.
    messages: list[dict] = [{"role": "user", "content": question}]

    for _ in range(MAX_TOOL_CALLS):
        response = call_llm(
            messages, system=SYSTEM_PROMPT, tools=[search.SEARCH_PDF_TOOL]
        )
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_use_blocks = [
                block for block in response.content if block.type == "tool_use"
            ]

            result_block = []

            for block in tool_use_blocks:
                query = block.input["query"]
                result = search.search_pdf(query, chunks, vectors)

                result_block.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result}
                )

            messages.append({"role": "user", "content": result_block})
            continue
        else:
            return extraction(response)

    fallback = call_llm(
            messages, system=FALLBACK_PROMPT, effort="medium"
        )
    return extraction(fallback)