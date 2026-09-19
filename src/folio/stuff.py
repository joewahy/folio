""" "Stuff the whole PDF into context" mode -- no retrieval, no chunking,
no embeddings, no agent loop.

Concatenates every page's text into one block, page numbers inline as
[p. N] markers, and asks Claude the question in a single call_llm()
call. Exists as a cost/quality comparison against the RAG path
(agent.py) -- see the README's "RAG vs. stuff mode" section for the
measured tradeoff.
"""

from __future__ import annotations

from folio.agent import extraction as extract_text
from folio.extraction import Page
from folio.llm import call_llm

SYSTEM_PROMPT = """
Read the reference material before answering. The question to be answered is located last, after the reference material. Cite pages where answer was found in the answer. If the reference material doesn't answer the question, say so instead of guessing.
"""

def ask(question: str, pages: list[Page], cache: bool = True) -> str:
    page_text = "Reference Material: "
    for page in pages:
        page_text += f"[p. {page.number}]\n{page.text}\n\n"

    if cache:
        # The reference material is its own content block with a cache
        # breakpoint, separate from the question. cache_control caches
        # everything up to and including the marked block (system prompt +
        # this block), so repeated questions in the same session reuse that
        # cached prefix instead of re-billing the whole PDF as input tokens
        # every time -- only the trailing question block is fresh per call.
        content = [
            {
                "type": "text",
                "text": page_text,
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": f"Question: {question}"},
        ]
    else:
        content = page_text + f"\nQuestion: {question}"

    messages: list[dict] = [{"role": "user", "content": content}]

    response = call_llm(messages, system=SYSTEM_PROMPT)
    return extract_text(response)
