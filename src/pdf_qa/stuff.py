""" "Stuff the whole PDF into context" mode -- no retrieval, no chunking,
no embeddings, no agent loop.

Concatenates every page's text into one block, page numbers inline as
[p. N] markers, and asks Claude the question in a single call_llm()
call. Exists as a cost/quality comparison against the RAG path
(agent.py) -- see the README's "RAG vs. stuff mode" section for the
measured tradeoff.
"""

from __future__ import annotations

from pdf_qa.agent import extraction
from pdf_qa.extraction import Page
from pdf_qa.llm import call_llm

SYSTEM_PROMPT = """
Read the page_text's reference material before answering. The question to be answered is located last, after the reference material. Cite pages where answer was found in the answer.
"""

def ask(question: str, pages: list[Page]) -> str:
    page_text = "Reference Material: "
    for page in pages:
        page_text += f"[p. {page.number}]\n{page.text}\n\n"
    page_text += f"\nQuestion: {question}"
    messages: list[dict] = [{"role": "user", "content": page_text}]

    response = call_llm(messages, system=SYSTEM_PROMPT)
    return extraction(response)
