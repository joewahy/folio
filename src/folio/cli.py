"""CLI entry point: `folio <path-to-pdf>` extracts the PDF, builds a
search index, and drops you into a Q&A loop -- or, with `--mode stuff`,
skips the index entirely and stuffs the whole document into context
instead (see stuff.py).
"""

from __future__ import annotations

import argparse

import numpy as np
from dotenv import load_dotenv

from folio import agent, chunking, embeddings, extraction, stuff


def build_index(pdf_path: str) -> tuple[list, np.ndarray]:
    """Extract, chunk, and embed a PDF once up front.

    Embedding every chunk eagerly at startup (rather than lazily, only
    for chunks the agent's search actually touches) is the simplest
    design and the one wired in here -- cheap at this scale. Revisit if
    you start feeding in much larger documents.

    Vectors are normalized to unit length here, once, so search.py's
    search_pdf() can score a query against the whole corpus with a
    single dot product instead of re-normalizing every chunk on every
    question.
    """
    pages = extraction.extract_pages(pdf_path)
    chunks = chunking.chunk_pages(pages)
    if not chunks:
        # embed_texts([]) would otherwise reach OpenAI's API and fail
        # with a raw 400 BadRequestError ("input cannot be an empty
        # array") -- confirmed via evals/test_messy_pdfs.py against a
        # scanned/image-only PDF. Fail here instead, with a message that
        # actually explains what's wrong.
        raise ValueError(
            f"No extractable text found in {pdf_path!r}. This is likely a "
            "scanned or image-only PDF -- folio reads text directly from "
            "the PDF and does not do OCR."
        )
    vectors = np.array(embeddings.embed_texts([chunk.text for chunk in chunks]))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors


def qa_loop(pdf_path: str) -> None:
    print(f"Indexing {pdf_path}...")
    chunks, vectors = build_index(pdf_path)
    print(f"Ready. {len(chunks)} chunks indexed. Ask a question (Ctrl+D to quit).\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        answer = agent.ask(question, chunks, vectors)
        print(f"\n{answer}\n")


def stuff_qa_loop(pdf_path: str) -> None:
    print(f"Extracting {pdf_path}...")
    pages = extraction.extract_pages(pdf_path)
    print(f"Ready. {len(pages)} pages loaded, no index needed. Ask a question (Ctrl+D to quit).\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue

        answer = stuff.ask(question, pages)
        print(f"\n{answer}\n")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Agentic PDF Q&A")
    parser.add_argument("pdf_path", help="Path to the PDF to ask questions about")
    parser.add_argument(
        "--mode",
        choices=["rag", "stuff"],
        default="rag",
        help="rag: agentic search over chunks (default). "
        "stuff: whole PDF in context, no retrieval, no embeddings.",
    )
    args = parser.parse_args()

    if args.mode == "stuff":
        stuff_qa_loop(args.pdf_path)
    else:
        qa_loop(args.pdf_path)


if __name__ == "__main__":
    main()
