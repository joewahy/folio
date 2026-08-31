"""Eval script: sweep chunking's chunk size (holding overlap fixed)
against the 40-question corpus (evals/corpus.py), to see whether chunk
size actually moves RAG citation accuracy, cost, or latency.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) because
this is what the README's chunk-size numbers actually come from.

Only RAG mode is swept here -- stuff mode never chunks, so chunk size is
irrelevant to it (see eval_modes.py for the RAG-vs-stuff comparison).

Deliberately does NOT touch chunking.py: chunk_pages() hardcodes
size=500, overlap=100 as local variables, not parameters, and making
that configurable is a chunking-algorithm decision left to the project
owner (see chunking.py's own docstring). chunk_pages_sized() below is a
parameterized copy of the exact same sliding-window logic, kept local to
this one experiment instead of changing the real implementation.

Overlap is held fixed at 100 across every size -- chunk size is the only
variable under test, per normal controlled-comparison practice. Sweeping
overlap too would be a second, separate experiment.

Costs real money, and more of it than eval_modes.py: it builds a fresh
RAG index (embeddings) and runs all 40 questions through agent.ask() for
EACH size -- 3 sizes x 40 questions x ~2 calls/question is on the order
of 240 real Claude calls, plus 3 x 4 embedding calls. Set
ANTHROPIC_MODEL=claude-haiku-4-5 in .env first.

Usage:
    python evals/eval_chunk_sizes.py
"""

from __future__ import annotations

import numpy as np
from corpus import DOCUMENTS, require_pdfs
from dotenv import load_dotenv
from eval_modes import (
    add_into,
    cited_pages,
    make_tracking_call_llm,
    new_stats,
    summary_line,
)

import pdf_qa.agent as agent
from pdf_qa import embeddings
from pdf_qa.chunking import Chunk
from pdf_qa.extraction import Page, extract_pages

SIZES = [250, 500, 1000]
OVERLAP = 100  # held fixed -- only `size` is swept


def chunk_pages_sized(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    """Parameterized copy of chunking.chunk_pages()'s windowing logic --
    same algorithm, with size/overlap exposed as arguments instead of
    chunking.py's hardcoded 500/100, so this sweep can vary them without
    touching the real implementation.
    """
    chunks: list[Chunk] = []
    chunk_index = 0

    if overlap < size:
        for page in pages:
            if len(page.text) != 0:
                pieces = []
                curr_index = 0
                while curr_index < len(page.text):
                    if curr_index + size < len(page.text):
                        pieces.append(page.text[curr_index : curr_index + size])
                    else:
                        pieces.append(page.text[curr_index:])
                        break
                    curr_index += size - overlap

                for piece in pieces:
                    chunks.append(
                        Chunk(text=piece, page_number=page.number, chunk_index=chunk_index)
                    )
                    chunk_index += 1

    return chunks


def build_index_sized(pages: list[Page], size: int, overlap: int):
    chunks = chunk_pages_sized(pages, size, overlap)
    vectors = np.array(embeddings.embed_texts([c.text for c in chunks]))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return chunks, vectors


def run_size(size: int, pages_by_doc: dict[str, list[Page]]) -> None:
    print(f"\n{'=' * 70}\n=== chunk size {size} (overlap {OVERLAP}) ===\n{'=' * 70}")
    grand = new_stats()
    grand_hits = grand_asked = 0

    for doc in DOCUMENTS:
        chunks, vectors = build_index_sized(pages_by_doc[doc.name], size, OVERLAP)
        doc_totals = new_stats()
        doc_hits = 0
        print(f"\n--- {doc.name}  ({len(chunks)} chunks) ---")

        for question, expected_pages in doc.questions:
            stats = new_stats()
            agent.call_llm = make_tracking_call_llm(stats)

            answer = agent.ask(question, chunks, vectors)

            ok = bool(expected_pages & cited_pages(answer))
            doc_hits += ok
            add_into(doc_totals, stats)

            want = ",".join(str(p) for p in sorted(expected_pages))
            print(
                f"[{'OK  ' if ok else 'MISS'}] want p.{want:<6} "
                f"{stats['calls']} call(s), {stats['input_tokens']:>7}in/{stats['output_tokens']:>4}out tok, "
                f"{stats['elapsed']:5.1f}s -- {question}"
            )
            print(f"       -> {' '.join(answer.split())}")

        print(summary_line(f"  {doc.name}", doc_hits, len(doc.questions), doc_totals))
        add_into(grand, doc_totals)
        grand_hits += doc_hits
        grand_asked += len(doc.questions)

    print("\n" + summary_line(f"size {size} TOTAL", grand_hits, grand_asked, grand))


if __name__ == "__main__":
    load_dotenv()
    require_pdfs()
    pages_by_doc = {doc.name: extract_pages(doc.path) for doc in DOCUMENTS}

    for size in SIZES:
        run_size(size, pages_by_doc)
