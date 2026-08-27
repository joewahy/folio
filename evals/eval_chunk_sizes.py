"""Eval script: sweep chunking's chunk size (holding overlap fixed)
against the same question set as eval_modes.py, to see whether chunk
size actually moves RAG citation accuracy/cost/latency on
scratch/sample.pdf.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) because
this is what the README's chunk-size numbers actually come from.

Only RAG mode is swept here -- stuff mode never chunks, so chunk size is
irrelevant to it (see eval_modes.py for the RAG-vs-stuff comparison).

Deliberately does NOT touch chunking.py: chunk_pages() hardcodes
size=500, overlap=100 as local variables, not parameters, and making
that configurable is a chunking-algorithm decision left to you (see
chunking.py's own docstring). chunk_pages_sized() below is a
parameterized copy of the exact same sliding-window logic, kept local
to this one experiment instead of changing the real implementation.

Overlap is held fixed at 100 across every size in the sweep -- chunk
size is the only variable under test, per normal controlled-comparison
practice. Sweeping overlap too would be a second, separate experiment.

Costs real money, more than eval_modes.py's single run: builds a full
RAG index (embeddings) and runs all 9 questions through agent.ask() for
EACH size -- 3 sizes x 9 questions x ~2 calls/question = ~54 real Claude
calls, plus 3 embedding calls (one per size, batched).

Usage:
    python evals/eval_chunk_sizes.py
"""

from __future__ import annotations

import numpy as np
from dotenv import load_dotenv

import pdf_qa.agent as agent
from eval_modes import QUESTIONS, cited_pages, make_tracking_call_llm, new_stats
from pdf_qa import embeddings
from pdf_qa.chunking import Chunk
from pdf_qa.extraction import Page, extract_pages

PDF_PATH = "scratch/sample.pdf"
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


def run_size(size: int, pages: list[Page]) -> None:
    chunks, vectors = build_index_sized(pages, size, OVERLAP)
    print(f"\n=== chunk size {size} (overlap {OVERLAP}, {len(chunks)} chunks) ===")

    totals = new_stats()
    hits = 0

    for question, expected_page in QUESTIONS:
        stats = new_stats()
        agent.call_llm = make_tracking_call_llm(stats)

        answer = agent.ask(question, chunks, vectors)

        ok = expected_page in cited_pages(answer)
        hits += ok
        for key in totals:
            totals[key] += stats[key]

        print(
            f"[{'OK  ' if ok else 'MISS'}] expected p.{expected_page:<3} "
            f"{stats['calls']} call(s), {stats['input_tokens']:>6}in/{stats['output_tokens']:>4}out tok, "
            f"{stats['elapsed']:5.1f}s -- {question}"
        )

    print(
        f"\nsize {size}: {hits}/{len(QUESTIONS)} correctly cited. "
        f"Totals: {totals['calls']} calls, {totals['input_tokens']} in / "
        f"{totals['output_tokens']} out tokens, {totals['elapsed']:.1f}s"
    )


if __name__ == "__main__":
    load_dotenv()
    pages = extract_pages(PDF_PATH)

    for size in SIZES:
        run_size(size, pages)
