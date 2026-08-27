"""Eval script: compare RAG mode (agent.ask) against stuff mode
(stuff.ask) on cost, latency, and citation accuracy across a small,
hand-labeled question set tied to scratch/sample.pdf's actual content.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) because
this is what the README's RAG-vs-stuff numbers actually come from.

Costs real money: it builds the RAG index (OpenAI embeddings) once, then
runs every question through both modes for real against the Claude API
-- unlike test_agent.py/test_stuff.py, nothing here is mocked. RAG mode
in particular can make several call_llm round trips per question (the
tool-use loop), so this adds up faster than a single question would.
Consider setting ANTHROPIC_MODEL=claude-sonnet-5 (or claude-haiku-4-5)
in .env before running this -- llm.py defaults to claude-opus-5, which
is the most expensive option and probably overkill for an eval loop.

"Accuracy" here is a cheap heuristic, not a rigorous grader: did the
expected page number show up somewhere in the answer's citations, in
either "p. N"/"pp. N-M" or prose "page N" form (range-aware, since a
real stuff-mode answer cited "[p. 5-6]" for a quote spanning a page
break). It'll catch a mode citing the wrong page or no page at all, but
it says nothing about whether the prose itself is actually correct --
read the printed answers, don't just trust the OK/MISS column.

Usage:
    python evals/eval_modes.py
"""

from __future__ import annotations

import re
import time

from dotenv import load_dotenv

import pdf_qa.agent as agent
import pdf_qa.stuff as stuff
from pdf_qa.cli import build_index
from pdf_qa.extraction import extract_pages
from pdf_qa.llm import call_llm as real_call_llm

PDF_PATH = "scratch/sample.pdf"

# (question, expected page) -- picked against sample.pdf's actual text,
# one per major section of the review sheet.
QUESTIONS = [
    ("What is functional decomposition?", 1),
    ("What is an abstract data type (ADT)?", 2),
    ("What is abstraction in OOP?", 3),
    ("Why would you want to use a Set instead of a List?", 6),
    ("What are the two types of Sets, and how do they differ?", 6),
    ("How does .equals() typically check whether two objects are equal?", 9),
    ("Which classes implement the Map interface?", 10),
    ("Why should exceptions be thrown as early as possible?", 10),
    ("What is encapsulation and how do we achieve it in Java?", 8),
]

# Claude sometimes writes ranges with a typographic en/em dash ("p. 5-6")
# rather than a plain hyphen -- matters, since matching only "-" silently
# drops the second page of a real range instead of erroring loudly.
_PN_RE = re.compile(r"\bpp?\.\s*(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE)
_PAGE_RE = re.compile(r"\bpages?\s+(\d+)(?:\s*[-–—]\s*(\d+))?", re.IGNORECASE)


def cited_pages(answer: str) -> set[int]:
    """Every page number cited in an answer, expanding "N-M" ranges."""
    pages = set()
    for pattern in (_PN_RE, _PAGE_RE):
        for match in pattern.finditer(answer):
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            pages.update(range(start, end + 1))
    return pages


def new_stats() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "elapsed": 0.0}


def make_tracking_call_llm(stats: dict):
    def tracking_call_llm(messages, **kwargs):
        start = time.perf_counter()
        response = real_call_llm(messages, **kwargs)
        stats["elapsed"] += time.perf_counter() - start
        stats["calls"] += 1
        stats["input_tokens"] += response.usage.input_tokens
        stats["output_tokens"] += response.usage.output_tokens
        return response

    return tracking_call_llm


def run_mode(name: str, ask_fn) -> None:
    print(f"\n=== {name} ===")
    totals = new_stats()
    hits = 0

    for question, expected_page in QUESTIONS:
        stats = new_stats()
        # Patching both is harmless -- only the one this mode actually
        # calls does anything, and it keeps this loop mode-agnostic.
        agent.call_llm = make_tracking_call_llm(stats)
        stuff.call_llm = make_tracking_call_llm(stats)

        answer = ask_fn(question)

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
        f"\n{name}: {hits}/{len(QUESTIONS)} correctly cited. "
        f"Totals: {totals['calls']} calls, {totals['input_tokens']} in / "
        f"{totals['output_tokens']} out tokens, {totals['elapsed']:.1f}s"
    )


if __name__ == "__main__":
    load_dotenv()

    print(f"Indexing {PDF_PATH} for RAG mode...")
    chunks, vectors = build_index(PDF_PATH)
    pages = extract_pages(PDF_PATH)

    run_mode("RAG", lambda q: agent.ask(q, chunks, vectors))
    run_mode("STUFF", lambda q: stuff.ask(q, pages))
