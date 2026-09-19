"""Eval script: compare RAG mode (agent.ask) against stuff mode
(stuff.ask), cached and uncached, on cost, latency, and citation accuracy
across the 40-question corpus (evals/corpus.py) -- four documents of
different shape, ten hand-labeled questions each.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) because
this is what the README's RAG-vs-stuff numbers actually come from.

Costs real money: it builds a RAG index (OpenAI embeddings) for each of
the four PDFs once, then runs every question through three modes for real
against the Claude API -- nothing here is mocked. RAG mode can make
several call_llm round trips per question (the tool-use loop); stuff mode
resends the whole document every question, so the 76-page NIST PDF in
particular is token-heavy in stuff mode -- run twice, once with
stuff.ask(cache=False) as the uncached baseline and once with
cache=True, since the ten questions per document run back-to-back
(within the cache's 5-minute TTL), which is exactly the pattern
cache_control benefits from. Set ANTHROPIC_MODEL=claude-haiku-4-5 (or
claude-sonnet-5) in .env before running -- llm.py defaults to
claude-opus-5, overkill for an eval loop.

"Accuracy" here is a cheap heuristic, not a rigorous grader: did any of
a question's labeled pages show up in the answer's citations, in "p. N"/
"pp. N-M" or prose "page N" form (range-aware). It catches a mode citing
the wrong page or no page at all, but says nothing about whether the
prose itself is correct -- read the printed answers, don't just trust
the OK/MISS column.

Set EVAL_DOCS to a comma-separated subset of document names (see
corpus.py's DOCUMENTS for the four names) to run a cheaper sanity check
before paying for the full corpus -- e.g. the two small synthetic docs
instead of all four. Results from a scoped run aren't the measured
numbers this README quotes (those come from the full corpus, unscoped);
it's for spot-checking a change cheaply before committing to the real run.

Usage:
    python evals/eval_modes.py
    EVAL_DOCS=notes_primer,thermostat_manual python evals/eval_modes.py
"""

from __future__ import annotations

import os
import re
import time

from corpus import DOCUMENTS, require_pdfs
from dotenv import load_dotenv

import pdf_qa.agent as agent
import pdf_qa.stuff as stuff
from pdf_qa.cli import build_index
from pdf_qa.extraction import extract_pages
from pdf_qa.llm import call_llm as real_call_llm

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
    return {
        "calls": 0,
        "input_tokens": 0,
        "cache_write_tokens": 0,
        "cache_read_tokens": 0,
        "output_tokens": 0,
        "elapsed": 0.0,
    }


def make_tracking_call_llm(stats: dict):
    def tracking_call_llm(messages, **kwargs):
        start = time.perf_counter()
        response = real_call_llm(messages, **kwargs)
        stats["elapsed"] += time.perf_counter() - start
        stats["calls"] += 1
        # cache_creation/cache_read tokens are billed at different rates than
        # a plain input token (~1.25x and ~0.1x respectively) and the API
        # reports them separately from input_tokens -- lumping them together
        # would hide exactly what a cached vs. uncached comparison needs to see.
        stats["input_tokens"] += response.usage.input_tokens
        stats["cache_write_tokens"] += response.usage.cache_creation_input_tokens
        stats["cache_read_tokens"] += response.usage.cache_read_input_tokens
        stats["output_tokens"] += response.usage.output_tokens
        return response

    return tracking_call_llm


def add_into(totals: dict, stats: dict) -> None:
    for key in totals:
        totals[key] += stats[key]


def summary_line(label: str, hits: int, asked: int, totals: dict) -> str:
    return (
        f"{label}: {hits}/{asked} correctly cited. "
        f"{totals['calls']} calls, {totals['input_tokens']} in "
        f"(+{totals['cache_write_tokens']} cache write / "
        f"{totals['cache_read_tokens']} cache read) / "
        f"{totals['output_tokens']} out tokens, {totals['elapsed']:.1f}s"
    )


def run_mode(name: str, ask_for) -> None:
    """ask_for(doc) -> (question -> answer) for one document."""
    print(f"\n{'=' * 70}\n=== {name} ===\n{'=' * 70}")
    grand = new_stats()
    grand_hits = grand_asked = 0

    for doc in DOCUMENTS:
        ask_fn = ask_for(doc)
        doc_totals = new_stats()
        doc_hits = 0
        print(f"\n--- {doc.name}  ({doc.shape}) ---")

        for question, expected_pages in doc.questions:
            stats = new_stats()
            agent.call_llm = make_tracking_call_llm(stats)
            stuff.call_llm = make_tracking_call_llm(stats)

            answer = ask_fn(question)

            ok = bool(expected_pages & cited_pages(answer))
            doc_hits += ok
            add_into(doc_totals, stats)

            want = ",".join(str(p) for p in sorted(expected_pages))
            print(
                f"[{'OK  ' if ok else 'MISS'}] want p.{want:<6} "
                f"{stats['calls']} call(s), {stats['input_tokens']:>7}in "
                f"(+{stats['cache_write_tokens']:>6}cw/{stats['cache_read_tokens']:>6}cr)"
                f"/{stats['output_tokens']:>4}out tok, "
                f"{stats['elapsed']:5.1f}s -- {question}"
            )
            # Print the answer too: OK/MISS is a substring heuristic and has
            # been wrong in both directions before (the injection-substring
            # and en-dash bugs in the README) -- the column is not the eval.
            print(f"       -> {' '.join(answer.split())}")

        print(summary_line(f"  {doc.name}", doc_hits, len(doc.questions), doc_totals))
        add_into(grand, doc_totals)
        grand_hits += doc_hits
        grand_asked += len(doc.questions)

    print("\n" + summary_line(f"{name} TOTAL", grand_hits, grand_asked, grand))


if __name__ == "__main__":
    load_dotenv()
    require_pdfs()

    doc_filter = os.environ.get("EVAL_DOCS")
    if doc_filter:
        wanted = {name.strip() for name in doc_filter.split(",")}
        unknown = wanted - {d.name for d in DOCUMENTS}
        if unknown:
            raise ValueError(
                f"EVAL_DOCS names not in corpus.DOCUMENTS: {sorted(unknown)}"
            )
        DOCUMENTS = [d for d in DOCUMENTS if d.name in wanted]

    print(f"Documents: {[d.name for d in DOCUMENTS]}")
    print("Building RAG indexes (OpenAI embeddings, once per document)...")
    indexes = {}
    pages = {}
    for doc in DOCUMENTS:
        print(f"  {doc.name} ...")
        indexes[doc.name] = build_index(doc.path)
        pages[doc.name] = extract_pages(doc.path)

    run_mode(
        "RAG",
        lambda doc: (lambda q: agent.ask(q, *indexes[doc.name])),
    )
    run_mode(
        "STUFF (uncached)",
        lambda doc: (lambda q: stuff.ask(q, pages[doc.name], cache=False)),
    )
    run_mode(
        "STUFF (cached)",
        lambda doc: (lambda q: stuff.ask(q, pages[doc.name], cache=True)),
    )
