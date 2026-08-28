"""Eval script: exercise two prompt behaviors that have so far only been
reasoned about, never actually verified against real model output.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked in evals/ since this is the evidence behind
whatever claim gets made about SYSTEM_PROMPT/FALLBACK_PROMPT actually
working, not just reading correctly.

1. Abstention: both agent.SYSTEM_PROMPT and stuff.SYSTEM_PROMPT now say
   "say so instead of guessing" if the material doesn't answer the
   question -- checked against BOTH modes, not just agent.py. Ask
   something completely unrelated to sample.pdf's actual content (a
   Java data structures review sheet) and check whether the answer says
   so, rather than confidently answering from Claude's own general
   knowledge. This matters specifically because RAG mode's search_pdf
   has no relevance threshold -- it always returns its top_k=5 chunks no
   matter how irrelevant, so nothing upstream stops the model from being
   handed useless context and trying to answer anyway; stuff mode has no
   such excuse since it sees the entire document either way.

2. Fallback: FALLBACK_PROMPT has never fired in any real eval run in
   this repo -- every real question so far resolved in 2-3 call_llm
   calls, well under MAX_TOOL_CALLS=5. To exercise it without relying
   on an adversarial question and hoping the model cooperates, this
   monkeypatches agent.MAX_TOOL_CALLS to 1 for one call -- the loop's
   single iteration is consumed by the mandatory first search
   (SYSTEM_PROMPT: "Always call search_pdf before answering"), so the
   fallback fires deterministically regardless of the question. This
   confirms it actually took the fallback branch by checking the last
   call_llm invocation's kwargs (no `tools`, effort="medium" -- the
   fallback call's real signature per agent.py), not just by counting
   calls, since a normal 2-call round trip and a 1-search-then-fallback
   round trip both make exactly 2 calls.

Costs a small amount of real money (a handful of real Claude calls).

Usage:
    python evals/test_abstention_and_fallback.py
"""

from __future__ import annotations

from dotenv import load_dotenv

import pdf_qa.agent as agent
import pdf_qa.stuff as stuff
from pdf_qa.cli import build_index
from pdf_qa.extraction import extract_pages
from pdf_qa.llm import call_llm as real_call_llm

PDF_PATH = "scratch/sample.pdf"

ABSTENTION_PHRASES = [
    "n't answer", "n't contain", "n't cover", "n't appear", "n't found",
    "n't mention",  # catches doesn't/don't/can't/couldn't + verb, singular
    # or plural subject ("the passages don't answer" vs "it doesn't")
    "does not", "do not",  # the uncontracted forms of all of the above --
    # missed a real "does not answer this question" during testing
    "no information", "not covered", "not addressed", "not mentioned",
    "no content about", "there is no content", "not in the document",
    "not in this document", "cannot be determined", "can not be determined",
    # A fixed phrase list will always undershoot real model phrasing.
    # This one has now missed twice during actual testing (once on a
    # contraction it didn't expect, once on a fully-spelled-out "does
    # not"), on top of the en-dash and COMPROMISED false positives
    # elsewhere in evals/. Treat a FLAG here as "read the printed
    # answer," never as a reliable negative -- these checks are a
    # first-pass triage, not a source of truth.
]


def make_tracking_call_llm(calls_log: list):
    def tracking_call_llm(messages, **kwargs):
        response = real_call_llm(messages, **kwargs)
        calls_log.append(
            {"has_tools": "tools" in kwargs, "effort": kwargs.get("effort")}
        )
        return response

    return tracking_call_llm


def check_abstention(label: str, answer: str) -> None:
    print(f"\n-- {label} answer --\n{answer}")
    abstained = any(phrase in answer.lower() for phrase in ABSTENTION_PHRASES)
    print(
        f"\n[{'PASS' if abstained else 'FLAG -- check manually'}] {label}: "
        f"abstention phrase {'found' if abstained else 'NOT found'} in answer"
    )


def test_abstention(chunks, vectors, pages) -> None:
    print("\n=== abstention on an unanswerable question (both modes) ===")
    question = "What is the boiling point of liquid nitrogen in Celsius?"
    print(f"Q: {question} (nothing in sample.pdf covers this)")

    agent.call_llm = make_tracking_call_llm([])
    rag_answer = agent.ask(question, chunks, vectors)
    check_abstention("RAG", rag_answer)

    stuff.call_llm = make_tracking_call_llm([])
    stuff_answer = stuff.ask(question, pages)
    check_abstention("STUFF", stuff_answer)


def test_fallback(chunks, vectors) -> None:
    print("\n=== forcing the fallback path with MAX_TOOL_CALLS=1 ===")
    question = "What is functional decomposition?"
    print(
        f"Q: {question} (an easy, normally-answerable question -- forcing "
        "the fallback anyway to see how it behaves for real)"
    )

    original_max = agent.MAX_TOOL_CALLS
    calls_log: list = []
    agent.call_llm = make_tracking_call_llm(calls_log)
    agent.MAX_TOOL_CALLS = 1
    try:
        answer = agent.ask(question, chunks, vectors)
    finally:
        agent.MAX_TOOL_CALLS = original_max

    print(f"\n-- answer --\n{answer}")
    print(f"\ncalls made: {len(calls_log)}")

    took_fallback = (
        bool(calls_log)
        and not calls_log[-1]["has_tools"]
        and calls_log[-1]["effort"] == "medium"
    )
    print(
        f"[{'PASS' if took_fallback else 'FAIL'}] last call matches the "
        "fallback branch's signature (no tools, effort='medium')"
    )

    # "incomplete" alone missed a real answer that said "cut short" instead
    # -- same lesson as ABSTENTION_PHRASES, checking both wordings the
    # prompt itself uses ("may be incomplete... cut short").
    mentions_incomplete = any(
        phrase in answer.lower() for phrase in ("incomplete", "cut short")
    )
    print(
        f"[{'PASS' if mentions_incomplete else 'FLAG -- check manually'}] "
        "answer mentions being incomplete"
    )


if __name__ == "__main__":
    load_dotenv()
    print(f"Indexing {PDF_PATH}...")
    chunks, vectors = build_index(PDF_PATH)
    pages = extract_pages(PDF_PATH)

    test_abstention(chunks, vectors, pages)
    test_fallback(chunks, vectors)
