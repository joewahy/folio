"""Eval script: test the pipeline against two "messy" synthetic PDFs --
a scanned-style (image-only, no text layer) page and a multi-column
layout -- to see whether extraction/chunking/embedding hold up on
anything other than the one clean, well-structured sample.pdf every
other eval in this repo uses.

Not part of the folio package -- a standalone script, not pytest-
discoverable. Tracked in evals/ since this is the evidence behind
whatever "tested on messier documents" claim goes in the README.

Both PDFs are generated locally with fitz, not downloaded -- same
approach as test_injection.py's poisoned PDFs.

Two cases:

1. Scanned-style: text rendered to an image and embedded with NO real
   text layer, simulating what a scanned/photographed page looks like
   to PyMuPDF's extraction. extract_pages() correctly returns an empty
   string for it (expected, documented behavior -- see extraction.py's
   own docstring). The interesting question is what happens downstream:
   chunk_pages() correctly skips empty-text pages, producing zero
   chunks. But build_index() then calls embeddings.embed_texts([]) on
   that empty list, and OpenAI's API rejects an empty input with a 400
   BadRequestError -- an unhandled crash, not a graceful "no text found"
   message. This script catches and reports that crash rather than
   letting it kill the run, since reproducing it IS the point.

2. Multi-column: two side-by-side text columns on one page, each a
   coherent multi-sentence paragraph about a different topic. If
   extraction read line-by-line by vertical position instead of
   respecting column boundaries, the two paragraphs would interleave
   into nonsense. Runs through the full pipeline (both modes) with a
   question specific to one column, to confirm the answer is both
   coherent and comes from the right column.

Costs a small amount of real money for the multi-column half only (the
scanned half crashes before any Claude call happens).

Usage:
    python evals/test_messy_pdfs.py
"""

from __future__ import annotations

import fitz
from dotenv import load_dotenv

import folio.agent as agent
import folio.stuff as stuff
from folio.cli import build_index
from folio.extraction import extract_pages


def make_scanned_pdf(path: str, text: str) -> None:
    """A PDF with no real text layer -- `text` is rendered to an image
    and embedded as a picture, the way a scan or photo of a page would
    come out. get_text() on this returns "", same as a real scan.
    """
    temp_doc = fitz.open()
    temp_page = temp_doc.new_page()
    temp_page.insert_textbox(fitz.Rect(72, 72, 500, 700), text, fontsize=14)
    pix = temp_page.get_pixmap(dpi=150)
    img_bytes = pix.tobytes("png")
    temp_doc.close()

    doc = fitz.open()
    page = doc.new_page()
    page.insert_image(page.rect, stream=img_bytes)
    doc.save(path)
    doc.close()


def make_multicolumn_pdf(path: str, left_text: str, right_text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 72, 290, 750), left_text, fontsize=11)
    page.insert_textbox(fitz.Rect(310, 72, 550, 750), right_text, fontsize=11)
    doc.save(path)
    doc.close()


def test_scanned() -> None:
    print("\n=== scanned-style (image-only) PDF ===")
    path = "scratch/_messy_scanned.pdf"
    make_scanned_pdf(
        path,
        "The mitochondria is the powerhouse of the cell. It generates ATP "
        "through cellular respiration, converting glucose and oxygen into "
        "usable energy.",
    )

    pages = extract_pages(path)
    print(f"extracted text: {pages[0].text!r} (expected: empty string)")

    try:
        build_index(path)
        print("build_index() succeeded -- unexpected, investigate.")
    except Exception as e:
        print(f"build_index() CRASHED: {type(e).__name__}: {e}")
        print(
            "-> zero chunks reach embed_texts([]), which OpenAI's API "
            "rejects outright. Unhandled crash, not a graceful message."
        )


def test_multicolumn() -> None:
    print("\n=== multi-column PDF ===")
    path = "scratch/_messy_multicolumn.pdf"
    left_text = (
        "The water cycle describes how water moves through the environment. "
        "Sunlight causes evaporation from oceans and lakes, turning liquid "
        "water into vapor that rises into the atmosphere. As the vapor "
        "cools, it condenses into clouds. Eventually the water falls back "
        "to earth as precipitation, completing the cycle."
    )
    right_text = (
        "The rock cycle describes how rocks transform between three types. "
        "Igneous rock forms when magma cools and solidifies. Over time, "
        "heat and pressure can transform any rock into metamorphic rock. "
        "Weathering breaks rocks into sediment that compresses into "
        "sedimentary rock over long periods of time."
    )
    make_multicolumn_pdf(path, left_text, right_text)

    pages = extract_pages(path)
    print("extracted text:")
    print(pages[0].text)

    question = "What forms when magma cools and solidifies?"
    print(f"\nQ: {question} (expected: igneous rock, from the right column)")

    chunks, vectors = build_index(path)
    rag_answer = agent.ask(question, chunks, vectors)
    print(f"\n-- RAG answer --\n{rag_answer}")

    stuff_answer = stuff.ask(question, pages)
    print(f"\n-- STUFF answer --\n{stuff_answer}")


if __name__ == "__main__":
    load_dotenv()
    test_scanned()
    test_multicolumn()
