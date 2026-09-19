"""PDF text extraction, page by page.

Extraction is kept page-by-page (rather than one big concatenated string)
so every downstream chunk can be tagged with the page number(s) it came
from -- that's what makes citations like "(p. 12)" possible later.
"""

from __future__ import annotations

from dataclasses import dataclass

import fitz  # PyMuPDF


@dataclass
class Page:
    number: int  # 1-indexed, matches how humans refer to pages
    text: str


def extract_pages(pdf_path: str) -> list[Page]:
    """Extract text from every page of a PDF.

    Returns one Page per page in the document, in order. Pages with no
    extractable text (e.g. pure-image scans) come back with text == "".
    """
    doc = fitz.open(pdf_path)
    try:
        return [Page(number=i + 1, text=page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()
