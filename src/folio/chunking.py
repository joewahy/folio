"""Chunking: splitting extracted page text into pieces before embedding.

chunk_pages() below implements fixed-size chunking with overlap, computed
per page (chunks never span a page boundary). Decision space considered:

- Chunk size vs. overlap. Bigger chunks give the embedding more context per
  vector (better semantic matches for vague queries) but dilute precision
  and cost more tokens once retrieved. Overlap between consecutive chunks
  avoids splitting a sentence/idea across a chunk boundary, at the cost of
  redundant embeddings.
- Splitting unit. Fixed character/token count is simple but can cut
  mid-sentence. Splitting on paragraph or sentence boundaries is more
  "semantically clean" but produces uneven chunk sizes.
- Page boundaries. Do you let a chunk span two pages? If you do, a single
  chunk maps to a page range instead of one page number, which complicates
  citations. If you don't, you lose context that's split across a page
  break (e.g. a paragraph continuing onto the next page).
- What metadata each chunk needs. At minimum you'll want enough to cite it
  back to the user (which page(s)) and enough to embed it (the text
  itself) -- see the Chunk dataclass below.
"""

from __future__ import annotations

from dataclasses import dataclass

from folio.extraction import Page


@dataclass
class Chunk:
    text: str
    page_number: int
    chunk_index: int


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    chunk_index = 0
    size = 500
    overlap = 100

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
                        Chunk(
                            text=piece, page_number=page.number, chunk_index=chunk_index
                        )
                    )
                    chunk_index += 1

    return chunks
