# pdf-qa

Ask questions about a PDF. Instead of a fixed retrieve-then-generate
pipeline, Claude gets a `search_pdf` tool and decides for itself whether
and how many times to call it before answering.

## Setup

```bash
pip install -e .
cp .env.example .env
```

Fill in `.env` with an [Anthropic API key](https://console.anthropic.com/settings/keys) (the LLM) and an [OpenAI API key](https://platform.openai.com/api-keys) (embeddings only).

## Usage

```bash
pdf-qa path/to/your.pdf
```

Drops you into a `>` prompt. Ask questions, `Ctrl+D` to quit.

## How it works

1. `extraction.py` pulls text out of the PDF, page by page.
2. `chunking.py` splits each page into overlapping chunks.
3. `embeddings.py` embeds every chunk once, up front.
4. You ask a question. Claude (`agent.py`) decides whether to call
   `search_pdf` (`search.py`), which does a cosine-similarity lookup over
   the chunk embeddings and returns the most relevant passages, page
   numbers included.
5. Claude answers, citing the pages it used. It can call `search_pdf`
   more than once per question if it needs to.

## Project layout

```
src/pdf_qa/
├── extraction.py   PDF -> per-page text (PyMuPDF)
├── chunking.py     page text -> overlapping chunks
├── embeddings.py   OpenAI embedding wrapper
├── search.py       search_pdf tool: cosine similarity over chunk embeddings
├── llm.py          Claude Messages API wrapper
├── agent.py        the tool-use loop -- decides when to search, builds the final answer
├── stuff.py        whole-PDF-in-context mode: no retrieval, no chunking, no embeddings
└── cli.py          `pdf-qa <pdf>` entry point (--mode rag | stuff)
```

## Status

Both modes work end to end: `--mode rag` (default) and `--mode stuff`.

## RAG vs. stuff mode

`stuff.py` exists to answer a concrete question: is the retrieval step actually
worth its complexity, or would just pasting the whole document into context do
just as well? `scratch/eval_modes.py` runs both modes against the same 9
hand-labeled questions over `scratch/sample.pdf` (a 12-page document), checking
whether each answer cites the page the answer actually came from.

| | RAG | Stuff |
|---|---|---|
| Citation accuracy | 9/9 | 9/9 |
| API calls | 18 (2/question) | 9 (1/question) |
| Total tokens | 23,032 (19,271 in / 3,761 out) | 42,264 (39,949 in / 2,315 out) |
| Wall time | 54.3s | 32.9s |

On this document, accuracy is a tie. The real tradeoff is cost vs. latency:
RAG uses **~55% of the tokens** stuff mode does, at the cost of **~1.65x the
wall-clock time** (two API round trips per question -- a search call, then an
answer call -- instead of one). Stuff mode's token cost scales with document
size per question; RAG's stays roughly flat, bounded by how many chunks it
retrieves. On a 12-page document the gap is already meaningful; on a much
larger one, stuff mode's cost would keep climbing while RAG's wouldn't.

Reproduce with `python scratch/eval_modes.py` (needs `.env` set up; makes real
API calls, so it costs a small amount to run).

## Chunk size sweep

`chunking.py` defaults to 500-character chunks with 100-character overlap.
`scratch/eval_chunk_sizes.py` checks whether that's actually a good default by
running RAG mode's same 9 questions at chunk sizes 250, 500, and 1000
(overlap held fixed at 100, so size is the only variable).

| Chunk size | Chunks | Citation accuracy | API calls | Total tokens | Wall time |
|---|---|---|---|---|---|
| 250 | 83 | 9/9 | 20 | 29,632 | 83.7s |
| 500 | 35 | 9/9 | 18 | 30,295 | 83.7s |
| 1000 | 19 | 9/9 | 18 | 40,713 | 91.6s |

Accuracy didn't move at all across sizes on this document -- the interesting
result is cost. **1000 is the clear loser**, costing ~35% more tokens than
either smaller size, since each retrieved chunk carries roughly twice the raw
text, resent as input tokens on every round trip. **250 and 500 land almost
identically on cost, for different reasons**: 250 has more than double the
chunks of 500, but 2 of the 9 questions needed a third `call_llm` round trip
at that size instead of two -- the extra retrieval round trip resends the
whole growing conversation as input tokens, erasing the savings smaller
chunks should have produced. Net result: 500 isn't an arbitrary choice, it's
empirically the sweet spot between "too small, sometimes needs a second
search" and "too big, bloats every retrieval," while matching both on
accuracy.

Reproduce with `python scratch/eval_chunk_sizes.py` (same cost caveat as
above, roughly 3x the spend since it repeats the sweep across three sizes).
