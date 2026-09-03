"""Streamlit UI for pdf-qa.

A thin wrapper, nothing more: it uploads a PDF, builds the index once, and
calls the exact same `agent.ask()` / `stuff.ask()` the CLI does. No
retrieval, chunking, or prompt logic lives here -- if the terminal REPL
answers a question a certain way, so does this.

Run locally:

    pip install -e ".[ui]"
    streamlit run streamlit_app.py

Deploy (Streamlit Community Cloud): point it at this file and set
ANTHROPIC_API_KEY and OPENAI_API_KEY (plus optional ANTHROPIC_MODEL) in the
app's Secrets -- the shim below copies them into os.environ so the same
`os.environ[...]` reads in llm.py / embeddings.py work unchanged.
"""

from __future__ import annotations

import hashlib
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from pdf_qa import agent, stuff
from pdf_qa.cli import build_index
from pdf_qa.extraction import extract_pages

# Keys come from a local .env in dev; on Streamlit Cloud they arrive via
# st.secrets instead. Copy whatever's there into os.environ so the rest of
# the package (which only ever reads os.environ) doesn't need to know the
# difference. Reading st.secrets with no secrets file configured raises.
load_dotenv()
try:
    _secrets = dict(st.secrets)
except StreamlitSecretNotFoundError:
    _secrets = {}
for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_MODEL"):
    if _k not in os.environ and _k in _secrets:
        os.environ[_k] = str(_secrets[_k])

MODELS = ["claude-sonnet-5", "claude-haiku-4-5", "claude-opus-5"]


def _write_temp_pdf(pdf_bytes: bytes) -> str:
    """extraction/build_index take a path, not bytes -- stage the upload."""
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as f:
        f.write(pdf_bytes)
    return path


# Both builders are keyed on the PDF's content hash (the leading str arg);
# the raw bytes are passed underscore-prefixed so Streamlit doesn't try to
# hash them itself. Net effect: upload a PDF once, pay the embedding /
# extraction cost once, no matter how many questions or reruns follow.
@st.cache_resource(show_spinner=False)
def rag_index(pdf_sha: str, _pdf_bytes: bytes):
    path = _write_temp_pdf(_pdf_bytes)
    try:
        return build_index(path)  # (chunks, vectors); raises ValueError if no text
    finally:
        os.unlink(path)


@st.cache_resource(show_spinner=False)
def stuff_pages(pdf_sha: str, _pdf_bytes: bytes):
    path = _write_temp_pdf(_pdf_bytes)
    try:
        return extract_pages(path)
    finally:
        os.unlink(path)


st.set_page_config(page_title="pdf-qa", page_icon="📄", layout="centered")
st.title("pdf-qa")
st.caption(
    "Ask questions about a PDF. **RAG** searches over chunks; **stuff** puts the "
    "whole document in context. Answers cite the page they came from."
)

with st.sidebar:
    st.subheader("Settings")
    mode = st.radio(
        "Mode",
        ["rag", "stuff"],
        format_func=lambda m: {"rag": "RAG (search over chunks)", "stuff": "Stuff (whole PDF in context)"}[m],
        help="Same two modes as `pdf-qa --mode rag|stuff`.",
    )
    _env_model = os.environ.get("ANTHROPIC_MODEL")
    model = st.selectbox(
        "Claude model",
        MODELS,
        index=MODELS.index(_env_model) if _env_model in MODELS else 0,
        help="The CLI defaults to claude-opus-5. Cheaper models are fine for a demo.",
    )
    os.environ["ANTHROPIC_MODEL"] = model

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai = bool(os.environ.get("OPENAI_API_KEY"))
    st.write(f"Anthropic key: {'Connected' if has_anthropic else 'Not Connected'}")
    st.write(f"OpenAI key: {'Connected' if has_openai else 'Not Connected'}  \n<small>(embeddings, RAG only)</small>", unsafe_allow_html=True)
    if st.button("Clear chat"):
        st.session_state.messages = []

if not has_anthropic or (mode == "rag" and not has_openai):
    st.warning(
        "Missing API key(s). Set `ANTHROPIC_API_KEY`"
        + ("" if mode == "stuff" else " and `OPENAI_API_KEY`")
        + " in `.env` (local) or the app's Secrets (deployed)."
    )
    st.stop()

uploaded = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
if uploaded is None:
    st.info("Upload a PDF to start.")
    st.stop()

pdf_bytes = uploaded.getvalue()
pdf_sha = hashlib.sha256(pdf_bytes).hexdigest()

# A new PDF invalidates the running conversation -- the old answers were
# about a different document.
if st.session_state.get("pdf_sha") != pdf_sha:
    st.session_state.pdf_sha = pdf_sha
    st.session_state.messages = []

try:
    if mode == "rag":
        with st.spinner("Indexing PDF (extract → chunk → embed)…"):
            chunks, vectors = rag_index(pdf_sha, pdf_bytes)
        st.caption(f"`{uploaded.name}` — {len(chunks)} chunks indexed")
    else:
        pages = stuff_pages(pdf_sha, pdf_bytes)
        st.caption(f"`{uploaded.name}` — {len(pages)} pages, no index (stuff mode)")
except ValueError as e:
    # build_index raises this for a scanned / image-only PDF (no text layer).
    st.error(str(e))
    st.stop()

st.session_state.setdefault("messages", [])
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if question := st.chat_input("Ask about the PDF…"):
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        spinner = "Searching the PDF…" if mode == "rag" else "Reading the whole PDF…"
        try:
            with st.spinner(spinner):
                if mode == "rag":
                    answer = agent.ask(question, chunks, vectors)
                else:
                    answer = stuff.ask(question, pages)
        except Exception as e:  # noqa: BLE001 -- show any API/runtime error in the chat, don't crash the page
            answer = f"**Error:** {type(e).__name__}: {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
