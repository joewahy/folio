"""Eval script: exercise stuff.ask()'s message-building without hitting
the Claude API.

Not part of the folio package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) as the
mocked-structure test backing stuff.py.

Unlike test_search.py, this doesn't need .env or real API credits: it
monkeypatches call_llm with a fake that just captures whatever `messages`
stuff.ask() built and returns a canned end_turn response. The point isn't
"does Claude answer well" (that needs a real call, see below) -- it's
"did we actually send Claude what we think we sent it", which is exactly
where the message-role and ordering bugs upstream lived.

Runs stuff.ask() twice, once with cache=True and once with cache=False --
eval_modes.py needs both to compare cached vs. uncached cost, so both
message shapes are checked here.

Checks:
- exactly one message, role "user" (no accidental multi-turn/assistant
  content)
- cache=True: content is two blocks, reference material (cached) then
  question (uncached) -- material first, question last, with a cache
  breakpoint on the material only
- cache=False: content is a single string, reference material followed by
  the question (the pre-caching shape, still used as the uncached baseline)
- both fake pages' text and "[p. N]" markers show up in the reference
  material (whichever shape it's in)
- the question text shows up separately from the reference material, not
  inside it
- page 1's text and page 2's "[p. 2]" marker aren't glued together with
  no whitespace between them

Once this passes, run a real check against scratch/sample.pdf, e.g.:
    python -c "
    from dotenv import load_dotenv; load_dotenv()
    from folio.extraction import extract_pages
    from folio.stuff import ask
    pages = extract_pages('scratch/sample.pdf')
    print(ask('<a question you know the answer to>', pages))
    "

Usage:
    python evals/test_stuff.py
"""

from types import SimpleNamespace

import folio.stuff as stuff
from folio.extraction import Page


def fake_end_turn_response():
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="Here's your answer. [p. 1]")],
    )


_calls = []  # captures the `messages` list ask() passed to call_llm()


def fake_call_llm(messages, **kwargs):
    _calls.append(messages)
    print("--- call_llm invoked with messages: ---")
    for m in messages:
        print(m)
    print()
    return fake_end_turn_response()


def check_message_shape(messages, question, pages, cached: bool):
    ok = True

    if len(messages) != 1:
        print(f"FAIL: expected exactly 1 message, got {len(messages)}")
        return
    msg = messages[0]
    if msg["role"] != "user":
        print(f"FAIL: expected role 'user', got {msg['role']!r}")
        ok = False

    content = msg["content"]

    if cached:
        if len(content) != 2:
            print(f"FAIL: expected 2 content blocks, got {len(content)}")
            return
        material, question_block = content[0], content[1]

        if material.get("cache_control") != {"type": "ephemeral"}:
            print("FAIL: reference material block missing cache_control")
            ok = False
        if "cache_control" in question_block:
            print("FAIL: question block should not be cached (it changes every call)")
            ok = False

        material_text = material["text"]
        question_text = question_block["text"]
    else:
        if not isinstance(content, str):
            print(f"FAIL: expected content to be a plain string, got {type(content)}")
            return
        material_text = content
        question_text = content

    for page in pages:
        marker = f"[p. {page.number}]"
        if marker not in material_text:
            print(f"FAIL: missing page marker {marker!r}")
            ok = False
        if page.text not in material_text:
            print(f"FAIL: missing text for page {page.number}")
            ok = False

    if question not in question_text:
        print("FAIL: question text not found")
        ok = False
    if cached and question in material_text:
        print("FAIL: question leaked into the cached reference-material block")
        ok = False

    page1_end = material_text.find(pages[0].text) + len(pages[0].text)
    page2_marker = material_text.find(f"[p. {pages[1].number}]")
    if page2_marker != -1 and page2_marker > page1_end:
        gap = material_text[page1_end:page2_marker]
        if not gap.strip("\n"):
            pass  # only whitespace between pages -- good
        else:
            print(f"FAIL: unexpected non-whitespace between pages: {gap!r}")
            ok = False
    elif page2_marker != -1 and material_text[page1_end:page2_marker] == "":
        print("FAIL: page 1 text and page 2 marker are glued together with no separator")
        ok = False

    if not cached:
        # the uncached shape puts the question after the material, not before
        last_marker_idx = material_text.rfind(f"[p. {pages[-1].number}]")
        if material_text.find(question) < last_marker_idx:
            print("FAIL: question appears before reference material, not after")
            ok = False

    if ok:
        print(f"PASS: message shape looks right ({'cached' if cached else 'uncached'}).")


if __name__ == "__main__":
    stuff.call_llm = fake_call_llm

    pages = [
        Page(number=1, text="The sky is blue because of Rayleigh scattering."),
        Page(number=2, text="Water boils at 100 degrees Celsius at sea level."),
    ]
    question = "Why is the sky blue?"

    _calls.clear()
    answer = stuff.ask(question, pages, cache=True)
    print("Final answer (cached):", answer, "\n")
    check_message_shape(_calls[0], question, pages, cached=True)

    _calls.clear()
    answer = stuff.ask(question, pages, cache=False)
    print("Final answer (uncached):", answer, "\n")
    check_message_shape(_calls[0], question, pages, cached=False)
