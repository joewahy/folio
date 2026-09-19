"""Eval script: exercise stuff.ask()'s message-building without hitting
the Claude API.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) as the
mocked-structure test backing stuff.py.

Unlike test_search.py, this doesn't need .env or real API credits: it
monkeypatches call_llm with a fake that just captures whatever `messages`
stuff.ask() built and returns a canned end_turn response. The point isn't
"does Claude answer well" (that needs a real call, see below) -- it's
"did we actually send Claude what we think we sent it", which is exactly
where the message-role and ordering bugs upstream lived.

Checks:
- exactly one message, role "user" (no accidental multi-turn/assistant
  content)
- content is two blocks: reference material (cached) then question
  (uncached) -- matches stuff.py's documented ordering: material first,
  question last, with a cache breakpoint on the material only
- both fake pages' text and "[p. N]" markers show up in the reference block
- the question text shows up in the question block, not the reference block
- page 1's text and page 2's "[p. 2]" marker aren't glued together with
  no whitespace between them

Once this passes, run a real check against scratch/sample.pdf, e.g.:
    python -c "
    from dotenv import load_dotenv; load_dotenv()
    from pdf_qa.extraction import extract_pages
    from pdf_qa.stuff import ask
    pages = extract_pages('scratch/sample.pdf')
    print(ask('<a question you know the answer to>', pages))
    "

Usage:
    python evals/test_stuff.py
"""

from types import SimpleNamespace

import pdf_qa.stuff as stuff
from pdf_qa.extraction import Page


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


def check_message_shape(messages, question, pages):
    ok = True

    if len(messages) != 1:
        print(f"FAIL: expected exactly 1 message, got {len(messages)}")
        ok = False
    else:
        msg = messages[0]
        if msg["role"] != "user":
            print(f"FAIL: expected role 'user', got {msg['role']!r}")
            ok = False

        content = msg["content"]

        if len(content) != 2:
            print(f"FAIL: expected 2 content blocks, got {len(content)}")
            ok = False
            content = None

        if content is not None:
            material, question_block = content[0], content[1]

            if material.get("cache_control") != {"type": "ephemeral"}:
                print("FAIL: reference material block missing cache_control")
                ok = False
            if "cache_control" in question_block:
                print("FAIL: question block should not be cached (it changes every call)")
                ok = False

            material_text = material["text"]
            question_text = question_block["text"]

            for page in pages:
                marker = f"[p. {page.number}]"
                if marker not in material_text:
                    print(f"FAIL: missing page marker {marker!r}")
                    ok = False
                if page.text not in material_text:
                    print(f"FAIL: missing text for page {page.number}")
                    ok = False

            if question not in question_text:
                print("FAIL: question text not found in question block")
                ok = False
            if question in material_text:
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

    if ok:
        print("PASS: message shape looks right.")


if __name__ == "__main__":
    stuff.call_llm = fake_call_llm

    pages = [
        Page(number=1, text="The sky is blue because of Rayleigh scattering."),
        Page(number=2, text="Water boils at 100 degrees Celsius at sea level."),
    ]
    question = "Why is the sky blue?"

    answer = stuff.ask(question, pages)
    print("Final answer:", answer, "\n")

    check_message_shape(_calls[0], question, pages)
