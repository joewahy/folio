"""Eval script: exercise agent.ask()'s tool-call handling without
hitting the Claude or OpenAI APIs.

Not part of the pdf_qa package -- a standalone script, not pytest-
discoverable. Tracked (unlike scratch/, which is gitignored) as the
mocked-structure test backing agent.py's tool-use loop.

This monkeypatches call_llm and search_pdf with fakes, so it costs
nothing and needs no .env. Two scenarios:

1. The fake first response has TWO tool_use blocks in one turn (Claude
   asking two searches at once, e.g. for a two-part question) -- the case
   the multi-tool-call batching is supposed to handle. It prints the
   message ask() actually builds in response, plus an explicit check for
   two things the real API requires: every tool_use_id from the previous
   turn has a matching tool_result, and the tool_result message's content
   is a list (not a bare block).

2. Claude asks to search on every single turn, exhausting
   MAX_TOOL_CALLS without ever reaching "end_turn" -- the case the
   tools-free fallback call is supposed to handle. Checks that ask()
   makes exactly one extra call_llm() invocation after the loop, that
   the extra call doesn't offer search_pdf as a tool, that it requests
   effort="medium", and that ask() actually returns its text (rather
   than None -- the two missing `return`s this scenario would have
   caught).

Usage:
    python evals/test_agent.py
"""

from types import SimpleNamespace

import pdf_qa.agent as agent


def fake_tool_use_response():
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(type="tool_use", id="toolu_01A", name="search_pdf", input={"query": "first question"}),
            SimpleNamespace(type="tool_use", id="toolu_01B", name="search_pdf", input={"query": "second question"}),
        ],
    )


def fake_end_turn_response():
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="Here's your answer. [p. 1]")],
    )


_responses = [fake_tool_use_response(), fake_end_turn_response()]
_calls = []  # captures the `messages` list ask() passed on each call_llm() invocation


def fake_call_llm(messages, **kwargs):
    _calls.append(messages)
    print("--- call_llm invoked with messages: ---")
    for m in messages:
        print(m)
    print()
    return _responses.pop(0)


def fake_search_pdf(query, chunks, vectors, top_k=5):
    return f"[p. 1] fake search result for: {query!r}"


def check_batching(messages):
    """Does the tool_result turn actually satisfy the real API's contract?"""
    assistant_msg = messages[-2]
    user_msg = messages[-1]

    tool_use_ids = {b.id for b in assistant_msg["content"] if b.type == "tool_use"}
    content = user_msg["content"]

    if not isinstance(content, list):
        print(f"FAIL: tool_result message content is a {type(content).__name__}, not a list: {content!r}")
        return

    result_ids = {c["tool_use_id"] for c in content}
    missing = tool_use_ids - result_ids
    if missing:
        print(f"FAIL: no tool_result sent back for tool_use_id(s): {missing}")
    else:
        print("PASS: every tool_use_id from the previous turn has a matching tool_result.")


def fake_tool_use_response_single(i):
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[
            SimpleNamespace(
                type="tool_use",
                id=f"toolu_loop{i}",
                name="search_pdf",
                input={"query": f"round {i} query"},
            ),
        ],
    )


def fake_fallback_response():
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text="Best guess without further searching.")],
    )


# MAX_TOOL_CALLS tool_use responses in a row -- Claude never reaches
# "end_turn" inside the loop -- followed by one end_turn response for the
# tools-free fallback call.
_fallback_responses = [
    fake_tool_use_response_single(i) for i in range(agent.MAX_TOOL_CALLS)
] + [fake_fallback_response()]
_fallback_calls = []  # captures (messages, kwargs) for every call_llm() invocation


def fake_call_llm_exhausting(messages, **kwargs):
    _fallback_calls.append((messages, kwargs))
    return _fallback_responses.pop(0)


def check_fallback(answer, calls):
    """Does ask() actually make one extra tools-free call after MAX_TOOL_CALLS exhausts, and return its text?"""
    expected_calls = agent.MAX_TOOL_CALLS + 1
    if len(calls) != expected_calls:
        print(f"FAIL: expected {expected_calls} call_llm() invocations (MAX_TOOL_CALLS + 1 fallback), got {len(calls)}")
        return

    _, last_kwargs = calls[-1]

    if last_kwargs.get("tools"):
        print(f"FAIL: fallback call still offered a tool: {last_kwargs['tools']!r}")
    else:
        print("PASS: fallback call did not offer search_pdf as a tool.")

    if last_kwargs.get("effort") != "medium":
        print(f"FAIL: fallback call effort was {last_kwargs.get('effort')!r}, expected 'medium'")
    else:
        print("PASS: fallback call requested effort='medium'.")

    if answer != "Best guess without further searching.":
        print(f"FAIL: ask() returned {answer!r}, expected the fallback call's text")
    else:
        print("PASS: ask() returned the fallback call's text (not None).")


if __name__ == "__main__":
    # Scenario 1: two tool calls batched into one turn.
    agent.call_llm = fake_call_llm
    agent.search.search_pdf = fake_search_pdf

    answer = agent.ask("does this handle two tool calls in one turn?", chunks=[], vectors=None)
    print("Final answer:", answer, "\n")

    # _calls[1] is the `messages` list as it stood on the SECOND call_llm()
    # call -- i.e. right after ask() processed the tool_use turn and
    # appended its response, which is exactly what we want to inspect.
    check_batching(_calls[1])

    print()

    # Scenario 2: Claude never stops asking to search -- MAX_TOOL_CALLS
    # exhausts and ask() should fall through to the tools-free fallback.
    agent.call_llm = fake_call_llm_exhausting
    agent.search.search_pdf = fake_search_pdf

    fallback_answer = agent.ask("a question that never gets resolved", chunks=[], vectors=None)
    print("Fallback answer:", fallback_answer, "\n")
    check_fallback(fallback_answer, _fallback_calls)
