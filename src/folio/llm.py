"""Wrapper around the Claude Messages API.

Mechanical only: given a list of messages and (optionally) tool
definitions, call the API and return the raw response. Deciding whether
Claude should call a tool, executing it, and feeding the result back is
the agent loop's job -- see agent.py.
"""

from __future__ import annotations

import os

from anthropic import Anthropic
from anthropic.types import Message

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def _model() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-opus-5")


def call_llm(
    messages: list[dict],
    *,
    system: str | None = None,
    tools: list[dict] | None = None,
    max_tokens: int = 8192,
    effort: str | None = None,
) -> Message:
    """Send messages (and optional tool definitions) to Claude.

    Returns the raw anthropic Message object. Check `response.stop_reason`
    ("tool_use" vs "end_turn") and walk `response.content` to see whether
    Claude answered directly or is asking to call a tool -- that branching
    lives in the agent loop, not here.

    Note: on Claude Opus 5, thinking is on by default (we don't pass a
    `thinking` param), and max_tokens caps thinking + response text
    together -- raise it if you see truncated answers. `effort`
    ("low"/"medium"/"high"/"xhigh"/"max") controls how much of that budget
    thinking uses; omit it to get the default ("high").
    """
    client = _get_client()
    kwargs: dict = {
        "model": _model(),
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    if effort:
        kwargs["output_config"] = {"effort": effort}

    return client.messages.create(**kwargs)
