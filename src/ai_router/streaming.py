"""Helpers for turning a growing DOM text into streamed deltas."""

from __future__ import annotations

from collections.abc import AsyncIterator

# A streamed piece: ("content" | "reasoning", text appended since last piece).
Delta = tuple[str, str]

# Control marker a stream yields once it is past setup (logged in, prompt about
# to be sent): the HTTP layer can commit to a 200 response at that point.
READY: Delta = ("ready", "")


def prefix_delta(sent: str, full: str) -> tuple[str, str]:
    """Return (new_sent, delta) for a text that is expected to only grow.

    Chat UIs append tokens, so ``full`` normally extends ``sent`` and the delta
    is the new suffix. If the renderer rewrites earlier text (non-append), we
    emit nothing rather than send a delta that contradicts what the client
    already has; the caller's final reconcile picks up any remaining suffix.
    """
    if full.startswith(sent):
        return full, full[len(sent):]
    return sent, ""


async def collect(stream: AsyncIterator[Delta]) -> tuple[str, str | None]:
    """Drain a delta stream into (content, reasoning-or-None)."""
    content: list[str] = []
    reasoning: list[str] = []
    async for kind, text in stream:
        if kind == "reasoning":
            reasoning.append(text)
        elif kind == "content":
            content.append(text)
        # other kinds (READY) are control markers, not text
    joined = "".join(reasoning)
    return "".join(content), (joined or None)
