"""Decode Gemini's answer stream (WebChannel backchannel, base64 protobuf).

gemini.google.com answers over a WebChannel to
``geminiweb-pa.clients6.google.com/v1/processSession``: the prompt is POSTed on
the forward channel and the reply arrives on a long-poll GET (the backchannel,
``TYPE=xmlhttp``) opened when the page loads. Its body is a series of
``<length>\\n<json>`` chunks; each JSON array holds ``[seq, [payload]]`` items
whose payload is a base64 protobuf (or ``"noop"``).

Field paths below were mapped from live captures (2026-11; no public schema):
``2.7.1`` is the turn — ``.1`` is 1 on its final frame — and ``2.7.1.4.4`` the
candidate, whose ``5`` (response) carries ``2.1`` (answer markdown with raw
LaTeX, cumulative), ``38.1.1`` (thinking summary, cumulative) and ``31``
(an "immersive" Canvas document: ``10`` file name, ``33.3`` its source).
"""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from ai_router.streaming import Delta, prefix_delta

# Placeholders the answer text uses for rich chips (Canvas documents, images…).
CHIP_RE = re.compile(r"http://googleusercontent\.com/(\w+)/\d+")
_CHIP_PREFIX = "http://googleusercontent.com/"


class WebChannelBuffer:
    """Split a WebChannel backchannel body into its ``[seq, payload]`` items."""

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[list[Any]]:
        self._buf += text
        out: list[list[Any]] = []
        while True:
            nl = self._buf.find("\n")
            if nl < 0 or not self._buf[:nl].strip().isdigit():
                if nl >= 0:  # not a length line: resync past it
                    self._buf = self._buf[nl + 1:]
                    continue
                return out
            n = int(self._buf[:nl])
            chunk = self._buf[nl + 1 : nl + 1 + n]
            if len(chunk) < n:
                return out
            self._buf = self._buf[nl + 1 + n :]
            try:
                items = json.loads(chunk)
            except ValueError:
                continue
            out.extend(i for i in items if isinstance(i, list) and len(i) == 2)


# --- Minimal schema-less protobuf reader -------------------------------------

def _varint(b: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = b[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if byte < 0x80:
            return result, i


def pb_fields(b: bytes) -> list[tuple[int, int | bytes]]:
    """(field number, value) pairs; varints as int, length-delimited as bytes."""
    out: list[tuple[int, int | bytes]] = []
    i = 0
    while i < len(b):
        key, i = _varint(b, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(b, i)
            out.append((field, v))
        elif wire == 2:
            n, i = _varint(b, i)
            if i + n > len(b):
                raise ValueError("truncated field")
            out.append((field, b[i : i + n]))
            i += n
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
    return out


def pb_get(b: bytes, *path: int) -> list[int | bytes]:
    """All values at ``path`` (repeated fields fan out); [] if absent/unparsable."""
    level: list[int | bytes] = [b]
    for field in path:
        nxt: list[int | bytes] = []
        for msg in level:
            if not isinstance(msg, bytes):
                continue
            try:
                nxt.extend(v for f, v in pb_fields(msg) if f == field)
            except (ValueError, IndexError):
                continue
        level = nxt
    return level


def pb_str(b: bytes, *path: int) -> str | None:
    for v in pb_get(b, *path):
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8")
            except UnicodeDecodeError:
                continue
    return None


def pb_int(b: bytes, *path: int) -> int | None:
    for v in pb_get(b, *path):
        if isinstance(v, int):
            return v
    return None


TURN = (2, 7, 1)
CANDIDATE = (*TURN, 4, 4)
RESPONSE = (*CANDIDATE, 5)


class GeminiStreamParser:
    """Turn backchannel payloads into ("reasoning"|"content", delta).

    Answer and thinking arrive as cumulative snapshots, so deltas are the
    appended part (prefix_delta). Chip placeholders are removed from the answer
    and a Canvas document's source is appended at the end as a code block.
    """

    def __init__(self) -> None:
        self.done = False
        self.got_any = False
        self.progress = False  # the turn started: status title, thinking or answer
        self.model: str | None = None
        self.chips: set[str] = set()
        self.documents: dict[str, str] = {}  # file name -> source
        self._answer = ""
        self._thinking = ""
        self._sent_answer = ""
        self._sent_thinking = ""

    def feed(self, payload: Any) -> list[Delta]:
        if not (isinstance(payload, list) and payload and isinstance(payload[0], str)):
            return []  # "noop" / control items
        if payload[0] == "noop":
            return []
        try:
            raw = base64.b64decode(payload[0] + "=" * (-len(payload[0]) % 4))
        except ValueError:
            return []
        if not pb_get(raw, *CANDIDATE):
            return []
        self.model = pb_str(raw, *CANDIDATE, 43) or self.model
        if pb_str(raw, *CANDIDATE, 7, 6, 1):  # "Solving a quadratic equation…" status
            self.progress = True
        answer = pb_str(raw, *RESPONSE, 2, 1)
        if answer:
            self.got_any = True
            self._answer = answer
        thinking = pb_str(raw, *RESPONSE, 38, 1, 1)
        if thinking:
            self.got_any = True
            self._thinking = thinking
        for doc in pb_get(raw, *RESPONSE, 31):
            if isinstance(doc, bytes):
                name = pb_str(doc, 10) or pb_str(doc, 1) or "document"
                source = pb_str(doc, 33, 3)
                if source:
                    self.documents[name] = source
        if pb_int(raw, *TURN, 1) == 1:
            self.done = True
        self.progress = self.progress or self.got_any
        return self._deltas()

    def finish(self) -> list[Delta]:
        """Flush held-back text and append Canvas documents (call once at the end)."""
        self.done = True
        out = self._deltas()
        for name, source in self.documents.items():
            lang = name.rsplit(".", 1)[-1] if "." in name else ""
            out.append(("content", f"\n\n`{name}`\n\n```{lang}\n{source.rstrip()}\n```\n"))
        self.documents.clear()
        return out

    def _clean_answer(self) -> str:
        def drop(m: re.Match) -> str:
            self.chips.add(m.group(1))
            return ""

        # Same transform on every snapshot, so earlier output stays a prefix.
        text = re.sub(r"\n{3,}", "\n\n", CHIP_RE.sub(drop, self._answer))
        if not self.done:
            # Hold back a trailing, still-growing chip URL so it never streams out.
            last = text.rsplit("\n", 1)[-1]
            tail = last.lstrip()
            if tail and (_CHIP_PREFIX.startswith(tail) or tail.startswith(_CHIP_PREFIX)):
                text = text[: len(text) - len(last)]
        return text

    def _deltas(self) -> list[Delta]:
        out: list[Delta] = []
        self._sent_thinking, d = prefix_delta(self._sent_thinking, self._thinking)
        if d:
            out.append(("reasoning", d))
        self._sent_answer, d = prefix_delta(self._sent_answer, self._clean_answer())
        if d:
            out.append(("content", d))
        return out
