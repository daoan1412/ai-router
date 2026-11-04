from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from ai_router.adapters.deepseek.selectors import FAILURE_STATUSES, RATE_LIMIT_MARKERS
from ai_router.browser.profile import StreamDone

_RATE_LIMIT_STATUSES = (401, 403, 429)


def _iter_sse_events(body: str) -> Iterator[tuple[str | None, dict[str, Any] | None]]:
    """Yield (event_name, data_payload) pairs from an SSE body."""
    current_event: str | None = None
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
            continue
        if line.startswith("data:"):
            payload = line[len("data:") :].strip()
            data: dict[str, Any] | None = None
            if payload:
                try:
                    parsed = json.loads(payload)
                    if isinstance(parsed, dict):
                        data = parsed
                except json.JSONDecodeError:
                    pass
            yield current_event, data
            current_event = None


def _patch_failure(data: dict[str, Any]) -> bool:
    if data.get("o") != "SET":
        return False
    path = data.get("p")
    value = data.get("v")
    if path == "response/status" and isinstance(value, str):
        return value.upper() in FAILURE_STATUSES
    return False


def _patch_finished(data: dict[str, Any]) -> bool:
    if data.get("o") == "SET":
        if data.get("p") == "response/status" and data.get("v") == "FINISHED":
            return True
    if data.get("o") == "BATCH" and isinstance(data.get("v"), list):
        for item in data["v"]:
            if isinstance(item, dict) and item.get("v") == "FINISHED":
                if item.get("p") in ("quasi_status", "response/status", "status"):
                    return True
    return False


_FRAGMENT_KIND = {"THINK": "reasoning", "RESPONSE": "content"}


class DeepSeekSSEParser:
    """Rebuild DeepSeek's reply from its /completion SSE, as text deltas.

    The stream is a series of JSON patches against a ``response`` object
    (observed 2026-11):
      - ``{"v": {"response": {... "fragments": [{type, content}]}}}`` snapshot
      - ``{"p": "response/fragments/-1/content", "o": "APPEND", "v": "tok"}``
      - ``{"v": "tok"}`` — no path: continues the previous path/op (most tokens)
      - ``{"p": "response/fragments", "o": "APPEND", "v": [fragment]}``
      - ``{"p": "response/status", "o": "SET", "v": "FINISHED"}`` / BATCH quasi_status
    THINK fragments are the DeepThink reasoning, RESPONSE fragments the answer;
    both are raw markdown/LaTeX. ``feed`` returns ("reasoning"|"content", delta).
    """

    def __init__(self) -> None:
        self.fragments: list[dict[str, Any]] = []
        self.done = False
        self.error: str | None = None
        self.got_any = False
        self.other_types: set[str] = set()
        self._path: str | None = None
        self._op = "APPEND"
        self._sent = {"reasoning": "", "content": ""}

    def text(self, kind: str) -> str:
        return "\n\n".join(
            f.get("content") or ""
            for f in self.fragments
            if _FRAGMENT_KIND.get(f.get("type") or "") == kind and f.get("content")
        )

    def feed(self, obj: Any) -> list[tuple[str, str]]:
        if not isinstance(obj, dict):
            return []
        code = obj.get("code")
        if isinstance(code, int) and code != 0 and "msg" in obj:  # JSON error body
            self.error = f"{code}: {obj.get('msg')}"
            return []
        if "v" not in obj:
            return []  # ready / update_session / title / close payloads
        self._apply(obj)
        out = []
        for kind in ("reasoning", "content"):
            sent, d = self._sent[kind], ""
            full = self.text(kind)
            if full.startswith(sent):
                d = full[len(sent):]
                self._sent[kind] = full
            if d:
                out.append((kind, d))
        return out

    def _apply(self, obj: dict[str, Any]) -> None:
        value = obj["v"]
        path = obj.get("p")
        if path is None and isinstance(value, dict) and "response" in value:
            frags = value["response"].get("fragments") or []
            self.fragments = [dict(f) for f in frags if isinstance(f, dict)]
            self._note_types()
            self.got_any = True
            return
        if path is not None:
            self._path = path
            self._op = obj.get("o") or ("APPEND" if path.endswith("/content") else self._op)
        elif "o" in obj:
            self._op = obj["o"]
        if self._path is None:
            return
        self.got_any = True
        if _patch_failure({"p": self._path, "o": self._op, "v": value}):
            self.error = f"stream status {value}"
            return
        if _patch_finished({"p": self._path, "o": self._op, "v": value}):
            self.done = True
            return
        if self._path == "response/fragments" and self._op == "APPEND":
            for f in value if isinstance(value, list) else [value]:
                if isinstance(f, dict):
                    self.fragments.append(dict(f))
            self._note_types()
            return
        prefix = "response/fragments/"
        if self._path.startswith(prefix) and self._path.endswith("/content"):
            idx_s = self._path[len(prefix) : -len("/content")]
            try:
                frag = self.fragments[int(idx_s)]
            except (ValueError, IndexError):
                return
            if self._op == "SET":
                frag["content"] = str(value)
            else:
                frag["content"] = (frag.get("content") or "") + str(value)

    def _note_types(self) -> None:
        for f in self.fragments:
            t = f.get("type")
            if t and t not in _FRAGMENT_KIND:
                self.other_types.add(t)  # e.g. a search/tool fragment


def parse_stream_done(status: int, body: str) -> StreamDone:
    """Classify a finished chat.deepseek.com /completion SSE body.

    Success requires a FINISHED status signal AND event: close.
    Answer text is read from the DOM by StateReducer — not from this parser.
    """
    if status >= 400:
        lower = body.lower()
        if status in _RATE_LIMIT_STATUSES or any(
            m in lower for m in RATE_LIMIT_MARKERS
        ):
            return StreamDone(
                done=True,
                ok=False,
                error_kind="rate_limit",
                error_text=f"HTTP {status}: {body[:200]}",
            )
        return StreamDone(
            done=True,
            ok=False,
            error_kind="error",
            error_text=f"HTTP {status}: {body[:200]}",
        )

    saw_finished = False
    saw_close = False

    for event_name, data in _iter_sse_events(body):
        if event_name == "close":
            saw_close = True
        if data is None:
            continue
        if _patch_failure(data):
            return StreamDone(
                done=True,
                ok=False,
                error_kind="error",
                error_text=f"Stream status: {data.get('v')}",
            )
        if _patch_finished(data):
            saw_finished = True

    if saw_close and saw_finished:
        return StreamDone(done=True, ok=True)
    if saw_close and not saw_finished:
        return StreamDone(done=False, ok=False)
    return StreamDone(done=False, ok=False)
