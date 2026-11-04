"""Read a page's streaming HTTP response live, at the browser level (CDP).

Chat UIs render a model's markdown/LaTeX into DOM that rewrites itself as it
streams, and hide parts of the reply (e.g. reasoning) behind panels. The
response the page itself receives is the clean source. CDPStreamTap follows it
with ``Network.streamResourceContent``: the page keeps its own copy untouched,
and page scripts (which may hold a pristine ``fetch``) can't bypass it.
"""

from __future__ import annotations

import asyncio
import base64
import codecs
import json
import re
from typing import Any

from playwright.async_api import Page

# Tap items: ("open", http_status) | ("text", decoded chunk) | ("end", None)
#            | ("error", message)
TapItem = tuple[str, Any]


class SSELineBuffer:
    """Incrementally split Server-Sent-Events text into JSON ``data:`` payloads.

    Chunks can end mid-line; the partial tail is kept until the next feed.
    Comment lines, other fields, ``[DONE]`` and non-JSON payloads are skipped.
    """

    def __init__(self) -> None:
        self._buf = ""

    def feed(self, text: str) -> list[Any]:
        self._buf += text
        *lines, self._buf = self._buf.split("\n")
        return self._parse(lines)

    def flush(self) -> list[Any]:
        lines, self._buf = [self._buf], ""
        return self._parse(lines)

    @staticmethod
    def _parse(lines: list[str]) -> list[Any]:
        out = []
        for line in lines:
            line = line.rstrip("\r")
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            try:
                out.append(json.loads(payload))
            except ValueError:
                continue
        return out


class CDPStreamTap:
    """Stream the body of the first response whose URL matches ``url_match``
    (a substring, or a compiled regex searched in the URL).

    Usage: ``await tap.start()`` before triggering the request, then poll
    ``await tap.next(timeout)``; always ``await tap.stop()``.
    """

    def __init__(self, page: Page, url_match: str | re.Pattern[str]) -> None:
        self.page = page
        self.url_match = url_match
        self.status: int | None = None
        self._cdp = None
        self._rid: str | None = None
        self._queue: asyncio.Queue[TapItem] = asyncio.Queue()
        self._decoder = codecs.getincrementaldecoder("utf-8")()
        # Chunks seen before streaming is enabled are returned as bufferedData;
        # hold any event chunks that race ahead of it so order is preserved.
        self._enabled = False
        self._held: list[bytes] = []
        self._ended_early = False
        self._ended = False

    async def start(self) -> None:
        self._cdp = await self.page.context.new_cdp_session(self.page)
        await self._cdp.send("Network.enable")
        self._cdp.on("Network.responseReceived", self._on_response)
        self._cdp.on("Network.dataReceived", self._on_data)
        self._cdp.on("Network.loadingFinished", self._on_end)
        self._cdp.on("Network.loadingFailed", self._on_end)

    async def stop(self) -> None:
        if self._cdp is not None:
            try:
                await self._cdp.detach()
            except Exception:  # noqa: BLE001 - page/browser may already be gone
                pass
            self._cdp = None

    async def next(self, timeout: float) -> TapItem | None:
        """Next tap item, or None if nothing arrives within ``timeout``."""
        try:
            return await asyncio.wait_for(self._queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    # --- CDP event handlers ---------------------------------------------------

    def _matches(self, url: str) -> bool:
        if isinstance(self.url_match, str):
            return self.url_match in url
        return self.url_match.search(url) is not None

    def _on_response(self, params: dict) -> None:
        if self._rid is not None or not self._matches(params["response"]["url"]):
            return
        self._rid = params["requestId"]
        self.status = int(params["response"]["status"])
        self._queue.put_nowait(("open", self.status))
        asyncio.ensure_future(self._enable(self._rid))

    async def _enable(self, rid: str) -> None:
        try:
            r = await self._cdp.send("Network.streamResourceContent", {"requestId": rid})
            self._emit(base64.b64decode(r.get("bufferedData") or ""))
        except Exception:  # noqa: BLE001 - e.g. already finished: read it whole
            try:
                r = await self._cdp.send("Network.getResponseBody", {"requestId": rid})
                body = r.get("body") or ""
                self._emit(base64.b64decode(body) if r.get("base64Encoded") else body.encode())
            except Exception as exc:  # noqa: BLE001
                self._queue.put_nowait(("error", str(exc)))
        finally:
            self._enabled = True
            for raw in self._held:
                self._emit(raw)
            self._held.clear()
            if self._ended_early:
                self._finish()

    def _on_data(self, params: dict) -> None:
        if params["requestId"] != self._rid or not params.get("data"):
            return
        raw = base64.b64decode(params["data"])
        if self._enabled:
            self._emit(raw)
        else:
            self._held.append(raw)

    def _on_end(self, params: dict) -> None:
        if params["requestId"] != self._rid:
            return
        if self._enabled:
            self._finish()
        else:
            self._ended_early = True

    def _emit(self, raw: bytes) -> None:
        text = self._decoder.decode(raw)
        if text:
            self._queue.put_nowait(("text", text))

    def _finish(self) -> None:
        if self._ended:
            return
        self._ended = True
        tail = self._decoder.decode(b"", final=True)
        if tail:
            self._queue.put_nowait(("text", tail))
        self._queue.put_nowait(("end", None))
