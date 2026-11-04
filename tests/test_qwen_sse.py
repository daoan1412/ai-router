from __future__ import annotations

import asyncio
import base64
from pathlib import Path

from ai_router.adapters.qwen.auth import QwenSSEParser, format_steps
from ai_router.browser.sse_tap import CDPStreamTap, SSELineBuffer

# A real chat.qwen.ai completion stream (ids replaced, usage dropped):
# 4 cumulative thinking_summary steps, then 140 raw-markdown answer deltas.
FIXTURE = (Path(__file__).parent / "fixtures" / "qwen_completion_sse.txt").read_text(encoding="utf-8")


def _parse_all(chunk_size: int) -> tuple[str, str, QwenSSEParser]:
    lines, parser = SSELineBuffer(), QwenSSEParser()
    content, reasoning = [], []
    for i in range(0, len(FIXTURE), chunk_size):
        for obj in lines.feed(FIXTURE[i : i + chunk_size]):
            for kind, d in parser.feed(obj):
                (reasoning if kind == "reasoning" else content).append(d)
    for obj in lines.flush():
        for kind, d in parser.feed(obj):
            (reasoning if kind == "reasoning" else content).append(d)
    return "".join(content), "".join(reasoning), parser


def test_sse_line_buffer_is_chunking_independent():
    whole = SSELineBuffer().feed(FIXTURE)
    for size in (1, 7, 64, 1000):
        buf, got = SSELineBuffer(), []
        for i in range(0, len(FIXTURE), size):
            got += buf.feed(FIXTURE[i : i + size])
        got += buf.flush()
        assert got == whole


def test_sse_line_buffer_skips_noise():
    buf = SSELineBuffer()
    out = buf.feed(': comment\nevent: x\ndata: {"a": 1}\r\ndata: [DONE]\ndata: not json\n')
    assert out == [{"a": 1}]


def test_parser_answer_is_raw_markdown_with_latex():
    content, _, parser = _parse_all(97)
    assert parser.done
    assert content.startswith("# Bài toán tự tạo")
    assert "$$P = \\frac{a^2}{b} + \\frac{b^2}{c} + \\frac{c^2}{a}$$" in content
    assert content.rstrip().endswith("Đạt được khi $a = b = c = 2$.")  # nothing cut off
    # Rendered KaTeX glyphs (what the DOM reader duplicated) never appear in
    # the raw stream — it carries only the TeX source.
    assert "\U0001d44e" not in content  # MATHEMATICAL ITALIC SMALL A


def test_parser_reasoning_matches_final_snapshot():
    _, reasoning, parser = _parse_all(97)
    assert reasoning.startswith("Interpreting the user's request")
    assert "Brainstorming mathematical problem concepts" in reasoning
    assert reasoning.count("\n\n") == 3  # 4 steps
    assert reasoning == parser._sent_reasoning


def test_parser_ignores_second_ab_response():
    p = QwenSSEParser()
    a = {"response_id": "A", "choices": [{"delta": {"phase": "answer", "content": "one"}}]}
    b = {"response_id": "B", "choices": [{"delta": {"phase": "answer", "content": "two"}}]}
    assert p.feed(a) == [("content", "one")]
    assert p.feed(b) == []


def test_parser_errors_and_other_phases():
    p = QwenSSEParser()
    p.feed({"success": False, "data": {"code": "RateLimited"}})
    assert p.error and "RateLimited" in p.error
    q = QwenSSEParser()
    q.feed({"choices": [{"delta": {"phase": "web_search", "content": ""}}]})
    assert q.other_phases == {"web_search"}


def test_parser_summary_snapshot_rebuild():
    p = QwenSSEParser()

    def ev(titles, thoughts):
        extra = {"summary_title": {"content": titles}, "summary_thought": {"content": thoughts}}
        return {"choices": [{"delta": {"phase": "thinking_summary", "extra": extra}}]}

    d1 = p.feed(ev(["T1"], ["b1"]))
    d2 = p.feed(ev(["T1", "T2"], ["b1", "b2"]))
    d3 = p.feed(ev(["T1", "T2"], ["b1", "b2"]))  # repeated snapshot -> nothing new
    assert d1 == [("reasoning", "T1\nb1")]
    assert d2 == [("reasoning", "\n\nT2\nb2")]
    assert d3 == []
    assert format_steps([{"title": "T1", "body": "b1"}, {"title": "T2", "body": "b2"}]) == "T1\nb1\n\nT2\nb2"


# --- CDPStreamTap ordering (fake CDP session) ---------------------------------


class _FakeCDP:
    def __init__(self, buffered: bytes, gate: asyncio.Event):
        self.handlers = {}
        self.buffered = buffered
        self.gate = gate

    def on(self, event, fn):
        self.handlers[event] = fn

    async def send(self, method, params=None):
        if method == "Network.streamResourceContent":
            await self.gate.wait()  # let event chunks race ahead of the reply
            return {"bufferedData": base64.b64encode(self.buffered).decode()}
        return {}

    async def detach(self):
        pass


class _FakeContext:
    def __init__(self, cdp):
        self.cdp = cdp

    async def new_cdp_session(self, _page):
        return self.cdp


class _FakePage:
    def __init__(self, cdp):
        self.context = _FakeContext(cdp)


async def test_tap_keeps_order_when_events_race_buffered_data():
    gate = asyncio.Event()
    # "é" split across the buffered/event boundary to exercise the UTF-8 decoder.
    full = "data: café\n"
    raw = full.encode()
    cut = raw.index("é".encode()) + 1
    cdp = _FakeCDP(raw[:cut], gate)
    tap = CDPStreamTap(_FakePage(cdp), "/completions")
    await tap.start()
    h = cdp.handlers
    h["Network.responseReceived"]({"requestId": "r1", "response": {"url": "https://x/completions?a", "status": 200}})
    h["Network.responseReceived"]({"requestId": "r2", "response": {"url": "https://x/completions?b", "status": 200}})
    await asyncio.sleep(0)
    # These arrive before streamResourceContent returns: must be held, not lost/reordered.
    h["Network.dataReceived"]({"requestId": "r1", "data": base64.b64encode(raw[cut:]).decode()})
    h["Network.dataReceived"]({"requestId": "r2", "data": base64.b64encode(b"other").decode()})
    h["Network.loadingFinished"]({"requestId": "r1"})
    gate.set()

    items = []
    while True:
        item = await tap.next(1.0)
        assert item is not None
        items.append(item)
        if item[0] == "end":
            break
    assert items[0] == ("open", 200)
    assert "".join(v for k, v in items if k == "text") == full
    await tap.stop()
