from __future__ import annotations

from pathlib import Path

from ai_router.adapters.deepseek.stream import DeepSeekSSEParser
from ai_router.browser.sse_tap import SSELineBuffer

# Real chat.deepseek.com /completion streams for
# "Giải phương trình x^2 - 5x + 6 = 0, dùng LaTeX, ngắn gọn." — DeepThink on/off.
FIX = Path(__file__).parent / "fixtures"
THINK = (FIX / "deepseek_completion_think_sse.txt").read_text(encoding="utf-8")
NOTHINK = (FIX / "deepseek_completion_nothink_sse.txt").read_text(encoding="utf-8")


def _run(stream: str, chunk: int = 53) -> tuple[str, str, DeepSeekSSEParser]:
    lines, parser = SSELineBuffer(), DeepSeekSSEParser()
    got = {"content": [], "reasoning": []}
    for i in range(0, len(stream), chunk):
        for obj in lines.feed(stream[i : i + chunk]):
            for kind, d in parser.feed(obj):
                got[kind].append(d)
    for obj in lines.flush():
        for kind, d in parser.feed(obj):
            got[kind].append(d)
    return "".join(got["content"]), "".join(got["reasoning"]), parser


def test_think_stream_splits_reasoning_and_raw_latex_answer():
    content, reasoning, parser = _run(THINK)
    assert parser.done and parser.error is None
    assert reasoning.startswith("We need answer Vietnamese.")
    assert "factors (x-2)(x-3)=0" in reasoning
    assert content.startswith("\\[\nx^2-5x+6=0\n\\iff (x-2)(x-3)=0")
    assert content.endswith("Vậy \\(S=\\{2;3\\}\\).")
    # Raw TeX only — the DOM reader produced rendered glyphs like 𝑥 instead.
    assert "\U0001d465" not in content  # MATHEMATICAL ITALIC SMALL X


def test_nothink_stream_has_no_reasoning():
    content, reasoning, parser = _run(NOTHINK)
    assert parser.done
    assert reasoning == ""
    assert "\\Delta=(-5)^2-4\\cdot1\\cdot6=25-24=1" in content
    assert content.endswith("\\(\\boxed{x=2;\\ x=3}\\).")


def test_chunking_does_not_change_result():
    base = _run(THINK, 10_000)[:2]
    for size in (1, 7, 200):
        assert _run(THINK, size)[:2] == base


def test_pathless_value_continues_previous_path():
    p = DeepSeekSSEParser()
    p.feed({"v": {"response": {"fragments": [{"type": "RESPONSE", "content": "A"}]}}})
    assert p.feed({"p": "response/fragments/-1/content", "o": "APPEND", "v": "B"}) == [("content", "B")]
    assert p.feed({"v": "C"}) == [("content", "C")]
    # New fragment, then a path given without "o" on a content path => append.
    p.feed({"p": "response/fragments", "o": "APPEND", "v": [{"type": "RESPONSE", "content": "D"}]})
    assert p.feed({"p": "response/fragments/-1/content", "v": "E"}) == [("content", "E")]
    assert p.text("content") == "ABC\n\nDE"


def test_failure_status_and_json_error_body():
    p = DeepSeekSSEParser()
    p.feed({"p": "response/status", "o": "SET", "v": "FAILED"})
    assert p.error and "FAILED" in p.error
    q = DeepSeekSSEParser()
    q.feed({"code": 40003, "msg": "rate limited", "data": None})
    assert q.error == "40003: rate limited"


def test_unknown_fragment_type_is_reported():
    p = DeepSeekSSEParser()
    p.feed({"v": {"response": {"fragments": [{"type": "SEARCH", "content": "…"}]}}})
    assert p.other_types == {"SEARCH"}
    assert p.text("content") == ""
