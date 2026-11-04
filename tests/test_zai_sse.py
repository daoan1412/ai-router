from __future__ import annotations

from pathlib import Path

from ai_router.adapters.zai.auth import ZaiSSEParser
from ai_router.browser.sse_tap import SSELineBuffer

# Real chat.z.ai guest completion (GLM-5.3-Flash) for
# "Giải phương trình x^2 - 5x + 6 = 0, dùng LaTeX, ngắn gọn." (token stripped).
STREAM = (Path(__file__).parent / "fixtures" / "zai_completion_sse.txt").read_text(encoding="utf-8")


def _run(chunk: int) -> tuple[str, str, ZaiSSEParser]:
    lines, parser = SSELineBuffer(), ZaiSSEParser()
    got = {"content": [], "reasoning": []}
    for i in range(0, len(STREAM), chunk):
        for obj in lines.feed(STREAM[i : i + chunk]):
            for kind, d in parser.feed(obj):
                got[kind].append(d)
    for obj in lines.flush():
        for kind, d in parser.feed(obj):
            got[kind].append(d)
    return "".join(got["content"]), "".join(got["reasoning"]), parser


def test_thinking_then_raw_latex_answer():
    content, reasoning, parser = _run(41)
    assert parser.done and parser.error is None and not parser.other_phases
    assert reasoning.startswith("The user asks in Vietnamese to solve")
    assert "\\begin{cases} x = 2 \\\\ x = 3 \\end{cases}$$" in content
    assert content.endswith("**Vậy** $x \\in \\{2;\\,3\\}$.")
    assert "\U0001d465" not in content  # no rendered math glyphs, TeX source only


def test_chunking_independent():
    assert _run(1)[:2] == _run(10_000)[:2]


def test_error_payloads():
    p = ZaiSSEParser()
    p.feed({"type": "chat:completion", "data": {"error": {"code": 429, "detail": "Too many requests"}}})
    assert p.error and "429" in p.error
    q = ZaiSSEParser()
    q.feed({"detail": "Unauthorized"})
    assert q.error == "Unauthorized"


def test_unknown_phase_recorded_not_emitted():
    p = ZaiSSEParser()
    assert p.feed({"type": "chat:completion", "data": {"phase": "tool_call", "delta_content": "x"}}) == []
    assert p.other_phases == {"tool_call"}
