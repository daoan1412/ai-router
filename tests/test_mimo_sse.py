from __future__ import annotations

import json
from pathlib import Path

from ai_router.adapters.mimo.auth import MimoSSEParser, force_no_search
from ai_router.browser.sse_tap import SSELineBuffer

# Real MiMo Studio completion (MiMo-V2.6-Pro) for
# "Giải phương trình x^2 - 5x + 6 = 0, dùng LaTeX, ngắn gọn." (URL token stripped).
STREAM = (Path(__file__).parent / "fixtures" / "mimo_chat_sse.txt").read_text(encoding="utf-8")


def _run(chunk: int) -> tuple[str, str, MimoSSEParser]:
    lines, parser = SSELineBuffer(), MimoSSEParser()
    got = {"content": [], "reasoning": []}
    for i in range(0, len(STREAM), chunk):
        for obj in lines.feed(STREAM[i : i + chunk]):
            for kind, d in parser.feed(obj):
                got[kind].append(d)
    for obj in lines.flush():
        for kind, d in parser.feed(obj):
            got[kind].append(d)
    return "".join(got["content"]), "".join(got["reasoning"]), parser


def test_reasoning_then_raw_latex_answer():
    content, reasoning, parser = _run(37)
    assert parser.done and parser.error is None and not parser.other_types
    assert reasoning.startswith("The user wants to solve the quadratic equation")
    assert content.startswith("$$x^2 - 5x + 6 = 0 \\iff (x-2)(x-3) = 0$$")
    assert content.endswith("Nghiệm: $S = \\{2,\\, 3\\}$.")
    assert "[DONE]" not in content and "21565835" not in content  # finish / dialogId ignored


def test_chunking_independent():
    assert _run(1)[:2] == _run(10_000)[:2]


def test_error_and_unknown_types():
    p = MimoSSEParser()
    p.feed({"code": 401, "msg": "not login"})
    assert p.error == "not login"
    q = MimoSSEParser()
    assert q.feed({"messageType": "tool", "content": "search results"}) == []
    assert q.other_types == {"tool"}
    assert q.feed({"code": 0, "msg": "ok"}) == [] and q.error is None


def test_force_no_search_rewrites_only_when_needed():
    on = json.dumps({"query": "hi", "modelConfig": {"webSearchStatus": "enabled", "model": "mimo-v2.6-pro"}})
    out = json.loads(force_no_search(on))
    assert out["modelConfig"] == {"webSearchStatus": "disabled", "model": "mimo-v2.6-pro"}
    off = json.dumps({"modelConfig": {"webSearchStatus": "disabled"}})
    assert force_no_search(off) == off
    assert force_no_search("not json") == "not json"
