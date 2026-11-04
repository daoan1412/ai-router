from __future__ import annotations

import base64
from pathlib import Path

from ai_router.adapters.gemini.chat import model_name
from ai_router.adapters.gemini.stream import (
    GeminiStreamParser,
    WebChannelBuffer,
    pb_fields,
    pb_get,
    pb_str,
)

FIX = Path(__file__).parent / "fixtures"
# Real gemini.google.com backchannels (3.1 Pro) for "Giải phương trình x^2 - 5x + 6 = 0,
# dùng LaTeX, ngắn gọn." — a temporary chat, and one where Pro made a Canvas document.
# The location field (candidate .6) was removed from both.
PRO = (FIX / "gemini_backchannel_pro.txt").read_text(encoding="utf-8")
CANVAS = (FIX / "gemini_backchannel_canvas.txt").read_text(encoding="utf-8")


def _run(stream: str, chunk: int) -> tuple[str, str, GeminiStreamParser, list]:
    buf, parser = WebChannelBuffer(), GeminiStreamParser()
    got = {"content": [], "reasoning": []}
    for i in range(0, len(stream), chunk):
        for _seq, payload in buf.feed(stream[i : i + chunk]):
            for kind, d in parser.feed(payload):
                got[kind].append(d)
    for kind, d in parser.finish():
        got[kind].append(d)
    return "".join(got["content"]), "".join(got["reasoning"]), parser, got["content"]


def test_pro_answer_raw_latex_and_thinking():
    content, reasoning, parser, pieces = _run(PRO, 113)
    assert parser.done and parser.model == "3.1 Pro" and not parser.chips
    assert content.startswith("Để giải phương trình $x^2 - 5x + 6 = 0$")
    assert "$(x - 2)(x - 3) = 0$" in content and content.endswith("$x = 3$.")
    assert reasoning.startswith("**Solved the equation**")
    assert len(pieces) > 1  # streamed in several deltas, not one blob


def test_chunking_independent():
    assert _run(PRO, 1)[:2] == _run(PRO, 100_000)[:2]
    assert _run(CANVAS, 7)[:2] == _run(CANVAS, 100_000)[:2]


def test_canvas_chip_removed_and_document_appended():
    content, _, parser, _ = _run(CANVAS, 501)
    assert parser.chips == {"immersive_entry_chip"}
    assert "googleusercontent.com" not in content
    assert "`giai_phuong_trinh.tex`\n\n```tex\n\\documentclass" in content
    assert content.rstrip().endswith("\\end{document}\n```")


def test_location_not_in_fixtures():
    for stream in (PRO, CANVAS):
        buf = WebChannelBuffer()
        for _seq, payload in buf.feed(stream):
            if isinstance(payload, list) and payload and payload[0] != "noop":
                raw = base64.b64decode(payload[0] + "=" * (-len(payload[0]) % 4))
                assert not pb_get(raw, 2, 7, 1, 4, 4, 6)


def test_webchannel_buffer_partial_and_noop():
    buf = WebChannelBuffer()
    data = '14\n[[1,["noop"]]]14\n[[2,["CgA="]]]'
    assert buf.feed(data[:10]) == []
    assert buf.feed(data[10:]) == [[1, ["noop"]], [2, ["CgA="]]]
    assert GeminiStreamParser().feed(["noop"]) == []


def test_protobuf_reader():
    # field 1 varint 150, field 2 string "hi", field 3 message {1: "x"}
    msg = bytes([0x08, 0x96, 0x01, 0x12, 0x02]) + b"hi" + bytes([0x1A, 0x03, 0x0A, 0x01]) + b"x"
    assert pb_fields(msg)[0] == (1, 150)
    assert pb_str(msg, 2) == "hi" and pb_str(msg, 3, 1) == "x"
    assert pb_get(msg, 9) == []


def test_stall_timeout_grows_with_model():
    from ai_router.adapters.gemini.chat import stall_timeout_for

    assert stall_timeout_for("Flash") < stall_timeout_for("Pro") < stall_timeout_for("Argon")
    assert stall_timeout_for(None) == 60.0


def test_model_name_strips_version():
    assert model_name("3.8 Flash") == "Flash"
    assert model_name("3.5 Flash-Lite") == "Flash-Lite"
    assert model_name("4 Argon") == "Argon"
