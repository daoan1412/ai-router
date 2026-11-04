from __future__ import annotations

from ai_router.adapters.qwen.auth import format_steps


def test_format_steps_title_and_body():
    out = format_steps(
        [
            {"title": "Assessing the problem", "body": "I need 17*23."},
            {"title": "Computing", "body": "17*20 + 17*3 = 391"},
            {"title": "Thinking completed", "body": ""},
        ]
    )
    assert out == "Assessing the problem\nI need 17*23.\n\nComputing\n17*20 + 17*3 = 391"


def test_format_steps_skips_title_only_cards():
    # A step whose body hasn't streamed yet, and the trailing marker.
    assert format_steps([{"title": "Planning", "body": ""}, {"title": "Thinking completed"}]) == ""


def test_format_steps_grows_by_appending_only():
    # Streaming relies on each tick's text extending the previous one.
    t1 = format_steps([{"title": "A", "body": "x"}])
    t2 = format_steps([{"title": "A", "body": "xyz"}, {"title": "B", "body": ""}])
    t3 = format_steps([{"title": "A", "body": "xyz"}, {"title": "B", "body": "w"}])
    assert t2.startswith(t1) and t3.startswith(t2)


def test_format_steps_body_without_title():
    assert format_steps([{"title": "", "body": "just text"}]) == "just text"
