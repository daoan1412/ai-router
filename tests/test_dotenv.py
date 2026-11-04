from __future__ import annotations

import os

from ai_router.dotenv import load_dotenv, set_env_vars


def test_load_dotenv_sets_without_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('A=1\nB="two"\n# comment\nexport C=3\n', encoding="utf-8")
    monkeypatch.delenv("A", raising=False)
    monkeypatch.setenv("B", "real")  # real env should win

    parsed = load_dotenv(env)

    assert parsed == {"A": "1", "B": "two", "C": "3"}
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "real"  # not overridden
    assert os.environ["C"] == "3"


def test_set_env_vars_upserts_and_preserves(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("KEEP=yes\nDEEPSEEK_EMAIL=old@x.com\n", encoding="utf-8")
    monkeypatch.delenv("DEEPSEEK_EMAIL", raising=False)

    set_env_vars({"DEEPSEEK_EMAIL": "new@x.com", "DEEPSEEK_PASSWORD": "pw"}, env)

    text = env.read_text(encoding="utf-8")
    assert "KEEP=yes" in text
    assert "DEEPSEEK_EMAIL=new@x.com" in text
    assert "DEEPSEEK_PASSWORD=pw" in text
    assert text.count("DEEPSEEK_EMAIL=") == 1  # replaced, not duplicated
    assert os.environ["DEEPSEEK_EMAIL"] == "new@x.com"
