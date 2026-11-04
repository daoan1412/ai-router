from __future__ import annotations

import asyncio
import json

import pytest
from starlette.testclient import TestClient

import ai_router.openai_api as api
from ai_router.config import load_config
from ai_router.streaming import READY, collect, prefix_delta


def test_prefix_delta_appends():
    assert prefix_delta("", "He") == ("He", "He")
    assert prefix_delta("He", "Hello") == ("Hello", "llo")
    assert prefix_delta("Hello", "Hello") == ("Hello", "")


def test_prefix_delta_ignores_rewrite():
    # Renderer rewrote earlier text: emit nothing rather than contradict the client.
    assert prefix_delta("Hello", "Help") == ("Hello", "")


async def _agen(items):
    for item in items:
        yield item


async def test_collect_splits_content_and_reasoning():
    content, reasoning = await collect(
        _agen([("reasoning", "think "), ("reasoning", "more"), ("content", "4"), ("content", "!")])
    )
    assert content == "4!"
    assert reasoning == "think more"


async def test_collect_no_reasoning_is_none():
    assert await collect(_agen([("content", "hi")])) == ("hi", None)


async def test_collect_ignores_ready_marker():
    assert await collect(_agen([READY, ("content", "ok")])) == ("ok", None)


def test_delta_for_and_chunk_shape():
    assert api.delta_for(("content", "a")) == {"content": "a"}
    assert api.delta_for(("reasoning", "r")) == {"reasoning_content": "r"}
    chunk = api.build_chunk("id1", 1, "m", {"content": "a"})
    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["choices"][0] == {"index": 0, "delta": {"content": "a"}, "finish_reason": None}
    assert api.format_sse({"a": "é"}) == 'data: {"a": "é"}\n\n'


class FakeManager:
    """Stands in for the browser-backed manager."""

    deltas = [("reasoning", "Let me "), ("reasoning", "think."), ("content", "17×23 "), ("content", "= 391")]
    fail_with: Exception | None = None
    delay_before_first: float = 0.0
    closed = False

    def __init__(self, cfg, *, headless=False):
        pass

    async def start(self):
        pass

    async def stop(self):
        pass

    async def stream(self, provider, prompt, *, deep_think=False, model_choice=None):
        try:
            if FakeManager.fail_with is not None:
                raise FakeManager.fail_with  # e.g. not logged in: before READY
            yield READY
            if FakeManager.delay_before_first:
                await asyncio.sleep(FakeManager.delay_before_first)  # silent "thinking"
            for d in FakeManager.deltas:
                yield d
        finally:
            FakeManager.closed = True

    async def complete(self, provider, prompt, *, deep_think=False, model_choice=None):
        return await collect(self.stream(provider, prompt))


@pytest.fixture
def client(monkeypatch):
    FakeManager.fail_with = None
    FakeManager.delay_before_first = 0.0
    FakeManager.closed = False
    monkeypatch.setattr(api, "ProviderChatManager", FakeManager)
    with TestClient(api.create_app(load_config(), keepalive_s=0.05)) as c:
        yield c


def _events(resp_text: str) -> list:
    out = []
    for line in resp_text.splitlines():
        if line.startswith("data: "):
            payload = line[len("data: "):]
            out.append(payload if payload == "[DONE]" else json.loads(payload))
    return out


def test_stream_emits_one_chunk_per_delta(client):
    body = {
        "model": "deepseek/deepseek-r1-0528",
        "stream": True,
        "messages": [{"role": "user", "content": "17*23?"}],
    }
    resp = client.post("/v1/chat/completions", json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    ev = _events(resp.text)
    assert ev[-1] == "[DONE]"
    chunks = ev[:-1]
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    deltas = [c["choices"][0]["delta"] for c in chunks[1:-1]]
    assert deltas == [
        {"reasoning_content": "Let me "},
        {"reasoning_content": "think."},
        {"content": "17×23 "},
        {"content": "= 391"},
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    assert len({c["id"] for c in chunks}) == 1  # one id for the whole stream
    assert FakeManager.closed  # provider generator closed -> lock released


def test_stream_not_logged_in_is_401_not_200(client):
    FakeManager.fail_with = api.NotLoggedInError("qwen")
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen/qwen3.7-plus", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "not_logged_in"


def test_locked_model_is_401_with_its_own_code(client):
    from ai_router.adapters.zai.auth import ModelLockedError

    FakeManager.fail_with = ModelLockedError("Z.ai model 'GLM-5.3' needs a signed-in account")
    for stream in (True, False):
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "z-ai/glm-5.3", "stream": stream, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "model_requires_login"


def test_stream_sends_keepalive_while_model_is_silent(client):
    # Qwen can think for minutes with nothing in the DOM: the connection must
    # stay alive with SSE comments, and the real deltas still follow.
    FakeManager.delay_before_first = 0.3
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "qwen/qwen3.7-plus", "stream": True, "messages": [{"role": "user", "content": "x"}]},
    )
    assert resp.status_code == 200
    assert ": keep-alive" in resp.text
    contents = [
        c["choices"][0]["delta"].get("content")
        for c in _events(resp.text)[:-1]
        if "content" in c["choices"][0]["delta"]
    ]
    assert "".join(contents) == "17×23 = 391"
    assert FakeManager.closed


def test_non_stream_joins_deltas(client):
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek/deepseek-r1-0528", "messages": [{"role": "user", "content": "x"}]},
    )
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "17×23 = 391"
    assert msg["reasoning_content"] == "Let me think."
