from __future__ import annotations

import httpx
import pytest

from ai_router.mailtemp import MailtempClient, MailtempError, _extract_address


def _client_with(handler) -> MailtempClient:
    return MailtempClient(
        api_key="k",
        base_url="https://mailtemp.test",
        transport=httpx.MockTransport(handler),
    )


def test_extract_address_shapes():
    assert _extract_address({"mailbox": {"address": "a@b.com"}}) == "a@b.com"
    assert _extract_address({"address": "a@b.com"}) == "a@b.com"
    assert _extract_address({"email": "a@b.com"}) == "a@b.com"
    assert _extract_address({"data": {"address": "a@b.com"}}) == "a@b.com"
    assert _extract_address({"nope": 1}) is None


def test_no_key_detection():
    assert MailtempClient(api_key="").has_key is False
    assert MailtempClient(api_key="x").has_key is True


async def test_create_random_mailbox_parses_address():
    def handler(request):
        return httpx.Response(200, json={"mailbox": {"address": "rand@mailtemp.test"}})

    client = _client_with(handler)
    mailbox = await client.create_random_mailbox()
    assert mailbox.address == "rand@mailtemp.test"


async def test_peek_code_text():
    def handler(request):
        return httpx.Response(200, text="123456")

    client = _client_with(handler)
    assert await client.peek_code("a@b.com") == "123456"


async def test_peek_code_empty_when_none():
    def handler(request):
        return httpx.Response(200, text="")

    client = _client_with(handler)
    assert await client.peek_code("a@b.com") == ""


async def test_401_raises_with_hint():
    def handler(request):
        return httpx.Response(401, json={"error": "API key không hợp lệ."})

    client = _client_with(handler)
    with pytest.raises(MailtempError, match="MAILTEMP_API_KEY"):
        await client.create_random_mailbox()


async def test_wait_for_code_polls_until_present():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, text="" if calls["n"] < 2 else "999000")

    client = _client_with(handler)
    code = await client.wait_for_code("a@b.com", timeout=5, interval=0.01)
    assert code == "999000"
