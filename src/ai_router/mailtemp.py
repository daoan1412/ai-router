"""Client for the mailtemp.vn (TMail) public API.

Used to provision a disposable mailbox and poll it for the verification code a
provider (e.g. DeepSeek) emails during sign-up. The API needs an API key sent
in the ``x-api-key`` header; keyless calls return HTTP 401.

Docs: https://mailtemp.vn/api-docs
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import httpx

DEFAULT_BASE_URL = "https://mailtemp.vn"


class MailtempError(RuntimeError):
    """Raised when the mailtemp.vn API returns an error or cannot be reached."""


@dataclass
class Mailbox:
    address: str


def _extract_address(payload: object) -> str | None:
    """Pull the mailbox address out of a loosely-specified JSON response.

    The public docs don't pin the response schema, so probe the shapes seen in
    the wild: ``{"mailbox": {"address": ...}}``, ``{"address": ...}``,
    ``{"email": ...}``, and the same nested under ``data``.
    """
    if not isinstance(payload, dict):
        return None
    for container in (payload, payload.get("mailbox"), payload.get("data")):
        if not isinstance(container, dict):
            continue
        for key in ("address", "email", "mail", "name"):
            value = container.get(key)
            if isinstance(value, str) and "@" in value:
                return value
    return None


class MailtempClient:
    """Thin async wrapper over the mailtemp.vn endpoints we use."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        timeout: float = 20.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("MAILTEMP_API_KEY")
        self.base_url = (base_url or os.getenv("MAILTEMP_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.timeout = timeout
        self._transport = transport  # injected in tests; None uses the network

    @property
    def has_key(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        url = f"{self.base_url}{path}"
        async with httpx.AsyncClient(timeout=self.timeout, transport=self._transport) as client:
            try:
                resp = await client.request(method, url, headers=self._headers(), **kwargs)
            except httpx.HTTPError as exc:  # network / DNS / timeout
                raise MailtempError(f"mailtemp request failed: {exc}") from exc
        if resp.status_code == 401:
            raise MailtempError(
                "mailtemp.vn rejected the API key (401). Set MAILTEMP_API_KEY in your .env."
            )
        if resp.status_code == 429:
            raise MailtempError("mailtemp.vn rate limit hit (429). Wait and retry.")
        return resp

    async def create_random_mailbox(self) -> Mailbox:
        """POST /api/messages/random — return a fresh disposable address."""
        resp = await self._request("POST", "/api/messages/random")
        if resp.status_code >= 400:
            raise MailtempError(
                f"create_random_mailbox failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        address = _extract_address(_safe_json(resp))
        if not address:
            raise MailtempError(
                f"create_random_mailbox: no address in response: {resp.text[:200]}"
            )
        return Mailbox(address=address)

    async def list_messages(self, email: str) -> list[dict]:
        """GET /api/messages/{email} — the mailbox inbox."""
        resp = await self._request("GET", f"/api/messages/{email}")
        if resp.status_code == 404:
            return []
        if resp.status_code >= 400:
            raise MailtempError(
                f"list_messages failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        data = _safe_json(resp)
        if isinstance(data, dict):
            msgs = data.get("messages")
            if isinstance(msgs, list):
                return msgs
        if isinstance(data, list):
            return data
        return []

    async def peek_code(self, email: str) -> str:
        """GET /api/code/{email}?format=text — latest OTP, or '' if none yet."""
        resp = await self._request("GET", f"/api/code/{email}", params={"format": "text"})
        if resp.status_code == 404:
            return ""
        if resp.status_code >= 400:
            raise MailtempError(
                f"peek_code failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        return resp.text.strip()

    async def wait_for_code(
        self,
        email: str,
        *,
        timeout: float = 120.0,
        interval: float = 2.0,
    ) -> str:
        """Poll the code endpoint until a non-empty OTP arrives or we time out."""
        deadline = asyncio.get_event_loop().time() + timeout
        last_err: Exception | None = None
        while asyncio.get_event_loop().time() < deadline:
            try:
                code = await self.peek_code(email)
            except MailtempError as exc:
                last_err = exc
                code = ""
            if code:
                return code
            await asyncio.sleep(interval)
        if last_err:
            raise MailtempError(
                f"no code for {email} within {timeout:.0f}s (last error: {last_err})"
            )
        raise MailtempError(f"no code for {email} within {timeout:.0f}s")


def _safe_json(resp: httpx.Response) -> object:
    try:
        return resp.json()
    except ValueError:
        return None
