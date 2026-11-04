"""Keyless mailtemp.vn access by driving its website with Playwright.

The public API needs an x-api-key (keyless → 401), but the website creates a
mailbox on load with no key. This module reads the auto-generated address and
polls the inbox, reading the verification code out of the message-body iframe
(mailtemp renders each email in an ``about:srcdoc`` frame, so the code is not in
the top document's text).
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable

from playwright.async_api import Page

MT_URL = "https://mailtemp.vn/"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

EMPTY_MARKER = "Chưa có thư"
REFRESH_LABEL = "Làm mới hộp thư"
NEW_ADDRESS_LABEL = "Địa chỉ mới"

Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


def parse_address(body_text: str) -> str | None:
    """Pull the mailbox address from the page's innerText.

    mailtemp renders it as the local part and ``@domain`` on separate lines, so
    join those two lines rather than globbing (which would swallow the adjacent
    "Chép"/copy label into the TLD).
    """
    lines = [ln.strip() for ln in body_text.split("\n") if ln.strip()]
    for idx, ln in enumerate(lines):
        if ln.startswith("@") and idx > 0:
            candidate = lines[idx - 1] + ln
            if EMAIL_RE.fullmatch(candidate):
                return candidate
    for ln in lines:
        if EMAIL_RE.fullmatch(ln):
            return ln
    return None


def extract_code(text: str, *, min_digits: int = 6) -> str | None:
    """Return the first OTP-looking token: >= min_digits digits, or XXX-XXX.

    The floor of min_digits (default 6, DeepSeek's length) keeps a 4-digit year
    inside a timestamp like "4/11/2026" from being read as the code.
    """
    pat = re.compile(rf"(?<!\d)(\d{{{min_digits},8}}|\d{{3}}-\d{{3}})(?!\d)")
    m = pat.search(text)
    return m.group(1) if m else None


async def _body_text(page: Page) -> str:
    try:
        return await page.evaluate("() => document.body ? document.body.innerText : ''")
    except Exception:  # noqa: BLE001
        return ""


async def _click_new_address(page: Page) -> None:
    for finder in (
        lambda: page.get_by_role("button", name=NEW_ADDRESS_LABEL, exact=False).first,
        lambda: page.get_by_text(NEW_ADDRESS_LABEL, exact=False).first,
    ):
        loc = finder()
        try:
            if await loc.count():
                await loc.click(timeout=3000)
                return
        except Exception:  # noqa: BLE001
            continue


async def create_mailbox(page: Page, *, fresh: bool = False, log: Logger = _noop) -> str:
    """Load mailtemp.vn and return the current temporary address.

    fresh=True clicks "Địa chỉ mới" until the address actually changes, so a
    persisted profile can't hand back a previous (non-empty) mailbox.
    """
    await page.goto(MT_URL, wait_until="networkidle")
    await page.wait_for_timeout(2500)
    address = parse_address(await _body_text(page))

    if fresh:
        prev = address
        for _ in range(4):
            await _click_new_address(page)
            await page.wait_for_timeout(2000)
            address = parse_address(await _body_text(page))
            if address and address != prev:
                break
        else:
            log("  [warn] could not rotate to a new mailtemp address; using current one")

    if not address:
        raise RuntimeError("mailtemp.vn: could not read the mailbox address from the page")
    log(f"mailtemp address: {address}")
    return address


async def _read_code_from_frames(page: Page, min_digits: int) -> str | None:
    for fr in page.frames:
        try:
            txt = await fr.evaluate("() => document.body ? document.body.innerText : ''")
        except Exception:  # noqa: BLE001
            continue
        if not txt.strip():
            continue
        code = extract_code(txt, min_digits=min_digits)
        if code:
            return code
    return None


async def wait_for_code(
    page: Page,
    *,
    subject_hint: str = "verification code",
    timeout: float = 150.0,
    interval: float = 3.0,
    min_digits: int = 6,
    log: Logger = _noop,
) -> str:
    """Poll the inbox, open the newest matching message, read its code."""
    deadline = asyncio.get_event_loop().time() + timeout
    opened = False
    while asyncio.get_event_loop().time() < deadline:
        try:
            r = page.get_by_text(REFRESH_LABEL, exact=False).first
            if await r.count():
                await r.click(timeout=2000)
        except Exception:  # noqa: BLE001
            pass
        await page.wait_for_timeout(1000)

        body = await _body_text(page)
        if EMPTY_MARKER not in body:
            if not opened:
                try:
                    msg = page.get_by_text(subject_hint, exact=False).first
                    if await msg.count():
                        await msg.click(timeout=2500)
                        opened = True
                        await page.wait_for_timeout(1500)
                except Exception:  # noqa: BLE001
                    pass
            code = await _read_code_from_frames(page, min_digits)
            if code:
                log(f"verification code: {code}")
                return code
        await page.wait_for_timeout(interval * 1000)
    raise RuntimeError(f"mailtemp.vn: no verification code within {timeout:.0f}s")
