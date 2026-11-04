"""Gemini account state: signed in or guest, and which models the picker unlocks.

A guest still gets the prompt box (and 3.5 Flash-Lite), so "prompt box present"
is not a login signal: signed in = the Google account switcher link is there,
guest = a "Sign in" link is.
"""

from __future__ import annotations

import asyncio

from playwright.async_api import Page

from ai_router.adapters.gemini.selectors import GEMINI_URL, SEL_SIGN_IN

# Login link the user opens in their own browser; lands on Gemini once signed in.
GEMINI_LOGIN_URL = (
    "https://accounts.google.com/ServiceLogin?continue=https%3A%2F%2Fgemini.google.com%2Fapp"
)
SEL_ACCOUNT = 'a[href*="accounts.google.com/SignOutOptions"]'
SEL_MODEL_BUTTON = '[data-test-id="bard-mode-menu-button"]'


async def signed_in(page: Page, timeout: float = 10.0) -> bool:
    """True once the account switcher shows, False once a "Sign in" link does."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if await page.locator(SEL_ACCOUNT).count():
            return True
        if await page.locator(SEL_SIGN_IN).count():
            return False
        await page.wait_for_timeout(500)
    return False


async def list_models(page: Page) -> list[dict]:
    """Picker entries: ``{"label", "description", "available"}`` (the guest
    "Sign in / Get more access" entry and settings entries are dropped)."""
    button = page.locator(SEL_MODEL_BUTTON).first
    try:
        await button.wait_for(state="visible", timeout=15_000)
    except Exception:  # noqa: BLE001
        return []
    await page.keyboard.press("Escape")  # a welcome / promo dialog can cover the picker
    try:
        await button.click(timeout=5_000)
    except Exception:  # noqa: BLE001 - still covered: click it through the DOM
        await page.evaluate("(s) => document.querySelector(s).click()", SEL_MODEL_BUTTON)
    await page.wait_for_timeout(1200)
    items = await page.evaluate(
        r"""() => [...document.querySelectorAll('[role=menuitem],[role=menuitemradio]')]
            .filter(e => e.offsetParent !== null)
            .map(e => ({lines: (e.innerText || '').split('\n').map(s => s.trim()).filter(Boolean),
                        disabled: e.disabled || e.getAttribute('aria-disabled') === 'true'}))"""
    )
    await page.keyboard.press("Escape")
    out = []
    for it in items:
        lines = it["lines"]
        # Models are labelled by version ("3.5 Flash-Lite", "4 Argon"); this
        # drops "Sign in" and settings entries such as the thinking-level submenu.
        if not lines or not lines[0][:1].isdigit():
            continue
        out.append({
            "label": lines[0],
            "description": " ".join(lines[1:]),
            "available": not it["disabled"],
        })
    return out


async def account_report(page: Page) -> tuple[bool, list[dict]]:
    """Load Gemini and return (signed_in, models)."""
    await page.goto(GEMINI_URL, wait_until="domcontentloaded")
    is_in = await signed_in(page)
    return is_in, await list_models(page)
