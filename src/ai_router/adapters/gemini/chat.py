"""Gemini chat for the OpenAI-compatible API: model choice, no memory, live stream.

Each request loads a fresh page and, when signed in, switches to a
**temporary chat**: it doesn't use or update Gemini's memory / personal
context (past chats, saved info, home location) and isn't kept in history —
without touching the account's settings. Image / video generation stay off
(their composer toggles default to off). The answer is read from the
WebChannel backchannel (see stream.py), opened at page load, so the tap starts
before navigation.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable

from playwright.async_api import Page

from ai_router.adapters.base import ModelLockedError
from ai_router.adapters.gemini.account import SEL_MODEL_BUTTON, signed_in
from ai_router.adapters.gemini.selectors import GEMINI_URL, SEL_PROMPT_INPUT
from ai_router.adapters.gemini.stream import GeminiStreamParser, WebChannelBuffer
from ai_router.streaming import Delta

BACKCHANNEL_RE = re.compile(r"/v1/processSession\?.*TYPE=xmlhttp")
TEMP_CHAT_LABELS = ("Cuộc trò chuyện tạm thời", "Temporary chat")

Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


# Seconds without any sign of the turn starting before the prompt is resent.
# Bigger models can think silently for a long time (Argon: up to ~2 min).
STALL_TIMEOUTS = {"Flash-Lite": 30.0, "Flash": 30.0, "Pro": 60.0, "Argon": 150.0}


def stall_timeout_for(model_choice: str | None) -> float:
    return STALL_TIMEOUTS.get(model_choice or "", 60.0)


def model_name(label: str) -> str:
    """Picker label without its version: "3.8 Flash" → "Flash", "4 Argon" → "Argon"."""
    parts = label.split(" ", 1)
    return parts[1] if len(parts) == 2 and parts[0][:1].isdigit() else label


async def enable_temporary_chat(page: Page) -> bool:
    """Switch the fresh chat to temporary; True once its banner shows."""
    sel = ", ".join(f'button[aria-label="{label}"]' for label in TEMP_CHAT_LABELS)
    button = page.locator(sel).first
    try:
        # The button renders a moment after the prompt box.
        await button.wait_for(state="visible", timeout=15_000)
    except Exception:  # noqa: BLE001
        return False
    for _attempt in range(2):
        await button.click()
        for _ in range(20):
            await page.wait_for_timeout(300)
            main = await page.evaluate(
                "() => (document.querySelector('main') || document.body).innerText"
            )
            if any(t in main for t in TEMP_CHAT_LABELS):
                return True
    return False


async def select_model(page: Page, name: str, log: Logger = _noop) -> str:
    """Pick the picker entry whose name (label minus version) is ``name``.

    Returns the full label; ModelLockedError if that entry is disabled.
    """
    button = page.locator(SEL_MODEL_BUTTON).first
    await button.wait_for(state="visible", timeout=15_000)
    await page.keyboard.press("Escape")  # a welcome / promo dialog can cover the picker
    try:
        await button.click(timeout=5_000)
    except Exception:  # noqa: BLE001 - still covered: click it through the DOM
        await page.evaluate("(s) => document.querySelector(s).click()", SEL_MODEL_BUTTON)
    await page.wait_for_timeout(800)
    state = await page.evaluate(
        r"""(name) => {
            const items = [...document.querySelectorAll('[role=menuitem],[role=menuitemradio]')]
                .filter(e => e.offsetParent !== null);
            for (const e of items) {
                const label = (e.innerText || '').split('\n')[0].trim();
                const bare = /^\d/.test(label) ? label.split(' ').slice(1).join(' ') : label;
                if (bare !== name) continue;
                if (e.disabled || e.getAttribute('aria-disabled') === 'true') return {state: 'disabled', label};
                e.click();
                return {state: 'clicked', label};
            }
            return {state: 'missing', label: null};
        }""",
        name,
    )
    await page.wait_for_timeout(600)
    if state["state"] != "clicked":
        await page.keyboard.press("Escape")
        if state["state"] == "disabled":
            raise ModelLockedError(
                f"Gemini model {state['label']!r} needs a signed-in Google account. "
                "Run `ai-router browser login-gemini`."
            )
        raise RuntimeError(f"Gemini model {name!r} not in the picker")
    log(f"  model -> {state['label']}")
    return state["label"]


async def open_chat(
    page: Page,
    *,
    model_choice: str | None = None,
    log: Logger = _noop,
    require_temporary: bool = True,
):
    """Load a fresh (temporary) chat with ``model_choice`` picked; return the
    started backchannel tap for stream_reply. Raises (ModelLockedError, …)
    before anything is sent, so the API can still answer with an HTTP error."""
    from ai_router.browser.sse_tap import CDPStreamTap

    tap = CDPStreamTap(page, BACKCHANNEL_RE, follow=True)
    await tap.start()  # before navigation: the backchannel opens on page load
    try:
        await page.goto(GEMINI_URL, wait_until="domcontentloaded")
        await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=20_000)
        if await signed_in(page):
            if not await enable_temporary_chat(page):
                if require_temporary:
                    raise RuntimeError(
                        "Gemini: could not switch to a temporary chat (memory would be on)"
                    )
                log("  [warn] Gemini temporary chat not enabled")
        if model_choice:
            await select_model(page, model_choice, log)
    except BaseException:
        await tap.stop()
        raise
    return tap


async def chat_stream(
    page: Page,
    prompt: str,
    *,
    model_choice: str | None = None,
    log: Logger = _noop,
    timeout: float = 600.0,
) -> AsyncIterator[Delta]:
    """Fresh (temporary) Gemini chat, one prompt; yield ("reasoning"|"content", delta)."""
    tap = await open_chat(page, model_choice=model_choice, log=log)
    async for d in stream_reply(
        page, tap, prompt, model_choice=model_choice, log=log, timeout=timeout
    ):
        yield d


async def stream_reply(
    page: Page,
    tap,
    prompt: str,
    *,
    model_choice: str | None = None,
    log: Logger = _noop,
    timeout: float = 600.0,
    stall_timeout: float | None = None,
    retries: int = 2,
) -> AsyncIterator[Delta]:
    """Send ``prompt`` in the chat open_chat prepared and stream the reply.

    Gemini sometimes accepts a prompt and then sends nothing. If no progress —
    status title, thinking or answer — shows within ``stall_timeout`` s (per
    model by default, see STALL_TIMEOUTS), a fresh temporary chat is opened
    and the prompt resent, up to ``retries`` times.
    """
    from ai_router.adapters.gemini.wait import type_prompt
    from ai_router.logger import trace

    if stall_timeout is None:
        stall_timeout = stall_timeout_for(model_choice)
    try:
        for attempt in range(retries + 1):
            if attempt:
                log(f"  [warn] Gemini sent nothing for {stall_timeout:.0f}s; resending")
                trace("gemini_resend", attempt=attempt)
                await tap.stop()
                tap = await open_chat(page, model_choice=model_choice, log=log)
            await type_prompt(page, prompt)
            await page.wait_for_timeout(300)  # let the editor settle before sending
            log(f"Submitting prompt: {prompt[:80]!r}")
            await page.locator(SEL_PROMPT_INPUT).first.press("Enter")
            trace("gemini_sent", attempt=attempt)
            stall_at = asyncio.get_event_loop().time() + stall_timeout
            parser = None
            async for d in _follow(tap, stall_at, timeout, log, trace):
                if isinstance(d, GeminiStreamParser):
                    parser = d
                else:
                    yield d
            if parser is not None:
                break
        else:
            raise RuntimeError(f"Gemini: no reply after {retries + 1} attempts")
        if not parser.got_any:
            raise RuntimeError("Gemini: no answer seen on the backchannel")
        if parser.chips:
            log(f"  [warn] Gemini used rich content: {sorted(parser.chips)}")
    finally:
        await tap.stop()


async def _follow(tap, stall_at: float, timeout: float, log: Logger, trace):
    """Yield deltas from the backchannel, then the finished parser. Ends without
    the parser if the turn never started before ``stall_at`` (caller resends)."""
    loop = asyncio.get_event_loop()
    frames = WebChannelBuffer()
    parser = GeminiStreamParser()
    deadline = loop.time() + timeout
    while not parser.done:
        limit = deadline if parser.progress else min(deadline, stall_at)
        remaining = limit - loop.time()
        item = await tap.next(remaining) if remaining > 0 else None
        if item is None:
            if not parser.progress:
                return  # stalled: let the caller resend
            log("  [warn] Gemini stream timed out")
            break
        kind, value = item
        if kind == "open":
            trace("gemini_backchannel_open", status=value)
            frames = WebChannelBuffer()  # a (re)connected backchannel
            if value >= 400:
                raise RuntimeError(f"Gemini backchannel HTTP {value}")
        elif kind == "text":
            for seq, payload in frames.feed(value):
                deltas = parser.feed(payload)
                trace(
                    "gemini_frame", seq=seq, answer_len=len(parser._answer),
                    thinking_len=len(parser._thinking), done=parser.done,
                )
                for d in deltas:
                    yield d
        elif kind == "error":
            log(f"  [warn] stream tap: {value}")
        # "end": WebChannel reconnects its backchannel; keep following.
    for d in parser.finish():
        yield d
    yield parser
