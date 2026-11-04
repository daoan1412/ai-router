"""Automated DeepSeek email sign-up for ai-router.

Selectors below were captured from the live chat.deepseek.com/sign_in page
(region: "Only email registration is supported"). The flow runs *headed* so the
user can solve a Cloudflare Turnstile if one appears; the verification code is
supplied by ``get_code`` (mailtemp website or API) so this module stays
independent of where the mailbox lives.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from playwright.async_api import Page

from ai_router.browser.pwcompat import PlaywrightTimeout

from ai_router.adapters.deepseek.selectors import DEEPSEEK_URL, SEL_PROMPT_INPUT
from ai_router.adapters.deepseek.wait import is_challenge_visible, read_response_snapshot
from ai_router.streaming import Delta, collect, prefix_delta

SIGN_IN_URL = "https://chat.deepseek.com/sign_in"

# --- Verified sign-up form selectors -----------------------------------------
SEL_EMAIL_INPUT = (
    'input[placeholder="Email address"], '
    'input[placeholder*="email" i], '
    'input[placeholder*="Phone number / email" i]'
)
SEL_PASSWORD_INPUT = 'input[placeholder="Password"], input[type="password"]'
SEL_CONFIRM_INPUT = 'input[placeholder="Confirm password"]'
SEL_CODE_INPUT = (
    'input[placeholder="Code"], input[autocomplete="one-time-code"], input[type="tel"]'
)
SEL_TERMS_CHECKBOX = 'input[type="checkbox"]'

CodeProvider = Callable[[], Awaitable[str]]
Logger = Callable[[str], None]


def _noop(_: str) -> None:  # pragma: no cover - trivial
    pass


async def _logged_in(page: Page) -> bool:
    try:
        await page.wait_for_selector(SEL_PROMPT_INPUT, timeout=2_000)
        return True
    except PlaywrightTimeout:
        return False


async def _fill(page: Page, selector: str, value: str, log: Logger, label: str) -> bool:
    loc = page.locator(selector).first
    try:
        if await loc.count() == 0:
            log(f"  [skip] {label}: field not found")
            return False
        await loc.click(timeout=3_000)
        await loc.fill(value)
        log(f"  filled {label}")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] {label}: {exc}")
        return False


async def _click_text(page: Page, labels: list[str], log: Logger, *, button: bool = False) -> bool:
    for label in labels:
        finders = []
        if button:
            finders.append(lambda ln=label: page.get_by_role("button", name=ln, exact=True))
        finders.append(lambda ln=label: page.get_by_text(ln, exact=False))
        for finder in finders:
            loc = finder().first
            try:
                if await loc.count() == 0:
                    continue
                await loc.click(timeout=3_000)
                log(f"  clicked {label!r}")
                return True
            except Exception:  # noqa: BLE001
                continue
    return False


async def _wait_logged_in(page: Page, log: Logger, timeout: float) -> bool:
    log(
        f"Waiting up to {timeout:.0f}s for login. If a Cloudflare check or extra "
        "step appears in the window, complete it by hand..."
    )
    deadline = asyncio.get_event_loop().time() + timeout
    warned = False
    while asyncio.get_event_loop().time() < deadline:
        if await _logged_in(page):
            return True
        if not warned and await is_challenge_visible(page):
            log("  Cloudflare challenge detected — please solve it in the window.")
            warned = True
        await asyncio.sleep(1.5)
    return False


async def deepseek_sign_up(
    page: Page,
    *,
    email: str,
    password: str,
    get_code: CodeProvider,
    log: Logger = _noop,
    login_timeout: float = 300.0,
) -> str:
    """Sign up a DeepSeek account by email. Returns the outcome string.

    "already_logged_in" | "signed_up" | "needs_manual"
    """
    await page.goto(DEEPSEEK_URL, wait_until="domcontentloaded")
    if await _logged_in(page):
        log("Already logged in — skipping sign-up.")
        return "already_logged_in"

    log("Opening sign-in page...")
    try:
        await page.goto(SIGN_IN_URL, wait_until="domcontentloaded")
        await page.wait_for_timeout(1500)
    except PlaywrightTimeout:
        pass

    # Switch to the sign-up form (reveals Confirm password + Code fields).
    await _click_text(page, ["Sign up"], log)
    await page.wait_for_timeout(1000)

    log(f"Entering email {email} ...")
    await _fill(page, SEL_EMAIL_INPUT, email, log, "email")

    if await _click_text(page, ["Send code", "获取验证码", "Gửi mã"], log):
        log("Requested code — waiting for it to arrive in the mailbox...")
        try:
            code = await get_code()
            await _fill(page, SEL_CODE_INPUT, code, log, "code")
        except Exception as exc:  # noqa: BLE001
            log(f"  [warn] code retrieval failed: {exc} (type it manually if you have it)")
    else:
        log("  [skip] 'Send code' not found — complete the email step manually.")

    log("Setting password...")
    await _fill(page, SEL_PASSWORD_INPUT, password, log, "password")
    await _fill(page, SEL_CONFIRM_INPUT, password, log, "confirm password")

    # Accept terms if a checkbox is present.
    try:
        cb = page.locator(SEL_TERMS_CHECKBOX).first
        if await cb.count() > 0 and not await cb.is_checked():
            await cb.check(timeout=2_000)
            log("  accepted terms")
    except Exception:  # noqa: BLE001
        pass

    # Submit: the in-form "Sign up" button (exact, to avoid the tab link).
    await _click_text(page, ["Sign up"], log, button=True)

    if await _wait_logged_in(page, log, login_timeout):
        log("Logged in successfully.")
        return "signed_up"
    log("Not logged in after automated attempt.")
    return "needs_manual"


# The reasoning/thinking block DeepSeek keeps in the DOM (persists post-answer).
# Only produced when the DeepThink toggle is on.
SEL_THINK = ".ds-think-content, [class*='think-content']"


async def _set_toggle(page: Page, label: str, on: bool, log: Logger = _noop) -> None:
    """Force a composer toggle ("DeepThink" / "Search") on or off.

    Toggle states persist across chats on the account, so they have to be set
    explicitly both ways — otherwise a previous R1 request leaves DeepThink on
    and a later 'deepseek-chat' request still thinks.
    """
    try:
        toggle = page.locator(".ds-toggle-button", has_text=label).first
        if await toggle.count() == 0:
            return
        cls = await toggle.get_attribute("class") or ""
        if ("ds-toggle-button--selected" in cls) != on:
            await toggle.click(timeout=3000)
            await page.wait_for_timeout(400)
            log(f"  {label} {'on' if on else 'off'}")
    except Exception as exc:  # noqa: BLE001
        log(f"  [warn] could not set {label}: {exc}")


async def _read_reasoning(page: Page) -> str | None:
    """Text of the last assistant turn's thinking block, if present."""
    blocks = page.locator(SEL_THINK)
    count = await blocks.count()
    if not count:
        return None
    try:
        text = (await blocks.nth(count - 1).inner_text()).strip()
    except Exception:  # noqa: BLE001
        return None
    return text or None


async def chat_stream(
    page: Page,
    prompt: str,
    *,
    log: Logger = _noop,
    timeout: float = 600.0,
    new_chat: bool = True,
    deep_think: bool = False,
    poll: float = 0.3,
    open_timeout: float = 20.0,
) -> AsyncIterator[Delta]:
    """Send one prompt and yield ("reasoning"|"content", delta) as DeepSeek generates.

    Reads the page's own /completion stream via CDP: DeepThink reasoning
    (THINK fragments) then the answer (RESPONSE fragments), both as raw
    markdown/LaTeX. If that request isn't seen within ``open_timeout``, falls
    back to reading the rendered page (where math comes out as glyphs).
    """
    from ai_router.adapters.deepseek.selectors import DEEPSEEK_COMPLETION_RE, SEL_SUBMIT_BUTTON
    from ai_router.adapters.deepseek.stream import DeepSeekSSEParser
    from ai_router.adapters.deepseek.wait import ensure_new_chat
    from ai_router.browser.sse_tap import CDPStreamTap, SSELineBuffer

    if new_chat:
        await ensure_new_chat(page)
    await _set_toggle(page, "DeepThink", deep_think, log)
    await _set_toggle(page, "Search", False, log)  # never let answers web-search
    count_before, _ = await read_response_snapshot(page)

    tap = CDPStreamTap(page, DEEPSEEK_COMPLETION_RE)
    await tap.start()
    try:
        inp = page.locator(SEL_PROMPT_INPUT).first
        await inp.click()
        await inp.fill("")
        await inp.type(prompt, delay=15)
        log(f"Submitting prompt: {prompt!r}")
        submit = page.locator(SEL_SUBMIT_BUTTON).first
        if await submit.count() > 0 and not await submit.is_disabled():
            await submit.click()
        else:
            await inp.press("Enter")

        opened = await tap.next(open_timeout)
        if opened is None or opened[0] != "open":
            log("  completion stream not seen; reading the rendered page instead")
            async for d in _dom_stream(page, count_before, timeout=timeout, poll=poll):
                yield d
            return

        status = opened[1]
        lines = SSELineBuffer()
        parser = DeepSeekSSEParser()
        raw_head = ""  # for error reporting when the body isn't SSE
        deadline = asyncio.get_event_loop().time() + timeout
        while not parser.done:
            remaining = deadline - asyncio.get_event_loop().time()
            item = await tap.next(remaining) if remaining > 0 else None
            if item is None:
                log("  [warn] DeepSeek stream timed out")
                break
            kind, value = item
            if kind == "text":
                if len(raw_head) < 2000:
                    raw_head += value
                for obj in lines.feed(value):
                    for d in parser.feed(obj):
                        yield d
                if parser.error:
                    raise RuntimeError(f"DeepSeek error: {parser.error}")
            elif kind == "end":
                for obj in lines.flush():
                    for d in parser.feed(obj):
                        yield d
                break
            elif kind == "error":
                log(f"  [warn] stream tap: {value}")

        if status >= 400 or (not parser.got_any and raw_head.strip()):
            raise RuntimeError(f"DeepSeek HTTP {status}: {raw_head.strip()[:300]}")
        if parser.error:
            raise RuntimeError(f"DeepSeek error: {parser.error}")
        if parser.other_types:
            log(f"  [warn] DeepSeek produced non-answer fragments: {sorted(parser.other_types)}")
    finally:
        await tap.stop()


async def _dom_stream(
    page: Page, count_before: int, *, timeout: float, poll: float
) -> AsyncIterator[Delta]:
    """Fallback: stream from the rendered page (.ds-think-content + answer)."""
    from ai_router.adapters.deepseek.wait import is_stop_visible

    sent_content = ""
    sent_reasoning = ""
    last_text = ""
    stable_ticks = 0
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        generating = await is_stop_visible(page)
        sent_reasoning, d = prefix_delta(sent_reasoning, await _read_reasoning(page) or "")
        if d:
            yield ("reasoning", d)
        count, text = await read_response_snapshot(page)
        if count > count_before:
            sent_content, d = prefix_delta(sent_content, text)
            if d:
                yield ("content", d)
            if text and not generating:
                if text == last_text:
                    stable_ticks += 1
                    if stable_ticks >= 4:
                        break
                else:
                    stable_ticks = 0
                    last_text = text
            else:
                stable_ticks = 0
        await asyncio.sleep(poll)

    # Final reconcile: pick up anything appended after the last tick.
    _, d = prefix_delta(sent_reasoning, await _read_reasoning(page) or "")
    if d:
        yield ("reasoning", d)
    count, text = await read_response_snapshot(page)
    if count > count_before:
        _, d = prefix_delta(sent_content, text)
        if d:
            yield ("content", d)


async def chat(
    page: Page,
    prompt: str,
    *,
    log: Logger = _noop,
    timeout: float = 600.0,
    new_chat: bool = True,
    deep_think: bool = False,
) -> tuple[str, str | None]:
    """Send one prompt; return (answer, reasoning).

    reasoning is DeepSeek's thinking block (.ds-think-content), populated when
    deep_think=True enables the DeepThink toggle; otherwise None. answer always
    excludes the thinking block.
    """
    return await collect(
        chat_stream(
            page, prompt, log=log, timeout=timeout, new_chat=new_chat, deep_think=deep_think
        )
    )


async def smoke_chat(page: Page, prompt: str, log: Logger = _noop, timeout: float = 180.0) -> str:
    """Back-compat wrapper used by provision-deepseek: returns just the answer."""
    answer, _ = await chat(page, prompt, log=log, timeout=timeout, new_chat=True)
    return answer
